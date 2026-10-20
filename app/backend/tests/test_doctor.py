import asyncio
import threading

import httpx

from haiku.rag.config.models import AppConfig, DatabaseEntry, LanceDBConfig
from haiku_rag_app import doctor
from haiku_rag_app.main import create_app


async def _settled(client, source: str, kind_path: str = "") -> dict:
    for _ in range(600):
        body = (await client.get(f"/api/doctor/{source}{kind_path}")).json()
        run = body.get("run") or body.get("projection")
        if run and run["status"] in ("done", "failed"):
            return run
        await asyncio.sleep(0.02)
    raise AssertionError("the job did not settle")


async def test_a_doctor_run_reports_its_checks_and_progress(client):
    assert (await client.get("/api/doctor/alpha")).json() == {"run": None}
    started = await client.post("/api/doctor/alpha/run", params={"providers": "false"})
    assert started.status_code == 202
    assert started.json()["status"] == "queued"
    run = await _settled(client, "alpha")
    assert run["status"] == "done"
    names = {check["name"] for check in run["result"]["results"]}
    assert {"tables_present", "duplicate_documents", "unembedded_chunks"} <= names
    assert "api_keys" not in names
    assert "Scanning chunk vectors" in run["progress"]
    assert run["started_at"] and run["finished_at"]


async def test_jobs_run_one_at_a_time_and_a_duplicate_is_refused(
    app, client, monkeypatch
):
    release = threading.Event()

    async def held(*args, **kwargs):
        release.wait(5)
        from haiku.rag.doctor import DoctorReport

        return DoctorReport()

    monkeypatch.setattr(doctor, "run_doctor", held)
    await client.post("/api/doctor/alpha/run")
    for _ in range(100):
        if (await client.get("/api/doctor/alpha")).json()["run"]["status"] == "running":
            break
        await asyncio.sleep(0.01)
    again = await client.post("/api/doctor/alpha/run")
    assert again.status_code == 409
    queued = await client.post("/api/doctor/beta/run")
    assert queued.status_code == 202
    assert (await client.get("/api/doctor/beta")).json()["run"]["status"] == "queued"
    release.set()
    assert (await _settled(client, "alpha"))["status"] == "done"
    assert (await _settled(client, "beta"))["status"] == "done"


async def test_the_request_loop_answers_during_a_blocking_job(client, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    async def blocking(*args, **kwargs):
        entered.set()
        release.wait(5)
        from haiku.rag.doctor import DoctorReport

        return DoctorReport()

    monkeypatch.setattr(doctor, "run_doctor", blocking)
    await client.post("/api/doctor/alpha/run")
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert (await client.get("/health")).status_code == 200
        state = await client.get("/api/doctor/alpha")
        assert state.status_code == 200
        assert state.json()["run"]["status"] == "running"
    finally:
        release.set()
        assert (await _settled(client, "alpha"))["status"] == "done"


async def test_a_failing_run_is_recorded_with_its_error(client, monkeypatch):
    async def failing(*args, **kwargs):
        raise RuntimeError("vectors unreadable")

    monkeypatch.setattr(doctor, "run_doctor", failing)
    await client.post("/api/doctor/alpha/run")
    run = await _settled(client, "alpha")
    assert run["status"] == "failed"
    assert run["error"] == "vectors unreadable"


async def test_an_unknown_database_is_not_found(client):
    assert (await client.get("/api/doctor/gamma")).status_code == 404
    assert (await client.post("/api/doctor/gamma/run")).status_code == 404


async def test_a_database_read_at_a_tag_is_refused(tmp_path):
    config = AppConfig(
        lancedb=LanceDBConfig(
            databases={
                "pinned": DatabaseEntry(
                    location=str(tmp_path / "pinned.lancedb"), tag="release-1"
                )
            }
        )
    )
    app = create_app(config, sessions_path=tmp_path / "sessions.db")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as http:
        response = await http.post("/api/doctor/pinned/run")
    assert response.status_code == 400
    assert "live state" in response.json()["error"]
