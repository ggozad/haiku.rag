import asyncio
import threading

from haiku.rag.config.models import AppConfig, DatabaseEntry, LanceDBConfig
from haiku_rag_app import doctor
from tests.support import served, settled


async def _running(client, source: str) -> dict:
    # A job's status changes on the request loop after its work has started.
    for _ in range(100):
        run = (await client.get(f"/api/doctor/{source}")).json()["run"]
        if run["status"] == "running":
            return run
        await asyncio.sleep(0.01)
    raise AssertionError("the job did not start")


async def test_a_doctor_run_reports_its_checks_and_progress(client):
    assert (await client.get("/api/doctor/alpha")).json() == {"run": None}
    started = await client.post("/api/doctor/alpha/run", params={"providers": "false"})
    assert started.status_code == 202
    assert started.json()["status"] == "queued"
    run = await settled(client, "alpha")
    assert run["status"] == "done"
    names = {check["name"] for check in run["result"]["results"]}
    assert {"tables_present", "duplicate_documents", "unembedded_chunks"} <= names
    assert "api_keys" not in names
    assert "Scanning chunk vectors" in run["progress"]
    assert run["started_at"] and run["finished_at"]


async def test_blocking_jobs_run_one_at_a_time_off_the_request_loop(
    client, monkeypatch
):
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
        await _running(client, "alpha")
        again = await client.post("/api/doctor/alpha/run")
        assert again.status_code == 409
        queued = await client.post("/api/doctor/beta/run")
        assert queued.status_code == 202
        beta = (await client.get("/api/doctor/beta")).json()["run"]
        assert beta["status"] == "queued"
    finally:
        release.set()
        assert (await settled(client, "alpha"))["status"] == "done"
    assert (await settled(client, "beta"))["status"] == "done"


async def test_a_failing_run_is_recorded_with_its_error(client, monkeypatch):
    async def failing(*args, **kwargs):
        raise RuntimeError("vectors unreadable")

    monkeypatch.setattr(doctor, "run_doctor", failing)
    await client.post("/api/doctor/alpha/run")
    run = await settled(client, "alpha")
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
    async with served(config, tmp_path / "sessions.db") as (_, http):
        response = await http.post("/api/doctor/pinned/run")
    assert response.status_code == 400
    assert "live state" in response.json()["error"]
