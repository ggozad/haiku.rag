from datetime import UTC, datetime
from unittest.mock import MagicMock

import httpx
import pytest
from httpx import ASGITransport

from haiku.rag.config.models import AppConfig, IngesterConfig, LanceDBConfig
from haiku.rag.ingester.api.server import APIState, build_app
from haiku.rag.ingester.queue.models import JobOp, JobStatus
from tests.ingester.conftest import finished_job


@pytest.fixture
def state(tmp_path, jobs, sync):
    config = AppConfig(
        lancedb=LanceDBConfig(databases={"files": str(tmp_path / "files.lancedb")}),
        ingester=IngesterConfig(folders_register=tmp_path / "folders.yaml"),
    )
    return APIState(config=config, job_repo=jobs, sync_repo=sync)


def _client(state: APIState) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=ASGITransport(app=build_app(state)), base_url="http://testserver"
    )


async def test_folders_round_trip(tmp_path, state):
    books = tmp_path / "books"
    books.mkdir()
    reloads = []
    state.request_reload = lambda: reloads.append(1)

    async with _client(state) as client:
        assert (await client.get("/folders")).json() == {"folders": []}

        created = await client.post(
            "/folders", json={"path": str(books), "include": ["*.pdf"]}
        )
        assert created.status_code == 201
        assert created.json() == {
            "id": "books",
            "database": "files",
            "path": str(books.resolve()),
            "include": ["*.pdf"],
            "exclude": [],
            "status": None,
        }

        changed = await client.patch("/folders/books", json={"exclude": ["old/"]})
        assert changed.json()["exclude"] == ["old/"]
        assert changed.json()["include"] == ["*.pdf"]

        listed = await client.get("/folders")
        assert [f["id"] for f in listed.json()["folders"]] == ["books"]

        removed = await client.delete("/folders/books", params={"keep_documents": True})
        assert removed.json() == {"removed": "books"}
    assert reloads == [1, 1, 1]


async def test_a_served_folder_reports_its_status(tmp_path, state, jobs, sync):
    books = tmp_path / "books"
    books.mkdir()
    state.source_databases = {"books": "files"}
    poller = MagicMock(
        source_id="books", last_polled_at=datetime(2026, 11, 3, tzinfo=UTC)
    )
    state.pollers = MagicMock(pollers=[MagicMock(source_id="other"), poller])
    await finished_job(jobs, JobStatus.DEAD, "file:///d.pdf", source_id="books")
    await jobs.enqueue("books", "file:///q.pdf", JobOp.UPSERT)
    await sync.upsert("books", "file:///a.pdf", revision="r1", ingested=True)

    async with _client(state) as client:
        await client.post("/folders", json={"path": str(books)})
        [folder] = (await client.get("/folders")).json()["folders"]

    assert folder["status"] == {
        "watching": True,
        "documents": 1,
        "queued": 1,
        "failed": 1,
        "last_scan_at": "2026-11-03T00:00:00Z",
    }


async def test_a_served_folder_without_pollers_has_no_scan_time(tmp_path, state):
    books = tmp_path / "books"
    books.mkdir()
    state.source_databases = {"books": "files"}
    async with _client(state) as client:
        await client.post("/folders", json={"path": str(books)})
        [folder] = (await client.get("/folders")).json()["folders"]
    assert folder["status"]["last_scan_at"] is None


async def test_folder_errors(tmp_path, state):
    books = tmp_path / "books"
    books.mkdir()
    async with _client(state) as client:
        await client.post("/folders", json={"path": str(books)})
        conflict = await client.post("/folders", json={"path": str(books)})
        assert conflict.status_code == 409
        missing = await client.post("/folders", json={"path": str(tmp_path / "x")})
        assert missing.status_code == 400
        assert (await client.patch("/folders/nope", json={})).status_code == 404
        assert (await client.delete("/folders/nope")).status_code == 404


async def test_folders_need_a_register(jobs, sync):
    state = APIState(config=AppConfig(), job_repo=jobs, sync_repo=sync)
    async with _client(state) as client:
        response = await client.get("/folders")
    assert response.status_code == 503
    assert "folders_register" in response.json()["detail"]
