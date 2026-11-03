"""The search service endpoints of the ingester API over a seeded database."""

import httpx
import pytest
from httpx import ASGITransport

from haiku.rag.config.models import AppConfig, LanceDBConfig
from haiku.rag.ingester.api.server import APIState, build_app
from haiku.rag.ingester.queue.models import JobOp, JobStatus
from haiku.rag.service.search import SearchService
from haiku.rag.service.throttle import Conditions, Throttle
from tests.ingester.conftest import finished_job
from tests.service.test_search import _seed, constant_embedder  # noqa: F401


@pytest.fixture
async def state(tmp_path, jobs, sync):
    config = AppConfig(
        lancedb=LanceDBConfig(databases={"files": str(tmp_path / "files.lancedb")})
    )
    ids = await _seed(tmp_path, config)
    search = SearchService(config)
    state = APIState(
        config=config,
        job_repo=jobs,
        sync_repo=sync,
        source_databases={"guitar": "files"},
        search=search,
    )
    state.ids = ids  # ty: ignore[unresolved-attribute]
    yield state
    await search.aclose()


def _client(state: APIState) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=ASGITransport(app=build_app(state)), base_url="http://testserver"
    )


async def test_search(state):
    async with _client(state) as client:
        response = await client.get(
            "/search",
            params={
                "q": "travis picking",
                "type": "fts",
                "folder": "guitar",
                "limit": 5,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "travis picking"
    assert {hit["page_numbers"][0] for hit in body["hits"]} == {40, 41}
    hit = next(h for h in body["hits"] if h["page_numbers"] == [40])
    assert hit["pages"][0]["boxes"] == [
        {"left": 50.0, "top": 100.0, "right": 550.0, "bottom": 200.0}
    ]
    assert hit["snippet_highlights"][0] == [0, 6]


async def test_search_refuses_an_unknown_database(state):
    async with _client(state) as client:
        response = await client.get("/search", params={"q": "x", "database": "nope"})
    assert response.status_code == 400


async def test_search_within_a_document(state):
    notes = state.ids["notes"]
    async with _client(state) as client:
        response = await client.get(
            f"/documents/{notes}/search",
            params={"q": "travis", "type": "fts", "database": "files", "context": True},
        )
        everywhere = await client.get(
            f"/documents/{notes}/search", params={"q": "travis", "type": "fts"}
        )
    [hit] = response.json()["hits"]
    assert hit["document_id"] == notes
    assert hit["line"] == 4
    assert "Warm up slowly." in hit["context"]
    assert [h["document_id"] for h in everywhere.json()["hits"]] == [notes]


async def test_documents_and_outlines(tmp_path, state):
    book = state.ids["book"]
    async with _client(state) as client:
        document = (await client.get(f"/documents/{book}")).json()
        outline = (
            await client.get(f"/documents/{book}/outline", params={"database": "files"})
        ).json()
        missing = await client.get("/documents/nope")
        unknown = await client.get(f"/documents/{book}", params={"database": "nope"})
    assert document["document_id"] == book
    assert document["database"] == "files"
    assert document["folder_id"] == "guitar"
    assert document["path"] == str(tmp_path / "Country Guitar.pdf")
    assert document["document_type"] == "pdf"
    assert document["content_type"] == "application/pdf"
    assert outline == {"document_id": book, "outline": []}
    assert missing.status_code == 404
    assert unknown.status_code == 400


async def test_an_outline_nests_headings(tmp_path, jobs, sync):
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel

    from haiku.rag.client import HaikuRAG
    from haiku.rag.store.models.chunk import Chunk

    config = AppConfig(
        lancedb=LanceDBConfig(databases={"f": str(tmp_path / "f.lancedb")})
    )
    doc = DoclingDocument(name="h")
    doc.add_heading("Chapter 1", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="intro")
    doc.add_heading("Section 1.1", level=2)
    doc.add_text(label=DocItemLabel.TEXT, text="body")
    async with HaikuRAG(config=config, create=True, sources=["f"]) as rag:
        stored = await rag.import_document(
            doc,
            [
                Chunk(
                    content="intro",
                    embedding=[0.1] * config.embeddings.model.vector_dim,
                )
            ],
            uri="file:///h.md",
        )
    search = SearchService(config)
    state = APIState(config=config, job_repo=jobs, sync_repo=sync, search=search)
    try:
        async with _client(state) as client:
            outline = (await client.get(f"/documents/{stored.id}/outline")).json()
    finally:
        await search.aclose()
    [chapter] = outline["outline"]
    assert chapter["title"] == "Chapter 1"
    assert chapter["level"] == 1
    assert [child["title"] for child in chapter["children"]] == ["Section 1.1"]


async def test_status(state, jobs):
    await finished_job(jobs, JobStatus.DEAD, "file:///d.pdf", source_id="guitar")
    await jobs.enqueue("guitar", "file:///q.pdf", JobOp.UPSERT)
    throttle = Throttle(state.config.ingester.service.throttle)
    throttle._conditions = Conditions(load_per_core=99.0)
    state.throttle = throttle
    async with _client(state) as client:
        body = (await client.get("/status")).json()
    assert body == {
        "indexing": True,
        "queued": 1,
        "in_flight": 0,
        "failed": 1,
        "paused": True,
        "pause_reason": "system busy",
        "embedder_loaded": True,
        "databases": [{"name": "files", "documents": 2}],
    }


async def test_status_without_a_throttle(state):
    async with _client(state) as client:
        body = (await client.get("/status")).json()
    assert body["paused"] is False
    assert body["pause_reason"] is None
    assert body["indexing"] is False


async def test_failures_are_listed_and_retried(state, jobs):
    first = await finished_job(
        jobs, JobStatus.DEAD, "file:///a.pdf", source_id="guitar", error="timed out"
    )
    second = await finished_job(
        jobs, JobStatus.DEAD, "file:///b.pdf", source_id="guitar", error="crashed"
    )
    unserved = await finished_job(
        jobs, JobStatus.DEAD, "https://x/c", source_id="web", error="404"
    )
    async with _client(state) as client:
        listed = (await client.get("/failures")).json()["failures"]
        only = (await client.get("/failures", params={"folder": "web"})).json()
        one = await client.post(f"/failures/{first.id}/retry")
        rest = await client.post("/failures/retry")

    by_id = {f["job_id"]: f for f in listed}
    assert by_id[first.id]["path"] is not None
    assert by_id[first.id]["reason"] == "timed out"
    assert by_id[first.id]["database"] == "files"
    assert by_id[first.id]["folder_id"] == "guitar"
    assert by_id[unserved.id]["path"] is None
    assert by_id[unserved.id]["database"] is None
    assert [f["job_id"] for f in only["failures"]] == [unserved.id]
    assert one.json() == {"retried": [first.id]}
    # The unserved source's job has no worker to claim it, so it stays.
    assert rest.json() == {"retried": [second.id]}


async def test_search_needs_the_service(jobs, sync):
    state = APIState(config=AppConfig(), job_repo=jobs, sync_repo=sync)
    async with _client(state) as client:
        response = await client.get("/search", params={"q": "x"})
    assert response.status_code == 503
