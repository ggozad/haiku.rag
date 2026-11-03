import hashlib
import logging

import httpx
import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.client import HaikuRAG
from haiku.rag.config import HTTPSourceConfig
from haiku.rag.ingester.exceptions import EmptyListingError
from haiku.rag.ingester.pollers.periodic import PeriodicPoller
from haiku.rag.ingester.queue.models import JobOp, JobStatus
from haiku.rag.ingester.reconcile import reconcile
from haiku.rag.ingester.workers.pool import WorkerPool
from haiku.rag.sources.http import HTTPSource
from haiku.rag.store.models.chunk import Chunk
from tests.conftest import capture_logs

_POLLER_LOGGER = logging.getLogger("haiku.rag.ingester.pollers.base")

SOURCE = "docs"
GONE = "https://example.com/gone.md"
KEPT = "https://example.com/kept.md"


@pytest.fixture
async def client(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        yield rag


async def _stored(client: HaikuRAG, uri: str) -> str:
    doc = DoclingDocument(name="lifecycle")
    doc.add_text(label=DocItemLabel.TEXT, text=f"Content of {uri}.")
    vector_dim = client.store.embedder.vector_dim
    stored = await client.import_document(
        doc,
        [Chunk(content=f"Content of {uri}.", embedding=[0.1] * vector_dim)],
        uri=uri,
    )
    assert stored.id is not None
    await client.set_document_source([stored.id], SOURCE)
    return stored.id


def _sweeper(client, jobs, sync, handler):
    """Sweeps GONE and KEPT, drains the queue and returns the sweep's jobs."""
    http = HTTPSource(
        source_id=SOURCE, urls=[GONE, KEPT], transport=httpx.MockTransport(handler)
    )
    config = HTTPSourceConfig(type="http", id=SOURCE, urls=[GONE, KEPT])
    poller = PeriodicPoller(source=http, config=config, job_repo=jobs, sync_repo=sync)
    pool = WorkerPool(
        clients={SOURCE: client}, job_repo=jobs, sync_repo=sync, sources=[http]
    )

    async def sweep_and_drain() -> list[tuple[JobOp, JobStatus]]:
        before = {job.id for job in await jobs.list_jobs(source_id=SOURCE)}
        assert await poller._sweep_once() is True
        while await pool.drain_once():
            pass
        return [
            (job.op, job.status)
            for job in await jobs.list_jobs(source_id=SOURCE)
            if job.id not in before
        ]

    return sweep_and_drain


async def test_a_url_that_stays_404_is_fetched_once_and_deleted(client, jobs, sync):
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, str(request.url)))
        if str(request.url) == KEPT:
            return httpx.Response(200, headers={"etag": '"k1"'})
        return httpx.Response(404)

    sweep_and_drain = _sweeper(client, jobs, sync, handler)
    for uri in (GONE, KEPT):
        await _stored(client, uri)
    await sync.upsert(SOURCE, GONE, revision="g1", ingested=True)
    await sync.upsert(SOURCE, KEPT, revision="k1", ingested=True)

    assert await sweep_and_drain() == [(JobOp.UPSERT, JobStatus.DEAD)]
    assert await client.get_document_by_uri(GONE) is not None

    assert await sweep_and_drain() == [(JobOp.DELETE, JobStatus.SUCCEEDED)]
    assert await client.get_document_by_uri(GONE) is None
    assert await client.get_document_by_uri(KEPT) is not None

    await reconcile(client, sync, [SOURCE])
    assert (await sync.get_revision_snapshot(SOURCE))[GONE] == "http-404"

    assert await sweep_and_drain() == [(JobOp.DELETE, JobStatus.SUCCEEDED)]
    assert [r for r in requests if r[0] == "GET"] == [("GET", GONE)]


async def test_a_url_that_recovers_without_validators_is_fetched_once(
    client, jobs, sync
):
    body = b"Content of the page."
    status = {"code": 404}
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, str(request.url)))
        if str(request.url) == KEPT:
            return httpx.Response(200, headers={"etag": '"k1"'})
        if status["code"] == 404:
            return httpx.Response(404)
        return httpx.Response(
            200, headers={"content-type": "text/markdown"}, content=body
        )

    sweep_and_drain = _sweeper(client, jobs, sync, handler)
    doc_id = await _stored(client, GONE)
    await _stored(client, KEPT)
    doc = await client.get_document_by_id(doc_id)
    assert doc is not None
    doc.metadata = {
        **doc.metadata,
        "md5": hashlib.md5(body, usedforsecurity=False).hexdigest(),
    }
    await client.document_repository.update_meta(doc)
    await sync.upsert(SOURCE, GONE, ingested=True)
    await sync.upsert(SOURCE, KEPT, revision="k1", ingested=True)

    assert await sweep_and_drain() == [(JobOp.UPSERT, JobStatus.DEAD)]
    assert (await sync.get_revision_snapshot(SOURCE))[GONE] == "http-404"

    status["code"] = 200
    assert await sweep_and_drain() == [(JobOp.UPSERT, JobStatus.SUCCEEDED)]
    assert GONE not in await sync.get_revision_snapshot(SOURCE)

    assert await sweep_and_drain() == []
    assert [r for r in requests if r[0] == "GET"] == [("GET", GONE), ("GET", GONE)]


def _always_404(client, jobs, sync):
    """A poller over one URL answering 404, and the pool draining its jobs."""
    http = HTTPSource(
        source_id=SOURCE,
        urls=[GONE],
        transport=httpx.MockTransport(lambda request: httpx.Response(404)),
    )
    config = HTTPSourceConfig(type="http", id=SOURCE, urls=[GONE])
    poller = PeriodicPoller(source=http, config=config, job_repo=jobs, sync_repo=sync)
    poller.documents = client
    pool = WorkerPool(
        clients={SOURCE: client}, job_repo=jobs, sync_repo=sync, sources=[http]
    )
    return poller, pool


async def test_a_lone_url_that_was_always_404_leaves_the_dead_letter_queue(
    client, jobs, sync
):
    """Every URL is gone, but none has a document, so the sweep proceeds."""
    poller, pool = _always_404(client, jobs, sync)

    for _ in range(2):
        assert await poller._sweep_once() is True
        while await pool.drain_once():
            pass

    assert [(job.op, job.status) for job in await jobs.list_jobs(source_id=SOURCE)] == [
        (JobOp.DELETE, JobStatus.SUCCEEDED)
    ]
    row = await sync.get_row(SOURCE, GONE)
    assert row is not None
    assert (row.revision, row.last_ingested_at) == ("http-404", None)


async def test_a_lone_gone_url_with_a_document_fails_the_sweep(client, jobs, sync):
    await _stored(client, GONE)
    await sync.upsert(SOURCE, GONE, revision="http-404", ingested=True)
    poller, _pool = _always_404(client, jobs, sync)

    with capture_logs(_POLLER_LOGGER, logging.ERROR):
        assert await poller._sweep_once() is False

    assert await jobs.list_jobs(source_id=SOURCE) == []
    assert await client.get_document_by_uri(GONE) is not None
    assert poller.last_error == str(EmptyListingError(SOURCE, 1, documents=True))
