"""End-to-end ingester tests: poller -> queue -> worker -> sync_state."""

import asyncio
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config import FSSourceConfig, HTTPSourceConfig
from haiku.rag.ingester.pollers.manager import PollerManager
from haiku.rag.ingester.queue.migrations import apply_migrations
from haiku.rag.ingester.queue.models import JobOp
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo
from haiku.rag.ingester.sources.http import HTTPSource
from haiku.rag.ingester.workers.pool import WorkerPool
from haiku.rag.store.models.document import Document


@pytest.fixture
async def conn(tmp_path):
    path = tmp_path / "queue.db"
    connection = await aiosqlite.connect(str(path))
    connection.row_factory = aiosqlite.Row
    await apply_migrations(connection)
    yield connection
    await connection.close()


@pytest.fixture
def queue_lock():
    return asyncio.Lock()


@pytest.fixture
def jobs(conn, queue_lock):
    return JobRepo(conn, lock=queue_lock)


@pytest.fixture
def sync(conn, queue_lock):
    return SyncStateRepo(conn, lock=queue_lock)


async def _wait_for(predicate, *, timeout: float = 5.0, interval: float = 0.05):
    """Poll `predicate` until it returns truthy or `timeout` elapses."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        result = (
            await predicate() if asyncio.iscoroutinefunction(predicate) else predicate()
        )
        if result:
            return result
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"predicate never became truthy within {timeout}s")
        await asyncio.sleep(interval)


def _mock_client(docs_root) -> AsyncMock:
    """A HaikuRAG mock that returns a fresh Document for each URI it's asked
    to ingest, mirroring real metadata shape (content_type + md5)."""
    client = AsyncMock(spec=HaikuRAG)

    counter = {"n": 0}

    async def _fake_create(uri, *_, metadata=None, **__):
        counter["n"] += 1
        return Document(
            id=f"doc-{counter['n']}",
            content="x",
            uri=uri,
            metadata={"content_type": "text/markdown", "md5": f"md5-{counter['n']}"},
        )

    client.create_document_from_source.side_effect = _fake_create
    return client


@pytest.mark.asyncio
async def test_e2e_initial_sweep_lands_succeeded_jobs(tmp_path, jobs, sync):
    """PollerManager + WorkerPool together: a file on disk at startup becomes
    a succeeded queue row and a sync_state entry."""
    (tmp_path / "a.md").write_text("hello")
    (tmp_path / "b.md").write_text("world")

    client = _mock_client(tmp_path)
    cfg = FSSourceConfig(
        type="fs",
        id="local",
        root=tmp_path,
        poll_interval_s=60.0,
    )

    manager = PollerManager(
        configs=[cfg],
        job_repo=jobs,
        sync_repo=sync,
        supported_extensions=[".md"],
    )
    pool = WorkerPool(
        client=client,
        job_repo=jobs,
        sync_repo=sync,
        worker_count=2,
        poll_idle_interval_s=0.05,
    )

    await pool.start()
    await manager.start()
    try:

        async def _two_succeeded() -> bool:
            counts = await jobs.counts_by_status()
            return counts.get("succeeded", 0) == 2

        await _wait_for(_two_succeeded, timeout=5.0)
    finally:
        await manager.stop()
        await pool.stop()

    counts = await jobs.counts_by_status()
    assert counts.get("succeeded", 0) == 2
    assert counts.get("queued", 0) == 0
    assert counts.get("dead", 0) == 0

    # The worker called create_document_from_source exactly twice — once per file.
    assert client.create_document_from_source.await_count == 2
    ingested_uris = {
        call.args[0] for call in client.create_document_from_source.await_args_list
    }
    assert ingested_uris == {
        (tmp_path / "a.md").as_uri(),
        (tmp_path / "b.md").as_uri(),
    }

    # sync_state holds last_seen_at + content_hash for each URI.
    row_a = await sync.get_row("local", (tmp_path / "a.md").as_uri())
    row_b = await sync.get_row("local", (tmp_path / "b.md").as_uri())
    assert row_a is not None and row_a.content_hash and row_a.last_ingested_at
    assert row_b is not None and row_b.content_hash and row_b.last_ingested_at


@pytest.mark.asyncio
async def test_e2e_handles_url_encoded_special_chars_in_path(tmp_path, jobs, sync):
    """File names containing characters that path.as_uri() URL-encodes (e.g.
    Next.js dynamic-route brackets like `[chunk_id]`) must survive the
    round-trip through the job queue without tripping the existence check
    inside create_document_from_source."""
    bracketed_dir = tmp_path / "[chunk_id]"
    bracketed_dir.mkdir()
    target = bracketed_dir / "route.ts"
    target.write_text("export default {};")

    client = _mock_client(tmp_path)
    cfg = FSSourceConfig(
        type="fs",
        id="local",
        root=tmp_path,
        poll_interval_s=60.0,
    )

    manager = PollerManager(
        configs=[cfg],
        job_repo=jobs,
        sync_repo=sync,
        supported_extensions=[".ts"],
    )
    pool = WorkerPool(
        client=client,
        job_repo=jobs,
        sync_repo=sync,
        worker_count=1,
        poll_idle_interval_s=0.05,
    )

    await pool.start()
    await manager.start()
    try:

        async def _one_succeeded() -> bool:
            counts = await jobs.counts_by_status()
            return counts.get("succeeded", 0) == 1

        await _wait_for(_one_succeeded, timeout=5.0)
    finally:
        await manager.stop()
        await pool.stop()

    counts = await jobs.counts_by_status()
    assert counts.get("succeeded", 0) == 1
    assert counts.get("dead", 0) == 0  # no PermanentError("File does not exist")

    # The URI in the queue is URL-encoded; the worker still finds the file.
    [call] = client.create_document_from_source.await_args_list
    assert "%5Bchunk_id%5D" in call.args[0]


@pytest.mark.asyncio
async def test_e2e_watchfiles_push_event_lands_as_job(tmp_path, jobs, sync):
    """FSPoller's watchfiles loop: a file *added* after startup should land
    as a queued job without waiting for the periodic sweep. No worker pool
    here — we're only asserting that watchfiles surfaces the event to the
    poller, which enqueues."""
    cfg = FSSourceConfig(
        type="fs",
        id="local",
        root=tmp_path,
        # poll_interval is far in the future so the periodic sweep CAN'T be
        # what picks up the new file — only watchfiles can.
        poll_interval_s=3600.0,
    )

    manager = PollerManager(
        configs=[cfg],
        job_repo=jobs,
        sync_repo=sync,
        supported_extensions=[".md"],
    )

    await manager.start()
    try:
        # Initial sweep saw an empty dir — give it a moment to settle, then
        # write a new file. watchfiles polls fs every ~50ms by default.
        async def _initial_sweep_done() -> bool:
            return manager.pollers[0].last_polled_at is not None

        await _wait_for(_initial_sweep_done, timeout=5.0)
        assert await jobs.counts_by_status() == {}

        (tmp_path / "new.md").write_text("after startup")

        async def _one_queued() -> bool:
            queued = await jobs.list_jobs(source_id="local")
            return any(j.uri == (tmp_path / "new.md").as_uri() for j in queued)

        await _wait_for(_one_queued, timeout=5.0)
    finally:
        await manager.stop()

    queued = await jobs.list_jobs(source_id="local")
    assert len(queued) == 1
    assert queued[0].op is JobOp.UPSERT
    assert queued[0].uri == (tmp_path / "new.md").as_uri()


@pytest.mark.asyncio
async def test_pre_existing_job_resolves_through_configured_source(
    tmp_path, jobs, sync
):
    """A job already in the queue at startup is processed through the
    configured Source adapter (with its headers / auth), not an adhoc
    HTTPSource. The Source list is built at PollerManager construction so
    the worker holds it before any start() call — no ordering required."""
    await jobs.enqueue("auth", "https://example.com/a.md", JobOp.UPSERT)

    client = _mock_client(tmp_path)
    cfg = HTTPSourceConfig(
        type="http",
        id="auth",
        urls=["https://example.com/a.md"],
        headers={"Authorization": "Bearer secret"},
    )
    manager = PollerManager(configs=[cfg], job_repo=jobs, sync_repo=sync)
    pool = WorkerPool(
        client=client,
        job_repo=jobs,
        sync_repo=sync,
        worker_count=1,
        poll_idle_interval_s=0.05,
        sources=manager.sources,
    )

    await manager.start()
    await pool.start()
    try:

        async def _one_succeeded() -> bool:
            counts = await jobs.counts_by_status()
            return counts.get("succeeded", 0) == 1

        await _wait_for(_one_succeeded, timeout=5.0)
    finally:
        await pool.stop()
        await manager.stop()

    kwargs = client.create_document_from_source.await_args.kwargs
    sources = kwargs.get("sources")
    assert sources is not None and len(sources) == 1
    assert isinstance(sources[0], HTTPSource)
    assert sources[0].headers == {"Authorization": "Bearer secret"}
