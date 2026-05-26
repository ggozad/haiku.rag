import asyncio
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.ingester.exceptions import PermanentError, TransientError
from haiku.rag.ingester.queue.migrations import apply_migrations
from haiku.rag.ingester.queue.models import JobOp, JobStatus
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo
from haiku.rag.ingester.workers.pool import WorkerPool
from haiku.rag.ingester.workers.retry import RetryPolicy
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


@pytest.fixture
def client():
    return AsyncMock(spec=HaikuRAG)


def _pool(client, jobs, sync, **kwargs) -> WorkerPool:
    return WorkerPool(
        client=client,
        job_repo=jobs,
        sync_repo=sync,
        worker_count=kwargs.pop("worker_count", 2),
        max_concurrent=kwargs.pop("max_concurrent", 2),
        poll_idle_interval_s=kwargs.pop("poll_idle_interval_s", 0.05),
        reaper_interval_s=kwargs.pop("reaper_interval_s", 60),
        claim_timeout_s=kwargs.pop("claim_timeout_s", 60),
        retry_policy=kwargs.pop("retry_policy", RetryPolicy()),
        sources=kwargs.pop("sources", None),
    )


# --- drain_once: covers _process logic deterministically ---


@pytest.mark.asyncio
async def test_drain_marks_job_succeeded_and_writes_sync_state(client, jobs, sync):
    client.create_document_from_source.return_value = Document(
        id="doc-1",
        content="x",
        uri="s3://b/k.md",
        metadata={"md5": "m1", "source_revision": "e1"},
    )
    job = await jobs.enqueue("src", "s3://b/k.md", JobOp.UPSERT, revision="e0")
    assert job is not None

    pool = _pool(client, jobs, sync)
    processed = await pool.drain_once()
    assert processed == 1

    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.SUCCEEDED

    snapshot = await sync.get_snapshot("src")
    assert snapshot == {"s3://b/k.md": "e1"}


@pytest.mark.asyncio
async def test_drain_delete_op_removes_sync_state(client, jobs, sync):
    await sync.upsert("src", "s3://b/k.md", revision="e1", content_hash="m1")
    client.get_document_by_uri.return_value = Document(
        id="doc-1", content="", uri="s3://b/k.md"
    )
    job = await jobs.enqueue("src", "s3://b/k.md", JobOp.DELETE)
    assert job is not None

    pool = _pool(client, jobs, sync)
    await pool.drain_once()

    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.SUCCEEDED
    client.delete_document.assert_awaited_once_with("doc-1")

    snapshot = await sync.get_snapshot("src")
    assert snapshot == {}


@pytest.mark.asyncio
async def test_permanent_error_marks_dead_no_reschedule(client, jobs, sync):
    client.create_document_from_source.side_effect = PermanentError("unsupported")
    job = await jobs.enqueue("src", "https://x/y.bin", JobOp.UPSERT)
    assert job is not None

    pool = _pool(client, jobs, sync)
    await pool.drain_once()

    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.DEAD
    assert refreshed.last_error == "unsupported"
    # sync_state is NOT written on failure
    assert await sync.get_snapshot("src") == {}


@pytest.mark.asyncio
async def test_transient_error_reschedules_below_max_attempts(client, jobs, sync):
    client.create_document_from_source.side_effect = TransientError("blip")
    job = await jobs.enqueue("src", "u", JobOp.UPSERT, max_attempts=3)
    assert job is not None

    # base_delay large enough that claim_next won't re-pick the job within
    # drain_once — we want exactly one process iteration.
    pool = _pool(
        client, jobs, sync, retry_policy=RetryPolicy(base_delay_s=60.0, jitter=0.0)
    )
    processed = await pool.drain_once()
    assert processed == 1

    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.QUEUED
    assert refreshed.last_error == "blip"
    assert refreshed.attempts == 1
    assert refreshed.scheduled_at > job.scheduled_at


@pytest.mark.asyncio
async def test_transient_error_at_max_attempts_marks_dead(client, jobs, sync, conn):
    client.create_document_from_source.side_effect = TransientError("blip")
    job = await jobs.enqueue("src", "u", JobOp.UPSERT, max_attempts=1)
    assert job is not None

    pool = _pool(
        client, jobs, sync, retry_policy=RetryPolicy(base_delay_s=0.0, jitter=0.0)
    )
    await pool.drain_once()

    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    # attempts started at 0, claim_next set it to 1 = max → dead
    assert refreshed.status is JobStatus.DEAD
    assert refreshed.attempts == 1


@pytest.mark.asyncio
async def test_unknown_exception_caught_and_marked_dead(client, jobs, sync):
    """An Exception subclass the pipeline classifier didn't recognise still
    gets marked dead by the pool's defensive `except Exception` net."""

    class _Weird(Exception):
        pass

    client.create_document_from_source.side_effect = _Weird("surprise")
    job = await jobs.enqueue("src", "u", JobOp.UPSERT, max_attempts=1)
    assert job is not None

    pool = _pool(
        client, jobs, sync, retry_policy=RetryPolicy(base_delay_s=0.0, jitter=0.0)
    )
    await pool.drain_once()
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.DEAD


@pytest.mark.asyncio
async def test_keyboard_interrupt_propagates_not_classified(client, jobs, sync):
    """KeyboardInterrupt / SystemExit / CancelledError signal runtime shutdown.
    The pipeline must not wrap them — the job stays 'claimed' for the reaper."""
    client.create_document_from_source.side_effect = KeyboardInterrupt("ctrl-c")
    job = await jobs.enqueue("src", "u", JobOp.UPSERT)
    assert job is not None

    pool = _pool(client, jobs, sync)
    with pytest.raises(KeyboardInterrupt):
        await pool.drain_once()

    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.CLAIMED


@pytest.mark.asyncio
async def test_drain_passes_configured_sources_to_client(client, jobs, sync):
    """The pool's `sources` list flows through run_job to
    client.create_document_from_source so resolve_fetcher can pick the
    configured authenticated source over an adhoc adapter."""
    from haiku.rag.ingester.sources.http import HTTPSource

    client.create_document_from_source.return_value = Document(
        id="d", content="x", uri="u", metadata={"md5": "m", "source_revision": "r"}
    )
    configured = HTTPSource(source_id="urls", headers={"Authorization": "Bearer abc"})
    await jobs.enqueue("src", "https://example.com/x", JobOp.UPSERT)

    pool = _pool(client, jobs, sync, sources=[configured])
    await pool.drain_once()

    kwargs = client.create_document_from_source.await_args.kwargs
    assert kwargs["sources"] == [configured]


# --- start / stop lifecycle ---


@pytest.mark.asyncio
async def test_workers_drain_queue_after_start(client, jobs, sync):
    client.create_document_from_source.return_value = Document(
        id="doc", content="x", uri="u", metadata={"md5": "m", "source_revision": "e"}
    )
    for i in range(5):
        await jobs.enqueue("src", f"u{i}", JobOp.UPSERT)

    pool = _pool(client, jobs, sync, worker_count=3, max_concurrent=3)
    await pool.start()
    try:
        # Wait until everything is succeeded or until a deadline.
        for _ in range(50):
            counts = await jobs.counts_by_status()
            if counts.get("succeeded", 0) == 5:
                break
            await asyncio.sleep(0.05)
    finally:
        await pool.stop()

    counts = await jobs.counts_by_status()
    assert counts.get("succeeded", 0) == 5


@pytest.mark.asyncio
async def test_shutdown_grace_lets_inflight_job_complete(client, jobs, sync):
    """A short-running job in flight when stop() is called must finish before
    the pool returns. Cancellation is the timeout path, not the default."""
    finished = asyncio.Event()

    async def _slow_then_finish(*args, **kwargs):
        await asyncio.sleep(0.1)
        finished.set()
        return Document(
            id="doc",
            content="x",
            uri="u",
            metadata={"md5": "m", "source_revision": "e"},
        )

    client.create_document_from_source.side_effect = _slow_then_finish
    await jobs.enqueue("src", "u", JobOp.UPSERT)

    pool = _pool(client, jobs, sync, worker_count=1, max_concurrent=1)
    await pool.start()
    # Yield long enough for the worker to claim and enter _process.
    await asyncio.sleep(0.02)
    await asyncio.wait_for(pool.stop(), timeout=5.0)

    assert finished.is_set()
    counts = await jobs.counts_by_status()
    assert counts.get("succeeded", 0) == 1


@pytest.mark.asyncio
async def test_shutdown_grace_timeout_releases_claim(client, jobs, sync):
    """When grace elapses and the worker is cancelled mid-job, the claim is
    released back to 'queued' so the next process can pick it up immediately
    — no waiting on the reaper's claim_timeout_s."""
    cancelled = asyncio.Event()

    async def _hangs_forever(*args, **kwargs):
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return Document(id="doc", content="x", uri="u")

    client.create_document_from_source.side_effect = _hangs_forever
    job = await jobs.enqueue("src", "u", JobOp.UPSERT)
    assert job is not None

    pool = _pool(client, jobs, sync, worker_count=1, max_concurrent=1)
    await pool.start()
    await asyncio.sleep(0.05)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(pool.stop(), timeout=0.2)

    assert cancelled.is_set()
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.QUEUED
    assert refreshed.claimed_by is None
    # claim_next incremented attempts to 1; release_if_claimed decremented it
    # back to 0 because a cancellation isn't a failed attempt.
    assert refreshed.attempts == 0


@pytest.mark.asyncio
async def test_cancel_cleanup_survives_second_cancel(client, jobs, sync, monkeypatch):
    """A second cancel arriving while the cancel-handler is awaiting
    release_if_claimed must not strand the claim. The shielded await may
    raise CancelledError, but the underlying SQL update keeps running and
    completes the release as an orphan task."""
    release_entered = asyncio.Event()
    release_done = asyncio.Event()

    real_release = jobs.release_if_claimed

    async def _slow_release(job_id):
        release_entered.set()
        # Long enough for the second cancel to arrive mid-update.
        await asyncio.sleep(0.2)
        result = await real_release(job_id)
        release_done.set()
        return result

    monkeypatch.setattr(jobs, "release_if_claimed", _slow_release)

    async def _hangs_forever(*args, **kwargs):
        await asyncio.sleep(60)
        return Document(id="doc", content="x", uri="u")

    client.create_document_from_source.side_effect = _hangs_forever
    job = await jobs.enqueue("src", "u", JobOp.UPSERT)
    assert job is not None

    pool = _pool(client, jobs, sync, worker_count=1, max_concurrent=1)
    await pool.start()
    try:
        await asyncio.sleep(0.05)
        worker_task = pool._workers[0]
        worker_task.cancel()
        # Wait until the worker is inside the shielded release call.
        await asyncio.wait_for(release_entered.wait(), timeout=1.0)
        # Second cancel mid-cleanup. Shield holds the SQL update upright.
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task
        # Background release Task still alive; let it finish.
        await asyncio.wait_for(release_done.wait(), timeout=1.0)
    finally:
        await pool.stop()

    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.QUEUED
    assert refreshed.claimed_by is None


@pytest.mark.asyncio
async def test_double_start_raises(client, jobs, sync):
    pool = _pool(client, jobs, sync, worker_count=1)
    await pool.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            await pool.start()
    finally:
        await pool.stop()


# --- reaper ---


@pytest.mark.asyncio
async def test_reaper_resets_stale_claims(client, jobs, sync, conn):
    from datetime import UTC, datetime, timedelta

    job = await jobs.enqueue("src", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("worker-old")
    assert claimed is not None

    # Push claimed_at back so reap_stale picks it up.
    long_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    await conn.execute(
        "UPDATE jobs SET claimed_at = ? WHERE id = ?", (long_ago, job.id)
    )
    await conn.commit()

    pool = _pool(
        client,
        jobs,
        sync,
        worker_count=0,
        reaper_interval_s=0.05,
        claim_timeout_s=1,
    )
    await pool.start()
    try:
        for _ in range(30):
            refreshed = await jobs.get_job(job.id)
            if refreshed is not None and refreshed.status is JobStatus.QUEUED:
                break
            await asyncio.sleep(0.05)
    finally:
        await pool.stop()

    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.QUEUED
