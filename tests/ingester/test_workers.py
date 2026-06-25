import asyncio
from unittest.mock import AsyncMock

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.ingester.exceptions import PermanentError, TransientError
from haiku.rag.ingester.queue.models import JobOp, JobStatus
from haiku.rag.ingester.workers.pool import WorkerPool
from haiku.rag.ingester.workers.retry import RetryPolicy
from haiku.rag.store.models.document import Document


@pytest.fixture
def client():
    return AsyncMock(spec=HaikuRAG)


def _pool(client, jobs, sync, **kwargs) -> WorkerPool:
    return WorkerPool(
        client=client,
        job_repo=jobs,
        sync_repo=sync,
        worker_count=kwargs.pop("worker_count", 2),
        poll_idle_interval_s=kwargs.pop("poll_idle_interval_s", 0.05),
        reaper_interval_s=kwargs.pop("reaper_interval_s", 60),
        claim_timeout_s=kwargs.pop("claim_timeout_s", 60),
        retention_s=kwargs.pop("retention_s", None),
        retry_policy=kwargs.pop("retry_policy", RetryPolicy()),
        sources=kwargs.pop("sources", None),
    )


# --- worker identity ---


@pytest.mark.asyncio
async def test_worker_ids_are_unique_across_pools(client, jobs, sync):
    """Two pools built with default construction must not share worker ids;
    otherwise a stale worker from one pool can satisfy the claimed_by guard of
    a job re-claimed by another pool and clobber its result."""
    pool_a = _pool(client, jobs, sync)
    pool_b = _pool(client, jobs, sync)
    worker_a = pool_a._worker_id(0)
    worker_b = pool_b._worker_id(0)
    assert worker_a != worker_b

    job = await jobs.enqueue("src", "u", JobOp.UPSERT)
    assert job is not None

    claimed = await jobs.claim_next(worker_a)
    assert claimed is not None
    # Reaper resets the claim; pool B re-claims and finishes first.
    await jobs.reap_stale(lease_ttl_seconds=0)
    reclaimed = await jobs.claim_next(worker_b)
    assert reclaimed is not None and reclaimed.id == job.id
    assert await jobs.mark_succeeded(reclaimed.id, worker_b) is True

    # Pool A's slow worker finishes later: with unique ids this is a no-op, so
    # pool B's success is not clobbered.
    assert await jobs.mark_succeeded(job.id, worker_a) is False


# --- event-driven wakeup ---


@pytest.mark.asyncio
async def test_idle_worker_picks_up_job_quickly(client, jobs, sync):
    """An idle worker should wake up well under poll_idle_s when a job is
    enqueued, thanks to the job_available condition notification."""
    client.create_document_from_source.return_value = Document(
        id="d", content="x", uri="u", metadata={"md5": "m", "source_revision": "r"}
    )
    pool = _pool(client, jobs, sync, worker_count=1, poll_idle_interval_s=5.0)
    await pool.start()
    try:
        await jobs.enqueue("src", "u", JobOp.UPSERT)
        for _ in range(50):
            listed = await jobs.list_jobs(status=JobStatus.SUCCEEDED)
            if listed:
                break
            await asyncio.sleep(0.05)
        assert len(listed) == 1, "job was not picked up within 2.5s"
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_stop_completes_with_idle_workers(client, jobs, sync):
    """stop() must notify workers parked on job_available.wait() so they
    exit promptly. Without the notify, workers sleep for the full
    poll_idle_interval_s before noticing _stop."""
    pool = _pool(client, jobs, sync, worker_count=2, poll_idle_interval_s=10.0)
    await pool.start()
    await asyncio.sleep(0.1)
    try:
        await asyncio.wait_for(pool.stop(), timeout=2.0)
    except TimeoutError:
        pytest.fail("stop() did not complete within 2s — idle workers were not woken")
    assert pool.live_workers == 0


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

    snapshot = await sync.get_revision_snapshot("src")
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

    snapshot = await sync.get_revision_snapshot("src")
    assert snapshot == {}


@pytest.mark.asyncio
async def test_successful_delete_prunes_dead_jobs_for_same_uri(client, jobs, sync):
    """Once a DELETE resolves a URI, any earlier UPSERT failure for the same
    (source_id, uri) is stale — auto-prune keeps the DLQ free of resolved
    entries."""
    # Stage a prior dead UPSERT (file-not-found style).
    upsert = await jobs.enqueue("src", "file:///gone.md", JobOp.UPSERT)
    assert upsert is not None
    claimed = await jobs.claim_next("prev-worker")
    assert claimed is not None
    await jobs.mark_dead(claimed.id, "File does not exist", "prev-worker")
    assert (await jobs.get_job(upsert.id)).status is JobStatus.DEAD

    # Now run a DELETE for the same URI.
    client.get_document_by_uri.return_value = Document(
        id="doc-9", content="", uri="file:///gone.md"
    )
    delete = await jobs.enqueue("src", "file:///gone.md", JobOp.DELETE)
    assert delete is not None

    pool = _pool(client, jobs, sync)
    await pool.drain_once()

    assert (await jobs.get_job(delete.id)).status is JobStatus.SUCCEEDED
    # The stale dead UPSERT for the same URI is gone.
    assert await jobs.get_job(upsert.id) is None


@pytest.mark.asyncio
async def test_permanent_error_without_revision_writes_no_marker(client, jobs, sync):
    """A permanent failure on a revision-less job (e.g. HTTP without ETag) writes
    no suppression marker — get_revision_snapshot omits revision-less rows, so it
    would re-enqueue on the next sweep. Documents the revision-less caveat."""
    client.create_document_from_source.side_effect = PermanentError("unsupported")
    job = await jobs.enqueue("src", "https://x/y.bin", JobOp.UPSERT)
    assert job is not None

    pool = _pool(client, jobs, sync)
    await pool.drain_once()

    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.DEAD
    assert refreshed.last_error == "unsupported"
    assert await sync.get_revision_snapshot("src") == {}


@pytest.mark.asyncio
async def test_permanent_error_with_revision_records_marker(client, jobs, sync):
    """A permanent failure on a revisioned job records the failed revision in
    sync_state (ingested=False) so discovery sees it as UNCHANGED and stops
    re-enqueuing it every sweep, until the file's revision changes."""
    client.create_document_from_source.side_effect = PermanentError("encrypted")
    job = await jobs.enqueue("src", "file:///x/y.pdf", JobOp.UPSERT, revision="r0")
    assert job is not None

    pool = _pool(client, jobs, sync)
    await pool.drain_once()

    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.DEAD

    # Failed revision is recorded, so discovery treats the unchanged file as known.
    assert await sync.get_revision_snapshot("src") == {"file:///x/y.pdf": "r0"}
    # Recorded as a failure, not an ingestion.
    row = await sync.get_row("src", "file:///x/y.pdf")
    assert row is not None
    assert row.revision == "r0"
    assert row.last_ingested_at is None


@pytest.mark.asyncio
async def test_transient_exhausted_writes_no_marker(client, jobs, sync):
    """A transient failure that exhausts max_attempts goes dead but records no
    suppression marker, so it stays re-attemptable on the next sweep (transient =
    keep retrying once the service recovers)."""
    client.create_document_from_source.side_effect = TransientError("blip")
    job = await jobs.enqueue(
        "src", "file:///x/y.pdf", JobOp.UPSERT, revision="r0", max_attempts=1
    )
    assert job is not None

    pool = _pool(client, jobs, sync)
    await pool.drain_once()

    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.DEAD
    # No marker despite a revision being present — only PermanentError suppresses.
    assert await sync.get_revision_snapshot("src") == {}


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
    """The classifier's fallback wraps any unrecognised Exception into
    TransientError, so an unknown error still flows through reschedule/DLQ
    rather than crashing the worker task."""

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

    pool = _pool(client, jobs, sync, worker_count=3)
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

    pool = _pool(client, jobs, sync, worker_count=1)
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

    pool = _pool(client, jobs, sync, worker_count=1)
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
async def test_worker_loses_claim_to_reaper_does_not_write_sync_state(
    client, jobs, sync
):
    """If the reaper resets a slow worker's claim and another worker re-claims
    the job, the original worker's mark_succeeded must be a no-op and its
    sync_state.upsert must not run — otherwise we'd overwrite freshly-written
    state from the re-claiming worker."""
    client.create_document_from_source.return_value = Document(
        id="doc-A", content="x", uri="u", metadata={"md5": "A", "source_revision": "A"}
    )
    job = await jobs.enqueue("src", "u", JobOp.UPSERT)
    assert job is not None
    claimed_by_a = await jobs.claim_next("worker-A")
    assert claimed_by_a is not None
    # Reaper resets A's claim, worker-B re-claims.
    await jobs.reap_stale(lease_ttl_seconds=0)
    await jobs.claim_next("worker-B")

    pool = _pool(client, jobs, sync, worker_count=1)
    # Drive A's _process directly with A's (now stale) Job snapshot.
    await pool._process(claimed_by_a)

    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    # B still owns the claim — A's mark_succeeded was a no-op.
    assert refreshed.status is JobStatus.CLAIMED
    assert refreshed.claimed_by == "worker-B"
    # And sync_state must be untouched.
    assert await sync.get_revision_snapshot("src") == {}


@pytest.mark.asyncio
async def test_permanent_error_loses_claim_to_reaper_writes_no_marker(
    client, jobs, sync
):
    """If the reaper resets the claim and another worker re-claims before a
    permanent failure is recorded, the original worker's mark_dead is a no-op
    and it writes no failure marker — the re-claiming worker drives the
    outcome, so the stale worker must not stamp sync_state."""
    client.create_document_from_source.side_effect = PermanentError("encrypted")
    job = await jobs.enqueue("src", "u", JobOp.UPSERT, revision="r0")
    assert job is not None
    claimed_by_a = await jobs.claim_next("worker-A")
    assert claimed_by_a is not None
    # Reaper resets A's claim, worker-B re-claims.
    await jobs.reap_stale(lease_ttl_seconds=0)
    await jobs.claim_next("worker-B")

    pool = _pool(client, jobs, sync, worker_count=1)
    # Drive A's _process with A's (now stale) Job snapshot.
    await pool._process(claimed_by_a)

    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    # B still owns the claim — A's mark_dead was a no-op.
    assert refreshed.status is JobStatus.CLAIMED
    assert refreshed.claimed_by == "worker-B"
    # No failure marker written despite a revision being present.
    assert await sync.get_revision_snapshot("src") == {}


@pytest.mark.asyncio
async def test_cancel_cleanup_survives_second_cancel(client, jobs, sync, monkeypatch):
    """A second cancel arriving while the cancel-handler is awaiting
    release_if_claimed must not strand the claim. The shielded await may
    raise CancelledError, but the underlying SQL update keeps running and
    completes the release as an orphan task."""
    release_entered = asyncio.Event()
    release_done = asyncio.Event()

    real_release = jobs.release_if_claimed

    async def _slow_release(job_id, claimed_by):
        release_entered.set()
        # Long enough for the second cancel to arrive mid-update.
        await asyncio.sleep(0.2)
        result = await real_release(job_id, claimed_by)
        release_done.set()
        return result

    monkeypatch.setattr(jobs, "release_if_claimed", _slow_release)

    async def _hangs_forever(*args, **kwargs):
        await asyncio.sleep(60)
        return Document(id="doc", content="x", uri="u")

    client.create_document_from_source.side_effect = _hangs_forever
    job = await jobs.enqueue("src", "u", JobOp.UPSERT)
    assert job is not None

    pool = _pool(client, jobs, sync, worker_count=1)
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
async def test_drain_pending_releases_waits_for_orphan_releases(
    client, jobs, sync, monkeypatch
):
    """When the worker is cancelled twice (shutdown_grace_s timeout path) it
    exits before its release_if_claimed Task completes, leaving an orphan.
    drain_pending_releases waits for that orphan so the SQL update lands
    before the lifecycle owner closes the queue connection."""
    real_release = jobs.release_if_claimed
    release_entered = asyncio.Event()

    async def _slow_release(job_id, claimed_by):
        release_entered.set()
        await asyncio.sleep(0.15)
        return await real_release(job_id, claimed_by)

    monkeypatch.setattr(jobs, "release_if_claimed", _slow_release)

    async def _hangs_forever(*args, **kwargs):
        await asyncio.sleep(60)
        return Document(id="doc", content="x", uri="u")

    client.create_document_from_source.side_effect = _hangs_forever
    job = await jobs.enqueue("src", "u", JobOp.UPSERT)
    assert job is not None

    pool = _pool(client, jobs, sync, worker_count=1)
    await pool.start()
    try:
        await asyncio.sleep(0.05)
        worker_task = pool._workers[0]
        worker_task.cancel()
        await asyncio.wait_for(release_entered.wait(), timeout=1.0)
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

        # At this point the orphan release is still running. drain returns
        # the count that landed within the timeout.
        assert len(pool._pending_releases) == 1
        landed = await pool.drain_pending_releases(timeout=1.0)
        assert landed == 1
        assert pool._pending_releases == set()
    finally:
        await pool.stop()

    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.QUEUED


@pytest.mark.asyncio
async def test_drain_pending_releases_with_no_orphans_is_noop(client, jobs, sync):
    """Common case: nothing to drain — drain returns 0 immediately, no
    asyncio.wait against an empty set."""
    pool = _pool(client, jobs, sync, worker_count=1)
    assert await pool.drain_pending_releases() == 0


@pytest.mark.asyncio
async def test_live_workers_drops_when_a_worker_finishes(client, jobs, sync):
    """live_workers powers /health's degraded signal. When a worker task
    has completed (crashed or exited), it must no longer count."""
    pool = _pool(client, jobs, sync, worker_count=2)
    await pool.start()
    try:
        assert pool.live_workers == 2
        # Cancel one worker directly to simulate a crash.
        pool._workers[0].cancel()
        await asyncio.gather(pool._workers[0], return_exceptions=True)
        assert pool.live_workers == 1
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_double_start_raises(client, jobs, sync):
    pool = _pool(client, jobs, sync, worker_count=1)
    await pool.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            await pool.start()
    finally:
        await pool.stop()


# --- per-source circuit breaker ---


@pytest.mark.asyncio
async def test_breaker_opens_after_n_consecutive_transient_failures(client, jobs, sync):
    """N back-to-back TransientErrors from one source flips that source's
    breaker open. While open, _worker_loop excludes the source from
    claim_next so its other jobs don't burn attempts during the same outage."""
    from haiku.rag.ingester.workers.pool import _WORKER_BREAKER_THRESHOLD

    client.create_document_from_source.side_effect = TransientError("downstream down")
    # Enough jobs to trip the breaker on attempt 1 of each, with one extra
    # that should remain unclaimed.
    for i in range(_WORKER_BREAKER_THRESHOLD + 1):
        await jobs.enqueue("src", f"u{i}", JobOp.UPSERT, max_attempts=5)

    pool = _pool(
        client, jobs, sync, retry_policy=RetryPolicy(base_delay_s=60.0, jitter=0.0)
    )
    # Drain one job at a time so the breaker can tick before the next claim.
    for _ in range(_WORKER_BREAKER_THRESHOLD):
        await pool.drain_once()
    assert pool.breaker_open is True

    # drain_once claims without the breaker exclusion (it's intended for
    # tests), so it would still process more jobs. The exclusion lives in
    # _worker_loop: a fresh worker with this source's breaker open won't
    # claim its jobs.
    remaining_before = len(await jobs.list_jobs(status=JobStatus.QUEUED, limit=500))
    assert remaining_before >= 1


@pytest.mark.asyncio
async def test_breaker_pauses_worker_loop_claims(client, jobs, sync):
    """Worker loop honours the breaker: an open source is excluded from
    claim_next, so its queued jobs stay queued until the breaker closes."""
    pool = _pool(client, jobs, sync, worker_count=1, poll_idle_interval_s=0.02)
    # Force the source's breaker open without touching the queue.
    for _ in range(10):
        pool._breaker_for("src").record_failure()
    assert pool.breaker_open is True

    await jobs.enqueue("src", "u", JobOp.UPSERT)
    await pool.start()
    try:
        # Even with a queued job available and a live worker, the exclusion
        # keeps the job in 'queued' state.
        await asyncio.sleep(0.1)
        refreshed = await jobs.list_jobs(status=JobStatus.QUEUED, limit=10)
        assert len(refreshed) == 1
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_breaker_closes_on_successful_probe(client, jobs, sync):
    """After cooldown, the next probe is allowed through; if it succeeds,
    record_success clears the breaker so workers fully resume."""
    client.create_document_from_source.return_value = Document(
        id="d", content="x", uri="u", metadata={"md5": "m", "source_revision": "r"}
    )
    pool = _pool(client, jobs, sync)
    # Open the source's breaker, then collapse the cooldown so is_open returns
    # False on the next check (the breaker's three-state model probes after
    # cooldown).
    breaker = pool._breaker_for("src")
    for _ in range(10):
        breaker.record_failure()
    breaker._opened_at = 0.0  # type: ignore[attr-defined]
    assert pool.breaker_open is False  # cooldown elapsed → probe allowed

    await jobs.enqueue("src", "u", JobOp.UPSERT)
    await pool.drain_once()

    # The successful job ticks record_success which clears the breaker.
    assert pool.breaker_consecutive_failures == 0


@pytest.mark.asyncio
async def test_breaker_isolates_sources(client, jobs, sync):
    """An open breaker pauses only the failing source. Workers keep draining
    a healthy source's jobs while the failing source's jobs stay queued."""

    def _route(uri, *, sources=None, source_id=None, metadata_provider=None):
        if source_id == "bad":
            raise TransientError("downstream down")
        return Document(
            id="d", content="x", uri=uri, metadata={"md5": "m", "source_revision": "r"}
        )

    client.create_document_from_source.side_effect = _route

    for i in range(3):
        await jobs.enqueue("bad", f"b{i}", JobOp.UPSERT)
        await jobs.enqueue("good", f"g{i}", JobOp.UPSERT)

    pool = _pool(client, jobs, sync, worker_count=2, poll_idle_interval_s=0.02)
    # Open the bad source's breaker without touching the queue.
    for _ in range(10):
        pool._breaker_for("bad").record_failure()

    await pool.start()
    try:
        await asyncio.sleep(0.2)
        succeeded = await jobs.list_jobs(status=JobStatus.SUCCEEDED, limit=50)
        queued = await jobs.list_jobs(status=JobStatus.QUEUED, limit=50)
    finally:
        await pool.stop()

    assert {j.uri for j in succeeded} == {"g0", "g1", "g2"}
    assert {j.uri for j in queued} == {"b0", "b1", "b2"}


@pytest.mark.asyncio
async def test_breaker_ignores_permanent_errors(client, jobs, sync):
    """Permanent errors are about the document, not downstream — they
    shouldn't poison the breaker against unrelated jobs."""
    from haiku.rag.ingester.workers.pool import _WORKER_BREAKER_THRESHOLD

    client.create_document_from_source.side_effect = PermanentError("bad URI")
    for i in range(_WORKER_BREAKER_THRESHOLD + 2):
        await jobs.enqueue("src", f"u{i}", JobOp.UPSERT)

    pool = _pool(client, jobs, sync)
    await pool.drain_once()
    assert pool.breaker_open is False
    assert pool.breaker_consecutive_failures == 0


# --- sync_state write resilience ---


@pytest.mark.asyncio
async def test_sync_state_write_failure_does_not_crash_worker(
    client, jobs, sync, monkeypatch
):
    """If the sync_state write fails after mark_succeeded, the worker should
    log the error and continue rather than crashing. The job is already
    marked succeeded — a stale sync_state just means a redundant re-ingest
    on the next sweep."""
    client.create_document_from_source.return_value = Document(
        id="d", content="x", uri="u", metadata={"md5": "m", "source_revision": "r"}
    )
    await jobs.enqueue("src", "u", JobOp.UPSERT)

    original_upsert = sync.upsert

    async def _failing_upsert(*args, **kwargs):
        # Only fail for ingested=True (the post-success write)
        if kwargs.get("ingested"):
            raise OSError("disk full")
        return await original_upsert(*args, **kwargs)

    monkeypatch.setattr(sync, "upsert", _failing_upsert)

    pool = _pool(client, jobs, sync)
    # drain_once should complete without raising
    processed = await pool.drain_once()
    assert processed == 1

    # Job should still be marked succeeded
    listed = await jobs.list_jobs(status=JobStatus.SUCCEEDED)
    assert len(listed) == 1


@pytest.mark.asyncio
async def test_permanent_failure_marker_write_failure_does_not_crash_worker(
    client, jobs, sync, monkeypatch
):
    """If the permanent-failure sync_state marker write fails after mark_dead,
    the worker logs and continues rather than crashing. The job is already dead;
    a missing marker just means the file may re-enqueue on the next sweep."""
    client.create_document_from_source.side_effect = PermanentError("encrypted")
    await jobs.enqueue("src", "file:///x/y.pdf", JobOp.UPSERT, revision="r0")

    async def _failing_upsert(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(sync, "upsert", _failing_upsert)

    pool = _pool(client, jobs, sync)
    # drain_once should complete without raising despite the marker write failing.
    processed = await pool.drain_once()
    assert processed == 1

    # Job is still dead — the marker write failure must not undo that.
    listed = await jobs.list_jobs(status=JobStatus.DEAD)
    assert len(listed) == 1


# --- reaper ---


@pytest.mark.asyncio
async def test_boot_reap_resets_stale_pre_existing_claims(client, jobs, sync, conn):
    """A SIGKILL'd previous process leaves a stale claim (its lease stopped
    being renewed). WorkerPool.start() sweeps it so fresh workers can take it
    over."""
    from datetime import UTC, datetime, timedelta

    await jobs.enqueue("src", "u", JobOp.UPSERT)
    pre_claimed = await jobs.claim_next("ghost-worker")
    assert pre_claimed is not None
    assert pre_claimed.status is JobStatus.CLAIMED

    # The ghost stopped renewing an hour ago.
    long_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    await conn.execute(
        "UPDATE jobs SET claimed_at = ?, last_heartbeat_at = ? WHERE id = ?",
        (long_ago, long_ago, pre_claimed.id),
    )
    await conn.commit()

    pool = _pool(client, jobs, sync, worker_count=0)
    await pool.start()
    try:
        refreshed = await jobs.get_job(pre_claimed.id)
        assert refreshed is not None
        assert refreshed.status is JobStatus.QUEUED
        assert refreshed.claimed_by is None
        # attempts was incremented by claim_next; reap decrements it so the
        # next claim doesn't see a consumed retry it never actually used.
        assert refreshed.attempts == 0
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_boot_reap_leaves_a_peer_process_fresh_claim_alone(client, jobs, sync):
    """A peer process sharing the queue holds a freshly-claimed job. Our
    startup boot-reap must not wipe its live claim."""
    await jobs.enqueue("src", "u", JobOp.UPSERT)
    peer_claim = await jobs.claim_next("peer-worker")
    assert peer_claim is not None

    pool = _pool(client, jobs, sync, worker_count=0)
    await pool.start()
    try:
        refreshed = await jobs.get_job(peer_claim.id)
        assert refreshed is not None
        assert refreshed.status is JobStatus.CLAIMED
        assert refreshed.claimed_by == "peer-worker"
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_reaper_resets_stale_claims(client, jobs, sync, conn):
    from datetime import UTC, datetime, timedelta

    job = await jobs.enqueue("src", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("worker-old")
    assert claimed is not None

    # Push the lease back so reap_stale picks it up.
    long_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    await conn.execute(
        "UPDATE jobs SET claimed_at = ?, last_heartbeat_at = ? WHERE id = ?",
        (long_ago, long_ago, job.id),
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


@pytest.mark.asyncio
async def test_reaper_prunes_old_terminal_jobs(client, jobs, sync, conn):
    from datetime import UTC, datetime, timedelta

    job = await jobs.enqueue("src", "u", JobOp.UPSERT)
    await jobs.claim_next("w")
    await jobs.mark_succeeded(job.id, "w")

    # Backdate completed_at so prune_terminal picks it up.
    long_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    await conn.execute(
        "UPDATE jobs SET completed_at = ? WHERE id = ?", (long_ago, job.id)
    )
    await conn.commit()

    pool = _pool(
        client,
        jobs,
        sync,
        worker_count=0,
        reaper_interval_s=0.05,
        retention_s=1,
    )
    await pool.start()
    try:
        for _ in range(30):
            if await jobs.get_job(job.id) is None:
                break
            await asyncio.sleep(0.05)
    finally:
        await pool.stop()

    assert await jobs.get_job(job.id) is None


@pytest.mark.asyncio
async def test_reaper_skips_prune_when_retention_none(client, jobs, sync, conn):
    from datetime import UTC, datetime, timedelta

    job = await jobs.enqueue("src", "u", JobOp.UPSERT)
    await jobs.claim_next("w")
    await jobs.mark_succeeded(job.id, "w")

    long_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    await conn.execute(
        "UPDATE jobs SET completed_at = ? WHERE id = ?", (long_ago, job.id)
    )
    await conn.commit()

    pool = _pool(
        client,
        jobs,
        sync,
        worker_count=0,
        reaper_interval_s=0.05,
        retention_s=None,
    )
    await pool.start()
    try:
        await asyncio.sleep(0.3)
    finally:
        await pool.stop()

    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.SUCCEEDED
