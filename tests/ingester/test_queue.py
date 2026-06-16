import asyncio
from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy as sa

from haiku.rag.config import QueueConfig
from haiku.rag.ingester.queue.db import jobs as jobs_table
from haiku.rag.ingester.queue.migrations import (
    apply_migrations,
    make_engine,
    open_queue,
)
from haiku.rag.ingester.queue.models import JobOp, JobStatus, SyncRow
from haiku.rag.ingester.queue.repository import JobRepo, _insert

# --- migrations / schema ---


@pytest.mark.asyncio
async def test_apply_migrations_sets_schema_version(conn):
    cursor = await conn.execute("SELECT version FROM schema_version")
    row = await cursor.fetchone()
    assert row is not None
    assert row["version"] >= 1


@pytest.mark.asyncio
async def test_apply_migrations_is_idempotent(engine, conn):
    await apply_migrations(engine)
    await apply_migrations(engine)
    cursor = await conn.execute("SELECT COUNT(*) AS n FROM schema_version")
    row = await cursor.fetchone()
    assert row["n"] == 1


@pytest.mark.asyncio
async def test_open_queue_creates_file_and_schema(tmp_path):
    path = tmp_path / "subdir" / "queue.db"
    eng = await open_queue(QueueConfig(path=path))
    try:
        assert path.exists()
        async with eng.connect() as conn:
            version = (
                await conn.execute(sa.text("SELECT version FROM schema_version"))
            ).scalar()
        assert version is not None
    finally:
        await eng.dispose()


@pytest.mark.asyncio
async def test_make_engine_postgres_is_pre_ping():
    """The Postgres branch builds a pre-ping engine without connecting."""
    engine = make_engine(QueueConfig(dburi="postgresql+asyncpg://u:p@localhost/db"))
    try:
        assert engine.dialect.name == "postgresql"
        assert engine.pool._pre_ping is True
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sqlite_engine_serves_concurrent_connections(tmp_path):
    """The SQLite queue pool hands out more than one connection at a time so an
    API read does not starve while a worker holds a connection. WAL mode makes
    concurrent readers safe; the claim stays atomic via its single UPDATE."""
    engine = make_engine(QueueConfig(path=tmp_path / "queue.db"))
    await apply_migrations(engine)
    held: list = []
    try:

        async def acquire() -> None:
            conn = await engine.connect()
            held.append(conn)
            await conn.execute(sa.text("SELECT 1"))

        # All three checkouts are held at once; a single-connection pool blocks
        # the second and third until the checkout timeout.
        await asyncio.wait_for(
            asyncio.gather(*(acquire() for _ in range(3))), timeout=3
        )
        assert len(held) == 3
    finally:
        for conn in held:
            await conn.close()
        await engine.dispose()


def test_insert_uses_dialect_specific_construct():
    """_insert dispatches to the dialect's INSERT (which exposes on_conflict_*)."""
    from sqlalchemy.dialects.postgresql import Insert as PostgresInsert
    from sqlalchemy.dialects.sqlite import Insert as SqliteInsert

    assert isinstance(_insert(jobs_table, "postgresql"), PostgresInsert)
    assert isinstance(_insert(jobs_table, "sqlite"), SqliteInsert)


@pytest.mark.asyncio
async def test_open_queue_handles_path_with_url_chars(tmp_path):
    """A `?` (or `#`) is a valid POSIX filename char but has URL meaning.
    The queue must open the literal file, not a truncated one."""
    path = tmp_path / "queue?weird.db"
    eng = await open_queue(QueueConfig(path=path))
    try:
        assert path.exists()
        assert not (tmp_path / "queue").exists()
    finally:
        await eng.dispose()


# --- enqueue ---


@pytest.mark.asyncio
async def test_enqueue_creates_queued_job(jobs):
    job = await jobs.enqueue(
        "fs:/tmp",
        "file:///tmp/a.md",
        JobOp.UPSERT,
        revision="abc",
        content_hash="md5_value",
        extra={"key": "value"},
    )
    assert job is not None
    assert job.status is JobStatus.QUEUED
    assert job.source_id == "fs:/tmp"
    assert job.uri == "file:///tmp/a.md"
    assert job.op is JobOp.UPSERT
    assert job.revision == "abc"
    assert job.content_hash == "md5_value"
    assert job.attempts == 0
    assert job.extra == {"key": "value"}
    assert job.enqueued_at.tzinfo is not None


@pytest.mark.asyncio
async def test_enqueue_returns_none_on_live_conflict(jobs):
    first = await jobs.enqueue("s", "u", JobOp.UPSERT)
    second = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert first is not None
    assert second is None


@pytest.mark.asyncio
async def test_enqueue_after_dead_succeeds(jobs):
    first = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_dead(first.id, "error", "w")
    second = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert second is not None
    assert second.id != first.id


@pytest.mark.asyncio
async def test_enqueue_after_succeeded_succeeds(jobs):
    await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_succeeded(claimed.id, "w")
    second = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert second is not None


@pytest.mark.asyncio
async def test_has_pending_returns_false_on_empty_queue(jobs):
    assert await jobs.has_pending("src") is False


@pytest.mark.asyncio
async def test_has_pending_true_for_queued_job(jobs):
    await jobs.enqueue("src", "u", JobOp.UPSERT)
    assert await jobs.has_pending("src") is True


@pytest.mark.asyncio
async def test_has_pending_true_for_claimed_job(jobs):
    await jobs.enqueue("src", "u", JobOp.UPSERT)
    await jobs.claim_next("w")
    assert await jobs.has_pending("src") is True


@pytest.mark.asyncio
async def test_has_pending_false_for_terminal_states(jobs):
    await jobs.enqueue("src", "u-ok", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_succeeded(claimed.id, "w")

    await jobs.enqueue("src", "u-bad", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_dead(claimed.id, "permanent", "w")

    assert await jobs.has_pending("src") is False


@pytest.mark.asyncio
async def test_has_pending_is_per_source(jobs):
    """A backed-up source must not block an idle one."""
    await jobs.enqueue("busy", "u", JobOp.UPSERT)
    assert await jobs.has_pending("busy") is True
    assert await jobs.has_pending("idle") is False


@pytest.mark.asyncio
async def test_enqueue_drops_delete_when_upsert_is_live(jobs):
    """Stops a DELETE worker from removing a document a sibling UPSERT
    just ingested."""
    upsert = await jobs.enqueue("s", "u", JobOp.UPSERT)
    delete = await jobs.enqueue("s", "u", JobOp.DELETE)
    assert upsert is not None
    assert delete is None


@pytest.mark.asyncio
async def test_enqueue_drops_upsert_when_delete_is_live(jobs):
    """Symmetric: live DELETE for a URI blocks a fresh UPSERT for the
    same URI. The next sweep after DELETE completes will re-emit the
    UPSERT if the file is back."""
    delete = await jobs.enqueue("s", "u", JobOp.DELETE)
    upsert = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert delete is not None
    assert upsert is None


@pytest.mark.asyncio
async def test_enqueue_different_uris_independent_of_op(jobs):
    """Uniqueness is per-(source_id, uri), not per-(source_id, uri, op).
    Different URIs can be queued regardless of which op each one is."""
    a = await jobs.enqueue("s", "u-a", JobOp.UPSERT)
    b = await jobs.enqueue("s", "u-b", JobOp.DELETE)
    assert a is not None
    assert b is not None


# --- claim_next ---


@pytest.mark.asyncio
async def test_claim_next_returns_none_when_empty(jobs):
    assert await jobs.claim_next("w") is None


@pytest.mark.asyncio
async def test_claim_next_increments_attempts_and_records_worker(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert job.attempts == 0
    claimed = await jobs.claim_next("worker-1")
    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.status is JobStatus.CLAIMED
    assert claimed.attempts == 1
    assert claimed.claimed_by == "worker-1"
    assert claimed.claimed_at is not None


@pytest.mark.asyncio
async def test_claim_next_returns_oldest_first(jobs):
    j1 = await jobs.enqueue("s", "u1", JobOp.UPSERT)
    j2 = await jobs.enqueue("s", "u2", JobOp.UPSERT)
    first = await jobs.claim_next("w")
    second = await jobs.claim_next("w")
    assert first is not None
    assert second is not None
    assert first.id == j1.id
    assert second.id == j2.id


@pytest.mark.asyncio
async def test_claim_next_excludes_source_ids(jobs):
    a = await jobs.enqueue("a", "u", JobOp.UPSERT)
    b = await jobs.enqueue("b", "u", JobOp.UPSERT)
    assert a is not None
    assert b is not None
    first = await jobs.claim_next("w", exclude_source_ids={"a"})
    assert first is not None
    assert first.id == b.id
    # The "a" job is excluded, so nothing more is claimable.
    assert await jobs.claim_next("w", exclude_source_ids={"a"}) is None


@pytest.mark.asyncio
async def test_claim_next_empty_exclude_is_noop(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert job is not None
    claimed = await jobs.claim_next("w", exclude_source_ids=set())
    assert claimed is not None
    assert claimed.id == job.id


@pytest.mark.asyncio
async def test_claim_next_skips_future_scheduled(conn, jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    # Push scheduled_at into the future.
    future = (datetime.now(UTC) + timedelta(seconds=60)).isoformat()
    await conn.execute(
        "UPDATE jobs SET scheduled_at = ? WHERE id = ?", (future, job.id)
    )
    await conn.commit()
    assert await jobs.claim_next("w") is None


@pytest.mark.asyncio
async def test_claim_next_atomic_under_concurrency(jobs):
    enqueued = []
    for i in range(5):
        j = await jobs.enqueue("s", f"u{i}", JobOp.UPSERT)
        assert j is not None
        enqueued.append(j)

    results = await asyncio.gather(*(jobs.claim_next(f"w{i}") for i in range(10)))
    claimed = [r for r in results if r is not None]
    assert len(claimed) == 5
    assert len({c.id for c in claimed}) == 5
    assert {c.id for c in claimed} == {j.id for j in enqueued}


@pytest.mark.asyncio
async def test_claim_next_atomic_across_independent_engines(tmp_path):
    """Two engines on one SQLite file stand in for two ingester processes.
    The claim is a single UPDATE...WHERE id=(subquery), so it stays atomic
    across connections — pool_size=1 only serializes within one engine."""
    cfg = QueueConfig(path=tmp_path / "shared-queue.db")
    engine_a = await open_queue(cfg)
    engine_b = await open_queue(cfg)
    try:
        repo_a = JobRepo(engine_a)
        repo_b = JobRepo(engine_b)

        enqueued = []
        for i in range(10):
            job = await repo_a.enqueue("s", f"u{i}", JobOp.UPSERT)
            assert job is not None
            enqueued.append(job)

        # Claims alternate across the two engines, contending on the same file.
        results = await asyncio.gather(
            *((repo_a if i % 2 == 0 else repo_b).claim_next(f"w{i}") for i in range(20))
        )
        claimed = [r for r in results if r is not None]

        assert len(claimed) == 10
        assert len({c.id for c in claimed}) == 10
        assert {c.id for c in claimed} == {j.id for j in enqueued}
        for job in enqueued:
            refreshed = await repo_a.get_job(job.id)
            assert refreshed is not None
            assert refreshed.status is JobStatus.CLAIMED
            assert refreshed.attempts == 1
    finally:
        await engine_a.dispose()
        await engine_b.dispose()


# --- terminal transitions ---


@pytest.mark.asyncio
async def test_mark_succeeded(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_succeeded(claimed.id, "w")
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.SUCCEEDED
    assert refreshed.completed_at is not None


@pytest.mark.asyncio
async def test_mark_dead_records_error(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_dead(claimed.id, "permanent failure", "w")
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.DEAD
    assert refreshed.last_error == "permanent failure"


@pytest.mark.asyncio
async def test_mark_succeeded_with_claimed_by_guard_skips_when_resurrected(jobs):
    """Reaper race: A claims, reaper resets, B re-claims, A finishes. A's
    mark_succeeded must be a no-op so B's in-flight work isn't clobbered."""
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    a = await jobs.claim_next("worker-A")
    assert a is not None
    # Reaper resets A's stale claim.
    await jobs.reap_stale(claim_timeout_seconds=0)
    # B picks the job up.
    b = await jobs.claim_next("worker-B")
    assert b is not None and b.id == job.id and b.claimed_by == "worker-B"
    # A finally finishes and tries to mark succeeded — guarded, no-op.
    assert await jobs.mark_succeeded(job.id, "worker-A") is False
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.CLAIMED
    assert refreshed.claimed_by == "worker-B"
    # B's own mark_succeeded does land.
    assert await jobs.mark_succeeded(job.id, "worker-B") is True


@pytest.mark.asyncio
async def test_mark_dead_with_claimed_by_guard_skips_when_resurrected(jobs):
    """Same guard semantics as mark_succeeded for the dead transition."""
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    a = await jobs.claim_next("worker-A")
    assert a is not None
    await jobs.reap_stale(claim_timeout_seconds=0)
    b = await jobs.claim_next("worker-B")
    assert b is not None and b.claimed_by == "worker-B"
    assert await jobs.mark_dead(job.id, "boom", "worker-A") is False
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.CLAIMED
    assert refreshed.last_error is None


@pytest.mark.asyncio
async def test_reschedule_with_claimed_by_guard_skips_when_resurrected(jobs):
    """Worker A times out → reaper resets → worker B re-claims → A surfaces
    with a TransientError and calls reschedule. The guard turns A's call
    into a no-op so B's claim isn't stripped back to `queued`."""
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    a = await jobs.claim_next("worker-A")
    assert a is not None
    await jobs.reap_stale(claim_timeout_seconds=0)
    b = await jobs.claim_next("worker-B")
    assert b is not None and b.claimed_by == "worker-B"
    assert await jobs.reschedule(job.id, 30.0, "transient", "worker-A") is False
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.CLAIMED
    assert refreshed.claimed_by == "worker-B"
    assert refreshed.last_error is None


@pytest.mark.asyncio
async def test_release_if_claimed_with_claimed_by_guard_skips_when_resurrected(jobs):
    """Cancel-cleanup path: worker A is cancelled while reaper-resurrected.
    release_if_claimed must not strip worker B's fresh claim."""
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    a = await jobs.claim_next("worker-A")
    assert a is not None
    await jobs.reap_stale(claim_timeout_seconds=0)
    b = await jobs.claim_next("worker-B")
    assert b is not None and b.claimed_by == "worker-B"
    assert await jobs.release_if_claimed(job.id, "worker-A") is False
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.CLAIMED
    assert refreshed.claimed_by == "worker-B"


# --- reschedule + retry ---


@pytest.mark.asyncio
async def test_reschedule_pushes_scheduled_at_into_future(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.reschedule(claimed.id, 30.0, "transient", "w")
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.QUEUED
    assert refreshed.claimed_at is None
    assert refreshed.claimed_by is None
    assert refreshed.last_error == "transient"
    assert refreshed.scheduled_at > datetime.now(UTC)


@pytest.mark.asyncio
async def test_reschedule_then_claim_skips_until_due(conn, jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.reschedule(claimed.id, 60.0, "transient", "w")
    assert await jobs.claim_next("w") is None

    # Backdate scheduled_at to simulate the delay elapsing.
    past = datetime.now(UTC).isoformat()
    await conn.execute("UPDATE jobs SET scheduled_at = ? WHERE id = ?", (past, job.id))
    await conn.commit()
    assert await jobs.claim_next("w") is not None


@pytest.mark.asyncio
async def test_retry_revives_dead_job(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_dead(claimed.id, "boom", "w")

    revived = await jobs.retry(job.id)
    assert revived.status is JobStatus.QUEUED
    assert revived.attempts == 0
    assert revived.last_error is None
    assert revived.claimed_at is None
    assert revived.claimed_by is None
    assert revived.completed_at is None


@pytest.mark.asyncio
async def test_retry_unknown_raises(jobs):
    with pytest.raises(KeyError):
        await jobs.retry("not-a-real-id")


@pytest.mark.asyncio
async def test_retry_refuses_claimed_job(jobs):
    """Resetting a `claimed` row would race with the worker still
    processing it: claim_next would re-claim and a second worker would
    process the same URI."""
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    assert claimed.status is JobStatus.CLAIMED

    with pytest.raises(KeyError):
        await jobs.retry(job.id)

    # Row is untouched.
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.CLAIMED
    assert refreshed.attempts == 1


@pytest.mark.asyncio
async def test_retry_returns_live_sibling_instead_of_colliding(jobs):
    """Retrying a dead job when a live job already exists for the same
    (source_id, uri) is idempotent: it returns the live sibling rather than
    violating uq_jobs_live (which previously surfaced as a 500)."""
    first = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert first is not None
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_dead(claimed.id, "boom", "w")

    # A fresh live job for the same (source, uri) — e.g. re-discovered by a sweep.
    live = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert live is not None
    assert live.id != first.id

    result = await jobs.retry(first.id)
    assert result.id == live.id
    assert result.status is JobStatus.QUEUED
    # The originally-dead row stays dead — not duplicated into a second live job.
    refreshed = await jobs.get_job(first.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.DEAD


@pytest.mark.asyncio
async def test_retry_refuses_succeeded_job(jobs):
    """Succeeded rows should be re-ingested through the UPSERT path, not
    re-run from the queue."""
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_succeeded(claimed.id, "w")

    with pytest.raises(KeyError):
        await jobs.retry(job.id)

    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.SUCCEEDED


# --- cancel ---


@pytest.mark.asyncio
async def test_cancel_queued_removes_row(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert await jobs.cancel(job.id) is True
    assert await jobs.get_job(job.id) is None


@pytest.mark.asyncio
async def test_cancel_succeeded_returns_false(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_succeeded(claimed.id, "w")
    assert await jobs.cancel(job.id) is False


# --- reap_stale ---


@pytest.mark.asyncio
async def test_reap_stale_resets_old_claims(conn, jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    assert claimed.attempts == 1

    # Backdate claimed_at to simulate a crashed worker.
    long_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    await conn.execute(
        "UPDATE jobs SET claimed_at = ? WHERE id = ?", (long_ago, job.id)
    )
    await conn.commit()

    reset = await jobs.reap_stale(claim_timeout_seconds=60)
    assert reset == 1
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.QUEUED
    assert refreshed.claimed_at is None
    assert refreshed.claimed_by is None
    # claim_next incremented to 1; reap undoes that since a crashed worker
    # isn't a consumed attempt — matches release_if_claimed semantics.
    assert refreshed.attempts == 0


@pytest.mark.asyncio
async def test_reap_stale_leaves_fresh_claims_alone(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None

    reset = await jobs.reap_stale(claim_timeout_seconds=3600)
    assert reset == 0
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.CLAIMED


# --- prune_dead ---


@pytest.mark.asyncio
async def test_prune_dead_removes_matching_dead_rows(jobs):
    """A dead UPSERT becomes stale once a sibling DELETE has resolved the URI;
    prune_dead() removes it so the DLQ stops showing resolved entries."""
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert job is not None
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_dead(claimed.id, "boom", "w")

    pruned = await jobs.prune_dead("s", "u")
    assert pruned == 1
    assert await jobs.get_job(job.id) is None


@pytest.mark.asyncio
async def test_prune_dead_leaves_non_dead_rows_alone(jobs):
    """Queued/claimed/succeeded rows for the same (source, uri) are not
    touched — only `dead` is purged."""
    queued = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert queued is not None

    pruned = await jobs.prune_dead("s", "u")
    assert pruned == 0
    refreshed = await jobs.get_job(queued.id)
    assert refreshed is not None and refreshed.status is JobStatus.QUEUED


@pytest.mark.asyncio
async def test_prune_dead_scoped_to_matching_uri(jobs):
    """Dead rows for other URIs (and other sources) survive."""
    j1 = await jobs.enqueue("s", "u1", JobOp.UPSERT)
    assert j1 is not None
    await jobs.mark_dead((await jobs.claim_next("w")).id, "err", "w")

    j2 = await jobs.enqueue("s", "u2", JobOp.UPSERT)
    assert j2 is not None
    await jobs.mark_dead((await jobs.claim_next("w")).id, "err", "w")

    pruned = await jobs.prune_dead("s", "u1")
    assert pruned == 1
    assert await jobs.get_job(j1.id) is None
    assert await jobs.get_job(j2.id) is not None


# --- prune_terminal ---


@pytest.mark.asyncio
async def test_prune_terminal_deletes_old_terminal_rows(jobs, conn):
    """Succeeded and dead rows whose completed_at is older than the window
    are deleted so the table doesn't grow without bound."""
    ok = await jobs.enqueue("s", "u1", JobOp.UPSERT)
    await jobs.claim_next("w")
    await jobs.mark_succeeded(ok.id, "w")

    bad = await jobs.enqueue("s", "u2", JobOp.UPSERT)
    await jobs.claim_next("w")
    await jobs.mark_dead(bad.id, "boom", "w")

    long_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    await conn.execute(
        "UPDATE jobs SET completed_at = ? WHERE id IN (?, ?)",
        (long_ago, ok.id, bad.id),
    )
    await conn.commit()

    pruned = await jobs.prune_terminal(max_age_seconds=60)
    assert pruned == 2
    assert await jobs.get_job(ok.id) is None
    assert await jobs.get_job(bad.id) is None


@pytest.mark.asyncio
async def test_prune_terminal_keeps_recent_terminal_rows(jobs):
    """A freshly-completed succeeded row survives — only rows past the
    retention window are removed."""
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    await jobs.claim_next("w")
    await jobs.mark_succeeded(job.id, "w")

    pruned = await jobs.prune_terminal(max_age_seconds=3600)
    assert pruned == 0
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None and refreshed.status is JobStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_prune_terminal_ignores_non_terminal_rows(jobs):
    """Queued/claimed rows have no completed_at and are never pruned,
    regardless of the window."""
    queued = await jobs.enqueue("s", "u1", JobOp.UPSERT)
    claimed = await jobs.enqueue("s", "u2", JobOp.UPSERT)
    await jobs.claim_next("w")

    pruned = await jobs.prune_terminal(max_age_seconds=0)
    assert pruned == 0
    assert (await jobs.get_job(queued.id)) is not None
    assert (await jobs.get_job(claimed.id)) is not None


# --- release_if_claimed ---


@pytest.mark.asyncio
async def test_release_if_claimed_resets_claimed_job_and_decrements_attempts(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert job is not None
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    assert claimed.attempts == 1

    released = await jobs.release_if_claimed(job.id, "w")
    assert released is True

    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.QUEUED
    assert refreshed.claimed_at is None
    assert refreshed.claimed_by is None
    assert refreshed.attempts == 0


@pytest.mark.asyncio
async def test_release_if_claimed_noop_on_already_queued(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert job is not None

    released = await jobs.release_if_claimed(job.id, "w")
    assert released is False
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.QUEUED
    assert refreshed.attempts == 0


@pytest.mark.asyncio
async def test_release_if_claimed_noop_on_succeeded(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert job is not None
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_succeeded(job.id, "w")

    released = await jobs.release_if_claimed(job.id, "w")
    assert released is False
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.SUCCEEDED


# --- list / counts ---


@pytest.mark.asyncio
async def test_list_jobs_with_filters(jobs):
    j1 = await jobs.enqueue("s1", "u1", JobOp.UPSERT)
    j2 = await jobs.enqueue("s2", "u2", JobOp.UPSERT)
    j3 = await jobs.enqueue("s1", "u3", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_succeeded(claimed.id, "w")

    all_jobs = await jobs.list_jobs()
    assert len(all_jobs) == 3

    by_source = await jobs.list_jobs(source_id="s1")
    assert {j.id for j in by_source} == {j1.id, j3.id}

    by_status = await jobs.list_jobs(status=JobStatus.QUEUED)
    assert {j.id for j in by_status} == {j2.id, j3.id}

    by_uri = await jobs.list_jobs(uri="u3")
    assert {j.id for j in by_uri} == {j3.id}


@pytest.mark.asyncio
async def test_counts_by_status(jobs):
    await jobs.enqueue("s", "u1", JobOp.UPSERT)
    j2 = await jobs.enqueue("s", "u2", JobOp.UPSERT)
    j3 = await jobs.enqueue("s", "u3", JobOp.UPSERT)

    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_succeeded(claimed.id, "w")
    claimed_j2 = await jobs.claim_next("w")
    assert claimed_j2 is not None and claimed_j2.id == j2.id
    await jobs.mark_dead(j2.id, "err", "w")

    counts = await jobs.counts_by_status()
    assert counts == {"queued": 1, "succeeded": 1, "dead": 1}
    assert j3.id  # silence unused


# --- stats ---


@pytest.mark.asyncio
async def test_count_succeeded_since_only_includes_recent(jobs, conn):
    await jobs.enqueue("s", "old", JobOp.UPSERT)
    await jobs.enqueue("s", "new", JobOp.UPSERT)

    old_claim = await jobs.claim_next("w")
    assert old_claim is not None
    await jobs.mark_succeeded(old_claim.id, "w")
    long_ago = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    await conn.execute(
        "UPDATE jobs SET completed_at = ? WHERE id = ?", (long_ago, old_claim.id)
    )
    await conn.commit()

    new_claim = await jobs.claim_next("w")
    assert new_claim is not None
    await jobs.mark_succeeded(new_claim.id, "w")

    assert await jobs.count_succeeded_since(60) == 1
    assert await jobs.count_succeeded_since(86400) == 2


@pytest.mark.asyncio
async def test_oldest_queued_age_seconds_none_when_empty(jobs):
    assert await jobs.oldest_queued_age_seconds() is None


@pytest.mark.asyncio
async def test_oldest_queued_age_seconds_returns_oldest(jobs, conn):
    old = await jobs.enqueue("s", "old", JobOp.UPSERT)
    await jobs.enqueue("s", "new", JobOp.UPSERT)
    backdate = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()
    assert old is not None
    await conn.execute(
        "UPDATE jobs SET scheduled_at = ? WHERE id = ?", (backdate, old.id)
    )
    await conn.commit()

    age = await jobs.oldest_queued_age_seconds()
    assert age is not None
    assert 119 <= age <= 125


@pytest.mark.asyncio
async def test_oldest_queued_age_seconds_ignores_future_scheduled(jobs, conn):
    """A job whose scheduled_at is in the future (e.g. after a backoff
    reschedule) isn't ready to run, so it shouldn't count toward backlog age."""
    j = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert j is not None
    future = (datetime.now(UTC) + timedelta(seconds=600)).isoformat()
    await conn.execute("UPDATE jobs SET scheduled_at = ? WHERE id = ?", (future, j.id))
    await conn.commit()
    assert await jobs.oldest_queued_age_seconds() is None


@pytest.mark.asyncio
async def test_counts_by_source_groups_correctly(jobs):
    # Enqueue s2 first so claim_next picks it up before the s1 rows; then
    # mark_dead routes through the production claim→terminal transition.
    j3 = await jobs.enqueue("s2", "u3", JobOp.UPSERT)
    assert j3 is not None
    claimed = await jobs.claim_next("w")
    assert claimed is not None and claimed.id == j3.id
    await jobs.mark_dead(j3.id, "boom", "w")
    await jobs.enqueue("s1", "u1", JobOp.UPSERT)
    await jobs.enqueue("s1", "u2", JobOp.UPSERT)

    assert await jobs.counts_by_source("queued") == {"s1": 2}
    assert await jobs.counts_by_source("dead") == {"s2": 1}
    assert await jobs.counts_by_source("queued", "claimed") == {"s1": 2}


@pytest.mark.asyncio
async def test_counts_by_source_no_statuses_returns_empty(jobs):
    await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert await jobs.counts_by_source() == {}


# --- sync state ---


@pytest.mark.asyncio
async def test_sync_state_get_revision_snapshot_empty(sync):
    assert await sync.get_revision_snapshot("unknown") == {}


@pytest.mark.asyncio
async def test_sync_state_upsert_and_get(sync):
    await sync.upsert("s", "u1", revision="abc", content_hash="m1")
    await sync.upsert("s", "u2", revision="def", content_hash="m2")
    assert await sync.get_revision_snapshot("s") == {"u1": "abc", "u2": "def"}


@pytest.mark.asyncio
async def test_sync_state_upsert_overwrites(sync):
    await sync.upsert("s", "u1", revision="abc", content_hash="m1")
    await sync.upsert("s", "u1", revision="def", content_hash="m2", ingested=True)
    assert await sync.get_revision_snapshot("s") == {"u1": "def"}


@pytest.mark.asyncio
async def test_sync_state_delete_removes_entry(sync):
    await sync.upsert("s", "u1", revision="abc", content_hash="m1")
    await sync.delete("s", "u1")
    assert await sync.get_revision_snapshot("s") == {}


@pytest.mark.asyncio
async def test_sync_state_snapshot_scoped_per_source(sync):
    await sync.upsert("s1", "u", revision="abc", content_hash="m")
    await sync.upsert("s2", "u", revision="def", content_hash="m")
    assert await sync.get_revision_snapshot("s1") == {"u": "abc"}
    assert await sync.get_revision_snapshot("s2") == {"u": "def"}


@pytest.mark.asyncio
async def test_sync_state_revision_snapshot_excludes_null_revision_rows(sync):
    """get_revision_snapshot returns only rows with a stored revision —
    sources compare against this for UPSERT/UNCHANGED. Rows without a
    revision (HTTP without ETag, worker DLQ'd before completion) are
    visible via list_known_uris instead."""
    await sync.upsert("s", "u", revision=None, content_hash=None)
    assert await sync.get_revision_snapshot("s") == {}
    assert await sync.list_known_uris("s") == {"u"}


@pytest.mark.asyncio
async def test_sync_state_upsert_preserves_revision_when_none(sync):
    """upsert(revision=None) leaves an existing revision in place."""
    await sync.upsert("s", "u", revision="v1", content_hash="hash-v1")
    await sync.upsert("s", "u", revision=None, content_hash=None)
    row = await sync.get_row("s", "u")
    assert row is not None
    assert row.revision == "v1"
    assert row.content_hash == "hash-v1"
    assert await sync.get_revision_snapshot("s") == {"u": "v1"}


@pytest.mark.asyncio
async def test_sync_state_upsert_replaces_revision_when_provided(sync):
    """upsert with a non-None revision overwrites the existing one."""
    await sync.upsert("s", "u", revision="v1", content_hash="hash-v1")
    await sync.upsert("s", "u", revision="v2", content_hash="hash-v2", ingested=True)
    row = await sync.get_row("s", "u")
    assert row is not None
    assert row.revision == "v2"
    assert row.content_hash == "hash-v2"
    assert row.last_ingested_at is not None


@pytest.mark.asyncio
async def test_sync_state_batch_upsert_inserts_multiple_rows(sync):
    """batch_upsert writes many rows in a single transaction."""
    await sync.batch_upsert(
        [
            SyncRow("s", "u1", "rev1", "hash1", False),
            SyncRow("s", "u2", "rev2", "hash2", False),
            SyncRow("s", "u3", "rev3", None, True),
        ]
    )
    assert await sync.get_revision_snapshot("s") == {
        "u1": "rev1",
        "u2": "rev2",
        "u3": "rev3",
    }
    row = await sync.get_row("s", "u3")
    assert row is not None
    assert row.last_ingested_at is not None


@pytest.mark.asyncio
async def test_sync_state_batch_upsert_updates_existing(sync):
    """batch_upsert applies ON CONFLICT update semantics like upsert()."""
    await sync.upsert("s", "u1", revision="old", content_hash="old-hash")
    await sync.batch_upsert(
        [
            SyncRow("s", "u1", "new", "new-hash", False),
        ]
    )
    row = await sync.get_row("s", "u1")
    assert row is not None
    assert row.revision == "new"
    assert row.content_hash == "new-hash"


@pytest.mark.asyncio
async def test_sync_state_batch_upsert_preserves_revision_when_none(sync):
    """batch_upsert with revision=None leaves existing revision in place."""
    await sync.upsert("s", "u1", revision="keep", content_hash="keep-hash")
    await sync.batch_upsert(
        [
            SyncRow("s", "u1", None, None, False),
        ]
    )
    row = await sync.get_row("s", "u1")
    assert row is not None
    assert row.revision == "keep"
    assert row.content_hash == "keep-hash"


@pytest.mark.asyncio
async def test_sync_state_batch_upsert_empty_is_noop(sync):
    """batch_upsert with an empty list does nothing."""
    await sync.batch_upsert([])
    assert await sync.get_revision_snapshot("s") == {}
