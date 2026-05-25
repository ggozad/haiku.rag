import asyncio
from datetime import UTC, datetime, timedelta

import aiosqlite
import pytest

from haiku.rag.ingester.queue.migrations import apply_migrations, open_queue
from haiku.rag.ingester.queue.models import JobOp, JobStatus
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo


@pytest.fixture
async def conn(tmp_path):
    path = tmp_path / "queue.db"
    connection = await aiosqlite.connect(str(path))
    connection.row_factory = aiosqlite.Row
    await apply_migrations(connection)
    yield connection
    await connection.close()


@pytest.fixture
def jobs(conn):
    return JobRepo(conn)


@pytest.fixture
def sync(conn):
    return SyncStateRepo(conn)


# --- migrations / schema ---


@pytest.mark.asyncio
async def test_apply_migrations_sets_schema_version(conn):
    cursor = await conn.execute("SELECT version FROM schema_version")
    row = await cursor.fetchone()
    assert row is not None
    assert row["version"] >= 1


@pytest.mark.asyncio
async def test_apply_migrations_is_idempotent(conn):
    await apply_migrations(conn)
    await apply_migrations(conn)
    cursor = await conn.execute("SELECT COUNT(*) AS n FROM schema_version")
    row = await cursor.fetchone()
    assert row["n"] == 1


@pytest.mark.asyncio
async def test_open_queue_creates_file_and_schema(tmp_path):
    path = tmp_path / "subdir" / "queue.db"
    connection = await open_queue(path)
    try:
        assert path.exists()
        cursor = await connection.execute("SELECT version FROM schema_version")
        row = await cursor.fetchone()
        assert row is not None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_dlq_view_exposes_dead_jobs(conn, jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    await jobs.mark_dead(job.id, "permanent")

    cursor = await conn.execute("SELECT id FROM dlq")
    rows = await cursor.fetchall()
    assert [r["id"] for r in rows] == [job.id]


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
    await jobs.mark_dead(first.id, "error")
    second = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert second is not None
    assert second.id != first.id


@pytest.mark.asyncio
async def test_enqueue_after_succeeded_succeeds(jobs):
    await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_succeeded(claimed.id)
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
    await jobs.mark_succeeded(claimed.id)

    await jobs.enqueue("src", "u-bad", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_dead(claimed.id, "permanent")

    assert await jobs.has_pending("src") is False


@pytest.mark.asyncio
async def test_has_pending_is_per_source(jobs):
    """A backed-up source must not block an idle one."""
    await jobs.enqueue("busy", "u", JobOp.UPSERT)
    assert await jobs.has_pending("busy") is True
    assert await jobs.has_pending("idle") is False


@pytest.mark.asyncio
async def test_enqueue_different_ops_coexist(jobs):
    upsert = await jobs.enqueue("s", "u", JobOp.UPSERT)
    delete = await jobs.enqueue("s", "u", JobOp.DELETE)
    assert upsert is not None
    assert delete is not None


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


# --- terminal transitions ---


@pytest.mark.asyncio
async def test_mark_succeeded(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_succeeded(claimed.id)
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.SUCCEEDED
    assert refreshed.completed_at is not None


@pytest.mark.asyncio
async def test_mark_dead_records_error(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_dead(claimed.id, "permanent failure")
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.DEAD
    assert refreshed.last_error == "permanent failure"


# --- reschedule + retry ---


@pytest.mark.asyncio
async def test_reschedule_pushes_scheduled_at_into_future(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.reschedule(claimed.id, delay_seconds=30.0, error="transient")
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
    await jobs.reschedule(claimed.id, delay_seconds=60.0, error="transient")
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
    await jobs.mark_dead(claimed.id, "boom")

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
    await jobs.mark_succeeded(claimed.id)
    assert await jobs.cancel(job.id) is False


# --- reap_stale ---


@pytest.mark.asyncio
async def test_reap_stale_resets_old_claims(conn, jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None

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


# --- release_if_claimed ---


@pytest.mark.asyncio
async def test_release_if_claimed_resets_claimed_job_and_decrements_attempts(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert job is not None
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    assert claimed.attempts == 1

    released = await jobs.release_if_claimed(job.id)
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

    released = await jobs.release_if_claimed(job.id)
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
    await jobs.mark_succeeded(job.id)

    released = await jobs.release_if_claimed(job.id)
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
    await jobs.mark_succeeded(claimed.id)

    all_jobs = await jobs.list_jobs()
    assert len(all_jobs) == 3

    by_source = await jobs.list_jobs(source_id="s1")
    assert {j.id for j in by_source} == {j1.id, j3.id}

    by_status = await jobs.list_jobs(status=JobStatus.QUEUED)
    assert {j.id for j in by_status} == {j2.id, j3.id}


@pytest.mark.asyncio
async def test_counts_by_status(jobs):
    await jobs.enqueue("s", "u1", JobOp.UPSERT)
    j2 = await jobs.enqueue("s", "u2", JobOp.UPSERT)
    j3 = await jobs.enqueue("s", "u3", JobOp.UPSERT)

    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_succeeded(claimed.id)
    await jobs.mark_dead(j2.id, "err")

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
    await jobs.mark_succeeded(old_claim.id)
    long_ago = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    await conn.execute(
        "UPDATE jobs SET completed_at = ? WHERE id = ?", (long_ago, old_claim.id)
    )
    await conn.commit()

    new_claim = await jobs.claim_next("w")
    assert new_claim is not None
    await jobs.mark_succeeded(new_claim.id)

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
    await jobs.enqueue("s1", "u1", JobOp.UPSERT)
    await jobs.enqueue("s1", "u2", JobOp.UPSERT)
    j3 = await jobs.enqueue("s2", "u3", JobOp.UPSERT)
    assert j3 is not None
    await jobs.mark_dead(j3.id, "boom")

    assert await jobs.counts_by_source("queued") == {"s1": 2}
    assert await jobs.counts_by_source("dead") == {"s2": 1}
    assert await jobs.counts_by_source("queued", "claimed") == {"s1": 2}


@pytest.mark.asyncio
async def test_counts_by_source_no_statuses_returns_empty(jobs):
    await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert await jobs.counts_by_source() == {}


# --- sync state ---


@pytest.mark.asyncio
async def test_sync_state_get_snapshot_empty(sync):
    assert await sync.get_snapshot("unknown") == {}


@pytest.mark.asyncio
async def test_sync_state_upsert_and_get(sync):
    await sync.upsert("s", "u1", revision="abc", content_hash="m1")
    await sync.upsert("s", "u2", revision="def", content_hash="m2")
    assert await sync.get_snapshot("s") == {"u1": "abc", "u2": "def"}


@pytest.mark.asyncio
async def test_sync_state_upsert_overwrites(sync):
    await sync.upsert("s", "u1", revision="abc", content_hash="m1")
    await sync.upsert("s", "u1", revision="def", content_hash="m2", ingested=True)
    assert await sync.get_snapshot("s") == {"u1": "def"}


@pytest.mark.asyncio
async def test_sync_state_delete_removes_entry(sync):
    await sync.upsert("s", "u1", revision="abc", content_hash="m1")
    await sync.delete("s", "u1")
    assert await sync.get_snapshot("s") == {}


@pytest.mark.asyncio
async def test_sync_state_snapshot_scoped_per_source(sync):
    await sync.upsert("s1", "u", revision="abc", content_hash="m")
    await sync.upsert("s2", "u", revision="def", content_hash="m")
    assert await sync.get_snapshot("s1") == {"u": "abc"}
    assert await sync.get_snapshot("s2") == {"u": "def"}
