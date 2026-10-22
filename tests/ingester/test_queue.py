import asyncio
import sys
from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy as sa
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import QueuePool

from haiku.rag.config import QueueConfig
from haiku.rag.ingester.exceptions import QueueError
from haiku.rag.ingester.queue import migrations, repository
from haiku.rag.ingester.queue.db import SCHEMA_VERSION
from haiku.rag.ingester.queue.db import jobs as jobs_table
from haiku.rag.ingester.queue.migrations import (
    apply_migrations,
    make_engine,
    open_queue,
    verify_schema,
)
from haiku.rag.ingester.queue.models import JobOp, JobStatus, SyncRow
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo, _insert

# --- migrations / schema ---


async def test_apply_migrations_sets_schema_version(conn):
    cursor = await conn.execute("SELECT version FROM schema_version")
    row = await cursor.fetchone()
    assert row is not None
    assert row["version"] >= 1


async def test_apply_migrations_is_idempotent(engine, conn):
    await apply_migrations(engine)
    await apply_migrations(engine)
    cursor = await conn.execute("SELECT COUNT(*) AS n FROM schema_version")
    row = await cursor.fetchone()
    assert row["n"] == 1


_V1_JOBS_DDL = """
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    uri TEXT NOT NULL,
    op TEXT NOT NULL,
    content_hash TEXT,
    revision TEXT,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 5,
    last_error TEXT,
    extra TEXT,
    enqueued_at TEXT NOT NULL,
    scheduled_at TEXT NOT NULL,
    claimed_at TEXT,
    claimed_by TEXT,
    completed_at TEXT
)
"""

_V3_SYNC_STATE_DDL = """
CREATE TABLE sync_state (
    source_id TEXT NOT NULL,
    uri TEXT NOT NULL,
    revision TEXT,
    content_hash TEXT,
    last_seen_at TEXT NOT NULL,
    last_ingested_at TEXT,
    PRIMARY KEY (source_id, uri)
)
"""

_V3_JOBS_INDEX_DDL = [
    "CREATE UNIQUE INDEX uq_jobs_live ON jobs (source_id, uri) "
    "WHERE status IN ('queued', 'claimed')",
    "CREATE UNIQUE INDEX uq_jobs_blocking_op ON jobs "
    "(source_id, uri, op, coalesce(revision, '')) "
    "WHERE status IN ('queued', 'claimed') "
    "OR status = 'dead' AND conversion_stalled IS 1",
    "CREATE INDEX idx_jobs_claimable ON jobs (scheduled_at) WHERE status = 'queued'",
    "CREATE INDEX idx_jobs_succeeded_completed ON jobs (completed_at) "
    "WHERE status = 'succeeded'",
]


async def _downgrade_to_v3(engine) -> None:
    """Strip a current queue back to schema 3: no `namespace` anywhere."""
    async with engine.begin() as conn:
        for index in jobs_table.indexes:
            await conn.execute(sa.text(f"DROP INDEX {index.name}"))
        await conn.execute(sa.text("ALTER TABLE jobs DROP COLUMN namespace"))
        for ddl in _V3_JOBS_INDEX_DDL:
            await conn.execute(sa.text(ddl))
        await conn.execute(sa.text("DROP TABLE sync_state"))
        await conn.execute(sa.text(_V3_SYNC_STATE_DDL))
        await conn.execute(sa.text("UPDATE schema_version SET version = 3"))


async def test_migration_v1_to_v2_adds_and_backfills_heartbeat(tmp_path):
    """A real v1 DB (no last_heartbeat_at, schema_version=1) gains the column
    and has it backfilled from claimed_at for in-flight rows only."""
    db = tmp_path / "v1.db"
    engine = create_async_engine(URL.create("sqlite+aiosqlite", database=str(db)))
    try:
        when = "2026-01-01T00:00:00+00:00"
        claimed_when = "2026-01-01T00:05:00+00:00"
        async with engine.begin() as conn:
            await conn.execute(sa.text(_V1_JOBS_DDL))
            await conn.execute(
                sa.text("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
            )
            await conn.execute(sa.text("INSERT INTO schema_version VALUES (1)"))
            await conn.execute(
                sa.text(
                    "INSERT INTO jobs (id, source_id, uri, op, status, attempts, "
                    "max_attempts, enqueued_at, scheduled_at, claimed_at, claimed_by) "
                    "VALUES ('claimed-1', 's', 'u1', 'upsert', 'claimed', 1, 5, "
                    f"'{when}', '{when}', '{claimed_when}', 'w')"
                )
            )
            await conn.execute(
                sa.text(
                    "INSERT INTO jobs (id, source_id, uri, op, status, attempts, "
                    "max_attempts, enqueued_at, scheduled_at) "
                    "VALUES ('queued-1', 's', 'u2', 'upsert', 'queued', 0, 5, "
                    f"'{when}', '{when}')"
                )
            )

        version = await apply_migrations(engine)
        assert version == SCHEMA_VERSION

        async with engine.connect() as conn:
            cols = (await conn.execute(sa.text("PRAGMA table_info(jobs)"))).fetchall()
            assert "last_heartbeat_at" in {c[1] for c in cols}
            rows = {
                row[0]: row[1]
                for row in (
                    await conn.execute(
                        sa.text("SELECT id, last_heartbeat_at FROM jobs")
                    )
                ).fetchall()
            }
        # Backfilled from claimed_at for the in-flight row, left NULL otherwise.
        assert rows["claimed-1"] == claimed_when
        assert rows["queued-1"] is None
    finally:
        await engine.dispose()


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


async def test_make_engine_postgres_pool():
    """The Postgres branch builds a pre-ping engine holding at most five
    connections, without connecting."""
    engine = make_engine(QueueConfig(dburi="postgresql+asyncpg://u:p@localhost/db"))
    try:
        assert engine.dialect.name == "postgresql"
        pool = engine.pool
        assert isinstance(pool, QueuePool)
        assert pool._pre_ping is True
        assert pool.size() == 2
        assert pool._max_overflow == 3
    finally:
        await engine.dispose()


async def test_verify_schema_accepts_the_current_version(engine):
    await verify_schema(engine)


async def test_verify_schema_names_migrate_when_behind(engine):
    async with engine.begin() as conn:
        await conn.execute(sa.text("UPDATE schema_version SET version = 3"))
    with pytest.raises(QueueError, match="schema 3.*queue migrate --dburi"):
        await verify_schema(engine)


async def test_verify_schema_names_the_upgrade_when_ahead(engine):
    async with engine.begin() as conn:
        await conn.execute(sa.text("UPDATE schema_version SET version = 99"))
    with pytest.raises(QueueError, match="schema 99.*upgrade haiku.rag"):
        await verify_schema(engine)


async def test_verify_schema_names_init_without_a_schema(tmp_path):
    engine = create_async_engine(
        URL.create("sqlite+aiosqlite", database=str(tmp_path / "empty.db"))
    )
    try:
        with pytest.raises(QueueError, match="no schema.*queue init --dburi"):
            await verify_schema(engine)
    finally:
        await engine.dispose()


async def test_verify_schema_names_init_on_an_empty_version_table(engine):
    async with engine.begin() as conn:
        await conn.execute(sa.text("DELETE FROM schema_version"))
    with pytest.raises(QueueError, match="no schema.*queue init --dburi"):
        await verify_schema(engine)


async def test_verify_schema_writes_nothing(engine, conn):
    """A read-only account can run it: the version row is untouched and no
    table is created."""
    async with engine.begin() as sql:
        await sql.execute(sa.text("UPDATE schema_version SET version = 3"))
        await sql.execute(sa.text("DROP TABLE sync_state"))
    with pytest.raises(QueueError):
        await verify_schema(engine)
    cursor = await conn.execute("SELECT version FROM schema_version")
    assert (await cursor.fetchone())["version"] == 3
    cursor = await conn.execute(
        "SELECT name FROM sqlite_master WHERE name = 'sync_state'"
    )
    assert await cursor.fetchone() is None


async def test_apply_migrations_refuses_a_newer_schema(engine, conn):
    async with engine.begin() as sql:
        await sql.execute(sa.text("UPDATE schema_version SET version = 99"))
    with pytest.raises(QueueError, match="schema 99.*upgrade haiku.rag"):
        await apply_migrations(engine)
    cursor = await conn.execute("SELECT version FROM schema_version")
    assert (await cursor.fetchone())["version"] == 99


async def test_open_queue_disposes_the_engine_it_refuses(tmp_path, monkeypatch):
    """A refused queue leaves no pool behind to hold the loop open."""
    config = QueueConfig(path=tmp_path / "queue.db")
    engine = await open_queue(config)
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text("UPDATE schema_version SET version = 99"))
    finally:
        await engine.dispose()

    built = []

    def _spy(queue):
        engine = make_engine(queue)
        built.append((engine, engine.pool))
        return engine

    monkeypatch.setattr(migrations, "make_engine", _spy)
    with pytest.raises(QueueError):
        await open_queue(config)
    [(engine, pool_before)] = built
    assert engine.pool is not pool_before


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


@pytest.mark.parametrize(
    "char",
    [
        "#",
        pytest.param(
            "?",
            marks=pytest.mark.skipif(
                sys.platform == "win32", reason="Windows forbids ? in a filename"
            ),
        ),
    ],
)
async def test_open_queue_handles_path_with_url_chars(tmp_path, char):
    """The queue opens a file whose name carries a URL-significant character."""
    path = tmp_path / f"queue{char}weird.db"
    eng = await open_queue(QueueConfig(path=path))
    try:
        assert path.exists()
        assert not (tmp_path / "queue").exists()
    finally:
        await eng.dispose()


# --- enqueue ---


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


async def test_enqueue_returns_none_on_live_conflict(jobs):
    first = await jobs.enqueue("s", "u", JobOp.UPSERT)
    second = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert first is not None
    assert second is None


async def test_enqueue_after_dead_succeeds(jobs):
    first = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_dead(first.id, "error", "w")
    second = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert second is not None
    assert second.id != first.id


async def test_enqueue_after_succeeded_succeeds(jobs):
    await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_succeeded(claimed.id, "w")
    second = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert second is not None


async def test_has_pending_returns_false_on_empty_queue(jobs):
    assert await jobs.has_pending("src") is False


async def test_has_pending_true_for_queued_job(jobs):
    await jobs.enqueue("src", "u", JobOp.UPSERT)
    assert await jobs.has_pending("src") is True


async def test_has_pending_true_for_claimed_job(jobs):
    await jobs.enqueue("src", "u", JobOp.UPSERT)
    await jobs.claim_next("w")
    assert await jobs.has_pending("src") is True


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


async def test_has_pending_is_per_source(jobs):
    """A backed-up source must not block an idle one."""
    await jobs.enqueue("busy", "u", JobOp.UPSERT)
    assert await jobs.has_pending("busy") is True
    assert await jobs.has_pending("idle") is False


async def test_enqueue_drops_delete_when_upsert_is_live(jobs):
    """Stops a DELETE worker from removing a document a sibling UPSERT
    just ingested."""
    upsert = await jobs.enqueue("s", "u", JobOp.UPSERT)
    delete = await jobs.enqueue("s", "u", JobOp.DELETE)
    assert upsert is not None
    assert delete is None


async def test_enqueue_drops_upsert_when_delete_is_live(jobs):
    """Symmetric: live DELETE for a URI blocks a fresh UPSERT for the
    same URI. The next sweep after DELETE completes will re-emit the
    UPSERT if the file is back."""
    delete = await jobs.enqueue("s", "u", JobOp.DELETE)
    upsert = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert delete is not None
    assert upsert is None


async def test_enqueue_different_uris_independent_of_op(jobs):
    """Uniqueness is per-(source_id, uri), not per-(source_id, uri, op).
    Different URIs can be queued regardless of which op each one is."""
    a = await jobs.enqueue("s", "u-a", JobOp.UPSERT)
    b = await jobs.enqueue("s", "u-b", JobOp.DELETE)
    assert a is not None
    assert b is not None


# --- claim_next ---


async def test_claim_next_returns_none_when_empty(jobs):
    assert await jobs.claim_next("w") is None


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


async def test_claim_next_returns_oldest_first(jobs):
    j1 = await jobs.enqueue("s", "u1", JobOp.UPSERT)
    j2 = await jobs.enqueue("s", "u2", JobOp.UPSERT)
    first = await jobs.claim_next("w")
    second = await jobs.claim_next("w")
    assert first is not None
    assert second is not None
    assert first.id == j1.id
    assert second.id == j2.id


async def test_claim_next_claims_only_the_given_sources(jobs):
    a = await jobs.enqueue("a", "u", JobOp.UPSERT)
    b = await jobs.enqueue("b", "u", JobOp.UPSERT)
    assert a is not None
    assert b is not None
    first = await jobs.claim_next("w", source_ids={"b"})
    assert first is not None
    assert first.id == b.id
    assert await jobs.claim_next("w", source_ids={"b"}) is None


async def test_claim_next_with_no_sources_claims_nothing(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert job is not None
    assert await jobs.claim_next("w", source_ids=set()) is None


async def test_cancel_unconfigured_deletes_live_jobs_of_other_sources(jobs):
    kept = await jobs.enqueue("kept", "u", JobOp.UPSERT)
    claimed = await jobs.enqueue("gone", "claimed", JobOp.UPSERT)
    assert await jobs.claim_next("w", source_ids={"gone"}) is not None
    dead = await jobs.enqueue("gone", "dead", JobOp.UPSERT)
    assert kept is not None and claimed is not None and dead is not None
    dead_claim = await jobs.claim_next("w", source_ids={"gone"})
    assert dead_claim is not None and dead_claim.id == dead.id
    assert await jobs.mark_dead(dead.id, "boom", "w")
    queued = await jobs.enqueue("gone", "queued", JobOp.DELETE)
    assert queued is not None

    assert await jobs.cancel_unconfigured({"kept"}) == 2

    remaining = {job.id for job in await jobs.list_jobs(limit=50)}
    assert remaining == {kept.id, dead.id}


async def test_cancel_sources_deletes_live_jobs_of_those_sources(jobs):
    kept = await jobs.enqueue("kept", "u", JobOp.UPSERT)
    claimed = await jobs.enqueue("gone", "claimed", JobOp.UPSERT)
    assert await jobs.claim_next("w", source_ids={"gone"}) is not None
    dead = await jobs.enqueue("gone", "dead", JobOp.UPSERT)
    assert kept is not None and claimed is not None and dead is not None
    dead_claim = await jobs.claim_next("w", source_ids={"gone"})
    assert dead_claim is not None and dead_claim.id == dead.id
    assert await jobs.mark_dead(dead.id, "boom", "w")
    queued = await jobs.enqueue("gone", "queued", JobOp.DELETE)
    assert queued is not None

    assert await jobs.cancel_sources({"gone"}) == 2

    remaining = {job.id for job in await jobs.list_jobs(limit=50)}
    assert remaining == {kept.id, dead.id}


async def test_claim_next_skips_future_scheduled(conn, jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    # Push scheduled_at into the future.
    future = (datetime.now(UTC) + timedelta(seconds=60)).isoformat()
    await conn.execute(
        "UPDATE jobs SET scheduled_at = ? WHERE id = ?", (future, job.id)
    )
    await conn.commit()
    assert await jobs.claim_next("w") is None


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


async def test_mark_succeeded(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_succeeded(claimed.id, "w")
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.SUCCEEDED
    assert refreshed.completed_at is not None


async def test_mark_dead_records_error(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_dead(claimed.id, "permanent failure", "w")
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.DEAD
    assert refreshed.last_error == "permanent failure"


async def test_mark_succeeded_with_claimed_by_guard_skips_when_resurrected(jobs):
    """Reaper race: A claims, reaper resets, B re-claims, A finishes. A's
    mark_succeeded must be a no-op so B's in-flight work isn't clobbered."""
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    a = await jobs.claim_next("worker-A")
    assert a is not None
    # Reaper resets A's stale claim.
    await jobs.reap_stale(lease_ttl_seconds=0)
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


async def test_mark_dead_with_claimed_by_guard_skips_when_resurrected(jobs):
    """Same guard semantics as mark_succeeded for the dead transition."""
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    a = await jobs.claim_next("worker-A")
    assert a is not None
    await jobs.reap_stale(lease_ttl_seconds=0)
    b = await jobs.claim_next("worker-B")
    assert b is not None and b.claimed_by == "worker-B"
    assert await jobs.mark_dead(job.id, "boom", "worker-A") is False
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.CLAIMED
    assert refreshed.last_error is None


async def test_reschedule_with_claimed_by_guard_skips_when_resurrected(jobs):
    """Worker A times out → reaper resets → worker B re-claims → A surfaces
    with a TransientError and calls reschedule. The guard turns A's call
    into a no-op so B's claim isn't stripped back to `queued`."""
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    a = await jobs.claim_next("worker-A")
    assert a is not None
    await jobs.reap_stale(lease_ttl_seconds=0)
    b = await jobs.claim_next("worker-B")
    assert b is not None and b.claimed_by == "worker-B"
    assert await jobs.reschedule(job.id, 30.0, "transient", "worker-A") is False
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.CLAIMED
    assert refreshed.claimed_by == "worker-B"
    assert refreshed.last_error is None


async def test_release_if_claimed_with_claimed_by_guard_skips_when_resurrected(jobs):
    """Cancel-cleanup path: worker A is cancelled while reaper-resurrected.
    release_if_claimed must not strip worker B's fresh claim."""
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    a = await jobs.claim_next("worker-A")
    assert a is not None
    await jobs.reap_stale(lease_ttl_seconds=0)
    b = await jobs.claim_next("worker-B")
    assert b is not None and b.claimed_by == "worker-B"
    assert await jobs.release_if_claimed(job.id, "worker-A") is False
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.CLAIMED
    assert refreshed.claimed_by == "worker-B"


# --- reschedule + retry ---


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


async def test_retry_unknown_raises(jobs):
    with pytest.raises(KeyError):
        await jobs.retry("not-a-real-id")


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


async def test_cancel_queued_removes_row(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert await jobs.cancel(job.id) is True
    assert await jobs.get_job(job.id) is None


async def test_cancel_succeeded_returns_false(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_succeeded(claimed.id, "w")
    assert await jobs.cancel(job.id) is False


# --- reap_stale ---


async def test_reap_stale_resets_old_claims(conn, jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    assert claimed.attempts == 1

    # Backdate the lease to simulate a worker that stopped renewing (crash).
    long_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    await conn.execute(
        "UPDATE jobs SET claimed_at = ?, last_heartbeat_at = ? WHERE id = ?",
        (long_ago, long_ago, job.id),
    )
    await conn.commit()

    reset = await jobs.reap_stale(lease_ttl_seconds=60)
    assert reset == 1
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.QUEUED
    assert refreshed.claimed_at is None
    assert refreshed.claimed_by is None
    assert refreshed.last_heartbeat_at is None
    # claim_next incremented to 1; reap undoes that since a reaped worker
    # isn't a consumed attempt — matches release_if_claimed semantics.
    assert refreshed.attempts == 0


async def test_reap_stale_falls_back_to_claimed_at_when_no_heartbeat(conn, jobs):
    """A claim made by a process that doesn't write the lease (an older version
    sharing the queue) has last_heartbeat_at IS NULL. Reaping must fall back to
    claimed_at instead of stranding it forever (NULL comparisons are falsy)."""
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None

    long_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    await conn.execute(
        "UPDATE jobs SET claimed_at = ?, last_heartbeat_at = NULL WHERE id = ?",
        (long_ago, job.id),
    )
    await conn.commit()

    reset = await jobs.reap_stale(lease_ttl_seconds=60)
    assert reset == 1
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.QUEUED


async def test_reap_stale_leaves_fresh_claims_alone(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None

    reset = await jobs.reap_stale(lease_ttl_seconds=3600)
    assert reset == 0
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.CLAIMED


async def test_claim_next_sets_both_timestamps(jobs):
    await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    assert claimed.claimed_at is not None
    assert claimed.last_heartbeat_at is not None


async def test_renew_claims_keeps_a_stale_claim_alive(conn, jobs):
    """A claim whose claimed_at is old but whose lease was just renewed must
    not be reaped — this is the slow-but-alive worker the reaper used to kill."""
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None

    # Job started long ago, but the worker is alive and renewing.
    long_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    await conn.execute(
        "UPDATE jobs SET claimed_at = ? WHERE id = ?", (long_ago, job.id)
    )
    await conn.commit()
    renewed = await jobs.renew_claims({job.id: "w"})
    assert renewed == 1

    reset = await jobs.reap_stale(lease_ttl_seconds=60)
    assert reset == 0
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.CLAIMED


async def test_renew_claims_invariant_only_touches_heartbeat(conn, jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    claimed = await jobs.claim_next("w")
    assert claimed is not None and claimed.claimed_at is not None

    # Backdate the heartbeat so renewal demonstrably advances it.
    old = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    await conn.execute(
        "UPDATE jobs SET last_heartbeat_at = ? WHERE id = ?", (old, job.id)
    )
    await conn.commit()
    before = await jobs.get_job(job.id)
    assert before is not None and before.last_heartbeat_at is not None

    assert await jobs.renew_claims({job.id: "w"}) == 1
    after = await jobs.get_job(job.id)
    assert after is not None and after.last_heartbeat_at is not None
    assert after.claimed_at == before.claimed_at
    assert after.last_heartbeat_at > before.last_heartbeat_at


async def test_renew_claims_skips_reclaimed_job(jobs):
    """Renewal is guarded on claimed_by: a job reaped and re-claimed by another
    worker must not be renewed by the original (lost) claimant."""
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    a = await jobs.claim_next("worker-A")
    assert a is not None
    await jobs.reap_stale(lease_ttl_seconds=0)
    b = await jobs.claim_next("worker-B")
    assert b is not None and b.claimed_by == "worker-B"

    assert await jobs.renew_claims({job.id: "worker-A"}) == 0


async def test_renew_claims_empty_is_noop(jobs):
    assert await jobs.renew_claims({}) == 0


# --- prune_unsuccessful ---


async def test_prune_unsuccessful_removes_matching_dead_rows(jobs):
    """A dead UPSERT becomes stale once a sibling DELETE has resolved the URI;
    prune_unsuccessful() removes it so the DLQ stops showing resolved entries."""
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert job is not None
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_dead(claimed.id, "boom", "w")

    pruned = await jobs.prune_unsuccessful("s", "u")
    assert pruned == 1
    assert await jobs.get_job(job.id) is None


async def test_prune_unsuccessful_leaves_non_dead_rows_alone(jobs):
    """Queued/claimed/succeeded rows for the same (source, uri) are not
    touched — only `dead` is purged."""
    queued = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert queued is not None

    pruned = await jobs.prune_unsuccessful("s", "u")
    assert pruned == 0
    refreshed = await jobs.get_job(queued.id)
    assert refreshed is not None and refreshed.status is JobStatus.QUEUED


async def test_prune_unsuccessful_scoped_to_matching_uri(jobs):
    """Dead rows for other URIs (and other sources) survive."""
    j1 = await jobs.enqueue("s", "u1", JobOp.UPSERT)
    assert j1 is not None
    await jobs.mark_dead((await jobs.claim_next("w")).id, "err", "w")

    j2 = await jobs.enqueue("s", "u2", JobOp.UPSERT)
    assert j2 is not None
    await jobs.mark_dead((await jobs.claim_next("w")).id, "err", "w")

    pruned = await jobs.prune_unsuccessful("s", "u1")
    assert pruned == 1
    assert await jobs.get_job(j1.id) is None
    assert await jobs.get_job(j2.id) is not None


# --- prune_terminal ---


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


async def test_release_if_claimed_noop_on_already_queued(jobs):
    job = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert job is not None

    released = await jobs.release_if_claimed(job.id, "w")
    assert released is False
    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.QUEUED
    assert refreshed.attempts == 0


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


async def test_oldest_queued_age_seconds_none_when_empty(jobs):
    assert await jobs.oldest_queued_age_seconds() is None


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


async def test_oldest_queued_age_seconds_ignores_future_scheduled(jobs, conn):
    """A job whose scheduled_at is in the future (e.g. after a backoff
    reschedule) isn't ready to run, so it shouldn't count toward backlog age."""
    j = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert j is not None
    future = (datetime.now(UTC) + timedelta(seconds=600)).isoformat()
    await conn.execute("UPDATE jobs SET scheduled_at = ? WHERE id = ?", (future, j.id))
    await conn.commit()
    assert await jobs.oldest_queued_age_seconds() is None


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


async def test_counts_by_source_no_statuses_returns_empty(jobs):
    await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert await jobs.counts_by_source() == {}


# --- sync state ---


async def test_sync_state_get_revision_snapshot_empty(sync):
    assert await sync.get_revision_snapshot("unknown") == {}


async def test_sync_state_upsert_and_get(sync):
    await sync.upsert("s", "u1", revision="abc", content_hash="m1")
    await sync.upsert("s", "u2", revision="def", content_hash="m2")
    assert await sync.get_revision_snapshot("s") == {"u1": "abc", "u2": "def"}


async def test_sync_state_upsert_overwrites(sync):
    await sync.upsert("s", "u1", revision="abc", content_hash="m1")
    await sync.upsert("s", "u1", revision="def", content_hash="m2", ingested=True)
    assert await sync.get_revision_snapshot("s") == {"u1": "def"}


async def test_sync_state_delete_removes_entry(sync):
    await sync.upsert("s", "u1", revision="abc", content_hash="m1")
    await sync.delete("s", "u1")
    assert await sync.get_revision_snapshot("s") == {}


async def test_sync_state_delete_sources_removes_their_rows(sync):
    await sync.upsert("gone", "u1", revision="abc", content_hash="m1")
    await sync.upsert("gone", "u2")
    await sync.upsert("kept", "u1", revision="def", content_hash="m2")

    await sync.delete_sources({"gone"})

    assert await sync.list_known_uris("gone") == set()
    assert await sync.get_revision_snapshot("kept") == {"u1": "def"}


async def test_sync_state_snapshot_scoped_per_source(sync):
    await sync.upsert("s1", "u", revision="abc", content_hash="m")
    await sync.upsert("s2", "u", revision="def", content_hash="m")
    assert await sync.get_revision_snapshot("s1") == {"u": "abc"}
    assert await sync.get_revision_snapshot("s2") == {"u": "def"}


async def test_sync_state_revision_snapshot_excludes_null_revision_rows(sync):
    """get_revision_snapshot returns only rows with a stored revision —
    sources compare against this for UPSERT/UNCHANGED. Rows without a
    revision (HTTP without ETag, worker DLQ'd before completion) are
    visible via list_known_uris instead."""
    await sync.upsert("s", "u", revision=None, content_hash=None)
    assert await sync.get_revision_snapshot("s") == {}
    assert await sync.list_known_uris("s") == {"u"}


async def test_sync_state_upsert_preserves_revision_when_none(sync):
    """upsert(revision=None) leaves an existing revision in place."""
    await sync.upsert("s", "u", revision="v1", content_hash="hash-v1")
    await sync.upsert("s", "u", revision=None, content_hash=None)
    row = await sync.get_row("s", "u")
    assert row is not None
    assert row.revision == "v1"
    assert row.content_hash == "hash-v1"
    assert await sync.get_revision_snapshot("s") == {"u": "v1"}


async def test_sync_state_upsert_replaces_revision_when_provided(sync):
    """upsert with a non-None revision overwrites the existing one."""
    await sync.upsert("s", "u", revision="v1", content_hash="hash-v1")
    await sync.upsert("s", "u", revision="v2", content_hash="hash-v2", ingested=True)
    row = await sync.get_row("s", "u")
    assert row is not None
    assert row.revision == "v2"
    assert row.content_hash == "hash-v2"
    assert row.last_ingested_at is not None


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


async def test_sync_state_batch_upsert_empty_is_noop(sync):
    """batch_upsert with an empty list does nothing."""
    await sync.batch_upsert([])
    assert await sync.get_revision_snapshot("s") == {}


async def test_migration_v2_to_v3_adds_conversion_stalled(tmp_path):
    """A v2 DB gains `conversion_stalled`, defaulted false for existing rows."""
    db = tmp_path / "v2.db"
    engine = create_async_engine(URL.create("sqlite+aiosqlite", database=str(db)))
    try:
        await apply_migrations(engine)
        await _downgrade_to_v3(engine)
        async with engine.begin() as conn:
            await conn.execute(sa.text("UPDATE schema_version SET version = 2"))
            await conn.execute(sa.text("DROP INDEX uq_jobs_blocking_op"))
            await conn.execute(
                sa.text("ALTER TABLE jobs DROP COLUMN conversion_stalled")
            )
            when = "2026-01-01T00:00:00+00:00"
            await conn.execute(
                sa.text(
                    "INSERT INTO jobs (id, source_id, uri, op, status, attempts, "
                    "max_attempts, enqueued_at, scheduled_at) "
                    "VALUES ('old-1', 's', 'u1', 'upsert', 'dead', 1, 5, "
                    f"'{when}', '{when}')"
                )
            )

        assert await apply_migrations(engine) == SCHEMA_VERSION

        async with engine.connect() as conn:
            cols = (await conn.execute(sa.text("PRAGMA table_info(jobs)"))).fetchall()
            assert "conversion_stalled" in {c[1] for c in cols}
            names = (
                await conn.execute(
                    sa.text("SELECT name FROM sqlite_master WHERE type = 'index'")
                )
            ).scalars()
            assert "uq_jobs_blocking_op" in set(names)
            killed = (
                await conn.execute(
                    sa.text("SELECT conversion_stalled FROM jobs WHERE id = 'old-1'")
                )
            ).scalar_one()
            assert not killed
    finally:
        await engine.dispose()


async def test_migration_v3_to_v4_adds_namespace(tmp_path):
    """A v3 queue gains `namespace` on `jobs` and `sync_state`, every existing
    row under the empty namespace, with the job indexes and the sync_state key
    rebuilt over it."""
    db = tmp_path / "v3.db"
    engine = create_async_engine(URL.create("sqlite+aiosqlite", database=str(db)))
    try:
        await apply_migrations(engine)
        await _downgrade_to_v3(engine)
        when = "2026-01-01T00:00:00+00:00"
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "INSERT INTO jobs (id, source_id, uri, op, status, attempts, "
                    "max_attempts, enqueued_at, scheduled_at) "
                    "VALUES ('old-1', 's', 'u1', 'upsert', 'queued', 0, 5, "
                    f"'{when}', '{when}')"
                )
            )
            await conn.execute(
                sa.text(
                    "INSERT INTO sync_state (source_id, uri, revision, "
                    "last_seen_at, last_ingested_at) "
                    f"VALUES ('s', 'u1', 'r1', '{when}', '{when}')"
                )
            )

        assert await apply_migrations(engine) == SCHEMA_VERSION

        async with engine.connect() as conn:
            job_ns = (
                await conn.execute(
                    sa.text("SELECT namespace FROM jobs WHERE id = 'old-1'")
                )
            ).scalar_one()
            sync_row = (
                await conn.execute(
                    sa.text(
                        "SELECT namespace, revision, last_ingested_at "
                        "FROM sync_state WHERE source_id = 's' AND uri = 'u1'"
                    )
                )
            ).one()
            key_columns = {
                row[1]
                for row in (
                    await conn.execute(sa.text("PRAGMA table_info(sync_state)"))
                ).fetchall()
                if row[5]
            }
            index_sql = {
                name: sql
                for name, sql in (
                    await conn.execute(
                        sa.text(
                            "SELECT name, sql FROM sqlite_master WHERE "
                            "type = 'index' AND tbl_name = 'jobs' "
                            "AND sql IS NOT NULL"
                        )
                    )
                ).all()
            }
        assert job_ns == ""
        assert tuple(sync_row) == ("", "r1", when)
        assert key_columns == {"namespace", "source_id", "uri"}
        assert set(index_sql) == {index.name for index in jobs_table.indexes}
        assert all("namespace" in sql for sql in index_sql.values())

        # The migrated queue serves a namespace-scoped repository.
        repo = JobRepo(engine)
        old = await repo.get_job("old-1")
        assert old is not None and old.status is JobStatus.QUEUED
    finally:
        await engine.dispose()


async def test_interrupted_migration_rolls_back_and_retries(tmp_path, monkeypatch):
    """A failure after the DDL of the v4 step leaves the v3 schema and its rows
    as they were, and the next run migrates."""
    db = tmp_path / "v3.db"
    engine = create_async_engine(URL.create("sqlite+aiosqlite", database=str(db)))
    try:
        await apply_migrations(engine)
        await _downgrade_to_v3(engine)
        when = "2026-01-01T00:00:00+00:00"
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "INSERT INTO sync_state (source_id, uri, revision, "
                    f"last_seen_at) VALUES ('s', 'u1', 'r1', '{when}')"
                )
            )

        class _Broken:
            @staticmethod
            def create(*_, **__):
                raise RuntimeError("disk full")

        monkeypatch.setattr(migrations, "sync_state", _Broken)
        with pytest.raises(RuntimeError, match="disk full"):
            await apply_migrations(engine)
        monkeypatch.undo()

        async with engine.connect() as conn:
            version = (
                await conn.execute(sa.text("SELECT version FROM schema_version"))
            ).scalar_one()
            columns = {
                row[1]
                for row in (
                    await conn.execute(sa.text("PRAGMA table_info(jobs)"))
                ).fetchall()
            }
            tables = set(
                (
                    await conn.execute(
                        sa.text("SELECT name FROM sqlite_master WHERE type = 'table'")
                    )
                ).scalars()
            )
            revision = (
                await conn.execute(
                    sa.text("SELECT revision FROM sync_state WHERE uri = 'u1'")
                )
            ).scalar_one()
        assert version == 3
        assert "namespace" not in columns
        assert tables == {"jobs", "sync_state", "schema_version"}
        assert revision == "r1"

        assert await apply_migrations(engine) == SCHEMA_VERSION
        row = await SyncStateRepo(engine).get_row("s", "u1")
        assert row is not None and row.revision == "r1"
    finally:
        await engine.dispose()


async def test_a_stalled_document_is_not_re_enqueued(jobs):
    """A revision-less document that stalled has no suppression marker, so the
    index must refuse the re-enqueue or every sweep pays the deadline again."""
    job = await jobs.enqueue("src", "https://x/y.pdf", JobOp.UPSERT)
    assert job is not None
    claimed = await jobs.claim_next("w1")
    assert claimed is not None
    assert await jobs.mark_dead(claimed.id, "stalled", "w1", conversion_stalled=True)

    assert await jobs.enqueue("src", "https://x/y.pdf", JobOp.UPSERT) is None


async def test_dlq_retry_lifts_the_suppression(jobs):
    """The operator's escape hatch: a retry clears the flag, so discovery can
    enqueue the URI again afterwards."""
    job = await jobs.enqueue("src", "https://x/y.pdf", JobOp.UPSERT)
    assert job is not None
    claimed = await jobs.claim_next("w1")
    assert claimed is not None
    await jobs.mark_dead(claimed.id, "stalled", "w1", conversion_stalled=True)

    dead = await jobs.get_job(claimed.id)
    assert dead is not None and dead.conversion_stalled is True

    revived = await jobs.retry(claimed.id)
    assert revived.conversion_stalled is False
    assert revived.status is JobStatus.QUEUED

    # Take the revived job to a terminal state so the live-job dedup is not
    # what answers the next enqueue.
    again = await jobs.claim_next("w2")
    assert again is not None and again.id == claimed.id
    assert await jobs.mark_succeeded(again.id, "w2")

    assert await jobs.enqueue("src", "https://x/y.pdf", JobOp.UPSERT) is not None


async def test_an_ordinary_dead_job_still_re_enqueues(jobs):
    """Only a worker-killing failure suppresses; the documented behaviour for
    every other dead job is unchanged."""
    job = await jobs.enqueue("src", "https://x/y.pdf", JobOp.UPSERT)
    assert job is not None
    claimed = await jobs.claim_next("w1")
    assert claimed is not None
    await jobs.mark_dead(claimed.id, "unsupported", "w1")

    assert await jobs.enqueue("src", "https://x/y.pdf", JobOp.UPSERT) is not None


async def test_retention_never_prunes_a_stalled_document(jobs):
    """The row is what stops rediscovery, so age must not remove it: age is not
    evidence the document changed."""
    job = await jobs.enqueue("src", "https://x/y.pdf", JobOp.UPSERT)
    assert job is not None
    claimed = await jobs.claim_next("w1")
    assert claimed is not None
    await jobs.mark_dead(claimed.id, "stalled", "w1", conversion_stalled=True)

    other = await jobs.enqueue("src", "https://x/z.pdf", JobOp.UPSERT)
    assert other is not None
    claimed_other = await jobs.claim_next("w1")
    assert claimed_other is not None
    await jobs.mark_dead(claimed_other.id, "unsupported", "w1")

    assert await jobs.prune_terminal(max_age_seconds=-1) == 1

    assert await jobs.get_job(claimed.id) is not None
    assert await jobs.get_job(claimed_other.id) is None


async def test_a_stalled_document_never_blocks_a_delete(jobs):
    """Removing the document has to stay possible, and a successful DELETE
    prunes the row that was suppressing rediscovery."""
    job = await jobs.enqueue("src", "https://x/y.pdf", JobOp.UPSERT)
    assert job is not None
    claimed = await jobs.claim_next("w1")
    assert claimed is not None
    await jobs.mark_dead(claimed.id, "stalled", "w1", conversion_stalled=True)

    deletion = await jobs.enqueue("src", "https://x/y.pdf", JobOp.DELETE)
    assert deletion is not None

    claimed_delete = await jobs.claim_next("w1")
    assert claimed_delete is not None
    assert await jobs.mark_succeeded(claimed_delete.id, "w1")
    assert await jobs.prune_unsuccessful("src", "https://x/y.pdf") == 1

    assert await jobs.enqueue("src", "https://x/y.pdf", JobOp.UPSERT) is not None


async def test_concurrent_mark_dead_cannot_leave_a_poison_upsert_queued(tmp_path):
    """`uq_jobs_blocking_op` is what makes this safe: whichever order the
    transition and the sweep commit in, the URI's upsert slot is occupied — by
    the claimed row before, by the tombstone after — so the insert conflicts
    either way. A read-then-insert check cannot promise that on SQLite, whose
    deferred transaction takes no write lock for the read.
    """
    db = tmp_path / "race.db"
    engine = await open_queue(
        QueueConfig(dburi=str(URL.create("sqlite+aiosqlite", database=str(db))))
    )
    try:
        jobs = JobRepo(engine)
        for attempt in range(25):
            uri = f"https://x/{attempt}.pdf"
            first = await jobs.enqueue("src", uri, JobOp.UPSERT)
            assert first is not None
            claimed = await jobs.claim_next("w1")
            assert claimed is not None

            enqueued, _ = await asyncio.gather(
                jobs.enqueue("src", uri, JobOp.UPSERT),
                jobs.mark_dead(claimed.id, "stalled", "w1", conversion_stalled=True),
            )
            assert enqueued is None, (
                f"attempt {attempt}: a poison upsert was queued alongside the tombstone"
            )
            live = [
                j
                for j in await jobs.list_jobs()
                if j.uri == uri and j.status is not JobStatus.DEAD
            ]
            assert live == []
    finally:
        await engine.dispose()


async def test_a_changed_revision_is_not_blocked_by_the_tombstone(jobs):
    """The slot carries the revision, so a source that replaced the file is
    ingested again."""
    first = await jobs.enqueue("src", "https://x/y.pdf", JobOp.UPSERT, revision="r0")
    assert first is not None
    claimed = await jobs.claim_next("w1")
    assert claimed is not None
    await jobs.mark_dead(claimed.id, "stalled", "w1", conversion_stalled=True)

    assert (
        await jobs.enqueue("src", "https://x/y.pdf", JobOp.UPSERT, revision="r0")
        is None
    )
    assert (
        await jobs.enqueue("src", "https://x/y.pdf", JobOp.UPSERT, revision="r1")
        is not None
    )


async def test_two_revisionless_stalls_still_collide(jobs):
    """`coalesce` is what makes this hold: a unique index treats NULLs as
    distinct, so a bare `revision` column would stop blocking exactly the rows
    the tombstone exists for."""
    first = await jobs.enqueue("src", "https://x/y.pdf", JobOp.UPSERT)
    assert first is not None
    claimed = await jobs.claim_next("w1")
    assert claimed is not None
    await jobs.mark_dead(claimed.id, "stalled", "w1", conversion_stalled=True)

    assert await jobs.enqueue("src", "https://x/y.pdf", JobOp.UPSERT) is None


async def test_fresh_and_migrated_schemas_agree(tmp_path):
    """`create_all` and the migration ladder from v2 must produce the same job
    indexes and the same `sync_state` table, or a migrated queue enforces
    something different from a new one."""
    fresh_db = tmp_path / "fresh.db"
    fresh = await open_queue(
        QueueConfig(dburi=str(URL.create("sqlite+aiosqlite", database=str(fresh_db))))
    )
    migrated_db = tmp_path / "migrated.db"
    migrated = await open_queue(
        QueueConfig(
            dburi=str(URL.create("sqlite+aiosqlite", database=str(migrated_db)))
        )
    )
    try:
        await _downgrade_to_v3(migrated)
        async with migrated.begin() as conn:
            await conn.execute(sa.text("UPDATE schema_version SET version = 2"))
            await conn.execute(sa.text("DROP INDEX uq_jobs_blocking_op"))
            await conn.execute(
                sa.text("ALTER TABLE jobs DROP COLUMN conversion_stalled")
            )
        await apply_migrations(migrated)

        def _norm(sql: str) -> str:
            return " ".join(sql.replace('"', "").lower().split())

        async def _ddl(engine) -> dict[str, str]:
            async with engine.connect() as conn:
                rows = (
                    await conn.execute(
                        sa.text(
                            "SELECT name, sql FROM sqlite_master WHERE "
                            "(type = 'index' OR name = 'sync_state') "
                            "AND sql IS NOT NULL"
                        )
                    )
                ).all()
            return {name: _norm(sql) for name, sql in rows}

        expected = await _ddl(fresh)
        assert set(expected) == {
            *(index.name for index in jobs_table.indexes),
            "sync_state",
        }
        assert await _ddl(migrated) == expected
    finally:
        await fresh.dispose()
        await migrated.dispose()


async def test_retry_names_the_tombstone_that_blocks_it(jobs):
    """Both rows can exist: one dead on max attempts, the URI re-enqueued, the
    next attempt stalled. Retrying the first names the row in the way."""
    from haiku.rag.ingester.exceptions import BlockingTombstoneError

    first = await jobs.enqueue("src", "https://x/y.pdf", JobOp.UPSERT)
    assert first is not None
    claimed = await jobs.claim_next("w1")
    assert claimed is not None
    await jobs.mark_dead(claimed.id, "out of attempts", "w1")

    second = await jobs.enqueue("src", "https://x/y.pdf", JobOp.UPSERT)
    assert second is not None
    claimed_second = await jobs.claim_next("w1")
    assert claimed_second is not None
    await jobs.mark_dead(claimed_second.id, "stalled", "w1", conversion_stalled=True)

    with pytest.raises(BlockingTombstoneError) as excinfo:
        await jobs.retry(claimed.id)
    assert excinfo.value.blocking_job_id == claimed_second.id


async def test_sync_state_invalidate_clears_revision_and_hash(sync):
    """invalidate() drops what upsert() coalesces, keeping the row."""
    await sync.upsert("s", "u", revision="v1", content_hash="hash-v1", ingested=True)

    await sync.invalidate("s", ["u"])

    row = await sync.get_row("s", "u")
    assert row is not None
    assert row.revision is None
    assert row.content_hash is None
    assert await sync.get_revision_snapshot("s") == {}
    assert await sync.list_known_uris("s") == {"u"}


async def test_sync_state_invalidate_ignores_unknown_row(sync):
    await sync.invalidate("s", ["missing"])
    assert await sync.list_known_uris("s") == set()


async def test_sync_state_invalidate_batches_beyond_the_bind_limit(sync, monkeypatch):
    """One transaction, bounded statements: a rebuilt database can invalidate
    more URIs than a dialect allows bind parameters for."""
    monkeypatch.setattr(repository, "INVALIDATE_BATCH", 2)
    uris = [f"u{n}" for n in range(5)]
    for uri in uris:
        await sync.upsert("s", uri, revision="v1")

    await sync.invalidate("s", uris)

    assert await sync.get_revision_snapshot("s") == {}
    assert await sync.list_known_uris("s") == set(uris)


async def test_sync_state_invalidate_accepts_no_uris(sync):
    await sync.upsert("s", "u", revision="v1")
    await sync.invalidate("s", [])
    assert await sync.get_revision_snapshot("s") == {"u": "v1"}


async def test_sync_state_lists_only_ingested_uris(sync):
    """A revision is written for a dead job too; last_ingested_at is not."""
    await sync.upsert("s", "done", revision="v1", ingested=True)
    await sync.upsert("s", "dead", revision="v1", ingested=False)
    await sync.upsert("s", "queued")

    assert await sync.list_ingested_uris("s") == {"done"}
    assert await sync.get_revision_snapshot("s") == {"done": "v1", "dead": "v1"}


# --- namespaces ---
#
# Two repositories over one queue file, under different namespaces, using the
# same source id and URI. Each test fails when the method it names loses its
# namespace filter, or writes a row without one.


@pytest.fixture
def own(engine):
    return JobRepo(engine, namespace="own")


@pytest.fixture
def foreign(engine):
    return JobRepo(engine, namespace="other")


@pytest.fixture
def own_sync(engine):
    return SyncStateRepo(engine, namespace="own")


@pytest.fixture
def foreign_sync(engine):
    return SyncStateRepo(engine, namespace="other")


async def _claimed(repo: JobRepo, uri: str = "u"):
    job = await repo.enqueue("s", uri, JobOp.UPSERT)
    assert job is not None
    claimed = await repo.claim_next("w")
    assert claimed is not None and claimed.id == job.id
    return claimed


async def _dead(repo: JobRepo, uri: str = "u", *, conversion_stalled: bool = False):
    claimed = await _claimed(repo, uri)
    assert await repo.mark_dead(
        claimed.id, "boom", "w", conversion_stalled=conversion_stalled
    )
    return claimed


async def test_enqueue_writes_the_namespace_and_keeps_live_slots_apart(
    own, foreign, conn
):
    mine = await own.enqueue("s", "u", JobOp.UPSERT)
    theirs = await foreign.enqueue("s", "u", JobOp.UPSERT)
    assert mine is not None and theirs is not None
    cursor = await conn.execute("SELECT id, namespace FROM jobs")
    rows = {row["id"]: row["namespace"] for row in await cursor.fetchall()}
    assert rows == {mine.id: "own", theirs.id: "other"}


async def test_claim_next_skips_foreign_jobs(own, foreign):
    theirs = await foreign.enqueue("s", "u", JobOp.UPSERT)
    assert theirs is not None
    assert await own.claim_next("w") is None
    claimed = await foreign.claim_next("w")
    assert claimed is not None and claimed.id == theirs.id


async def test_get_job_hides_foreign_jobs(own, foreign):
    theirs = await foreign.enqueue("s", "u", JobOp.UPSERT)
    assert theirs is not None
    assert await own.get_job(theirs.id) is None


async def test_mark_succeeded_ignores_foreign_claims(own, foreign):
    theirs = await _claimed(foreign)
    assert not await own.mark_succeeded(theirs.id, "w")
    refreshed = await foreign.get_job(theirs.id)
    assert refreshed is not None and refreshed.status is JobStatus.CLAIMED


async def test_mark_dead_ignores_foreign_claims(own, foreign):
    theirs = await _claimed(foreign)
    assert not await own.mark_dead(theirs.id, "boom", "w")
    refreshed = await foreign.get_job(theirs.id)
    assert refreshed is not None and refreshed.status is JobStatus.CLAIMED


async def test_reschedule_ignores_foreign_claims(own, foreign):
    theirs = await _claimed(foreign)
    assert not await own.reschedule(theirs.id, 60, "later", "w")
    refreshed = await foreign.get_job(theirs.id)
    assert refreshed is not None and refreshed.status is JobStatus.CLAIMED


async def test_retry_refuses_foreign_jobs(own, foreign):
    theirs = await _dead(foreign)
    with pytest.raises(KeyError):
        await own.retry(theirs.id)
    refreshed = await foreign.get_job(theirs.id)
    assert refreshed is not None and refreshed.status is JobStatus.DEAD


async def test_live_sibling_is_scoped(own, foreign):
    assert await foreign.enqueue("s", "u", JobOp.UPSERT) is not None
    assert await own._live_sibling("s", "u") is None


async def test_blocking_tombstone_is_scoped(own, foreign):
    await _dead(foreign, conversion_stalled=True)
    mine = await own.enqueue("s", "u", JobOp.UPSERT)
    assert mine is not None
    assert await own._blocking_tombstone(mine) is None


async def test_cancel_refuses_foreign_jobs(own, foreign):
    theirs = await foreign.enqueue("s", "u", JobOp.UPSERT)
    assert theirs is not None
    assert not await own.cancel(theirs.id)
    assert await foreign.get_job(theirs.id) is not None


async def test_cancel_sources_ignores_foreign_jobs(own, foreign):
    theirs = await foreign.enqueue("s", "u", JobOp.UPSERT)
    assert theirs is not None
    assert await own.cancel_sources({"s"}) == 0
    assert await foreign.get_job(theirs.id) is not None


async def test_list_jobs_hides_foreign_jobs(own, foreign):
    assert await foreign.enqueue("s", "u", JobOp.UPSERT) is not None
    mine = await own.enqueue("s", "u2", JobOp.UPSERT)
    assert mine is not None
    assert [job.id for job in await own.list_jobs()] == [mine.id]


async def test_has_pending_ignores_foreign_jobs(own, foreign):
    assert await foreign.enqueue("s", "u", JobOp.UPSERT) is not None
    assert not await own.has_pending("s")


async def test_counts_by_status_ignores_foreign_jobs(own, foreign):
    assert await foreign.enqueue("s", "u", JobOp.UPSERT) is not None
    assert await own.counts_by_status() == {}


async def test_counts_by_status_since_ignores_foreign_jobs(own, foreign):
    since = datetime.now(UTC) - timedelta(minutes=1)
    theirs = await _claimed(foreign)
    assert await foreign.mark_succeeded(theirs.id, "w")
    assert await own.counts_by_status_since(since) == {}


async def test_batch_progress_counts_since_ignores_foreign_jobs(own, foreign):
    since = datetime.now(UTC) - timedelta(minutes=1)
    assert await foreign.enqueue("s", "u", JobOp.UPSERT) is not None
    assert await own.batch_progress_counts_since(since) == {}


async def test_count_succeeded_since_ignores_foreign_jobs(own, foreign):
    theirs = await _claimed(foreign)
    assert await foreign.mark_succeeded(theirs.id, "w")
    assert await own.count_succeeded_since(3600) == 0


async def test_oldest_queued_age_ignores_foreign_jobs(own, foreign):
    assert await foreign.enqueue("s", "u", JobOp.UPSERT) is not None
    assert await own.oldest_queued_age_seconds() is None


async def test_counts_by_source_ignores_foreign_jobs(own, foreign):
    assert await foreign.enqueue("s", "u", JobOp.UPSERT) is not None
    assert await own.counts_by_source("queued", "claimed") == {}


async def test_release_if_claimed_ignores_foreign_claims(own, foreign):
    theirs = await _claimed(foreign)
    assert not await own.release_if_claimed(theirs.id, "w")
    refreshed = await foreign.get_job(theirs.id)
    assert refreshed is not None and refreshed.status is JobStatus.CLAIMED


async def test_prune_unsuccessful_ignores_foreign_jobs(own, foreign):
    theirs = await _dead(foreign)
    assert await own.prune_unsuccessful("s", "u") == 0
    assert await foreign.get_job(theirs.id) is not None


async def test_prune_terminal_ignores_foreign_jobs(own, foreign):
    theirs = await _claimed(foreign)
    assert await foreign.mark_succeeded(theirs.id, "w")
    assert await own.prune_terminal(max_age_seconds=0) == 0
    assert await foreign.get_job(theirs.id) is not None


async def test_reap_stale_ignores_foreign_claims(own, foreign):
    theirs = await _claimed(foreign)
    assert await own.reap_stale(lease_ttl_seconds=0) == 0
    refreshed = await foreign.get_job(theirs.id)
    assert refreshed is not None and refreshed.status is JobStatus.CLAIMED


async def test_renew_claims_ignores_foreign_claims(own, foreign):
    theirs = await _claimed(foreign)
    assert await own.renew_claims({theirs.id: "w"}) == 0


async def test_sync_upsert_keeps_namespaces_apart(own_sync, foreign_sync):
    await own_sync.upsert("s", "u", revision="mine")
    await foreign_sync.upsert("s", "u", revision="theirs")
    mine = await own_sync.get_row("s", "u")
    theirs = await foreign_sync.get_row("s", "u")
    assert mine is not None and mine.revision == "mine"
    assert theirs is not None and theirs.revision == "theirs"


async def test_revision_snapshot_ignores_foreign_rows(own_sync, foreign_sync):
    await foreign_sync.upsert("s", "u", revision="r1")
    assert await own_sync.get_revision_snapshot("s") == {}


async def test_list_known_uris_ignores_foreign_rows(own_sync, foreign_sync):
    await foreign_sync.upsert("s", "u")
    assert await own_sync.list_known_uris("s") == set()


async def test_batch_upsert_writes_the_namespace(own_sync, foreign_sync):
    await own_sync.batch_upsert([SyncRow("s", "u", "r1", None, True)])
    assert await own_sync.get_revision_snapshot("s") == {"u": "r1"}
    assert await foreign_sync.get_row("s", "u") is None


async def test_list_ingested_uris_ignores_foreign_rows(own_sync, foreign_sync):
    await foreign_sync.upsert("s", "u", ingested=True)
    assert await own_sync.list_ingested_uris("s") == set()


async def test_invalidate_ignores_foreign_rows(own_sync, foreign_sync):
    await foreign_sync.upsert("s", "u", revision="r1", content_hash="h1")
    await own_sync.invalidate("s", ["u"])
    theirs = await foreign_sync.get_row("s", "u")
    assert theirs is not None and theirs.revision == "r1"


async def test_sync_delete_ignores_foreign_rows(own_sync, foreign_sync):
    await foreign_sync.upsert("s", "u")
    await own_sync.delete("s", "u")
    assert await foreign_sync.get_row("s", "u") is not None


async def test_sync_delete_sources_ignores_foreign_rows(own_sync, foreign_sync):
    await foreign_sync.upsert("s", "u")
    await own_sync.delete_sources({"s"})
    assert await foreign_sync.get_row("s", "u") is not None


# --- rejected ---


async def _rejected(jobs, uri: str = "u"):
    job = await jobs.enqueue("s", uri, JobOp.UPSERT)
    assert job is not None
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    assert await jobs.mark_rejected(claimed.id, "u rejected: draft", "w")
    return job


async def test_mark_rejected_records_the_reason(jobs):
    job = await _rejected(jobs)

    refreshed = await jobs.get_job(job.id)
    assert refreshed is not None
    assert refreshed.status is JobStatus.REJECTED
    assert refreshed.last_error == "u rejected: draft"
    assert refreshed.completed_at is not None


async def test_retry_revives_a_rejected_job(jobs):
    job = await _rejected(jobs)

    revived = await jobs.retry(job.id)

    assert revived.status is JobStatus.QUEUED
    assert revived.last_error is None


async def test_prune_unsuccessful_removes_rejected_rows(jobs):
    job = await _rejected(jobs)

    assert await jobs.prune_unsuccessful("s", "u") == 1
    assert await jobs.get_job(job.id) is None


async def test_prune_terminal_deletes_old_rejected_rows(jobs, conn):
    job = await _rejected(jobs)
    long_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    await conn.execute(
        "UPDATE jobs SET completed_at = ? WHERE id = ?", (long_ago, job.id)
    )
    await conn.commit()

    assert await jobs.prune_terminal(max_age_seconds=60) == 1
    assert await jobs.get_job(job.id) is None


async def test_rejections_are_counted_apart_from_dead(jobs):
    since = datetime.now(UTC) - timedelta(seconds=1)
    await _rejected(jobs, "u1")
    dead = await jobs.enqueue("s", "u2", JobOp.UPSERT)
    assert dead is not None
    await jobs.claim_next("w")
    await jobs.mark_dead(dead.id, "boom", "w")

    expected = {"rejected": 1, "dead": 1}
    assert await jobs.counts_by_status_since(since) == expected
    assert await jobs.batch_progress_counts_since(since) == expected


async def test_migration_v4_to_v5_keeps_the_jobs(engine):
    """Schema 5 adds the `rejected` status, which needs no DDL: the version
    alone keeps an older release from reading a status it cannot parse."""
    when = "2026-01-01T00:00:00+00:00"
    async with engine.begin() as conn:
        await conn.execute(sa.text("UPDATE schema_version SET version = 4"))
        await conn.execute(
            sa.text(
                "INSERT INTO jobs (id, namespace, source_id, uri, op, status, "
                "attempts, max_attempts, enqueued_at, scheduled_at) "
                "VALUES ('old-1', '', 's', 'u1', 'upsert', 'queued', 0, 5, "
                f"'{when}', '{when}')"
            )
        )

    assert SCHEMA_VERSION == 5
    assert await apply_migrations(engine) == 5
    async with engine.connect() as conn:
        ids = (await conn.execute(sa.text("SELECT id FROM jobs"))).scalars().all()
    assert ids == ["old-1"]


# --- retry_failed ---


async def _failed(repo: JobRepo, uri: str = "u", op: JobOp = JobOp.UPSERT):
    job = await repo.enqueue("s", uri, op)
    assert job is not None
    claimed = await repo.claim_next("w")
    assert claimed is not None and claimed.id == job.id
    assert await repo.mark_dead(claimed.id, "boom", "w")
    return job


async def _succeeded(repo: JobRepo, uri: str = "u", op: JobOp = JobOp.UPSERT):
    job = await repo.enqueue("s", uri, op)
    assert job is not None
    claimed = await repo.claim_next("w")
    assert claimed is not None and claimed.id == job.id
    assert await repo.mark_succeeded(claimed.id, "w")
    return job


async def _status(repo: JobRepo, job_id: str) -> JobStatus:
    job = await repo.get_job(job_id)
    assert job is not None
    return job.status


async def test_retry_failed_revives_the_chosen_statuses_only(jobs):
    dead = await _failed(jobs, "a")
    rejected = await _rejected(jobs, "b")

    assert await jobs.retry_failed({JobStatus.DEAD}) == 1

    assert await _status(jobs, dead.id) is JobStatus.QUEUED
    assert await _status(jobs, rejected.id) is JobStatus.REJECTED
    assert await jobs.retry_failed({JobStatus.REJECTED}) == 1
    assert await _status(jobs, rejected.id) is JobStatus.QUEUED


async def test_retry_failed_revives_the_given_sources_only(jobs):
    served = await _failed(jobs, "a")
    other = await jobs.enqueue("other", "b", JobOp.UPSERT)
    assert other is not None
    assert await jobs.claim_next("w", source_ids={"other"}) is not None
    assert await jobs.mark_dead(other.id, "boom", "w")

    assert await jobs.retry_failed({JobStatus.DEAD}, source_ids={"s"}) == 1

    assert await _status(jobs, served.id) is JobStatus.QUEUED
    assert await _status(jobs, other.id) is JobStatus.DEAD


async def test_retry_failed_refuses_a_live_or_succeeded_status(jobs):
    with pytest.raises(ValueError, match="queued"):
        await jobs.retry_failed({JobStatus.DEAD, JobStatus.QUEUED})


async def test_retry_failed_makes_a_revived_job_claimable_at_once(jobs):
    dead = await _dead(jobs, conversion_stalled=True)

    assert await jobs.retry_failed({JobStatus.DEAD}) == 1

    revived = await jobs.get_job(dead.id)
    assert revived is not None
    assert revived.status is JobStatus.QUEUED
    assert revived.attempts == 0
    assert revived.last_error is None
    assert revived.completed_at is None
    assert revived.conversion_stalled is False
    claimed = await jobs.claim_next("w")
    assert claimed is not None and claimed.id == dead.id


@pytest.mark.parametrize(
    "failed_op, succeeded_op",
    [(JobOp.UPSERT, JobOp.DELETE), (JobOp.DELETE, JobOp.UPSERT)],
)
async def test_retry_failed_leaves_a_failure_a_later_success_resolved(
    jobs, failed_op, succeeded_op
):
    dead = await _failed(jobs, op=failed_op)
    await _succeeded(jobs, op=succeeded_op)

    assert await jobs.retry_failed({JobStatus.DEAD}) == 0
    assert await _status(jobs, dead.id) is JobStatus.DEAD


async def test_retry_failed_leaves_a_failure_ingested_after_its_success_row_aged_out(
    jobs, sync
):
    stalled = await _dead(jobs, conversion_stalled=True)
    await sync.upsert("s", "u", revision="r2", ingested=True)

    assert await jobs.retry_failed({JobStatus.DEAD}) == 0
    assert await _status(jobs, stalled.id) is JobStatus.DEAD


async def test_retry_failed_revives_a_failure_after_an_earlier_ingest(jobs, sync):
    await sync.upsert("s", "u", revision="r1", ingested=True)
    dead = await _failed(jobs)

    assert await jobs.retry_failed({JobStatus.DEAD}) == 1
    assert await _status(jobs, dead.id) is JobStatus.QUEUED


async def test_retry_failed_revives_only_the_latest_failure_of_a_uri(jobs):
    older = await _failed(jobs)
    newer = await _rejected(jobs)

    assert await jobs.retry_failed({JobStatus.DEAD, JobStatus.REJECTED}) == 1

    assert await _status(jobs, older.id) is JobStatus.DEAD
    assert await _status(jobs, newer.id) is JobStatus.QUEUED


async def test_retry_failed_revives_one_of_two_failures_with_equal_completed_at(
    jobs, conn
):
    first = await _failed(jobs)
    second = await _failed(jobs)
    await conn.execute(
        "UPDATE jobs SET completed_at = ? WHERE id IN (?, ?)",
        ("2026-01-01T00:00:00+00:00", first.id, second.id),
    )
    await conn.commit()

    assert await jobs.retry_failed({JobStatus.DEAD}) == 1
    statuses = {await _status(jobs, first.id), await _status(jobs, second.id)}
    assert statuses == {JobStatus.QUEUED, JobStatus.DEAD}


async def test_retry_failed_skips_a_failure_with_a_live_sibling(jobs):
    dead = await _failed(jobs)
    live = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert live is not None

    assert await jobs.retry_failed({JobStatus.DEAD}) == 0
    assert await _status(jobs, dead.id) is JobStatus.DEAD
    assert await _status(jobs, live.id) is JobStatus.QUEUED


async def test_retry_failed_skips_a_failure_superseded_after_selection(
    jobs, monkeypatch
):
    dead = await _failed(jobs)
    replacement = await jobs.enqueue("s", "u", JobOp.UPSERT)
    assert replacement is not None
    claimed = await jobs.claim_next("w")
    assert claimed is not None and claimed.id == replacement.id
    revive = JobRepo._revive

    async def _complete_then_revive(self, job_id, statuses):
        assert await jobs.mark_succeeded(replacement.id, "w")
        return await revive(self, job_id, statuses)

    monkeypatch.setattr(JobRepo, "_revive", _complete_then_revive)

    assert await jobs.retry_failed({JobStatus.DEAD}) == 0
    assert await _status(jobs, dead.id) is JobStatus.DEAD


async def test_retry_failed_skips_a_failure_pruned_after_selection(jobs, monkeypatch):
    dead = await _failed(jobs)
    revive = JobRepo._revive

    async def _prune_then_revive(self, job_id, statuses):
        assert await jobs.prune_unsuccessful("s", "u") == 1
        return await revive(self, job_id, statuses)

    monkeypatch.setattr(JobRepo, "_revive", _prune_then_revive)

    assert await jobs.retry_failed({JobStatus.DEAD}) == 0
    assert await jobs.get_job(dead.id) is None


async def test_retry_failed_stays_within_its_namespace(
    own, foreign, own_sync, foreign_sync
):
    mine = await _failed(own)
    theirs = await _failed(foreign)
    await _succeeded(foreign)
    await foreign_sync.upsert("s", "u", revision="r", ingested=True)

    assert await own.retry_failed({JobStatus.DEAD}) == 1

    assert await _status(own, mine.id) is JobStatus.QUEUED
    assert await _status(foreign, theirs.id) is JobStatus.DEAD


async def test_run_counts_cover_the_given_sources_only(jobs):
    since = datetime.now(UTC) - timedelta(seconds=1)
    for source_id in ("a", "b"):
        assert await jobs.enqueue(source_id, "queued", JobOp.UPSERT) is not None
        done = await jobs.enqueue(source_id, "done", JobOp.UPSERT)
        assert done is not None
        claimed = await jobs.claim_next("w", source_ids={source_id})
        assert claimed is not None
        assert await jobs.mark_succeeded(claimed.id, "w")

    assert await jobs.counts_by_status_since(since, source_ids={"a"}) == {
        "succeeded": 1
    }
    assert await jobs.batch_progress_counts_since(since, source_ids={"a"}) == {
        "queued": 1,
        "succeeded": 1,
    }


async def test_live_jobs_lists_every_queued_and_claimed_job_of_the_sources(jobs):
    for i in range(60):
        assert await jobs.enqueue("a", f"u{i}", JobOp.UPSERT) is not None
    claimed = await jobs.claim_next("w", source_ids={"a"})
    assert claimed is not None
    finished = await jobs.claim_next("w", source_ids={"a"})
    assert finished is not None and await jobs.mark_succeeded(finished.id, "w")
    assert await jobs.enqueue("b", "other", JobOp.UPSERT) is not None

    live = await jobs.live_jobs({"a"})

    assert len(live) == 59
    assert {job.source_id for job in live} == {"a"}
    assert {job.status for job in live} == {JobStatus.QUEUED, JobStatus.CLAIMED}
