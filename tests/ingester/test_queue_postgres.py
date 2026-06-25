"""Postgres-backed queue tests. Marked `integration` (excluded in CI). The
`postgres_dburi` fixture connects to the docker-compose `postgres` service
(`docker compose -f tests/docker/docker-compose.yml up -d`), or uses
HAIKU_RAG_TEST_PG_DBURI when set, and skips when neither is reachable. They
exercise the dialect-specific SQL the SQLite suite cannot: ON CONFLICT, COALESCE
upserts, the partial unique index, and FOR UPDATE SKIP LOCKED under real
concurrent connections.

Each test owns its engine for its whole body. asyncpg binds connections to the
loop that created them and pytest-asyncio uses a per-test loop, so opening the
engine inside the test (with NullPool, so no connection is reused across loops)
avoids cross-loop fixture handoff.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import make_url, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from haiku.rag.ingester.queue.db import metadata
from haiku.rag.ingester.queue.models import JobOp, JobStatus, SyncRow
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo

pytestmark = pytest.mark.integration


@asynccontextmanager
async def queue_engine(dburi: str):
    """An engine scoped to a throwaway Postgres schema, so concurrent xdist
    workers each get their own isolated copy of the queue tables."""
    schema = f"q_{uuid.uuid4().hex[:12]}"
    engine = create_async_engine(
        make_url(dburi),
        poolclass=NullPool,
        connect_args={"server_settings": {"search_path": schema}},
    )
    async with engine.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        await conn.run_sync(metadata.create_all)
    try:
        yield engine
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await engine.dispose()


@pytest.mark.asyncio
async def test_enqueue_dedup_via_partial_unique_index(postgres_dburi):
    """ON CONFLICT DO NOTHING against uq_jobs_live drops a second live job for
    the same (source_id, uri), regardless of op."""
    async with queue_engine(postgres_dburi) as engine:
        jobs = JobRepo(engine)
        first = await jobs.enqueue("s", "u", JobOp.UPSERT)
        second = await jobs.enqueue("s", "u", JobOp.DELETE)
        assert first is not None
        assert second is None


@pytest.mark.asyncio
async def test_enqueue_after_terminal_succeeds(postgres_dburi):
    """Once a job is terminal it no longer satisfies the partial index, so a
    re-enqueue for the same URI is allowed."""
    async with queue_engine(postgres_dburi) as engine:
        jobs = JobRepo(engine)
        first = await jobs.enqueue("s", "u", JobOp.UPSERT)
        assert first is not None
        claimed = await jobs.claim_next("w")
        assert claimed is not None
        await jobs.mark_succeeded(claimed.id, "w")
        second = await jobs.enqueue("s", "u", JobOp.UPSERT)
        assert second is not None and second.id != first.id


@pytest.mark.asyncio
async def test_skip_locked_claims_each_job_once(postgres_dburi):
    """FOR UPDATE SKIP LOCKED: many concurrent claims over real Postgres
    connections each take a distinct job, with none claimed twice. Without
    SKIP LOCKED, concurrent transactions would grab the same row."""
    async with queue_engine(postgres_dburi) as engine:
        jobs = JobRepo(engine)
        enqueued = []
        for i in range(20):
            job = await jobs.enqueue("s", f"u{i}", JobOp.UPSERT)
            assert job is not None
            enqueued.append(job)

        results = await asyncio.gather(*(jobs.claim_next(f"w{i}") for i in range(40)))
        claimed = [r for r in results if r is not None]

        assert len(claimed) == 20
        assert len({c.id for c in claimed}) == 20
        assert {c.id for c in claimed} == {j.id for j in enqueued}


@pytest.mark.asyncio
async def test_reap_stale_clamps_attempts(postgres_dburi):
    """The attempts-1 clamp (CASE) renders and runs on Postgres."""
    async with queue_engine(postgres_dburi) as engine:
        jobs = JobRepo(engine)
        job = await jobs.enqueue("s", "u", JobOp.UPSERT)
        assert job is not None
        claimed = await jobs.claim_next("w")
        assert claimed is not None and claimed.attempts == 1
        reset = await jobs.reap_stale(lease_ttl_seconds=0)
        assert reset == 1
        refreshed = await jobs.get_job(job.id)
        assert refreshed is not None
        assert refreshed.status is JobStatus.QUEUED
        assert refreshed.attempts == 0


@pytest.mark.asyncio
async def test_renew_claims_survives_reap_on_postgres(postgres_dburi):
    """The COALESCE lease threshold and the OR-of-(id, claimed_by) renewal
    predicate render and run on Postgres: a renewed claim outlives a reap even
    though its claimed_at is old."""
    async with queue_engine(postgres_dburi) as engine:
        jobs = JobRepo(engine)
        job = await jobs.enqueue("s", "u", JobOp.UPSERT)
        assert job is not None
        claimed = await jobs.claim_next("w")
        assert claimed is not None

        async with engine.begin() as conn:
            await conn.execute(
                text("UPDATE jobs SET claimed_at = :ts WHERE id = :id"),
                {"ts": "2000-01-01T00:00:00+00:00", "id": job.id},
            )
        assert await jobs.renew_claims({job.id: "w"}) == 1
        assert await jobs.reap_stale(lease_ttl_seconds=60) == 0
        refreshed = await jobs.get_job(job.id)
        assert refreshed is not None and refreshed.status is JobStatus.CLAIMED


@pytest.mark.asyncio
async def test_sync_state_upsert_coalesce_preserves_revision(postgres_dburi):
    """ON CONFLICT DO UPDATE with COALESCE leaves an existing revision in place
    when a later upsert passes revision=None."""
    async with queue_engine(postgres_dburi) as engine:
        sync = SyncStateRepo(engine)
        await sync.upsert("s", "u", revision="v1", content_hash="h1")
        await sync.upsert("s", "u", revision=None, content_hash=None, ingested=True)
        row = await sync.get_row("s", "u")
        assert row is not None
        assert row.revision == "v1"
        assert row.content_hash == "h1"
        assert row.last_ingested_at is not None


@pytest.mark.asyncio
async def test_sync_state_batch_upsert(postgres_dburi):
    async with queue_engine(postgres_dburi) as engine:
        sync = SyncStateRepo(engine)
        await sync.batch_upsert(
            [
                SyncRow("s", "u1", "r1", "h1", False),
                SyncRow("s", "u2", "r2", "h2", True),
            ]
        )
        assert await sync.get_revision_snapshot("s") == {"u1": "r1", "u2": "r2"}


@pytest.mark.asyncio
async def test_prune_terminal_removes_old_rows(postgres_dburi):
    async with queue_engine(postgres_dburi) as engine:
        jobs = JobRepo(engine)
        job = await jobs.enqueue("s", "u", JobOp.UPSERT)
        assert job is not None
        await jobs.claim_next("w")
        await jobs.mark_succeeded(job.id, "w")
        # completed_at is "now", so a zero-width window prunes it.
        pruned = await jobs.prune_terminal(max_age_seconds=0)
        assert pruned == 1
        assert await jobs.get_job(job.id) is None
