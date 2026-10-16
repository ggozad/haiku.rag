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
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from haiku.rag.client import HaikuRAG
from haiku.rag.client.scope import DatabaseScope
from haiku.rag.config import (
    APIConfig,
    AppConfig,
    FSSourceConfig,
    IngesterConfig,
    QueueConfig,
    RetryPolicyConfig,
    WorkerConfig,
)
from haiku.rag.ingester.app import IngesterApp
from haiku.rag.ingester.exceptions import QueueError
from haiku.rag.ingester.queue.db import SCHEMA_VERSION, metadata
from haiku.rag.ingester.queue.migrations import apply_migrations, open_queue
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


async def test_enqueue_dedup_via_partial_unique_index(postgres_dburi):
    """ON CONFLICT DO NOTHING against uq_jobs_live drops a second live job for
    the same (source_id, uri), regardless of op."""
    async with queue_engine(postgres_dburi) as engine:
        jobs = JobRepo(engine)
        first = await jobs.enqueue("s", "u", JobOp.UPSERT)
        second = await jobs.enqueue("s", "u", JobOp.DELETE)
        assert first is not None
        assert second is None


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


async def test_tombstone_conflict_is_enforced_by_the_index(postgres_dburi):
    """`uq_jobs_blocking_op` refuses a re-enqueued UPSERT while the tombstone
    holds the URI's upsert slot, and admits a DELETE, which holds a different
    one. Enforced at the index, so a worker committing `mark_dead` concurrently
    cannot open a window between a read and the insert."""
    async with queue_engine(postgres_dburi) as engine:
        jobs = JobRepo(engine)
        first = await jobs.enqueue("s", "u", JobOp.UPSERT)
        assert first is not None
        claimed = await jobs.claim_next("w")
        assert claimed is not None
        assert await jobs.mark_dead(claimed.id, "stalled", "w", conversion_stalled=True)

        assert await jobs.enqueue("s", "u", JobOp.UPSERT) is None
        assert await jobs.enqueue("s", "u", JobOp.DELETE) is not None


async def test_concurrent_mark_dead_cannot_leave_a_poison_upsert_queued(postgres_dburi):
    """The same invariant under real concurrent connections: whichever order
    the transition and the sweep commit in, the upsert slot is occupied."""
    async with queue_engine(postgres_dburi) as engine:
        jobs = JobRepo(engine)
        for attempt in range(15):
            uri = f"u{attempt}"
            first = await jobs.enqueue("s", uri, JobOp.UPSERT)
            assert first is not None
            claimed = await jobs.claim_next("w")
            assert claimed is not None

            enqueued, _ = await asyncio.gather(
                jobs.enqueue("s", uri, JobOp.UPSERT),
                jobs.mark_dead(claimed.id, "stalled", "w", conversion_stalled=True),
            )
            assert enqueued is None, f"attempt {attempt}: poison upsert queued"


_V3_JOBS_INDEX_DDL = [
    "CREATE UNIQUE INDEX uq_jobs_live ON jobs (source_id, uri) "
    "WHERE status IN ('queued', 'claimed')",
    "CREATE UNIQUE INDEX uq_jobs_blocking_op ON jobs "
    "(source_id, uri, op, coalesce(revision, '')) "
    "WHERE status IN ('queued', 'claimed') "
    "OR status = 'dead' AND conversion_stalled IS TRUE",
    "CREATE INDEX idx_jobs_claimable ON jobs (scheduled_at) WHERE status = 'queued'",
    "CREATE INDEX idx_jobs_succeeded_completed ON jobs (completed_at) "
    "WHERE status = 'succeeded'",
]


async def _downgrade_to_v3(engine: AsyncEngine) -> None:
    """Strip a current queue back to schema 3: no `namespace` anywhere."""
    async with engine.begin() as conn:
        await conn.execute(text("ALTER TABLE jobs DROP COLUMN namespace"))
        for ddl in _V3_JOBS_INDEX_DDL:
            await conn.execute(text(ddl))
        await conn.execute(
            text("ALTER TABLE sync_state DROP CONSTRAINT sync_state_pkey")
        )
        await conn.execute(text("ALTER TABLE sync_state DROP COLUMN namespace"))
        await conn.execute(
            text("ALTER TABLE sync_state ADD PRIMARY KEY (source_id, uri)")
        )
        await conn.execute(text("UPDATE schema_version SET version = 3"))


@asynccontextmanager
async def restricted_role(dburi: str):
    """A throwaway schema and a role allowed into it, as the docs prescribe.

    Yields the superuser's engine on that schema and the role's URL. The role
    gets table rights from `grant_dml` once the tables exist."""
    schema = f"q_{uuid.uuid4().hex[:12]}"
    role = f"rw_{schema}"
    admin = create_async_engine(
        make_url(dburi),
        poolclass=NullPool,
        connect_args={"server_settings": {"search_path": schema}},
    )
    async with admin.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        await conn.execute(text(f"CREATE ROLE {role} LOGIN PASSWORD 'pw'"))
        await conn.execute(text(f"ALTER ROLE {role} SET search_path = {schema}"))
        await conn.execute(text(f"GRANT USAGE ON SCHEMA {schema} TO {role}"))
    role_url = make_url(dburi).set(username=role, password="pw")
    try:
        yield admin, role_url.render_as_string(hide_password=False)
    finally:
        async with admin.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            await conn.execute(text(f"DROP ROLE {role}"))
        await admin.dispose()


async def grant_dml(admin: AsyncEngine, role_url: str) -> None:
    """The documented grants for the account the ingesters run as."""
    role = make_url(role_url).username
    async with admin.begin() as conn:
        await conn.execute(
            text(f"GRANT SELECT, INSERT, UPDATE, DELETE ON jobs, sync_state TO {role}")
        )
        await conn.execute(text(f"GRANT SELECT ON schema_version TO {role}"))


async def test_migration_v3_to_v4_keeps_a_populated_queue(postgres_dburi):
    """The Postgres branch of the v4 step: a populated v3 queue keeps its rows
    under the empty namespace, with the `sync_state` key and every job index
    rebuilt over `namespace`."""
    async with queue_engine(postgres_dburi) as engine:
        await apply_migrations(engine)
        await _downgrade_to_v3(engine)
        when = "2026-01-01T00:00:00+00:00"
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO jobs (id, source_id, uri, op, status, attempts, "
                    "max_attempts, enqueued_at, scheduled_at) VALUES "
                    "('old-1', 's', 'u1', 'upsert', 'queued', 0, 5, :when, :when)"
                ),
                {"when": when},
            )
            await conn.execute(
                text(
                    "INSERT INTO sync_state (source_id, uri, revision, "
                    "last_seen_at, last_ingested_at) "
                    "VALUES ('s', 'u1', 'r1', :when, :when)"
                ),
                {"when": when},
            )

        assert await apply_migrations(engine) == SCHEMA_VERSION

        async with engine.connect() as conn:
            job_ns = (
                await conn.execute(
                    text("SELECT namespace FROM jobs WHERE id = 'old-1'")
                )
            ).scalar_one()
            sync_row = (
                await conn.execute(
                    text(
                        "SELECT namespace, revision FROM sync_state "
                        "WHERE source_id = 's' AND uri = 'u1'"
                    )
                )
            ).one()
            key_columns = set(
                (
                    await conn.execute(
                        text(
                            "SELECT a.attname FROM pg_index i "
                            "JOIN pg_attribute a ON a.attrelid = i.indrelid "
                            "AND a.attnum = ANY(i.indkey) "
                            "WHERE i.indrelid = 'sync_state'::regclass "
                            "AND i.indisprimary"
                        )
                    )
                ).scalars()
            )
            index_defs = (
                await conn.execute(
                    text(
                        "SELECT indexdef FROM pg_indexes WHERE tablename = 'jobs' "
                        "AND schemaname = current_schema()"
                    )
                )
            ).scalars()
            namespaced = [d for d in index_defs if "(namespace," in d]
        assert job_ns == ""
        assert tuple(sync_row) == ("", "r1")
        assert key_columns == {"namespace", "source_id", "uri"}
        assert len(namespaced) == 4

        repo = JobRepo(engine)
        old = await repo.get_job("old-1")
        assert old is not None and old.status is JobStatus.QUEUED


async def test_namespaces_share_live_slots_and_sync_keys(postgres_dburi):
    """`uq_jobs_live` and the `sync_state` key include the namespace, and the
    upsert's ON CONFLICT target infers the new key."""
    async with queue_engine(postgres_dburi) as engine:
        mine, theirs = JobRepo(engine, "a"), JobRepo(engine, "b")
        assert await mine.enqueue("s", "u", JobOp.UPSERT) is not None
        assert await theirs.enqueue("s", "u", JobOp.UPSERT) is not None

        my_sync, their_sync = SyncStateRepo(engine, "a"), SyncStateRepo(engine, "b")
        await my_sync.upsert("s", "u", revision="r-a")
        await their_sync.upsert("s", "u", revision="r-b")
        await my_sync.upsert("s", "u", revision=None, ingested=True)
        my_row = await my_sync.get_row("s", "u")
        their_row = await their_sync.get_row("s", "u")
        assert my_row is not None and my_row.revision == "r-a"
        assert my_row.last_ingested_at is not None
        assert their_row is not None and their_row.revision == "r-b"
        assert their_row.last_ingested_at is None


async def test_concurrent_claims_stay_within_their_namespace(postgres_dburi):
    """Two ingesters claiming at once over real connections each take exactly
    their own jobs, for the same source and URIs."""
    async with queue_engine(postgres_dburi) as engine:
        mine, theirs = JobRepo(engine, "a"), JobRepo(engine, "b")
        my_ids, their_ids = set(), set()
        for i in range(10):
            my_job = await mine.enqueue("s", f"u{i}", JobOp.UPSERT)
            their_job = await theirs.enqueue("s", f"u{i}", JobOp.UPSERT)
            assert my_job is not None and their_job is not None
            my_ids.add(my_job.id)
            their_ids.add(their_job.id)

        results = await asyncio.gather(
            *(mine.claim_next(f"a{i}") for i in range(15)),
            *(theirs.claim_next(f"b{i}") for i in range(15)),
        )
        my_claims = {r.id for r in results[:15] if r is not None}
        their_claims = {r.id for r in results[15:] if r is not None}
        assert my_claims == my_ids
        assert their_claims == their_ids


async def test_open_queue_refuses_an_empty_server_queue(postgres_dburi):
    async with restricted_role(postgres_dburi) as (_, role_url):
        with pytest.raises(QueueError, match="no schema.*queue init --dburi"):
            await open_queue(QueueConfig(dburi=role_url, namespace="t"))


async def test_open_queue_refuses_a_server_queue_behind(postgres_dburi):
    async with restricted_role(postgres_dburi) as (admin, role_url):
        await apply_migrations(admin)
        await _downgrade_to_v3(admin)
        await grant_dml(admin, role_url)
        with pytest.raises(QueueError, match="schema 3.*queue migrate --dburi"):
            await open_queue(QueueConfig(dburi=role_url, namespace="t"))


@pytest.mark.vcr()
async def test_run_batch_as_a_dml_only_role(postgres_dburi, tmp_path):
    """The whole ingester path issues no DDL: startup verification, a sweep,
    a real document converted, embedded and stored, and the sync state behind
    it, all as a role granted only the documented DML."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("The first document of the corpus.")
    uri = (corpus / "a.md").as_uri()
    db_path = tmp_path / "db.lancedb"
    async with restricted_role(postgres_dburi) as (admin, role_url):
        await apply_migrations(admin)
        await grant_dml(admin, role_url)
        config = AppConfig(
            ingester=IngesterConfig(
                queue=QueueConfig(dburi=role_url, namespace="t"),
                sources=[
                    FSSourceConfig(
                        type="fs", id="local", root=corpus, poll_interval_s=3600.0
                    )
                ],
                workers=WorkerConfig(
                    worker_count=1,
                    poll_idle_interval_s=0.05,
                    retry=RetryPolicyConfig(max_attempts=1),
                ),
                api=APIConfig(enabled=False),
            )
        )

        report = await IngesterApp(
            config=config, scope=DatabaseScope.at(db_path)
        ).run_batch()

        assert (report.succeeded, report.dead) == (1, 0)
        async with HaikuRAG(db_path, config) as rag:
            document = await rag.get_document_by_uri(uri)
        assert document is not None
        assert document.metadata["source_id"] == "local"
        async with admin.connect() as conn:
            job_rows = (
                await conn.execute(text("SELECT namespace, status FROM jobs"))
            ).all()
            sync_rows = (
                await conn.execute(
                    text(
                        "SELECT namespace, source_id, uri, "
                        "last_ingested_at IS NOT NULL FROM sync_state"
                    )
                )
            ).all()
        assert job_rows == [("t", "succeeded")]
        assert sync_rows == [("t", "local", uri, True)]


async def test_concurrent_migrations_serialize(postgres_dburi):
    """Two deployment jobs migrating one empty queue at once both return, and
    one schema results."""
    async with restricted_role(postgres_dburi) as (admin, _):
        versions = await asyncio.gather(
            apply_migrations(admin), apply_migrations(admin)
        )
        assert versions == [SCHEMA_VERSION, SCHEMA_VERSION]
        async with admin.connect() as conn:
            count = (
                await conn.execute(text("SELECT count(*) FROM schema_version"))
            ).scalar_one()
        assert count == 1


async def _dead(jobs: JobRepo, uri: str, *, conversion_stalled: bool = False) -> str:
    job = await jobs.enqueue("s", uri, JobOp.UPSERT)
    assert job is not None
    claimed = await jobs.claim_next("w")
    assert claimed is not None and claimed.id == job.id
    assert await jobs.mark_dead(
        claimed.id, "boom", "w", conversion_stalled=conversion_stalled
    )
    return job.id


async def test_retry_failed_revives_the_latest_unresolved_failures(postgres_dburi):
    async with queue_engine(postgres_dburi) as engine:
        jobs = JobRepo(engine, namespace="own")
        sync = SyncStateRepo(engine, namespace="own")
        foreign = JobRepo(engine, namespace="other")
        foreign_sync = SyncStateRepo(engine, namespace="other")

        resolved = await _dead(jobs, "resolved")
        replacement = await jobs.enqueue("s", "resolved", JobOp.DELETE)
        assert replacement is not None
        claimed = await jobs.claim_next("w")
        assert claimed is not None
        assert await jobs.mark_succeeded(claimed.id, "w")

        aged_out = await _dead(jobs, "aged-out", conversion_stalled=True)
        await sync.upsert("s", "aged-out", revision="r2", ingested=True)

        await sync.upsert("s", "updated", revision="r1", ingested=True)
        updated = await _dead(jobs, "updated")

        older = await _dead(jobs, "twice")
        newer = await _dead(jobs, "twice")

        tied = [await _dead(jobs, "tied"), await _dead(jobs, "tied")]
        async with engine.begin() as conn:
            await conn.execute(
                text("UPDATE jobs SET completed_at = :at WHERE id IN (:a, :b)"),
                {"at": "2026-01-01T00:00:00+00:00", "a": tied[0], "b": tied[1]},
            )

        rejected = await jobs.enqueue("s", "rejected", JobOp.UPSERT)
        assert rejected is not None
        claimed = await jobs.claim_next("w")
        assert claimed is not None
        assert await jobs.mark_rejected(claimed.id, "rejected: draft", "w")

        await _dead(foreign, "mine")
        mine = await _dead(jobs, "mine")
        await _dead(foreign, "mine")
        await foreign_sync.upsert("s", "mine", revision="r", ingested=True)

        blocked = await _dead(jobs, "blocked")
        live = await jobs.enqueue("s", "blocked", JobOp.UPSERT)
        assert live is not None

        async def status(job_id: str) -> JobStatus:
            job = await jobs.get_job(job_id)
            assert job is not None
            return job.status

        assert await jobs.retry_failed({JobStatus.DEAD}) == 4
        assert await status(rejected.id) is JobStatus.REJECTED
        assert await jobs.retry_failed({JobStatus.REJECTED}) == 1

        assert await status(resolved) is JobStatus.DEAD
        assert await status(aged_out) is JobStatus.DEAD
        assert await status(updated) is JobStatus.QUEUED
        assert await status(older) is JobStatus.DEAD
        assert await status(newer) is JobStatus.QUEUED
        assert {await status(tied[0]), await status(tied[1])} == {
            JobStatus.QUEUED,
            JobStatus.DEAD,
        }
        assert await status(rejected.id) is JobStatus.QUEUED
        assert await status(blocked) is JobStatus.DEAD
        assert await status(mine) is JobStatus.QUEUED


async def test_retry_failed_skips_a_failure_pruned_after_selection(
    postgres_dburi, monkeypatch
):
    async with queue_engine(postgres_dburi) as engine:
        jobs = JobRepo(engine)
        dead = await _dead(jobs, "u")
        revive = JobRepo._revive

        async def _prune_then_revive(self, job_id, statuses):
            assert await jobs.prune_unsuccessful("s", "u") == 1
            return await revive(self, job_id, statuses)

        monkeypatch.setattr(JobRepo, "_revive", _prune_then_revive)

        assert await jobs.retry_failed({JobStatus.DEAD}) == 0
        assert await jobs.get_job(dead) is None


async def test_retry_failed_waits_for_a_completion_in_flight(postgres_dburi):
    """A replacement's success that commits while the revive runs is seen by
    the revive, under Read Committed."""
    async with queue_engine(postgres_dburi) as engine:
        jobs = JobRepo(engine)
        dead = await _dead(jobs, "u")
        replacement = await jobs.enqueue("s", "u", JobOp.UPSERT)
        assert replacement is not None
        claimed = await jobs.claim_next("w")
        assert claimed is not None

        async with engine.connect() as completing:
            await completing.execute(
                text(
                    "UPDATE jobs SET status = 'succeeded', completed_at = :at "
                    "WHERE id = :id"
                ),
                {"at": "9999-01-01T00:00:00+00:00", "id": replacement.id},
            )
            revive = asyncio.create_task(jobs.retry_failed({JobStatus.DEAD}))
            await asyncio.sleep(0.5)
            assert not revive.done()
            await completing.commit()

        assert await revive == 0
        job = await jobs.get_job(dead)
        assert job is not None and job.status is JobStatus.DEAD
