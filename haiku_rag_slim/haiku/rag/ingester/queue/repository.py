import asyncio
import json
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects import sqlite as sqlite_dialect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from haiku.rag.ingester.queue.db import jobs, sync_state
from haiku.rag.ingester.queue.models import (
    Job,
    JobOp,
    JobStatus,
    SyncRow,
    SyncStateRow,
)


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _insert(table: sa.Table, dialect: str):
    """Dialect-specific INSERT exposing on_conflict_* (and `.excluded`)."""
    if dialect == "postgresql":
        return postgresql.insert(table)
    return sqlite_dialect.insert(table)


def _attempts_minus_one() -> sa.ColumnElement[int]:
    """attempts - 1, floored at 0. Renders identically on both dialects."""
    return sa.case((jobs.c.attempts - 1 < 0, 0), else_=jobs.c.attempts - 1)


def _row_to_job(row: Mapping) -> Job:
    extra_text = row["extra"]
    return Job(
        id=row["id"],
        source_id=row["source_id"],
        uri=row["uri"],
        op=JobOp(row["op"]),
        content_hash=row["content_hash"],
        revision=row["revision"],
        status=JobStatus(row["status"]),
        attempts=row["attempts"],
        max_attempts=row["max_attempts"],
        last_error=row["last_error"],
        extra=json.loads(extra_text) if extra_text else None,
        enqueued_at=datetime.fromisoformat(row["enqueued_at"]),
        scheduled_at=datetime.fromisoformat(row["scheduled_at"]),
        claimed_at=_parse_dt(row["claimed_at"]),
        claimed_by=row["claimed_by"],
        last_heartbeat_at=_parse_dt(row["last_heartbeat_at"]),
        completed_at=_parse_dt(row["completed_at"]),
    )


def _row_to_sync_state(row: Mapping) -> SyncStateRow:
    return SyncStateRow(
        source_id=row["source_id"],
        uri=row["uri"],
        revision=row["revision"],
        content_hash=row["content_hash"],
        last_seen_at=datetime.fromisoformat(row["last_seen_at"]),
        last_ingested_at=_parse_dt(row["last_ingested_at"]),
    )


class JobRepo:
    def __init__(self, engine: AsyncEngine):
        self._engine = engine
        self._dialect = engine.dialect.name
        # Notified after a successful enqueue so workers in this process wake
        # immediately instead of polling on a fixed sleep interval. Workers in
        # other processes (a shared Postgres queue) fall back to polling.
        self.job_available = asyncio.Condition()

    async def enqueue(
        self,
        source_id: str,
        uri: str,
        op: JobOp = JobOp.UPSERT,
        *,
        revision: str | None = None,
        content_hash: str | None = None,
        max_attempts: int = 5,
        extra: dict | None = None,
    ) -> Job | None:
        """Enqueue an upsert/delete job. Returns the inserted Job, or None if a
        live (queued/claimed) job already exists for the same (source_id, uri).
        The partial unique index enforces atomicity."""
        job_id = str(uuid.uuid4())
        now = _utcnow_iso()
        extra_json = json.dumps(extra) if extra is not None else None
        stmt = (
            _insert(jobs, self._dialect)
            .values(
                id=job_id,
                source_id=source_id,
                uri=uri,
                op=op.value,
                content_hash=content_hash,
                revision=revision,
                status="queued",
                attempts=0,
                max_attempts=max_attempts,
                last_error=None,
                extra=extra_json,
                enqueued_at=now,
                scheduled_at=now,
            )
            .on_conflict_do_nothing()
            .returning(*jobs.c)
        )
        async with self._engine.begin() as conn:
            row = (await conn.execute(stmt)).mappings().one_or_none()
        if row is not None:
            async with self.job_available:
                self.job_available.notify_all()
        return _row_to_job(row) if row else None

    async def claim_next(
        self, worker_id: str, *, exclude_source_ids: set[str] | None = None
    ) -> Job | None:
        """Atomically claim the oldest queued job whose scheduled_at <= now.
        A single `UPDATE ... WHERE id = (SELECT ... LIMIT 1) RETURNING` keeps
        the claim atomic across connections: on Postgres the subquery adds
        `FOR UPDATE SKIP LOCKED`; on SQLite the whole statement runs under one
        write lock, so a racing connection re-evaluates the subquery against
        the committed state and finds the row already claimed.

        `exclude_source_ids` skips jobs from those sources (a paused breaker).
        Empty or None adds no clause, leaving the query unchanged."""
        now = _utcnow_iso()
        conditions = [jobs.c.status == "queued", jobs.c.scheduled_at <= now]
        if exclude_source_ids:
            conditions.append(jobs.c.source_id.notin_(sorted(exclude_source_ids)))
        candidate = (
            sa.select(jobs.c.id)
            .where(*conditions)
            .order_by(jobs.c.scheduled_at, jobs.c.id)
            .limit(1)
            .with_for_update(skip_locked=True)
            .scalar_subquery()
        )
        claim = (
            sa.update(jobs)
            .where(jobs.c.id == candidate)
            .values(
                status="claimed",
                claimed_at=now,
                claimed_by=worker_id,
                last_heartbeat_at=now,
                attempts=jobs.c.attempts + 1,
            )
            .returning(*jobs.c)
        )
        async with self._engine.begin() as conn:
            row = (await conn.execute(claim)).mappings().one_or_none()
        return _row_to_job(row) if row else None

    async def get_job(self, job_id: str) -> Job | None:
        async with self._engine.connect() as conn:
            row = (
                (await conn.execute(sa.select(jobs).where(jobs.c.id == job_id)))
                .mappings()
                .one_or_none()
            )
        return _row_to_job(row) if row else None

    async def mark_succeeded(self, job_id: str, claimed_by: str) -> bool:
        """Transition a still-claimed job to `succeeded`. Guarded on
        `status='claimed' AND claimed_by=?` so a reaper-resurrected job
        picked up by a different worker isn't clobbered by the original
        slow worker. Returns True when the row was updated."""
        stmt = (
            sa.update(jobs)
            .where(
                jobs.c.id == job_id,
                jobs.c.status == "claimed",
                jobs.c.claimed_by == claimed_by,
            )
            .values(status="succeeded", completed_at=_utcnow_iso())
            .returning(jobs.c.id)
        )
        async with self._engine.begin() as conn:
            row = (await conn.execute(stmt)).first()
        return row is not None

    async def mark_dead(self, job_id: str, error: str, claimed_by: str) -> bool:
        """Transition a still-claimed job to `dead`. See `mark_succeeded`
        for the guard semantics."""
        stmt = (
            sa.update(jobs)
            .where(
                jobs.c.id == job_id,
                jobs.c.status == "claimed",
                jobs.c.claimed_by == claimed_by,
            )
            .values(status="dead", completed_at=_utcnow_iso(), last_error=error)
            .returning(jobs.c.id)
        )
        async with self._engine.begin() as conn:
            row = (await conn.execute(stmt)).first()
        return row is not None

    async def reschedule(
        self, job_id: str, delay_seconds: float, error: str, claimed_by: str
    ) -> bool:
        """Reset a still-claimed job back to `queued` with a future
        scheduled_at. Guarded on `status='claimed' AND claimed_by=?` so a
        slow worker can't clobber a re-claim that happened after the reaper
        reset its claim. Returns True when the row was updated."""
        scheduled = (datetime.now(UTC) + timedelta(seconds=delay_seconds)).isoformat()
        stmt = (
            sa.update(jobs)
            .where(
                jobs.c.id == job_id,
                jobs.c.status == "claimed",
                jobs.c.claimed_by == claimed_by,
            )
            .values(
                status="queued",
                scheduled_at=scheduled,
                claimed_at=None,
                claimed_by=None,
                last_heartbeat_at=None,
                last_error=error,
            )
            .returning(jobs.c.id)
        )
        async with self._engine.begin() as conn:
            row = (await conn.execute(stmt)).first()
        return row is not None

    async def retry(self, job_id: str) -> Job:
        """Reset a `dead` or `queued` job: status='queued', attempts=0,
        error cleared, scheduled for immediate re-claim. Refuses `claimed`
        rows (would race with the worker still processing) and `succeeded`
        rows (re-ingest via UPSERT instead). Raises KeyError when the row
        is missing or in a non-retryable state.

        Idempotent against a live sibling: if a live (queued/claimed) job
        already exists for the same (source_id, uri), reviving the dead row
        would violate `uq_jobs_live`; instead the existing live job is returned
        and the dead row is left dead."""
        now = _utcnow_iso()
        stmt = (
            sa.update(jobs)
            .where(jobs.c.id == job_id, jobs.c.status.in_(["dead", "queued"]))
            .values(
                status="queued",
                attempts=0,
                last_error=None,
                claimed_at=None,
                claimed_by=None,
                last_heartbeat_at=None,
                completed_at=None,
                scheduled_at=now,
            )
            .returning(*jobs.c)
        )
        collided = False
        try:
            async with self._engine.begin() as conn:
                row = (await conn.execute(stmt)).mappings().one_or_none()
        except IntegrityError:
            collided = True
            row = None
        if row is not None:
            return _row_to_job(row)
        if collided:
            target = await self.get_job(job_id)
            if target is not None:
                live = await self._live_sibling(target.source_id, target.uri)
                if live is not None:
                    return live
        raise KeyError(f"Job {job_id!r} not found or not retryable")

    async def _live_sibling(self, source_id: str, uri: str) -> Job | None:
        """The live (queued/claimed) job for a (source_id, uri), if any.
        `uq_jobs_live` guarantees at most one."""
        query = (
            sa.select(jobs)
            .where(
                jobs.c.source_id == source_id,
                jobs.c.uri == uri,
                jobs.c.status.in_(["queued", "claimed"]),
            )
            .limit(1)
        )
        async with self._engine.connect() as conn:
            row = (await conn.execute(query)).mappings().one_or_none()
        return _row_to_job(row) if row else None

    async def cancel(self, job_id: str) -> bool:
        """True iff a queued/claimed row was removed; terminal jobs aren't
        cancellable (succeeded/dead rows are kept for history)."""
        stmt = (
            sa.delete(jobs)
            .where(jobs.c.id == job_id, jobs.c.status.in_(["queued", "claimed"]))
            .returning(jobs.c.id)
        )
        async with self._engine.begin() as conn:
            row = (await conn.execute(stmt)).first()
        return row is not None

    async def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        source_id: str | None = None,
        uri: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Job]:
        query = sa.select(jobs)
        if status is not None:
            query = query.where(jobs.c.status == status.value)
        if source_id is not None:
            query = query.where(jobs.c.source_id == source_id)
        if uri is not None:
            query = query.where(jobs.c.uri == uri)
        query = query.order_by(jobs.c.enqueued_at.desc()).limit(limit).offset(offset)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).mappings().all()
        return [_row_to_job(r) for r in rows]

    async def has_pending(self, source_id: str) -> bool:
        """True iff at least one queued/claimed job exists for the source.

        Cheap probe used by pollers to skip sweeps when there's already
        outstanding work — the queue's unique index would dedupe new enqueues
        anyway, so a sweep into a saturated queue is pure wasted listing work.
        """
        query = (
            sa.select(jobs.c.id)
            .where(
                jobs.c.source_id == source_id,
                jobs.c.status.in_(["queued", "claimed"]),
            )
            .limit(1)
        )
        async with self._engine.connect() as conn:
            row = (await conn.execute(query)).first()
        return row is not None

    async def counts_by_status(self) -> dict[str, int]:
        query = sa.select(jobs.c.status, sa.func.count().label("n")).group_by(
            jobs.c.status
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        return {status: n for status, n in rows}

    async def counts_by_status_since(self, since: datetime) -> dict[str, int]:
        """status -> count of jobs that reached a terminal state at or after
        `since` (by completed_at). Only succeeded/dead set completed_at, so
        those are the only keys returned. Lets a one-shot batch report the
        work it finished, independent of terminal rows from earlier runs."""
        query = (
            sa.select(jobs.c.status, sa.func.count().label("n"))
            .where(jobs.c.completed_at >= since.isoformat())
            .group_by(jobs.c.status)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        return {status: n for status, n in rows}

    async def batch_progress_counts_since(self, since: datetime) -> dict[str, int]:
        """Counts for a one-shot batch progress snapshot.

        Live rows are counted regardless of enqueue time because run-batch
        drains the whole pending queue. Terminal rows are counted only when
        this run completed them, matching BatchReport semantics.
        """
        query = (
            sa.select(jobs.c.status, sa.func.count().label("n"))
            .where(
                sa.or_(
                    jobs.c.status.in_(["queued", "claimed"]),
                    jobs.c.completed_at >= since.isoformat(),
                )
            )
            .group_by(jobs.c.status)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        return {status: n for status, n in rows}

    async def count_succeeded_since(self, seconds: int) -> int:
        """How many jobs reached `succeeded` in the last `seconds` seconds.
        Drives the dashboard's rolling-throughput chips."""
        threshold = (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat()
        query = sa.select(sa.func.count()).where(
            jobs.c.status == "succeeded", jobs.c.completed_at >= threshold
        )
        async with self._engine.connect() as conn:
            count = (await conn.execute(query)).scalar()
        return int(count or 0)

    async def oldest_queued_age_seconds(self) -> float | None:
        """Age (in seconds) of the oldest job sitting in `queued` whose
        scheduled_at is in the past. Returns None when nothing is waiting.
        Tells operators whether work is backing up."""
        now = datetime.now(UTC)
        query = sa.select(sa.func.min(jobs.c.scheduled_at)).where(
            jobs.c.status == "queued", jobs.c.scheduled_at <= now.isoformat()
        )
        async with self._engine.connect() as conn:
            oldest = (await conn.execute(query)).scalar()
        if oldest is None:
            return None
        return (now - datetime.fromisoformat(oldest)).total_seconds()

    async def counts_by_source(self, *statuses: str) -> dict[str, int]:
        """source_id → count of jobs in any of the given statuses. Drives the
        dashboard's per-source DLQ and backlog summaries."""
        if not statuses:
            return {}
        query = (
            sa.select(jobs.c.source_id, sa.func.count().label("n"))
            .where(jobs.c.status.in_(statuses))
            .group_by(jobs.c.source_id)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        return {source_id: n for source_id, n in rows}

    async def release_if_claimed(self, job_id: str, claimed_by: str) -> bool:
        """Reset a still-claimed job back to queued, immediately reclaimable.
        Guarded on `status='claimed' AND claimed_by=?` so the cancel-cleanup
        of a slow worker doesn't strip the claim of a different worker that
        re-claimed after a reaper reset. Decrements attempts to undo the
        increment from `claim_next`, since a cancellation isn't a failed
        attempt. Returns True if the row was released."""
        stmt = (
            sa.update(jobs)
            .where(
                jobs.c.id == job_id,
                jobs.c.status == "claimed",
                jobs.c.claimed_by == claimed_by,
            )
            .values(
                status="queued",
                claimed_at=None,
                claimed_by=None,
                last_heartbeat_at=None,
                scheduled_at=_utcnow_iso(),
                attempts=_attempts_minus_one(),
            )
            .returning(jobs.c.id)
        )
        async with self._engine.begin() as conn:
            row = (await conn.execute(stmt)).first()
        return row is not None

    async def prune_dead(self, source_id: str, uri: str) -> int:
        """Delete dead jobs for the given (source_id, uri). Called after a
        successful DELETE to clear stale UPSERT failures for the same URI —
        the document is gone, so a "couldn't ingest this" entry is no longer
        actionable. Returns the number of rows removed."""
        stmt = sa.delete(jobs).where(
            jobs.c.source_id == source_id,
            jobs.c.uri == uri,
            jobs.c.status == "dead",
        )
        async with self._engine.begin() as conn:
            result = await conn.execute(stmt)
        return result.rowcount or 0

    async def prune_terminal(self, max_age_seconds: int) -> int:
        """Delete terminal jobs (succeeded/dead) whose completed_at is older
        than max_age_seconds. General housekeeping so the table doesn't grow
        without bound. Returns the number of rows removed."""
        threshold = (datetime.now(UTC) - timedelta(seconds=max_age_seconds)).isoformat()
        stmt = sa.delete(jobs).where(
            jobs.c.status.in_(["succeeded", "dead"]),
            jobs.c.completed_at < threshold,
        )
        async with self._engine.begin() as conn:
            result = await conn.execute(stmt)
        return result.rowcount or 0

    async def reap_stale(self, lease_ttl_seconds: int) -> int:
        """Reset claimed jobs whose lease has gone stale back to `queued`. A
        live worker renews its lease via `renew_claims`, so a claim is stale
        only when its owner stopped renewing (crash, wedged loop, lost DB).
        Staleness is measured against the last heartbeat, falling back to
        claimed_at for a claim made by a process that doesn't write the lease
        (an older version sharing the queue). Decrements `attempts` to undo the
        increment from `claim_next` — a reaped worker isn't a consumed attempt."""
        threshold = (
            datetime.now(UTC) - timedelta(seconds=lease_ttl_seconds)
        ).isoformat()
        lease = sa.func.coalesce(jobs.c.last_heartbeat_at, jobs.c.claimed_at)
        stmt = (
            sa.update(jobs)
            .where(jobs.c.status == "claimed", lease < threshold)
            .values(
                status="queued",
                claimed_at=None,
                claimed_by=None,
                last_heartbeat_at=None,
                attempts=_attempts_minus_one(),
            )
        )
        async with self._engine.begin() as conn:
            result = await conn.execute(stmt)
        return result.rowcount or 0

    async def renew_claims(self, claims: Mapping[str, str]) -> int:
        """Refresh `last_heartbeat_at` for the given `job_id -> claimed_by`
        pairs, extending their lease so the reaper leaves them alone. Guarded
        on `status='claimed' AND claimed_by=?` per pair, so a job already
        reaped and re-claimed elsewhere is left untouched — renewal never
        resurrects a lost claim. Returns the number of rows renewed."""
        if not claims:
            return 0
        now = _utcnow_iso()
        pair_match = sa.or_(
            *(
                sa.and_(jobs.c.id == job_id, jobs.c.claimed_by == worker_id)
                for job_id, worker_id in claims.items()
            )
        )
        stmt = (
            sa.update(jobs)
            .where(jobs.c.status == "claimed", pair_match)
            .values(last_heartbeat_at=now)
        )
        async with self._engine.begin() as conn:
            result = await conn.execute(stmt)
        return result.rowcount or 0


class SyncStateRepo:
    def __init__(self, engine: AsyncEngine):
        self._engine = engine
        self._dialect = engine.dialect.name

    async def get_revision_snapshot(self, source_id: str) -> dict[str, str]:
        """uri -> revision map for URIs that have a stored revision. Sources
        compare current revision against this map to decide UPSERT vs
        UNCHANGED. A stored revision means the file was accounted for at that
        revision — successfully ingested OR permanently failed; both suppress
        re-enqueue until the revision changes. Rows without a revision (HTTP
        without ETag, or a worker that didn't complete) are excluded — they
        have no revision to compare against; the closing-loop DELETE diff uses
        list_known_uris instead."""
        query = sa.select(sync_state.c.uri, sync_state.c.revision).where(
            sync_state.c.source_id == source_id,
            sync_state.c.revision.is_not(None),
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        return {uri: revision for uri, revision in rows}

    async def list_known_uris(self, source_id: str) -> set[str]:
        """Every URI the source has ever produced. Used by the closing-loop
        diff in discover() so a URI previously seen but no longer visible
        (FS file deleted, HTTP URL removed from config) emits DELETE."""
        query = sa.select(sync_state.c.uri).where(sync_state.c.source_id == source_id)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        return {uri for (uri,) in rows}

    async def get_row(self, source_id: str, uri: str) -> SyncStateRow | None:
        query = sa.select(sync_state).where(
            sync_state.c.source_id == source_id, sync_state.c.uri == uri
        )
        async with self._engine.connect() as conn:
            row = (await conn.execute(query)).mappings().one_or_none()
        return _row_to_sync_state(row) if row else None

    def _upsert_stmt(
        self,
        source_id: str,
        uri: str,
        revision: str | None,
        content_hash: str | None,
        last_seen_at: str,
        last_ingested_at: str | None,
    ):
        ins = _insert(sync_state, self._dialect).values(
            source_id=source_id,
            uri=uri,
            revision=revision,
            content_hash=content_hash,
            last_seen_at=last_seen_at,
            last_ingested_at=last_ingested_at,
        )
        excluded = ins.excluded
        return ins.on_conflict_do_update(
            index_elements=[sync_state.c.source_id, sync_state.c.uri],
            set_={
                "revision": sa.func.coalesce(excluded.revision, sync_state.c.revision),
                "content_hash": sa.func.coalesce(
                    excluded.content_hash, sync_state.c.content_hash
                ),
                "last_seen_at": excluded.last_seen_at,
                "last_ingested_at": sa.func.coalesce(
                    excluded.last_ingested_at, sync_state.c.last_ingested_at
                ),
            },
        )

    async def upsert(
        self,
        source_id: str,
        uri: str,
        *,
        revision: str | None = None,
        content_hash: str | None = None,
        ingested: bool = False,
    ) -> None:
        """Insert-or-update the sync_state row. `ingested=True` stamps
        last_ingested_at; otherwise only last_seen_at is bumped.
        `revision=None` and `content_hash=None` leave any existing values
        untouched."""
        now = _utcnow_iso()
        stmt = self._upsert_stmt(
            source_id, uri, revision, content_hash, now, now if ingested else None
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def batch_upsert(self, rows: list[SyncRow]) -> None:
        """Batch insert-or-update sync_state rows in a single transaction."""
        if not rows:
            return
        now = _utcnow_iso()
        async with self._engine.begin() as conn:
            for source_id, uri, revision, content_hash, ingested in rows:
                stmt = self._upsert_stmt(
                    source_id,
                    uri,
                    revision,
                    content_hash,
                    now,
                    now if ingested else None,
                )
                await conn.execute(stmt)

    async def delete(self, source_id: str, uri: str) -> None:
        stmt = sa.delete(sync_state).where(
            sync_state.c.source_id == source_id, sync_state.c.uri == uri
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)
