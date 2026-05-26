import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta

import aiosqlite

from haiku.rag.ingester.queue.models import Job, JobOp, JobStatus, SyncStateRow


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _row_to_job(row: aiosqlite.Row) -> Job:
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
        completed_at=_parse_dt(row["completed_at"]),
    )


def _row_to_sync_state(row: aiosqlite.Row) -> SyncStateRow:
    return SyncStateRow(
        source_id=row["source_id"],
        uri=row["uri"],
        revision=row["revision"],
        content_hash=row["content_hash"],
        last_seen_at=datetime.fromisoformat(row["last_seen_at"]),
        last_ingested_at=_parse_dt(row["last_ingested_at"]),
    )


class JobRepo:
    def __init__(
        self,
        conn: aiosqlite.Connection,
        lock: asyncio.Lock | None = None,
    ):
        # Row access by name in helpers below.
        conn.row_factory = aiosqlite.Row
        self._conn = conn
        # Serialize repo calls on the shared connection so cursors from one
        # coroutine don't sit "in progress" when another tries to commit.
        # aiosqlite executes statements on a single worker thread, but
        # individual cursors don't finalize until closed or GC'd — SQLite
        # then refuses commit() with "SQL statements in progress". When
        # JobRepo and SyncStateRepo share the same connection, callers must
        # pass the same lock instance so cross-repo calls also serialize.
        self._lock = lock or asyncio.Lock()

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
        live (queued/claimed) job already exists for the same (source_id, uri,
        op). The partial unique index enforces atomicity."""
        job_id = str(uuid.uuid4())
        now = _utcnow_iso()
        extra_json = json.dumps(extra) if extra is not None else None
        async with self._lock:
            async with self._conn.execute(
                """
                INSERT INTO jobs (
                    id, source_id, uri, op, content_hash, revision, status,
                    attempts, max_attempts, last_error, extra,
                    enqueued_at, scheduled_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', 0, ?, NULL, ?, ?, ?)
                ON CONFLICT DO NOTHING
                RETURNING *
                """,
                (
                    job_id,
                    source_id,
                    uri,
                    op.value,
                    content_hash,
                    revision,
                    max_attempts,
                    extra_json,
                    now,
                    now,
                ),
            ) as cursor:
                row = await cursor.fetchone()
            await self._conn.commit()
        return _row_to_job(row) if row else None

    async def claim_next(self, worker_id: str) -> Job | None:
        """Atomically claim the oldest queued job whose scheduled_at <= now.
        Implemented as a single UPDATE ... RETURNING — no SELECT/UPDATE race."""
        now = _utcnow_iso()
        async with self._lock:
            async with self._conn.execute(
                """
                UPDATE jobs
                SET status = 'claimed',
                    claimed_at = ?,
                    claimed_by = ?,
                    attempts = attempts + 1
                WHERE id = (
                    SELECT id FROM jobs
                    WHERE status = 'queued' AND scheduled_at <= ?
                    ORDER BY scheduled_at
                    LIMIT 1
                )
                RETURNING *
                """,
                (now, worker_id, now),
            ) as cursor:
                row = await cursor.fetchone()
            await self._conn.commit()
        return _row_to_job(row) if row else None

    async def get_job(self, job_id: str) -> Job | None:
        async with self._lock:
            async with self._conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ) as cursor:
                row = await cursor.fetchone()
        return _row_to_job(row) if row else None

    async def mark_succeeded(self, job_id: str, claimed_by: str) -> bool:
        """Transition a still-claimed job to `succeeded`. Guarded on
        `status='claimed' AND claimed_by=?` so a reaper-resurrected job
        picked up by a different worker isn't clobbered by the original
        slow worker. Returns True when the row was updated."""
        async with self._lock:
            async with self._conn.execute(
                "UPDATE jobs SET status='succeeded', completed_at=? "
                "WHERE id=? AND status='claimed' AND claimed_by=? RETURNING id",
                (_utcnow_iso(), job_id, claimed_by),
            ) as cursor:
                row = await cursor.fetchone()
            await self._conn.commit()
        return row is not None

    async def mark_dead(self, job_id: str, error: str, claimed_by: str) -> bool:
        """Transition a still-claimed job to `dead`. See `mark_succeeded`
        for the guard semantics."""
        async with self._lock:
            async with self._conn.execute(
                "UPDATE jobs SET status='dead', completed_at=?, last_error=? "
                "WHERE id=? AND status='claimed' AND claimed_by=? RETURNING id",
                (_utcnow_iso(), error, job_id, claimed_by),
            ) as cursor:
                row = await cursor.fetchone()
            await self._conn.commit()
        return row is not None

    async def reschedule(
        self, job_id: str, delay_seconds: float, error: str, claimed_by: str
    ) -> bool:
        """Reset a still-claimed job back to `queued` with a future
        scheduled_at. Guarded on `status='claimed' AND claimed_by=?` so a
        slow worker can't clobber a re-claim that happened after the reaper
        reset its claim. Returns True when the row was updated."""
        scheduled = (datetime.now(UTC) + timedelta(seconds=delay_seconds)).isoformat()
        async with self._lock:
            async with self._conn.execute(
                """
                UPDATE jobs
                SET status='queued',
                    scheduled_at=?,
                    claimed_at=NULL,
                    claimed_by=NULL,
                    last_error=?
                WHERE id=? AND status='claimed' AND claimed_by=?
                RETURNING id
                """,
                (scheduled, error, job_id, claimed_by),
            ) as cursor:
                row = await cursor.fetchone()
            await self._conn.commit()
        return row is not None

    async def retry(self, job_id: str) -> Job:
        """Reset a `dead` or `queued` job: status='queued', attempts=0,
        error cleared, scheduled for immediate re-claim. Refuses `claimed`
        rows (would race with the worker still processing) and `succeeded`
        rows (re-ingest via UPSERT instead). Raises KeyError when the row
        is missing or in a non-retryable state."""
        now = _utcnow_iso()
        async with self._lock:
            async with self._conn.execute(
                """
                UPDATE jobs
                SET status='queued',
                    attempts=0,
                    last_error=NULL,
                    claimed_at=NULL,
                    claimed_by=NULL,
                    completed_at=NULL,
                    scheduled_at=?
                WHERE id=? AND status IN ('dead', 'queued')
                RETURNING *
                """,
                (now, job_id),
            ) as cursor:
                row = await cursor.fetchone()
            await self._conn.commit()
        if not row:
            raise KeyError(f"Job {job_id!r} not found or not retryable")
        return _row_to_job(row)

    async def cancel(self, job_id: str) -> bool:
        """Delete a queued or claimed job. Returns True if a row was removed."""
        async with self._lock:
            async with self._conn.execute(
                "DELETE FROM jobs WHERE id=? AND status IN ('queued', 'claimed') RETURNING id",
                (job_id,),
            ) as cursor:
                row = await cursor.fetchone()
            await self._conn.commit()
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
        clauses: list[str] = []
        params: list[object] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        if uri is not None:
            clauses.append("uri = ?")
            params.append(uri)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.extend([limit, offset])
        async with self._lock:
            async with self._conn.execute(
                f"SELECT * FROM jobs {where} ORDER BY enqueued_at DESC LIMIT ? OFFSET ?",
                params,
            ) as cursor:
                rows = await cursor.fetchall()
        return [_row_to_job(r) for r in rows]

    async def has_pending(self, source_id: str) -> bool:
        """True iff at least one queued/claimed job exists for the source.

        Cheap probe used by pollers to skip sweeps when there's already
        outstanding work — the queue's unique index would dedupe new enqueues
        anyway, so a sweep into a saturated queue is pure wasted listing work.
        """
        async with self._lock:
            async with self._conn.execute(
                "SELECT 1 FROM jobs WHERE source_id=? AND status IN ('queued','claimed') LIMIT 1",
                (source_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return row is not None

    async def counts_by_status(self) -> dict[str, int]:
        async with self._lock:
            async with self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
            ) as cursor:
                rows = await cursor.fetchall()
        return {row["status"]: row["n"] for row in rows}

    async def count_succeeded_since(self, seconds: int) -> int:
        """How many jobs reached `succeeded` in the last `seconds` seconds.
        Drives the dashboard's rolling-throughput chips."""
        threshold = (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat()
        async with self._lock:
            async with self._conn.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE status='succeeded' AND completed_at >= ?",
                (threshold,),
            ) as cursor:
                row = await cursor.fetchone()
        return int(row["n"]) if row else 0

    async def oldest_queued_age_seconds(self) -> float | None:
        """Age (in seconds) of the oldest job sitting in `queued` whose
        scheduled_at is in the past. Returns None when nothing is waiting.
        Tells operators whether work is backing up."""
        now = datetime.now(UTC)
        async with self._lock:
            async with self._conn.execute(
                "SELECT MIN(scheduled_at) AS oldest FROM jobs "
                "WHERE status='queued' AND scheduled_at <= ?",
                (now.isoformat(),),
            ) as cursor:
                row = await cursor.fetchone()
        if not row or row["oldest"] is None:
            return None
        return (now - datetime.fromisoformat(row["oldest"])).total_seconds()

    async def counts_by_source(self, *statuses: str) -> dict[str, int]:
        """source_id → count of jobs in any of the given statuses. Drives the
        dashboard's per-source DLQ and backlog summaries."""
        if not statuses:
            return {}
        placeholders = ",".join("?" * len(statuses))
        async with self._lock:
            async with self._conn.execute(
                f"SELECT source_id, COUNT(*) AS n FROM jobs "
                f"WHERE status IN ({placeholders}) GROUP BY source_id",
                statuses,
            ) as cursor:
                rows = await cursor.fetchall()
        return {row["source_id"]: row["n"] for row in rows}

    async def release_if_claimed(self, job_id: str, claimed_by: str) -> bool:
        """Reset a still-claimed job back to queued, immediately reclaimable.
        Guarded on `status='claimed' AND claimed_by=?` so the cancel-cleanup
        of a slow worker doesn't strip the claim of a different worker that
        re-claimed after a reaper reset. Decrements attempts to undo the
        increment from `claim_next`, since a cancellation isn't a failed
        attempt. Returns True if the row was released."""
        now = _utcnow_iso()
        async with self._lock:
            async with self._conn.execute(
                """
                UPDATE jobs
                SET status='queued',
                    claimed_at=NULL,
                    claimed_by=NULL,
                    scheduled_at=?,
                    attempts=MAX(0, attempts - 1)
                WHERE id=? AND status='claimed' AND claimed_by=?
                RETURNING id
                """,
                (now, job_id, claimed_by),
            ) as cursor:
                row = await cursor.fetchone()
            await self._conn.commit()
        return row is not None

    async def reap_stale(self, claim_timeout_seconds: int) -> int:
        """Reset claimed jobs whose claimed_at is older than the timeout
        back to `queued`. Decrements `attempts` to undo the increment from
        `claim_next` — a crashed worker isn't a consumed attempt."""
        threshold = (
            datetime.now(UTC) - timedelta(seconds=claim_timeout_seconds)
        ).isoformat()
        async with self._lock:
            cursor = await self._conn.execute(
                """
                UPDATE jobs
                SET status='queued',
                    claimed_at=NULL,
                    claimed_by=NULL,
                    attempts=MAX(0, attempts - 1)
                WHERE status='claimed' AND claimed_at < ?
                """,
                (threshold,),
            )
            rowcount = cursor.rowcount or 0
            await cursor.close()
            await self._conn.commit()
        return rowcount


class SyncStateRepo:
    def __init__(
        self,
        conn: aiosqlite.Connection,
        lock: asyncio.Lock | None = None,
    ):
        conn.row_factory = aiosqlite.Row
        self._conn = conn
        # Pass the same lock instance JobRepo uses when both wrap one
        # connection. See JobRepo for the SQLite cursor + commit constraint.
        self._lock = lock or asyncio.Lock()

    async def get_snapshot(self, source_id: str) -> dict[str, str]:
        """uri -> revision map for the source. Drops rows where revision is
        NULL (the poller can't compare against an absent revision)."""
        async with self._lock:
            async with self._conn.execute(
                "SELECT uri, revision FROM sync_state WHERE source_id=? AND revision IS NOT NULL",
                (source_id,),
            ) as cursor:
                rows = await cursor.fetchall()
        return {row["uri"]: row["revision"] for row in rows}

    async def get_row(self, source_id: str, uri: str) -> SyncStateRow | None:
        async with self._lock:
            async with self._conn.execute(
                "SELECT * FROM sync_state WHERE source_id=? AND uri=?",
                (source_id, uri),
            ) as cursor:
                row = await cursor.fetchone()
        return _row_to_sync_state(row) if row else None

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
        ingested_at = now if ingested else None
        async with self._lock:
            async with self._conn.execute(
                """
                INSERT INTO sync_state (
                    source_id, uri, revision, content_hash, last_seen_at, last_ingested_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, uri) DO UPDATE SET
                    revision = COALESCE(excluded.revision, revision),
                    content_hash = COALESCE(excluded.content_hash, content_hash),
                    last_seen_at = excluded.last_seen_at,
                    last_ingested_at = COALESCE(excluded.last_ingested_at, last_ingested_at)
                """,
                (source_id, uri, revision, content_hash, now, ingested_at),
            ):
                pass
            await self._conn.commit()

    async def delete(self, source_id: str, uri: str) -> None:
        async with self._lock:
            async with self._conn.execute(
                "DELETE FROM sync_state WHERE source_id=? AND uri=?",
                (source_id, uri),
            ):
                pass
            await self._conn.commit()
