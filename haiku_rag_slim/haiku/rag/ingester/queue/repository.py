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
    def __init__(self, conn: aiosqlite.Connection):
        # Row access by name in helpers below.
        conn.row_factory = aiosqlite.Row
        self._conn = conn
        # Serialize repo calls on the shared connection so cursors from one
        # coroutine don't sit "in progress" when another tries to commit.
        # aiosqlite executes statements on a single worker thread, but
        # individual cursors don't finalize until closed or GC'd — SQLite
        # then refuses commit() with "SQL statements in progress".
        self._lock = asyncio.Lock()

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

    async def mark_succeeded(self, job_id: str) -> None:
        async with self._lock:
            async with self._conn.execute(
                "UPDATE jobs SET status='succeeded', completed_at=? WHERE id=?",
                (_utcnow_iso(), job_id),
            ):
                pass
            await self._conn.commit()

    async def mark_dead(self, job_id: str, error: str) -> None:
        async with self._lock:
            async with self._conn.execute(
                "UPDATE jobs SET status='dead', completed_at=?, last_error=? WHERE id=?",
                (_utcnow_iso(), error, job_id),
            ):
                pass
            await self._conn.commit()

    async def reschedule(self, job_id: str, delay_seconds: float, error: str) -> None:
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
                WHERE id=?
                """,
                (scheduled, error, job_id),
            ):
                pass
            await self._conn.commit()

    async def retry(self, job_id: str) -> Job:
        """Rescue a dead job: status='queued', attempts=0, error cleared.
        Raises KeyError if the job doesn't exist."""
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
                WHERE id=?
                RETURNING *
                """,
                (now, job_id),
            ) as cursor:
                row = await cursor.fetchone()
            await self._conn.commit()
        if not row:
            raise KeyError(f"Job {job_id!r} not found")
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

    async def counts_by_status(self) -> dict[str, int]:
        async with self._lock:
            async with self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
            ) as cursor:
                rows = await cursor.fetchall()
        return {row["status"]: row["n"] for row in rows}

    async def reap_stale(self, claim_timeout_seconds: int) -> int:
        """Return claimed jobs whose claimed_at is older than the timeout to
        the queue. Used by the reaper to recover from crashed workers."""
        threshold = (
            datetime.now(UTC) - timedelta(seconds=claim_timeout_seconds)
        ).isoformat()
        async with self._lock:
            cursor = await self._conn.execute(
                """
                UPDATE jobs
                SET status='queued', claimed_at=NULL, claimed_by=NULL
                WHERE status='claimed' AND claimed_at < ?
                """,
                (threshold,),
            )
            rowcount = cursor.rowcount or 0
            await cursor.close()
            await self._conn.commit()
        return rowcount


class SyncStateRepo:
    def __init__(self, conn: aiosqlite.Connection):
        conn.row_factory = aiosqlite.Row
        self._conn = conn
        # See JobRepo for why we serialize on the shared connection.
        self._lock = asyncio.Lock()

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
        last_ingested_at; otherwise only last_seen_at is bumped."""
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
                    revision = excluded.revision,
                    content_hash = excluded.content_hash,
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
