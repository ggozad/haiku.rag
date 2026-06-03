import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine

SCHEMA_VERSION = 1

metadata = sa.MetaData()

jobs = sa.Table(
    "jobs",
    metadata,
    sa.Column("id", sa.Text, primary_key=True),
    sa.Column("source_id", sa.Text, nullable=False),
    sa.Column("uri", sa.Text, nullable=False),
    sa.Column("op", sa.Text, nullable=False),
    sa.Column("content_hash", sa.Text),
    sa.Column("revision", sa.Text),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("attempts", sa.Integer, nullable=False, server_default=sa.text("0")),
    sa.Column("max_attempts", sa.Integer, nullable=False, server_default=sa.text("5")),
    sa.Column("last_error", sa.Text),
    sa.Column("extra", sa.Text),
    sa.Column("enqueued_at", sa.Text, nullable=False),
    sa.Column("scheduled_at", sa.Text, nullable=False),
    sa.Column("claimed_at", sa.Text),
    sa.Column("claimed_by", sa.Text),
    sa.Column("completed_at", sa.Text),
)

# A (source_id, uri) pair can only have one live job (queued or claimed) at a
# time, regardless of op. Live UPSERT and DELETE for the same URI can't both
# exist — preventing a DELETE worker from removing a document a sibling UPSERT
# just ingested. Once succeeded or dead, the row no longer satisfies the WHERE
# clause and a re-enqueue is allowed.
_live = jobs.c.status.in_(["queued", "claimed"])
sa.Index(
    "uq_jobs_live",
    jobs.c.source_id,
    jobs.c.uri,
    unique=True,
    sqlite_where=_live,
    postgresql_where=_live,
)

_queued = jobs.c.status == "queued"
sa.Index(
    "idx_jobs_claimable",
    jobs.c.scheduled_at,
    sqlite_where=_queued,
    postgresql_where=_queued,
)

_succeeded = jobs.c.status == "succeeded"
sa.Index(
    "idx_jobs_succeeded_completed",
    jobs.c.completed_at,
    sqlite_where=_succeeded,
    postgresql_where=_succeeded,
)

sync_state = sa.Table(
    "sync_state",
    metadata,
    sa.Column("source_id", sa.Text, primary_key=True),
    sa.Column("uri", sa.Text, primary_key=True),
    sa.Column("revision", sa.Text),
    sa.Column("content_hash", sa.Text),
    sa.Column("last_seen_at", sa.Text, nullable=False),
    sa.Column("last_ingested_at", sa.Text),
)

schema_version = sa.Table(
    "schema_version",
    metadata,
    sa.Column("version", sa.Integer, primary_key=True),
)


def install_sqlite_pragmas(engine: AsyncEngine) -> None:
    """Register a connect listener that sets the per-connection pragmas the
    queue relies on. SQLite-only — Postgres has no equivalent and asyncpg
    rejects PRAGMA, so the listener is never attached for it."""
    if engine.dialect.name != "sqlite":
        return

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
        finally:
            cursor.close()
