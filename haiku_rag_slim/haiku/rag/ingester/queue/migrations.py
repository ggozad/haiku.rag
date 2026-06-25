import sqlalchemy as sa
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from haiku.rag.config.models import QueueConfig
from haiku.rag.ingester.queue.db import (
    SCHEMA_VERSION,
    install_sqlite_pragmas,
    metadata,
    schema_version,
)

__all__ = ["SCHEMA_VERSION", "apply_migrations", "make_engine", "open_queue"]


def make_engine(config: QueueConfig) -> AsyncEngine:
    """Build the queue's AsyncEngine from config. Uses `dburi` when set,
    otherwise a `sqlite+aiosqlite` URL pointing at the resolved `path`
    (creating the parent directory). SQLite runs in WAL mode with a small pool
    so reads (API stats/jobs) proceed concurrently with worker writes; the
    claim stays atomic via its single UPDATE statement, not the pool size.
    Postgres uses pool_pre_ping so a long-running ingester survives a DB
    restart or idle connection drop."""
    if config.dburi:
        url = make_url(config.dburi)
    else:
        path = config.path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # URL.create keeps the path literal — building a string and reparsing
        # would treat `?`/`#` in the filename as query/fragment.
        url = URL.create("sqlite+aiosqlite", database=str(path))

    if url.get_backend_name() == "sqlite":
        engine = create_async_engine(url, pool_size=5, max_overflow=5)
    else:
        engine = create_async_engine(url, pool_pre_ping=True)
    install_sqlite_pragmas(engine)
    return engine


async def apply_migrations(engine: AsyncEngine) -> int:
    """Idempotently create tables/indexes and pin schema_version.

    Returns the schema version after the call. Safe on a fresh DB or one
    already at the latest version.
    """
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        current = (
            await conn.execute(sa.select(schema_version.c.version).limit(1))
        ).scalar_one_or_none()
        if current is None:
            await conn.execute(sa.insert(schema_version).values(version=SCHEMA_VERSION))
        elif current < SCHEMA_VERSION:
            # create_all only creates missing tables/indexes, never adds columns
            # to an existing table, so column additions need explicit ALTERs.
            if current < 2:
                await conn.execute(
                    sa.text("ALTER TABLE jobs ADD COLUMN last_heartbeat_at TEXT")
                )
                await conn.execute(
                    sa.text(
                        "UPDATE jobs SET last_heartbeat_at = claimed_at "
                        "WHERE status = 'claimed'"
                    )
                )
            await conn.execute(sa.update(schema_version).values(version=SCHEMA_VERSION))
    return SCHEMA_VERSION


async def open_queue(config: QueueConfig) -> AsyncEngine:
    """Build the queue engine and ensure its schema is up to date."""
    engine = make_engine(config)
    await apply_migrations(engine)
    return engine
