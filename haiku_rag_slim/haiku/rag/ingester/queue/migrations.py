import sqlalchemy as sa
from sqlalchemy.engine import make_url
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
    (creating the parent directory). SQLite is capped to a single pooled
    connection so the claim stays atomic without row locks; Postgres uses
    pool_pre_ping so a long-running ingester survives a DB restart or idle
    connection drop."""
    if config.dburi:
        url = make_url(config.dburi)
    else:
        path = config.path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        url = make_url(f"sqlite+aiosqlite:///{path}")

    if url.get_backend_name() == "sqlite":
        engine = create_async_engine(url, pool_size=1, max_overflow=0)
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
        elif current < SCHEMA_VERSION:  # pragma: no cover - no migrations yet
            # No diff migrations exist yet — future versions add ALTER/UPDATE
            # statements between create_all and the version bump.
            await conn.execute(sa.update(schema_version).values(version=SCHEMA_VERSION))
    return SCHEMA_VERSION


async def open_queue(config: QueueConfig) -> AsyncEngine:
    """Build the queue engine and ensure its schema is up to date."""
    engine = make_engine(config)
    await apply_migrations(engine)
    return engine
