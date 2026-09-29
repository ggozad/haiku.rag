import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.schema import CreateIndex

from haiku.rag import sqlstore
from haiku.rag.config.models import QueueConfig
from haiku.rag.ingester.queue.db import (
    SCHEMA_VERSION,
    jobs,
    metadata,
    schema_version,
)

__all__ = ["SCHEMA_VERSION", "apply_migrations", "make_engine", "open_queue"]


def make_engine(config: QueueConfig) -> AsyncEngine:
    """Build the queue's AsyncEngine from config."""
    return sqlstore.make_engine(config.path, config.dburi)


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
            if current < 3:
                await conn.execute(
                    sa.text(
                        "ALTER TABLE jobs ADD COLUMN conversion_stalled "
                        "BOOLEAN NOT NULL DEFAULT FALSE"
                    )
                )
                # create_all made every other index; this one needs the column
                # that has just been added. Emitted from the same Index object
                # create_all uses, so a migrated queue enforces exactly what a
                # new one does.
                blocking = next(
                    i for i in jobs.indexes if i.name == "uq_jobs_blocking_op"
                )
                await conn.execute(CreateIndex(blocking, if_not_exists=True))
            await conn.execute(sa.update(schema_version).values(version=SCHEMA_VERSION))
    return SCHEMA_VERSION


async def open_queue(config: QueueConfig) -> AsyncEngine:
    """Build the queue engine and ensure its schema is up to date."""
    engine = make_engine(config)
    await apply_migrations(engine)
    return engine
