import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from haiku.rag import sqlstore
from haiku.rag.config.models import CurateStoreConfig
from haiku.rag.curate.store.db import SCHEMA_VERSION, metadata, schema_version


class UnsupportedStoreError(Exception):
    """The store was written by a newer haiku-curate."""


def make_engine(config: CurateStoreConfig) -> AsyncEngine:
    """Build the curate store's AsyncEngine from config."""
    return sqlstore.make_engine(config.path, config.dburi)


async def apply_migrations(engine: AsyncEngine) -> int:
    """Create missing tables and pin schema_version; returns the version after the call."""
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        current = (
            await conn.execute(sa.select(schema_version.c.version).limit(1))
        ).scalar_one_or_none()
        if current is not None and current > SCHEMA_VERSION:
            raise UnsupportedStoreError(
                f"store schema {current} is newer than this haiku-curate "
                f"supports ({SCHEMA_VERSION}); upgrade haiku-curate"
            )
        if current is None:
            await conn.execute(sa.insert(schema_version).values(version=SCHEMA_VERSION))
    return SCHEMA_VERSION


async def open_store(config: CurateStoreConfig) -> AsyncEngine:
    """Build the store engine and ensure its schema is up to date."""
    engine = make_engine(config)
    await apply_migrations(engine)
    return engine
