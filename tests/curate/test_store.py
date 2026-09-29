import pytest
import sqlalchemy as sa

from haiku.rag.config.models import CurateStoreConfig
from haiku.rag.curate.store.db import SCHEMA_VERSION, schema_version
from haiku.rag.curate.store.migrations import (
    UnsupportedStoreError,
    apply_migrations,
    make_engine,
    open_store,
)


async def test_open_store_creates_the_schema(tmp_path):
    engine = await open_store(CurateStoreConfig(path=tmp_path / "curate.db"))
    try:
        async with engine.connect() as conn:
            version = (await conn.execute(sa.select(schema_version.c.version))).scalar()
        assert version == SCHEMA_VERSION
        assert await apply_migrations(engine) == SCHEMA_VERSION
    finally:
        await engine.dispose()


async def test_store_from_newer_code_is_refused(tmp_path):
    engine = await open_store(CurateStoreConfig(path=tmp_path / "curate.db"))
    try:
        async with engine.begin() as conn:
            await conn.execute(
                sa.update(schema_version).values(version=SCHEMA_VERSION + 1)
            )
        with pytest.raises(
            UnsupportedStoreError,
            match=f"store schema {SCHEMA_VERSION + 1} is newer than this "
            f"haiku-curate supports \\({SCHEMA_VERSION}\\)",
        ):
            await apply_migrations(engine)
    finally:
        await engine.dispose()


def test_postgres_engine_is_built_without_connecting():
    engine = make_engine(CurateStoreConfig(dburi="postgresql+asyncpg://u:p@h/db"))
    assert engine.dialect.name == "postgresql"
