import logging
from importlib import metadata

import pytest

from haiku.rag.config.models import AppConfig
from haiku.rag.store import (
    ConfigMismatchError,
    MigrationRequiredError,
    ReadOnlyError,
    TagError,
    engine,
)
from haiku.rag.store.engine import Store
from haiku.rag.store.models import Document
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.repositories.settings import SettingsRepository
from haiku.rag.store.schema import INCOMPLETE_REBUILD_TABLE, IncompleteRebuildRecord
from tests.conftest import capture_logs


async def _tagged_database(path) -> dict[str, int]:
    """One document, a tag, then a second document. Returns the tag's versions."""
    async with Store(path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="before the tag"))
        await store.create_tag("before")
        await repo.create(Document(content="after the tag"))
        return (await store.list_tags())["before"].tables


async def test_store_reads_every_table_at_the_tag(temp_db_path):
    tagged = await _tagged_database(temp_db_path)

    async with Store(temp_db_path, tag="before") as store:
        assert store.is_read_only
        for name, table in store._tables().items():
            assert await table.version() == tagged[name], name
        docs = await DocumentRepository(store).list_all(include_content=True)
        assert [d.content for d in docs] == ["before the tag"]


async def test_store_at_a_tag_refuses_writes(temp_db_path):
    await _tagged_database(temp_db_path)

    async with Store(temp_db_path, tag="before") as store:
        with pytest.raises(ReadOnlyError):
            await DocumentRepository(store).create(Document(content="x"))


async def test_store_at_a_tag_cannot_create(temp_db_path):
    with pytest.raises(ValueError, match="create"):
        Store(temp_db_path, create=True, tag="before")


async def test_store_at_a_missing_tag_raises(temp_db_path):
    await _tagged_database(temp_db_path)

    with pytest.raises(TagError, match="Tag 'nope' does not exist"):
        async with Store(temp_db_path, tag="nope"):
            pass


async def test_store_at_a_partial_tag_raises(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        await store.chunks_table.tags.create(
            "partial", await store.chunks_table.version()
        )

    with pytest.raises(TagError, match="Tag 'partial' is partial") as raised:
        async with Store(temp_db_path, tag="partial"):
            pass
    assert "settings" in str(raised.value)


async def test_store_at_a_tag_validates_against_the_tagged_settings(temp_db_path):
    config = AppConfig()
    later = config.model_copy(deep=True)
    later.embeddings.model.vector_dim = config.embeddings.model.vector_dim + 1
    async with Store(temp_db_path, config=config, create=True) as store:
        await store.create_tag("before")
    async with Store(temp_db_path, config=later, skip_validation=True) as store:
        await SettingsRepository(store).save_current_settings()

    with pytest.raises(ConfigMismatchError):
        async with Store(temp_db_path, config=config, read_only=True):
            pass
    async with Store(temp_db_path, config=config, tag="before") as store:
        assert store.stored_embedding is not None
        assert store.stored_embedding[2] == config.embeddings.model.vector_dim


async def test_store_at_a_tag_before_a_migration_is_refused(temp_db_path):
    current = metadata.version("haiku.rag-slim")
    async with Store(temp_db_path, create=True) as store:
        await store.set_haiku_version("0.88.0")
        await store.create_tag("old")
        await store.set_haiku_version(current)

    with pytest.raises(MigrationRequiredError, match="copy") as raised:
        async with Store(temp_db_path, tag="old"):
            pass
    assert "Run 'haiku-rag migrate' to upgrade" not in str(raised.value)

    async with Store(temp_db_path, read_only=True):
        pass


async def test_store_at_a_tag_does_not_report_the_live_rebuild_state(temp_db_path):
    await _tagged_database(temp_db_path)
    async with Store(temp_db_path) as store:
        await store.db.create_table(
            INCOMPLETE_REBUILD_TABLE,
            data=[IncompleteRebuildRecord(mode="embed_only", vector_index=False)],
        )

    with capture_logs(engine.logger, logging.WARNING) as live:
        async with Store(temp_db_path, read_only=True):
            pass
    with capture_logs(engine.logger, logging.WARNING) as tagged:
        async with Store(temp_db_path, tag="before"):
            pass

    assert ["Database incomplete" in r.getMessage() for r in live] == [True]
    assert tagged == []
