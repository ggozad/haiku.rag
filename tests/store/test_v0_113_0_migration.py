from importlib import metadata

import pytest

from haiku.rag.store.engine import Store
from haiku.rag.store.exceptions import MigrationRequiredError
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.repositories.settings import SettingsRepository
from haiku.rag.store.upgrades import get_pending_upgrades


async def _at_previous_release(store: Store) -> None:
    """Drop the columns the step adds and record the release before it."""
    await store.documents_table.drop_columns(["original", "original_md5"])
    await store._write_settings(version="0.112.3", schema_version="0.91.0")


async def test_a_new_database_needs_no_migration(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        recorded = await SettingsRepository(store).get_current_settings()

    assert recorded["version"] == metadata.version("haiku.rag-slim") == "0.113.0"
    assert get_pending_upgrades(recorded["version"]) == []


async def test_migrate_adds_the_original_columns(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        created = await DocumentRepository(store).create(
            Document(content="kept", uri="test://kept")
        )
        await _at_previous_release(store)

    with pytest.raises(MigrationRequiredError):
        async with Store(temp_db_path, read_only=True):
            pass

    async with Store(temp_db_path, skip_migration_check=True) as store:
        await store.migrate()

    async with Store(temp_db_path, read_only=True) as store:
        rows = await store.documents_table.query().to_list()
        fields = {f.name: str(f.type) for f in await store.documents_table.schema()}

    assert (fields["original"], fields["original_md5"]) == ("large_binary", "string")
    assert [
        (r["id"], r["content"], r["original"], r["original_md5"]) for r in rows
    ] == [(created.id, "kept", None, None)]


async def test_rerunning_the_step_changes_nothing(temp_db_path):
    from haiku.rag.store.upgrades.v0_113_0 import _apply_original_columns

    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _at_previous_release(store)
        await _apply_original_columns(store)
        versions = await store.current_table_versions()

        await _apply_original_columns(store)

        assert await store.current_table_versions() == versions
