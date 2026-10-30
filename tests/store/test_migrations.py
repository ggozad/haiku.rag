import json
import logging
from importlib import metadata

import pytest

from haiku.rag.config import AppConfig
from haiku.rag.store.engine import Store, connect_lancedb, read_settings
from haiku.rag.store.engine import logger as engine_logger
from haiku.rag.store.exceptions import MigrationRequiredError, UpgradeRequiredError
from haiku.rag.store.upgrades import get_pending_upgrades, latest_schema_version
from tests.conftest import capture_logs


class TestMigrationCheck:
    async def test_new_database_sets_version(self, temp_db_path):
        """New database should set the current package version."""
        async with Store(temp_db_path, create=True) as store:
            version = await store.get_haiku_version()
            expected = metadata.version("haiku.rag-slim")
            assert version == expected

    @pytest.mark.parametrize("read_only", [False, True])
    async def test_open_does_not_change_version(self, temp_db_path, read_only):
        """Opening a database never writes the version, even when it differs.

        A version newer than all upgrade steps has no pending migrations, so the
        open succeeds, writable or read-only, and leaves the stored version alone.
        """
        async with Store(temp_db_path, create=True) as store:
            await store.set_haiku_version("100.0.0")

        async with Store(temp_db_path, read_only=read_only) as store:
            assert await store.get_haiku_version() == "100.0.0"

    @pytest.mark.parametrize("open_kwargs", [{}, {"create": True}, {"read_only": True}])
    async def test_pending_migrations_raises_error(self, temp_db_path, open_kwargs):
        """Pending migrations refuse the open, with create=True and read-only too."""
        async with Store(temp_db_path, create=True) as store:
            await store.set_haiku_version("0.19.0")

        with pytest.raises(MigrationRequiredError) as exc_info:
            async with Store(temp_db_path, **open_kwargs):
                pass
        assert "migrate" in str(exc_info.value).lower()

    async def test_skip_migration_check_bypasses_error(self, temp_db_path):
        """skip_migration_check=True should bypass migration error."""
        async with Store(temp_db_path, create=True) as store:
            await store.set_haiku_version("0.19.0")

        # Open with skip_migration_check should work
        async with Store(temp_db_path, skip_migration_check=True) as store:
            # Version should remain old (no auto-migration)
            assert await store.get_haiku_version() == "0.19.0"


class TestMigrateMethod:
    async def test_migrate_applies_pending_upgrades(self, temp_db_path):
        """Store.migrate() should apply pending upgrades and update version."""
        async with Store(temp_db_path, create=True) as store:
            await store.set_haiku_version("0.19.0")

        # Open with skip_migration_check to avoid error
        async with Store(temp_db_path, skip_migration_check=True) as store:
            old_version = await store.get_haiku_version()
            assert old_version == "0.19.0"

            # Run migration
            applied = await store.migrate()

            # Should have applied migrations
            assert isinstance(applied, list)
            assert len(applied) > 0
            for item in applied:
                assert isinstance(item, str)
            pending = get_pending_upgrades("0.19.0")
            assert isinstance(pending, list)
            assert len(pending) > 0

            # Version should be updated
            new_version = await store.get_haiku_version()
            expected = metadata.version("haiku.rag-slim")
            assert new_version == expected

    async def test_migrate_with_no_pending_returns_empty(self, temp_db_path):
        """Store.migrate() with no pending migrations returns empty list."""
        async with Store(temp_db_path, create=True) as store:
            # Already at current version
            pass

        async with Store(temp_db_path, skip_migration_check=True) as store:
            applied = await store.migrate()
            assert applied == []

    async def test_migrate_does_not_downgrade_future_version(self, temp_db_path):
        """migrate() must not move the version backwards.

        Running an older build's migrate against a DB stamped with a newer
        version (no pending upgrades) leaves the stored version untouched.
        """
        async with Store(temp_db_path, create=True) as store:
            await store.set_haiku_version("100.0.0")

        async with Store(temp_db_path, skip_migration_check=True) as store:
            applied = await store.migrate()
            assert applied == []
            assert await store.get_haiku_version() == "100.0.0"

    async def test_migrate_raises_read_only_error(self, temp_db_path):
        """Store.migrate() should raise ReadOnlyError in read-only mode."""
        from haiku.rag.store.exceptions import ReadOnlyError

        async with Store(temp_db_path, create=True) as store:
            await store.set_haiku_version("0.19.0")

        async with Store(
            temp_db_path, skip_migration_check=True, read_only=True
        ) as store:
            with pytest.raises(ReadOnlyError):
                await store.migrate()


async def _stamp_schema(store: Store, schema_version: str | None) -> None:
    """Write `schema_version` into the settings row; None removes it."""
    settings = await read_settings(store.settings_table)
    settings.pop("schema_version", None)
    if schema_version is not None:
        settings["schema_version"] = schema_version
    await store.settings_table.update(
        {"settings": json.dumps(settings)}, where="id = 'settings'"
    )


class TestSchemaVersion:
    async def test_init_records_the_newest_upgrade_step(self, temp_db_path):
        async with Store(temp_db_path, create=True) as store:
            stored = await read_settings(store.settings_table)
        assert stored["schema_version"] == latest_schema_version()

    async def test_migrate_records_it_with_nothing_pending(self, temp_db_path):
        async with Store(temp_db_path, create=True) as store:
            await _stamp_schema(store, None)

        async with Store(temp_db_path, skip_migration_check=True) as store:
            assert await store.migrate() == []
            stored = await read_settings(store.settings_table)
        assert stored["schema_version"] == latest_schema_version()

    async def test_recording_the_embedder_keeps_it(self, temp_db_path):
        from haiku.rag.store.repositories.settings import SettingsRepository

        async with Store(temp_db_path, create=True) as store:
            await _stamp_schema(store, "0.1.0")
            await SettingsRepository(store).save_current_settings()
            await store.set_haiku_version("0.92.0")
            stored = await read_settings(store.settings_table)
        assert stored["schema_version"] == "0.1.0"

    async def test_writable_open_of_a_newer_schema_raises(self, temp_db_path):
        async with Store(temp_db_path, create=True) as store:
            await _stamp_schema(store, "100.0.0")

        with pytest.raises(UpgradeRequiredError) as raised:
            async with Store(temp_db_path):
                pass
        assert "100.0.0" in str(raised.value)
        assert latest_schema_version() in str(raised.value)

    @pytest.mark.parametrize("skip_migration_check", [False, True])
    async def test_read_only_open_of_a_newer_schema_warns(
        self, temp_db_path, skip_migration_check
    ):
        async with Store(temp_db_path, create=True) as store:
            await _stamp_schema(store, "100.0.0")

        with capture_logs(engine_logger, logging.WARNING) as records:
            async with Store(
                temp_db_path,
                read_only=True,
                skip_migration_check=skip_migration_check,
            ):
                pass
        assert any("100.0.0" in record.getMessage() for record in records)

    async def test_open_without_a_schema_version_is_writable(self, temp_db_path):
        async with Store(temp_db_path, create=True) as store:
            await _stamp_schema(store, None)

        async with Store(temp_db_path) as store:
            assert not store.is_read_only

    async def test_migrate_refuses_a_newer_schema(self, temp_db_path):
        async with Store(temp_db_path, create=True) as store:
            await _stamp_schema(store, "100.0.0")

        async with Store(temp_db_path, skip_migration_check=True) as store:
            with pytest.raises(UpgradeRequiredError):
                await store.migrate()
            stored = await read_settings(store.settings_table)
        assert stored["schema_version"] == "100.0.0"

    async def test_no_open_creates_a_table_in_a_newer_schema(self, temp_db_path):
        async with Store(temp_db_path, create=True) as store:
            await _stamp_schema(store, "100.0.0")
            await store.db.drop_table("document_items")

        with pytest.raises(UpgradeRequiredError):
            async with Store(temp_db_path, skip_migration_check=True):
                pass

        db = await connect_lancedb(temp_db_path, AppConfig())
        try:
            assert "document_items" not in (await db.list_tables()).tables
        finally:
            db.close()
