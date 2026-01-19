from importlib import metadata

import pytest

from haiku.rag.store import Store
from haiku.rag.store.exceptions import MigrationRequiredError


class TestMigrationRequiredError:
    def test_migration_required_error_is_exception(self):
        """MigrationRequiredError should be a subclass of Exception."""
        assert issubclass(MigrationRequiredError, Exception)

    def test_migration_required_error_can_be_raised(self):
        """MigrationRequiredError can be raised and caught."""
        with pytest.raises(MigrationRequiredError) as exc_info:
            raise MigrationRequiredError("Run 'haiku-rag migrate' to upgrade")
        assert "migrate" in str(exc_info.value)


class TestMigrationCheck:
    def test_new_database_sets_version(self, temp_db_path):
        """New database should set the current package version."""
        store = Store(temp_db_path, create=True)
        version = store.get_haiku_version()
        expected = metadata.version("haiku.rag-slim")
        assert version == expected
        store.close()

    def test_existing_database_same_version_no_error(self, temp_db_path):
        """Opening a database with the same version should not error."""
        store = Store(temp_db_path, create=True)
        store.close()

        # Re-open - should work without error
        store = Store(temp_db_path)
        store.close()

    def test_version_bump_without_pending_migrations_updates_silently(
        self, temp_db_path
    ):
        """When version is outdated but no migrations pending, update version silently."""
        store = Store(temp_db_path, create=True)
        # Set an older version that has no pending migrations
        # (newer than all current upgrade steps)
        store.set_haiku_version("100.0.0")
        store.close()

        # Re-open - should update version silently, no error
        store = Store(temp_db_path)
        # Version should now be current
        version = store.get_haiku_version()
        expected = metadata.version("haiku.rag-slim")
        assert version == expected
        store.close()

    def test_pending_migrations_raises_error(self, temp_db_path):
        """When actual migrations are pending, should raise MigrationRequiredError."""
        store = Store(temp_db_path, create=True)
        # Set version to before the first upgrade step
        store.set_haiku_version("0.19.0")
        store.close()

        # Re-open should raise
        with pytest.raises(MigrationRequiredError) as exc_info:
            Store(temp_db_path)
        assert "migrate" in str(exc_info.value).lower()

    def test_pending_migrations_read_only_raises_error(self, temp_db_path):
        """Read-only mode with pending migrations should still raise."""
        store = Store(temp_db_path, create=True)
        store.set_haiku_version("0.19.0")
        store.close()

        with pytest.raises(MigrationRequiredError):
            Store(temp_db_path, read_only=True)

    def test_read_only_version_bump_without_migrations_ok(self, temp_db_path):
        """Read-only mode with version bump but no migrations should work."""
        store = Store(temp_db_path, create=True)
        # Set a version newer than all upgrade steps
        store.set_haiku_version("100.0.0")
        store.close()

        # Read-only open should work (version not updated, but no error)
        store = Store(temp_db_path, read_only=True)
        # Version should stay at the old value (can't update in read-only)
        assert store.get_haiku_version() == "100.0.0"
        store.close()

    def test_skip_migration_check_bypasses_error(self, temp_db_path):
        """skip_migration_check=True should bypass migration error."""
        store = Store(temp_db_path, create=True)
        store.set_haiku_version("0.19.0")
        store.close()

        # Open with skip_migration_check should work
        store = Store(temp_db_path, skip_migration_check=True)
        # Version should remain old (no auto-migration)
        assert store.get_haiku_version() == "0.19.0"
        store.close()


class TestMigrateMethod:
    def test_migrate_applies_pending_upgrades(self, temp_db_path):
        """Store.migrate() should apply pending upgrades and update version."""
        store = Store(temp_db_path, create=True)
        store.set_haiku_version("0.19.0")
        store.close()

        # Open with skip_migration_check to avoid error
        store = Store(temp_db_path, skip_migration_check=True)
        old_version = store.get_haiku_version()
        assert old_version == "0.19.0"

        # Run migration
        applied = store.migrate()

        # Should have applied migrations
        assert len(applied) > 0

        # Version should be updated
        new_version = store.get_haiku_version()
        expected = metadata.version("haiku.rag-slim")
        assert new_version == expected
        store.close()

    def test_migrate_returns_applied_upgrades(self, temp_db_path):
        """Store.migrate() should return list of applied upgrade descriptions."""
        store = Store(temp_db_path, create=True)
        store.set_haiku_version("0.19.0")
        store.close()

        store = Store(temp_db_path, skip_migration_check=True)
        applied = store.migrate()

        # Should return descriptions of applied upgrades
        assert isinstance(applied, list)
        for item in applied:
            assert isinstance(item, str)
        store.close()

    def test_migrate_with_no_pending_returns_empty(self, temp_db_path):
        """Store.migrate() with no pending migrations returns empty list."""
        store = Store(temp_db_path, create=True)
        # Already at current version
        store.close()

        store = Store(temp_db_path, skip_migration_check=True)
        applied = store.migrate()
        assert applied == []
        store.close()

    def test_migrate_raises_read_only_error(self, temp_db_path):
        """Store.migrate() should raise ReadOnlyError in read-only mode."""
        from haiku.rag.store.exceptions import ReadOnlyError

        store = Store(temp_db_path, create=True)
        store.set_haiku_version("0.19.0")
        store.close()

        store = Store(temp_db_path, skip_migration_check=True, read_only=True)
        with pytest.raises(ReadOnlyError):
            store.migrate()
        store.close()


class TestGetPendingUpgrades:
    def test_get_pending_upgrades_returns_list(self):
        """get_pending_upgrades() should return a list of Upgrade objects."""
        from haiku.rag.store.upgrades import get_pending_upgrades

        pending = get_pending_upgrades("0.19.0")
        assert isinstance(pending, list)
        # Should have at least the v0.20.0 upgrade
        assert len(pending) > 0

    def test_get_pending_upgrades_from_current_version_is_empty(self):
        """get_pending_upgrades() from current version should be empty."""
        from haiku.rag.store.upgrades import get_pending_upgrades

        current = metadata.version("haiku.rag-slim")
        pending = get_pending_upgrades(current)
        assert pending == []

    def test_get_pending_upgrades_from_future_version_is_empty(self):
        """get_pending_upgrades() from a future version should be empty."""
        from haiku.rag.store.upgrades import get_pending_upgrades

        pending = get_pending_upgrades("100.0.0")
        assert pending == []
