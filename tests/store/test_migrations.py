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
    @pytest.mark.asyncio
    async def test_new_database_sets_version(self, temp_db_path):
        """New database should set the current package version."""
        async with Store(temp_db_path, create=True) as store:
            version = await store.get_haiku_version()
            expected = metadata.version("haiku.rag-slim")
            assert version == expected

    @pytest.mark.asyncio
    async def test_existing_database_same_version_no_error(self, temp_db_path):
        """Opening a database with the same version should not error."""
        async with Store(temp_db_path, create=True):
            pass

        # Re-open - should work without error
        async with Store(temp_db_path):
            pass

    @pytest.mark.asyncio
    async def test_version_bump_without_pending_migrations_updates_silently(
        self, temp_db_path
    ):
        """When version is outdated but no migrations pending, update version silently."""
        async with Store(temp_db_path, create=True) as store:
            # Set an older version that has no pending migrations
            # (newer than all current upgrade steps)
            await store.set_haiku_version("100.0.0")

        # Re-open - should update version silently, no error
        async with Store(temp_db_path) as store:
            # Version should now be current
            version = await store.get_haiku_version()
            expected = metadata.version("haiku.rag-slim")
            assert version == expected

    @pytest.mark.asyncio
    async def test_pending_migrations_raises_error(self, temp_db_path):
        """When actual migrations are pending, should raise MigrationRequiredError."""
        async with Store(temp_db_path, create=True) as store:
            # Set version to before the first upgrade step
            await store.set_haiku_version("0.19.0")

        # Re-open should raise
        with pytest.raises(MigrationRequiredError) as exc_info:
            async with Store(temp_db_path) as store:
                pass
        assert "migrate" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_pending_migrations_read_only_raises_error(self, temp_db_path):
        """Read-only mode with pending migrations should still raise."""
        async with Store(temp_db_path, create=True) as store:
            await store.set_haiku_version("0.19.0")

        with pytest.raises(MigrationRequiredError):
            async with Store(temp_db_path, read_only=True) as store:
                pass

    @pytest.mark.asyncio
    async def test_read_only_version_bump_without_migrations_ok(self, temp_db_path):
        """Read-only mode with version bump but no migrations should work."""
        async with Store(temp_db_path, create=True) as store:
            # Set a version newer than all upgrade steps
            await store.set_haiku_version("100.0.0")

        # Read-only open should work (version not updated, but no error)
        async with Store(temp_db_path, read_only=True) as store:
            # Version should stay at the old value (can't update in read-only)
            assert await store.get_haiku_version() == "100.0.0"

    @pytest.mark.asyncio
    async def test_skip_migration_check_bypasses_error(self, temp_db_path):
        """skip_migration_check=True should bypass migration error."""
        async with Store(temp_db_path, create=True) as store:
            await store.set_haiku_version("0.19.0")

        # Open with skip_migration_check should work
        async with Store(temp_db_path, skip_migration_check=True) as store:
            # Version should remain old (no auto-migration)
            assert await store.get_haiku_version() == "0.19.0"


class TestMigrateMethod:
    @pytest.mark.asyncio
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
            assert len(applied) > 0

            # Version should be updated
            new_version = await store.get_haiku_version()
            expected = metadata.version("haiku.rag-slim")
            assert new_version == expected

    @pytest.mark.asyncio
    async def test_migrate_returns_applied_upgrades(self, temp_db_path):
        """Store.migrate() should return list of applied upgrade descriptions."""
        async with Store(temp_db_path, create=True) as store:
            await store.set_haiku_version("0.19.0")

        async with Store(temp_db_path, skip_migration_check=True) as store:
            applied = await store.migrate()

            # Should return descriptions of applied upgrades
            assert isinstance(applied, list)
            for item in applied:
                assert isinstance(item, str)

    @pytest.mark.asyncio
    async def test_migrate_with_no_pending_returns_empty(self, temp_db_path):
        """Store.migrate() with no pending migrations returns empty list."""
        async with Store(temp_db_path, create=True) as store:
            # Already at current version
            pass

        async with Store(temp_db_path, skip_migration_check=True) as store:
            applied = await store.migrate()
            assert applied == []

    @pytest.mark.asyncio
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
