import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.store.repositories.settings import (
    ConfigMismatchError,
)


def test_settings_table_populated_on_store_init(temp_db_path):
    """Test that settings table is populated with current config when store is initialized."""
    from haiku.rag.store.engine import Store
    from haiku.rag.store.repositories.settings import SettingsRepository

    store = Store(temp_db_path)
    settings_repo = SettingsRepository(store)

    db_settings = settings_repo.get_current_settings()
    config_dict = Config.model_dump(mode="json")

    # Remove version from db_settings since it's added automatically
    db_settings_without_version = {
        k: v for k, v in db_settings.items() if k != "version"
    }
    assert db_settings_without_version == config_dict

    store.close()


def test_settings_save_and_retrieve(temp_db_path):
    """Test saving and retrieving settings after config change."""
    from haiku.rag.store.engine import Store
    from haiku.rag.store.repositories.settings import SettingsRepository

    store = Store(temp_db_path)
    settings_repo = SettingsRepository(store)

    original_chunk_size = Config.CHUNK_SIZE
    Config.CHUNK_SIZE = 2 * original_chunk_size

    settings_repo.save_current_settings()
    retrieved_settings = settings_repo.get_current_settings()
    assert retrieved_settings["CHUNK_SIZE"] == 2 * original_chunk_size

    Config.CHUNK_SIZE = original_chunk_size
    store.close()


@pytest.mark.skip(reason="Config validation not fully implemented for LanceDB")
async def test_config_validation_on_db_load(temp_db_path):
    """Test that config validation fails when loading db with mismatched settings."""
    from haiku.rag.store.engine import Store
    from haiku.rag.store.repositories.settings import SettingsRepository

    # Create store and save settings
    store1 = Store(temp_db_path)
    store1.close()

    # Change config
    original_chunk_size = Config.CHUNK_SIZE
    Config.CHUNK_SIZE = 999

    try:
        # Loading the database should raise ConfigMismatchError
        with pytest.raises(ConfigMismatchError) as exc_info:
            Store(temp_db_path)

        assert "CHUNK_SIZE" in str(exc_info.value)
        assert "Consider rebuilding" in str(exc_info.value)

        # Rebuild
        async with HaikuRAG(db_path=temp_db_path, skip_validation=True) as client:
            async for _ in client.rebuild_database():
                pass  # Process all documents

        # Verify we can now load the database without exception (settings were updated)
        store2 = Store(temp_db_path)
        settings_repo2 = SettingsRepository(store2)
        db_settings = settings_repo2.get_current_settings()
        assert db_settings["CHUNK_SIZE"] == 999
        store2.close()

    finally:
        Config.CHUNK_SIZE = original_chunk_size
