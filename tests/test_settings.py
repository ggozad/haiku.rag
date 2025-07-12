import tempfile
from pathlib import Path

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.store.engine import Store
from haiku.rag.store.repositories.settings import (
    ConfigMismatchError,
    SettingsRepository,
)


def test_settings_table_populated_on_store_init():
    """Test that settings table is populated with current config when store is initialized."""

    store = Store(":memory:")
    settings_repo = SettingsRepository(store)

    db_settings = settings_repo.get()
    config_dict = Config.model_dump(mode="json")

    assert db_settings == config_dict

    store.close()


def test_settings_save_and_retrieve():
    """Test saving and retrieving settings after config change."""
    store = Store(":memory:")
    settings_repo = SettingsRepository(store)

    original_chunk_size = Config.CHUNK_SIZE
    Config.CHUNK_SIZE = 2 * original_chunk_size

    settings_repo.save()
    retrieved_settings = settings_repo.get()
    assert retrieved_settings["CHUNK_SIZE"] == 2 * original_chunk_size

    Config.CHUNK_SIZE = original_chunk_size
    store.close()


async def test_config_validation_on_db_load():
    """Test that config validation fails when loading db with mismatched settings."""
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
        db_path = Path(tmp.name)

        # Create store and save settings
        store1 = Store(db_path)
        SettingsRepository(store1)
        store1.close()

        # Change config
        original_chunk_size = Config.CHUNK_SIZE
        Config.CHUNK_SIZE = 999

        try:
            # Loading the database should raise ConfigMismatchError
            with pytest.raises(ConfigMismatchError) as exc_info:
                Store(db_path)

            assert "CHUNK_SIZE" in str(exc_info.value)
            assert "Consider rebuilding" in str(exc_info.value)

            # Rebuild
            async with HaikuRAG(db_path=db_path, skip_validation=True) as client:
                async for _ in client.rebuild_database():
                    pass  # Process all documents

            # Verify we can now load the database without exception (settings were updated)
            store2 = Store(db_path)
            settings_repo2 = SettingsRepository(store2)
            db_settings = settings_repo2.get()
            assert db_settings["CHUNK_SIZE"] == 999
            store2.close()

        finally:
            Config.CHUNK_SIZE = original_chunk_size
