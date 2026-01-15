import pytest

from haiku.rag.config import AppConfig, Config
from haiku.rag.store.repositories.settings import ConfigMismatchError


def test_settings_table_populated_on_store_init(temp_db_path):
    """Test that settings table is populated with current config when store is initialized."""
    from haiku.rag.store.engine import Store
    from haiku.rag.store.repositories.settings import SettingsRepository

    store = Store(temp_db_path, create=True)
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

    store = Store(temp_db_path, create=True)
    settings_repo = SettingsRepository(store)

    original_chunk_size = Config.processing.chunk_size
    Config.processing.chunk_size = 2 * original_chunk_size

    settings_repo.save_current_settings()
    retrieved_settings = settings_repo.get_current_settings()
    assert retrieved_settings["processing"]["chunk_size"] == 2 * original_chunk_size

    Config.processing.chunk_size = original_chunk_size
    store.close()


def test_monitor_filter_patterns_config():
    """Test that monitor filter patterns are available in config."""
    assert hasattr(Config.monitor, "ignore_patterns")
    assert hasattr(Config.monitor, "include_patterns")
    assert hasattr(Config.monitor, "directories")
    assert isinstance(Config.monitor.ignore_patterns, list)
    assert isinstance(Config.monitor.include_patterns, list)
    assert isinstance(Config.monitor.directories, list)


class TestValidateConfigCompatibility:
    """Tests for validate_config_compatibility method."""

    def test_empty_settings_saves_config(self, temp_db_path):
        """When settings row is missing, validation saves current config."""
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.settings import SettingsRepository

        store = Store(temp_db_path, create=True, skip_validation=True)
        settings_repo = SettingsRepository(store)

        # Clear settings to simulate empty state
        store.settings_table.delete("id = 'settings'")
        assert settings_repo.get_current_settings() == {}

        # Validation should save settings
        settings_repo.validate_config_compatibility()

        # Now settings should exist
        saved = settings_repo.get_current_settings()
        assert saved.get("embeddings", {}).get("model", {}).get("provider") is not None
        store.close()

    def test_compatible_config_no_error(self, temp_db_path):
        """Compatible config does not raise error."""
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.settings import SettingsRepository

        store = Store(temp_db_path, create=True)
        settings_repo = SettingsRepository(store)

        # Should not raise - same config
        settings_repo.validate_config_compatibility()
        store.close()

    def test_provider_mismatch_raises_error(self, temp_db_path):
        """Different embedding provider raises ConfigMismatchError."""
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.settings import SettingsRepository

        # Create store with default config (ollama)
        store = Store(temp_db_path, create=True)
        store.close()

        # Create new config with different provider
        new_config = AppConfig()
        new_config.embeddings.model.provider = "openai"

        store2 = Store(temp_db_path, config=new_config, skip_validation=True)
        settings_repo = SettingsRepository(store2)

        with pytest.raises(ConfigMismatchError) as exc_info:
            settings_repo.validate_config_compatibility()

        assert "embedding provider" in str(exc_info.value)
        assert "ollama" in str(exc_info.value)
        assert "openai" in str(exc_info.value)
        store2.close()

    def test_model_mismatch_raises_error(self, temp_db_path):
        """Different embedding model raises ConfigMismatchError."""
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.settings import SettingsRepository

        # Create store with default config
        store = Store(temp_db_path, create=True)
        store.close()

        # Create new config with different model
        new_config = AppConfig()
        new_config.embeddings.model.name = "different-model"

        store2 = Store(temp_db_path, config=new_config, skip_validation=True)
        settings_repo = SettingsRepository(store2)

        with pytest.raises(ConfigMismatchError) as exc_info:
            settings_repo.validate_config_compatibility()

        assert "embedding model" in str(exc_info.value)
        store2.close()

    def test_vector_dim_mismatch_raises_error(self, temp_db_path):
        """Different vector dimension raises ConfigMismatchError."""
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.settings import SettingsRepository

        # Create store with default config
        store = Store(temp_db_path, create=True)
        store.close()

        # Create new config with different vector dimension
        new_config = AppConfig()
        new_config.embeddings.model.vector_dim = 9999

        store2 = Store(temp_db_path, config=new_config, skip_validation=True)
        settings_repo = SettingsRepository(store2)

        with pytest.raises(ConfigMismatchError) as exc_info:
            settings_repo.validate_config_compatibility()

        assert "vector dimension" in str(exc_info.value)
        assert "9999" in str(exc_info.value)
        store2.close()
