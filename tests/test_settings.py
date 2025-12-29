from haiku.rag.config import Config


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
