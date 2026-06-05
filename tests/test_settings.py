import logging

import pytest

from haiku.rag.config import AppConfig, Config
from haiku.rag.store.repositories.settings import ConfigMismatchError


@pytest.mark.asyncio
async def test_settings_table_populated_on_store_init(temp_db_path):
    """Test that settings table is populated with current config when store is initialized."""
    from haiku.rag.store.engine import Store
    from haiku.rag.store.repositories.settings import SettingsRepository

    async with Store(temp_db_path, create=True) as store:
        settings_repo = SettingsRepository(store)

        db_settings = await settings_repo.get_current_settings()
        config_dict = Config.model_dump(mode="json")

        # Remove version from db_settings since it's added automatically
        db_settings_without_version = {
            k: v for k, v in db_settings.items() if k != "version"
        }
        assert db_settings_without_version == config_dict


@pytest.mark.asyncio
async def test_settings_save_and_retrieve(temp_db_path):
    """Test saving and retrieving settings after config change."""
    from haiku.rag.store.engine import Store
    from haiku.rag.store.repositories.settings import SettingsRepository

    async with Store(temp_db_path, create=True) as store:
        settings_repo = SettingsRepository(store)

        original_chunk_size = Config.processing.chunk_size
        Config.processing.chunk_size = 2 * original_chunk_size

        await settings_repo.save_current_settings()
        retrieved_settings = await settings_repo.get_current_settings()
        assert retrieved_settings["processing"]["chunk_size"] == 2 * original_chunk_size

        Config.processing.chunk_size = original_chunk_size


class TestValidateConfigCompatibility:
    """Tests for validate_config_compatibility method."""

    @pytest.mark.asyncio
    async def test_empty_settings_does_not_write(self, temp_db_path):
        """Validation never writes on open, even when the settings row is missing."""
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.settings import SettingsRepository

        async with Store(temp_db_path, create=True, skip_validation=True) as store:
            settings_repo = SettingsRepository(store)

            # Clear settings to simulate empty state
            await store.settings_table.delete("id = 'settings'")
            assert await settings_repo.get_current_settings() == {}

            # Validation must not write — nothing to validate against
            await settings_repo.validate_config_compatibility()

            assert await settings_repo.get_current_settings() == {}

    @pytest.mark.asyncio
    async def test_compatible_config_no_error(self, temp_db_path):
        """Compatible config does not raise error."""
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.settings import SettingsRepository

        async with Store(temp_db_path, create=True) as store:
            settings_repo = SettingsRepository(store)

            # Should not raise - same config
            await settings_repo.validate_config_compatibility()

    @pytest.mark.asyncio
    async def test_provider_drift_read_only_warns_without_writing(
        self, temp_db_path, caplog, monkeypatch
    ):
        """Provider drift (vector_dim matches) on a read-only store warns and continues.

        Same model served by a different stack (Ollama vs vLLM via openai-compat)
        legitimately differs in `provider`. A read-only open surfaces the change
        but must never modify the stored settings.
        """
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.settings import SettingsRepository

        # haiku.rag.logging.get_logger() sets propagate=False on the
        # `haiku.rag` logger. caplog's handler attaches to root by default,
        # so without restoring propagation the records never reach it.
        monkeypatch.setattr(logging.getLogger("haiku.rag"), "propagate", True)

        async with Store(temp_db_path, create=True):
            pass

        new_config = AppConfig()
        new_config.embeddings.model.provider = "openai"

        async with Store(
            temp_db_path, config=new_config, skip_validation=True, read_only=True
        ) as store2:
            settings_repo = SettingsRepository(store2)

            with caplog.at_level(logging.WARNING):
                await settings_repo.validate_config_compatibility()

            # Warning surfaced the change
            assert any(
                "provider" in r.getMessage()
                and "ollama" in r.getMessage()
                and "openai" in r.getMessage()
                for r in caplog.records
            )

            # Stored settings are untouched
            saved = await settings_repo.get_current_settings()
            assert saved["embeddings"]["model"]["provider"] == "ollama"

    @pytest.mark.asyncio
    async def test_provider_drift_writable_raises_without_writing(
        self, temp_db_path, caplog, monkeypatch
    ):
        """Provider drift on a writable store warns and raises, without writing."""
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.settings import (
            ConfigMismatchError,
            SettingsRepository,
        )

        monkeypatch.setattr(logging.getLogger("haiku.rag"), "propagate", True)

        async with Store(temp_db_path, create=True):
            pass

        new_config = AppConfig()
        new_config.embeddings.model.provider = "openai"

        async with Store(
            temp_db_path, config=new_config, skip_validation=True
        ) as store2:
            settings_repo = SettingsRepository(store2)

            with caplog.at_level(logging.WARNING):
                with pytest.raises(ConfigMismatchError):
                    await settings_repo.validate_config_compatibility()

            assert any(
                "provider" in r.getMessage()
                and "ollama" in r.getMessage()
                and "openai" in r.getMessage()
                for r in caplog.records
            )

            # Stored settings are untouched despite the writable open
            saved = await settings_repo.get_current_settings()
            assert saved["embeddings"]["model"]["provider"] == "ollama"

    @pytest.mark.asyncio
    async def test_model_drift_read_only_warns_without_writing(
        self, temp_db_path, caplog, monkeypatch
    ):
        """Model name drift (vector_dim matches) on a read-only store warns, no write."""
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.settings import SettingsRepository

        monkeypatch.setattr(logging.getLogger("haiku.rag"), "propagate", True)

        async with Store(temp_db_path, create=True):
            pass

        new_config = AppConfig()
        new_config.embeddings.model.name = "different-model"

        async with Store(
            temp_db_path, config=new_config, skip_validation=True, read_only=True
        ) as store2:
            settings_repo = SettingsRepository(store2)

            with caplog.at_level(logging.WARNING):
                await settings_repo.validate_config_compatibility()

            assert any(
                "model" in r.getMessage() and "different-model" in r.getMessage()
                for r in caplog.records
            )

            saved = await settings_repo.get_current_settings()
            assert saved["embeddings"]["model"]["name"] != "different-model"

    @pytest.mark.asyncio
    async def test_vector_dim_mismatch_raises_error(self, temp_db_path):
        """Different vector dimension raises ConfigMismatchError."""
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.settings import SettingsRepository

        # Create store with default config
        async with Store(temp_db_path, create=True):
            pass

        # Create new config with different vector dimension
        new_config = AppConfig()
        new_config.embeddings.model.vector_dim = 9999

        async with Store(
            temp_db_path, config=new_config, skip_validation=True
        ) as store2:
            settings_repo = SettingsRepository(store2)

            with pytest.raises(ConfigMismatchError) as exc_info:
                await settings_repo.validate_config_compatibility()

            assert "vector dimension" in str(exc_info.value)
            assert "9999" in str(exc_info.value)
