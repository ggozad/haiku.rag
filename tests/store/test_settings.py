import json
import logging

import pytest

from haiku.rag.config import AppConfig, get_config
from haiku.rag.store.exceptions import ConfigMismatchError

SECRETS = {
    "lancedb": {
        "api_key": "LANCE-KEY",
        "storage_options": {"aws_secret_access_key": "S3-KEY"},
    },
    "providers": {"docling_serve": {"api_key": "DOCLING-KEY"}},
    "embeddings": {
        "model": {"api_key": "EMBED-KEY", "base_url": "https://u:PW@embed.example/v1"}
    },
}


def _config_with_secrets() -> AppConfig:
    """The global config's embedder, with a secret in every field that can hold one."""
    base = get_config().model_dump(mode="json")
    base["lancedb"].update(SECRETS["lancedb"])
    base["providers"]["docling_serve"].update(SECRETS["providers"]["docling_serve"])
    base["embeddings"]["model"].update(SECRETS["embeddings"]["model"])
    return AppConfig.model_validate(base)


def _recorded(model) -> dict:
    return {
        "provider": model.provider,
        "name": model.name,
        "vector_dim": model.vector_dim,
    }


async def test_a_new_database_records_its_version_and_embedder_only(temp_db_path):
    from haiku.rag.store.engine import Store
    from haiku.rag.store.repositories.settings import SettingsRepository

    config = _config_with_secrets()
    async with Store(temp_db_path, config=config, create=True) as store:
        stored = await SettingsRepository(store).get_current_settings()

    assert set(stored) == {"version", "embeddings"}
    assert stored["embeddings"] == {"model": _recorded(config.embeddings.model)}
    for secret in ("LANCE-KEY", "S3-KEY", "DOCLING-KEY", "EMBED-KEY", "PW"):
        assert secret not in json.dumps(stored)


async def test_saving_settings_records_the_embedder_only(temp_db_path):
    from haiku.rag.store.engine import Store
    from haiku.rag.store.repositories.settings import SettingsRepository

    config = _config_with_secrets()
    async with Store(temp_db_path, config=config, create=True) as store:
        settings_repo = SettingsRepository(store)
        await store.settings_table.update(
            {
                "settings": json.dumps(
                    {**config.model_dump(mode="json"), "version": "1.0.0"}
                )
            },
            where="id = 'settings'",
        )

        await settings_repo.save_current_settings()

        assert await settings_repo.get_current_settings() == {
            "version": "1.0.0",
            "embeddings": {"model": _recorded(config.embeddings.model)},
        }


async def test_set_haiku_version_recreates_row_from_store_config(temp_db_path):
    """Recreating a missing settings row records the store's own embedder, not the
    process-global one's."""
    from haiku.rag.store.engine import Store
    from haiku.rag.store.repositories.settings import SettingsRepository

    config = _config_with_secrets()
    config.embeddings.model.name = "store-own-embedder"

    async with Store(temp_db_path, config=config, create=True) as store:
        settings_repo = SettingsRepository(store)

        await store.settings_table.delete("id = 'settings'")
        assert await settings_repo.get_current_settings() == {}
        assert await store.get_haiku_version() == "0.0.0"

        await store.set_haiku_version("1.2.3")

        assert await settings_repo.get_current_settings() == {
            "version": "1.2.3",
            "embeddings": {"model": _recorded(config.embeddings.model)},
        }


async def test_set_haiku_version_drops_everything_but_version_and_embedder(
    temp_db_path,
):
    from haiku.rag.store.engine import Store
    from haiku.rag.store.repositories.settings import SettingsRepository

    config = _config_with_secrets()
    async with Store(temp_db_path, config=config, create=True) as store:
        await store.settings_table.update(
            {
                "settings": json.dumps(
                    {**config.model_dump(mode="json"), "version": "1.0.0"}
                )
            },
            where="id = 'settings'",
        )

        await store.set_haiku_version("1.2.3")

        assert await SettingsRepository(store).get_current_settings() == {
            "version": "1.2.3",
            "embeddings": {"model": _recorded(config.embeddings.model)},
        }


class TestValidateConfigCompatibility:
    """Tests for validate_config_compatibility method."""

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

    async def test_compatible_config_no_error(self, temp_db_path):
        """Compatible config does not raise error."""
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.settings import SettingsRepository

        async with Store(temp_db_path, create=True) as store:
            settings_repo = SettingsRepository(store)

            # Should not raise - same config
            await settings_repo.validate_config_compatibility()

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

    async def test_vector_dim_mismatch_raises_error_read_only(self, temp_db_path):
        """vector_dim mismatch raises even read-only — search cannot work."""
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.settings import SettingsRepository

        async with Store(temp_db_path, create=True):
            pass

        new_config = AppConfig()
        new_config.embeddings.model.vector_dim = 9999

        async with Store(
            temp_db_path, config=new_config, skip_validation=True, read_only=True
        ) as store2:
            settings_repo = SettingsRepository(store2)

            with pytest.raises(ConfigMismatchError) as exc_info:
                await settings_repo.validate_config_compatibility()

            assert "9999" in str(exc_info.value)


async def test_save_current_settings_recreates_a_deleted_row(temp_db_path):
    from haiku.rag.store.engine import Store
    from haiku.rag.store.repositories.settings import SettingsRepository

    async with Store(temp_db_path, create=True, skip_validation=True) as store:
        settings_repo = SettingsRepository(store)

        await store.settings_table.delete("id = 'settings'")
        assert await settings_repo.get_current_settings() == {}

        await settings_repo.save_current_settings()

        recreated = await settings_repo.get_current_settings()
        assert recreated["embeddings"] == {
            "model": _recorded(store._config.embeddings.model)
        }
