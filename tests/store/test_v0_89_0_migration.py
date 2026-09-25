import json

import pytest

from haiku.rag.config import AppConfig, get_config
from haiku.rag.store.engine import Store
from haiku.rag.store.exceptions import MigrationRequiredError
from haiku.rag.store.repositories.settings import SettingsRepository


def _full_settings(version: str) -> dict:
    """A settings row as databases before 0.89.0 wrote it, secrets included."""
    config = get_config().model_dump(mode="json")
    config["lancedb"]["api_key"] = "LANCE-KEY"
    config["lancedb"]["storage_options"] = {"aws_secret_access_key": "S3-KEY"}
    config["embeddings"]["model"]["api_key"] = "EMBED-KEY"
    return {**config, "version": version}


async def _write_row(store: Store, settings: dict) -> None:
    await store.settings_table.update(
        {"settings": json.dumps(settings)}, where="id = 'settings'"
    )


async def test_purges_everything_but_version_and_embedder(temp_db_path):
    from haiku.rag.store.upgrades.v0_89_0 import _apply_record_embedder_only

    model = AppConfig().embeddings.model
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _write_row(store, _full_settings("0.88.2"))

        await _apply_record_embedder_only(store)

        assert await SettingsRepository(store).get_current_settings() == {
            "version": "0.88.2",
            "embeddings": {
                "model": {
                    "provider": model.provider,
                    "name": model.name,
                    "vector_dim": model.vector_dim,
                }
            },
        }


async def test_a_database_written_before_it_opens_only_after_migrate(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        await _write_row(store, _full_settings("0.88.2"))

    with pytest.raises(MigrationRequiredError):
        async with Store(temp_db_path, read_only=True):
            pass

    async with Store(temp_db_path, skip_migration_check=True) as store:
        await store.migrate()

    async with Store(temp_db_path, read_only=True) as store:
        stored = json.dumps(await SettingsRepository(store).get_current_settings())
    for secret in ("LANCE-KEY", "S3-KEY", "EMBED-KEY"):
        assert secret not in stored


def _files_holding(root, marker: bytes) -> int:
    return sum(1 for f in root.rglob("*") if f.is_file() and marker in f.read_bytes())


@pytest.mark.parametrize(("retention", "left"), [(0, 0), (None, 1)])
async def test_vacuum_at_zero_retention_removes_the_old_settings_from_disk(
    temp_db_path, retention, left
):
    from haiku.rag.client import HaikuRAG

    async with Store(temp_db_path, create=True) as store:
        await _write_row(store, _full_settings("0.88.2"))
    async with Store(temp_db_path, skip_migration_check=True) as store:
        await store.migrate()

    async with HaikuRAG(temp_db_path) as client:
        await client.vacuum(retention_seconds=retention)

    assert _files_holding(temp_db_path, b"LANCE-KEY") == left
