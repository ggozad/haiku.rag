from unittest.mock import AsyncMock, patch

import pytest

from haiku.rag.config import Config
from haiku.rag.config.models import AppConfig, LanceDBConfig
from haiku.rag.store.engine import ConnectionMode, Store, connect_lancedb


class TestConnectionMode:
    def test_local_when_uri_empty(self):
        config = AppConfig(lancedb=LanceDBConfig(uri=""))
        assert ConnectionMode.from_config(config) == ConnectionMode.LOCAL

    def test_cloud_when_db_uri(self):
        config = AppConfig(
            lancedb=LanceDBConfig(
                uri="db://my-database", api_key="key", region="us-east-1"
            )
        )
        assert ConnectionMode.from_config(config) == ConnectionMode.CLOUD

    def test_object_storage_s3(self):
        config = AppConfig(lancedb=LanceDBConfig(uri="s3://bucket/path"))
        assert ConnectionMode.from_config(config) == ConnectionMode.OBJECT_STORAGE

    def test_object_storage_gs(self):
        config = AppConfig(lancedb=LanceDBConfig(uri="gs://bucket/path"))
        assert ConnectionMode.from_config(config) == ConnectionMode.OBJECT_STORAGE

    def test_object_storage_az(self):
        config = AppConfig(lancedb=LanceDBConfig(uri="az://container/path"))
        assert ConnectionMode.from_config(config) == ConnectionMode.OBJECT_STORAGE

    def test_object_storage_hdfs(self):
        config = AppConfig(lancedb=LanceDBConfig(uri="hdfs://namenode/path"))
        assert ConnectionMode.from_config(config) == ConnectionMode.OBJECT_STORAGE

    def test_unknown_uri_treated_as_object_storage(self):
        config = AppConfig(lancedb=LanceDBConfig(uri="custom://something"))
        assert ConnectionMode.from_config(config) == ConnectionMode.OBJECT_STORAGE


class TestConnectLancedb:
    @pytest.mark.asyncio
    async def test_local_passes_db_path(self, temp_db_path):
        config = AppConfig(lancedb=LanceDBConfig(uri=""))
        with patch(
            "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
        ) as mock_connect:
            await connect_lancedb(config, db_path=temp_db_path)
            mock_connect.assert_called_once_with(temp_db_path)

    @pytest.mark.asyncio
    async def test_cloud_passes_uri_api_key_region(self):
        config = AppConfig(
            lancedb=LanceDBConfig(
                uri="db://my-database", api_key="test-key", region="us-west-2"
            )
        )
        with patch(
            "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
        ) as mock_connect:
            await connect_lancedb(config)
            mock_connect.assert_called_once_with(
                uri="db://my-database", api_key="test-key", region="us-west-2"
            )

    @pytest.mark.asyncio
    async def test_object_storage_passes_uri_and_storage_options(self):
        config = AppConfig(
            lancedb=LanceDBConfig(
                uri="s3://bucket/path",
                storage_options={
                    "endpoint": "http://minio:9000",
                    "region": "us-east-1",
                },
            )
        )
        with patch(
            "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
        ) as mock_connect:
            await connect_lancedb(config)
            mock_connect.assert_called_once_with(
                uri="s3://bucket/path",
                storage_options={
                    "endpoint": "http://minio:9000",
                    "region": "us-east-1",
                },
            )

    @pytest.mark.asyncio
    async def test_object_storage_without_storage_options(self):
        config = AppConfig(lancedb=LanceDBConfig(uri="s3://bucket/path"))
        with patch(
            "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
        ) as mock_connect:
            await connect_lancedb(config)
            mock_connect.assert_called_once_with(uri="s3://bucket/path")

    @pytest.mark.asyncio
    async def test_local_without_db_path_raises(self):
        config = AppConfig(lancedb=LanceDBConfig(uri=""))
        with pytest.raises(
            ValueError, match="No lancedb.uri configured and no db_path provided"
        ):
            await connect_lancedb(config)


class TestStoreConnectionMode:
    @pytest.mark.asyncio
    async def test_store_connection_mode_local(self, temp_db_path):
        async with Store(temp_db_path, create=True) as store:
            assert store._connection_mode == ConnectionMode.LOCAL

    @pytest.mark.asyncio
    async def test_store_connection_mode_cloud(self, temp_db_path):
        async with Store(temp_db_path, create=True) as store:
            with (
                patch.object(Config.lancedb, "uri", "db://test-database"),
                patch.object(Config.lancedb, "api_key", "test-api-key"),
                patch.object(Config.lancedb, "region", "us-east-1"),
            ):
                assert store._connection_mode == ConnectionMode.CLOUD

    @pytest.mark.asyncio
    async def test_store_connection_mode_object_storage(self, temp_db_path):
        async with Store(temp_db_path, create=True) as store:
            with patch.object(Config.lancedb, "uri", "s3://bucket/path"):
                assert store._connection_mode == ConnectionMode.OBJECT_STORAGE


class TestVacuumByConnectionMode:
    @pytest.mark.asyncio
    async def test_cloud_skips_vacuum(self, temp_db_path):
        async with Store(temp_db_path, create=True) as store:
            with (
                patch.object(Config.lancedb, "uri", "db://test-database"),
                patch.object(Config.lancedb, "api_key", "test-api-key"),
                patch.object(Config.lancedb, "region", "us-east-1"),
            ):
                with patch.object(
                    store.chunks_table, "optimize", new_callable=AsyncMock
                ) as mock_optimize:
                    await store.vacuum()
                    mock_optimize.assert_not_called()

    @pytest.mark.asyncio
    async def test_object_storage_runs_vacuum(self, temp_db_path):
        async with Store(temp_db_path, create=True) as store:
            with patch.object(Config.lancedb, "uri", "s3://bucket/path"):
                with patch.object(
                    store.chunks_table, "optimize", new_callable=AsyncMock
                ) as mock_optimize:
                    await store.vacuum()
                    mock_optimize.assert_called()

    @pytest.mark.asyncio
    async def test_local_runs_vacuum(self, temp_db_path):
        async with Store(temp_db_path, create=True) as store:
            with patch.object(Config.lancedb, "uri", ""):
                with patch.object(
                    store.chunks_table, "optimize", new_callable=AsyncMock
                ) as mock_optimize:
                    await store.vacuum()
                    mock_optimize.assert_called()


class TestVectorIndexByConnectionMode:
    @pytest.mark.asyncio
    async def test_cloud_skips_index_creation(self, temp_db_path):
        async with Store(temp_db_path, create=True) as store:
            with (
                patch.object(Config.lancedb, "uri", "db://test-database"),
                patch.object(Config.lancedb, "api_key", "test-api-key"),
                patch.object(Config.lancedb, "region", "us-east-1"),
            ):
                with patch.object(
                    store.chunks_table, "count_rows", new_callable=AsyncMock
                ) as mock_count:
                    await store._ensure_vector_index()
                    mock_count.assert_not_called()

    @pytest.mark.asyncio
    async def test_object_storage_runs_index_creation(self, temp_db_path):
        async with Store(temp_db_path, create=True) as store:
            with patch.object(Config.lancedb, "uri", "s3://bucket/path"):
                with patch.object(
                    store.chunks_table,
                    "count_rows",
                    new_callable=AsyncMock,
                    return_value=0,
                ) as mock_count:
                    await store._ensure_vector_index()
                    mock_count.assert_called()


class TestStoreSkipsPathValidationForRemote:
    @pytest.mark.asyncio
    async def test_skips_path_check_for_cloud(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist" / "db.lancedb"
        config = AppConfig(
            lancedb=LanceDBConfig(
                uri="db://test-database", api_key="key", region="us-east-1"
            )
        )
        with patch(
            "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
        ):
            with patch.object(Store, "_init_tables", new_callable=AsyncMock):
                async with Store(
                    nonexistent,
                    config=config,
                    create=True,
                    skip_validation=True,
                    skip_migration_check=True,
                ) as store:
                    assert store is not None

    @pytest.mark.asyncio
    async def test_skips_path_check_for_object_storage(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist" / "db.lancedb"
        config = AppConfig(
            lancedb=LanceDBConfig(
                uri="s3://bucket/path",
                storage_options={"endpoint": "http://localhost:9000"},
            )
        )
        with patch(
            "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
        ):
            with patch.object(Store, "_init_tables", new_callable=AsyncMock):
                async with Store(
                    nonexistent,
                    config=config,
                    create=True,
                    skip_validation=True,
                    skip_migration_check=True,
                ) as store:
                    assert store is not None
