from datetime import timedelta
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from haiku.rag.config.models import AppConfig, LanceDBConfig
from haiku.rag.store.engine import ConnectionMode, Store, connect_lancedb


class TestConnectionMode:
    """A location is classified by itself: a path is local, `db://` is LanceDB
    Cloud, any other scheme is object storage."""

    def test_a_path_is_local(self, tmp_path):
        assert ConnectionMode.of(tmp_path / "db.lancedb") == ConnectionMode.LOCAL

    def test_a_schemeless_string_is_local(self):
        assert ConnectionMode.of("/data/db.lancedb") == ConnectionMode.LOCAL

    def test_cloud_when_db_uri(self):
        assert ConnectionMode.of("db://my-database") == ConnectionMode.CLOUD

    @pytest.mark.parametrize(
        "uri",
        [
            "s3://bucket/path",
            "gs://bucket/path",
            "az://container/path",
            "hdfs://namenode/path",
            "custom://something",
        ],
    )
    def test_any_other_scheme_is_object_storage(self, uri):
        assert ConnectionMode.of(uri) == ConnectionMode.OBJECT_STORAGE


class TestConnectLancedb:
    @pytest.mark.asyncio
    async def test_local_passes_absolute_db_path(self, temp_db_path):
        with patch(
            "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
        ) as mock_connect:
            await connect_lancedb(temp_db_path, AppConfig())
            mock_connect.assert_awaited_once()
            assert mock_connect.call_args.args == (temp_db_path.absolute(),)

    @pytest.mark.asyncio
    async def test_local_resolves_relative_db_path(self, tmp_path, monkeypatch):
        from pathlib import Path

        monkeypatch.chdir(tmp_path)
        relative = Path("db/rag.lancedb")
        with patch(
            "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
        ) as mock_connect:
            await connect_lancedb(relative, AppConfig())
            mock_connect.assert_awaited_once()
            assert mock_connect.call_args.args == (relative.absolute(),)

    @pytest.mark.asyncio
    async def test_the_configured_uri_is_not_consulted(self, temp_db_path):
        """Storage connects to the location it is handed; placement is the
        caller's, and the configuration's own `uri` never redirects it."""
        config = AppConfig(
            lancedb=LanceDBConfig(databases={"elsewhere": "s3://elsewhere/db.lancedb"})
        )
        with patch(
            "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
        ) as mock_connect:
            await connect_lancedb(temp_db_path, config)
            assert mock_connect.call_args.args == (temp_db_path.absolute(),)
            assert "uri" not in mock_connect.call_args.kwargs

    @pytest.mark.asyncio
    async def test_cloud_passes_uri_api_key_region(self):
        config = AppConfig(
            lancedb=LanceDBConfig(api_key="test-key", region="us-west-2")
        )
        with patch(
            "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
        ) as mock_connect:
            await connect_lancedb("db://my-database", config)
            mock_connect.assert_awaited_once()
            kwargs = mock_connect.call_args.kwargs
            assert kwargs["uri"] == "db://my-database"
            assert kwargs["api_key"] == "test-key"
            assert kwargs["region"] == "us-west-2"

    @pytest.mark.asyncio
    async def test_object_storage_passes_uri_and_storage_options(self):
        config = AppConfig(
            lancedb=LanceDBConfig(
                storage_options={
                    "endpoint": "http://minio:9000",
                    "region": "us-east-1",
                },
            )
        )
        with patch(
            "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
        ) as mock_connect:
            await connect_lancedb("s3://bucket/path", config)
            mock_connect.assert_awaited_once()
            kwargs = mock_connect.call_args.kwargs
            assert kwargs["uri"] == "s3://bucket/path"
            assert kwargs["storage_options"] == {
                "endpoint": "http://minio:9000",
                "region": "us-east-1",
            }

    @pytest.mark.asyncio
    async def test_object_storage_without_storage_options(self):
        with patch(
            "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
        ) as mock_connect:
            await connect_lancedb("s3://bucket/path", AppConfig())
            mock_connect.assert_awaited_once()
            kwargs = mock_connect.call_args.kwargs
            assert kwargs["uri"] == "s3://bucket/path"
            assert "storage_options" not in kwargs


def _remote_store(location: str, config: AppConfig | None = None) -> Store:
    """A store over a remote location, opened against a mocked connection."""
    return Store(
        location,
        config=config,
        create=True,
        skip_validation=True,
        skip_migration_check=True,
    )


class TestStoreConnectionMode:
    @pytest.mark.asyncio
    async def test_store_connection_mode_local(self, temp_db_path):
        async with Store(temp_db_path, create=True) as store:
            assert store._connection_mode == ConnectionMode.LOCAL
            assert store.location == temp_db_path
            assert store.db_path == temp_db_path

    @pytest.mark.asyncio
    async def test_a_local_store_ignores_the_configured_uri(self, temp_db_path):
        config = AppConfig(
            lancedb=LanceDBConfig(databases={"elsewhere": "s3://elsewhere/db.lancedb"})
        )
        async with Store(temp_db_path, config=config, create=True) as store:
            assert store._connection_mode == ConnectionMode.LOCAL
            assert store.db_path == temp_db_path

    @pytest.mark.asyncio
    async def test_store_connection_mode_cloud(self):
        config = AppConfig(lancedb=LanceDBConfig(api_key="key", region="us-east-1"))
        with (
            patch(
                "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
            ),
            patch.object(Store, "_init_tables", new_callable=AsyncMock),
        ):
            async with _remote_store("db://test-database", config) as store:
                assert store._connection_mode == ConnectionMode.CLOUD
                assert store.location == "db://test-database"
                assert store.db_path is None

    @pytest.mark.asyncio
    async def test_store_connection_mode_object_storage(self):
        with (
            patch(
                "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
            ),
            patch.object(Store, "_init_tables", new_callable=AsyncMock),
        ):
            async with _remote_store("s3://bucket/path") as store:
                assert store._connection_mode == ConnectionMode.OBJECT_STORAGE
                assert store.db_path is None


def _remote_store_with_mock_tables(location: str) -> Store:
    """A remote store whose tables are mocks: the mode decision is under test,
    not the tables."""
    store = _remote_store(location)
    store.chunks_table = AsyncMock()
    return store


class TestVacuumByConnectionMode:
    @pytest.mark.asyncio
    async def test_cloud_skips_vacuum(self):
        with (
            patch(
                "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
            ),
            patch.object(Store, "_init_tables", new_callable=AsyncMock),
        ):
            async with _remote_store_with_mock_tables("db://test-database") as store:
                await store.vacuum()
                store.chunks_table.optimize.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_object_storage_runs_vacuum(self):
        with (
            patch(
                "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
            ),
            patch.object(Store, "_init_tables", new_callable=AsyncMock),
        ):
            async with _remote_store_with_mock_tables("s3://bucket/path") as store:
                store.chunks_table.tags.list = AsyncMock(return_value={})
                with patch.object(
                    store, "_tables", return_value={"chunks": store.chunks_table}
                ):
                    await store.vacuum()
                store.chunks_table.optimize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_local_runs_vacuum(self, temp_db_path):
        async with Store(temp_db_path, create=True) as store:
            with patch.object(
                store.chunks_table, "optimize", new_callable=AsyncMock
            ) as mock_optimize:
                await store.vacuum()
                mock_optimize.assert_called()


class TestVectorIndexByConnectionMode:
    @pytest.mark.asyncio
    async def test_cloud_skips_index_creation(self):
        with (
            patch(
                "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
            ),
            patch.object(Store, "_init_tables", new_callable=AsyncMock),
        ):
            async with _remote_store_with_mock_tables("db://test-database") as store:
                await store._ensure_vector_index()
                store.chunks_table.count_rows.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_object_storage_runs_index_creation(self):
        with (
            patch(
                "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
            ),
            patch.object(Store, "_init_tables", new_callable=AsyncMock),
        ):
            async with _remote_store_with_mock_tables("s3://bucket/path") as store:
                store.chunks_table.count_rows = AsyncMock(return_value=0)
                await store._ensure_vector_index()
                store.chunks_table.count_rows.assert_awaited_once()


class TestLocationIsFixed:
    @pytest.mark.asyncio
    async def test_a_store_keeps_the_location_it_opened(self, temp_db_path):
        """`db_path` and the connection mode derive from the location once; a
        store cannot be pointed elsewhere after it is built."""
        async with Store(temp_db_path, create=True) as store:
            with pytest.raises(AttributeError):
                store.location = "s3://bucket/path"  # type: ignore[misc]
            assert store.location == temp_db_path
            assert store._connection_mode == ConnectionMode.LOCAL


class TestInitFailureCleanup:
    @pytest.mark.asyncio
    async def test_store_aenter_closes_connection_on_init_failure(
        self, temp_db_path, monkeypatch
    ):
        """If _initialize raises after connect, __aenter__ must close the
        AsyncConnection so it doesn't leak (no __aexit__ runs in that case)."""
        mock_conn = AsyncMock()
        mock_conn.close = lambda: mock_conn.close_calls.append(True)  # type: ignore[attr-defined]
        mock_conn.close_calls = []  # type: ignore[attr-defined]

        async def fake_connect(*args, **kwargs):
            return mock_conn

        async def failing_init_tables(self, *args):
            raise RuntimeError("simulated table init failure")

        monkeypatch.setattr("haiku.rag.store.engine.connect_lancedb", fake_connect)
        monkeypatch.setattr(Store, "_init_tables", failing_init_tables)

        with pytest.raises(RuntimeError, match="simulated table init failure"):
            async with Store(temp_db_path, create=True) as store:
                assert store is not None

        assert mock_conn.close_calls == [True], (
            "AsyncConnection.close() was not called on init failure"
        )

    @pytest.mark.asyncio
    async def test_client_aenter_closes_store_on_init_failure(
        self, temp_db_path, monkeypatch
    ):
        """HaikuRAG.__aenter__ must close the store if _initialize fails."""
        from haiku.rag.client import HaikuRAG

        close_calls: list[bool] = []

        original_close = Store.close

        def tracking_close(self):
            close_calls.append(True)
            original_close(self)

        async def failing_init(self):
            # Set db so close() has something to close
            self.db = AsyncMock()
            self.db.close = lambda: None
            raise RuntimeError("simulated initialize failure")

        monkeypatch.setattr(Store, "_initialize", failing_init)
        monkeypatch.setattr(Store, "close", tracking_close)

        with pytest.raises(RuntimeError, match="simulated initialize failure"):
            async with HaikuRAG(temp_db_path, create=True):
                pass

        assert close_calls, "Store.close() was not called when _initialize raised"


class TestVectorIndexCreation:
    """_ensure_vector_index needs 256 rows of training data before it builds."""

    @staticmethod
    async def _seed_chunks(store, count: int) -> None:
        import random

        records = [
            store.ChunkRecord(
                document_id="doc-1",
                content=f"row {i}",
                content_fts=f"row {i}",
                metadata="{}",
                order=i,
                vector=[random.random() for _ in range(store.embedder.vector_dim)],
            )
            for i in range(count)
        ]
        await store.chunks_table.add(records)

    @pytest.mark.asyncio
    async def test_builds_index_once_enough_rows_exist(self, temp_db_path):
        async with Store(temp_db_path, create=True) as store:
            await self._seed_chunks(store, 256)

            await store._ensure_vector_index()

            indexes = await store.chunks_table.list_indices()
            assert any("vector" in idx.columns for idx in indexes)

    @pytest.mark.asyncio
    async def test_index_failure_is_warned_not_raised(self, temp_db_path):
        import logging

        from haiku.rag.store import engine as engine_module
        from tests.conftest import capture_logs

        async with Store(temp_db_path, create=True) as store:
            await self._seed_chunks(store, 256)

            async def boom(*_args, **_kwargs):
                raise RuntimeError("index build failed")

            with patch.object(store.chunks_table, "create_index", boom):
                with capture_logs(engine_module.logger, logging.WARNING) as records:
                    await store._ensure_vector_index()

            assert any("index build failed" in r.getMessage() for r in records)
            indexes = await store.chunks_table.list_indices()
            assert not any("vector" in idx.columns for idx in indexes)


class TestStoreMiscellany:
    @pytest.mark.asyncio
    async def test_create_makes_missing_parent_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "db.lancedb"

        async with Store(nested, create=True) as store:
            assert store._is_new_db is True

        assert nested.exists()

    @pytest.mark.asyncio
    async def test_stored_vector_dim_is_none_for_corrupt_settings(self, temp_db_path):
        async with Store(temp_db_path, create=True) as store:
            await store.settings_table.update(
                {"settings": "not json at all"}, where="id = 'settings'"
            )

            assert await store._read_stored_settings() == {}

    @pytest.mark.asyncio
    async def test_vacuum_skips_when_already_running(self, temp_db_path):
        import asyncio

        async with Store(temp_db_path, create=True) as store:
            async with store._vacuum_lock:
                # Bounded: a regression here blocks on the held lock, and the
                # timeout turns that deadlock into a clean failure.
                await asyncio.wait_for(store.vacuum(), timeout=5)

    @pytest.mark.asyncio
    async def test_history_rejects_unknown_table(self, temp_db_path):
        async with Store(temp_db_path, create=True) as store:
            with pytest.raises(ValueError, match="Unknown table"):
                await store.list_table_versions("not_a_table")


class TestSessionAndConsistency:
    @pytest.mark.asyncio
    async def test_session_is_shared_across_connections(self):
        config = AppConfig()
        with patch(
            "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
        ) as mock_connect:
            await connect_lancedb("s3://bucket/path", config)
            await connect_lancedb("s3://bucket/path", config)

        sessions = [c.kwargs["session"] for c in mock_connect.call_args_list]
        assert sessions[0] is sessions[1]

    @pytest.mark.asyncio
    async def test_cache_sizes_select_distinct_sessions(self):
        small = AppConfig(lancedb=LanceDBConfig(index_cache_size_bytes=1 << 20))
        large = AppConfig(lancedb=LanceDBConfig(index_cache_size_bytes=1 << 30))
        with patch(
            "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
        ) as mock_connect:
            await connect_lancedb("s3://bucket/path", small)
            await connect_lancedb("s3://bucket/path", large)

        sessions = [c.kwargs["session"] for c in mock_connect.call_args_list]
        assert sessions[0] is not sessions[1]

    @pytest.mark.asyncio
    async def test_both_cache_sizes_are_applied(self):
        config = AppConfig(
            lancedb=LanceDBConfig(
                index_cache_size_bytes=2 << 20,
                metadata_cache_size_bytes=4 << 20,
            )
        )
        with (
            patch(
                "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
            ),
            patch("haiku.rag.store.engine.lancedb.Session") as mock_session,
        ):
            await connect_lancedb("s3://bucket/path", config)

        mock_session.assert_called_once_with(
            index_cache_size_bytes=2 << 20, metadata_cache_size_bytes=4 << 20
        )

    @pytest.mark.asyncio
    async def test_read_consistency_interval_is_forwarded(self):
        config = AppConfig(lancedb=LanceDBConfig(read_consistency_interval_seconds=5))
        with patch(
            "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
        ) as mock_connect:
            await connect_lancedb("s3://bucket/path", config)

        assert mock_connect.call_args.kwargs["read_consistency_interval"] == timedelta(
            seconds=5
        )

    @pytest.mark.asyncio
    async def test_read_consistency_interval_omitted_when_disabled(self):
        config = AppConfig(
            lancedb=LanceDBConfig(read_consistency_interval_seconds=None)
        )
        with patch(
            "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
        ) as mock_connect:
            await connect_lancedb("s3://bucket/path", config)

        assert mock_connect.call_args.kwargs["read_consistency_interval"] is None

    @pytest.mark.asyncio
    async def test_local_connection_also_gets_session_and_consistency(self, tmp_path):
        config = AppConfig()
        with patch(
            "haiku.rag.store.engine.lancedb.connect_async", new_callable=AsyncMock
        ) as mock_connect:
            await connect_lancedb(tmp_path / "db.lancedb", config)

        assert mock_connect.call_args.kwargs["session"] is not None
        assert mock_connect.call_args.kwargs["read_consistency_interval"] == timedelta(
            seconds=30
        )


class TestLanceDBConfigValidation:
    def test_negative_values_are_rejected(self):
        """Negatives overflow or panic inside Lance, so reject them here."""
        with pytest.raises(ValidationError):
            LanceDBConfig(read_consistency_interval_seconds=-1)
        with pytest.raises(ValidationError):
            LanceDBConfig(index_cache_size_bytes=-1)
        with pytest.raises(ValidationError):
            LanceDBConfig(metadata_cache_size_bytes=-1)

    def test_zero_is_allowed(self):
        config = LanceDBConfig(
            read_consistency_interval_seconds=0, index_cache_size_bytes=0
        )
        assert config.read_consistency_interval_seconds == 0
