import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from haiku.rag.app import HaikuRAGApp
from haiku.rag.config.models import AppConfig, LanceDBConfig
from haiku.rag.store.engine import DocumentItemRecord


@pytest.mark.asyncio
async def test_app_info_outputs(temp_db_path, capsys):
    # Build a minimal LanceDB with settings, documents, and chunks without using Store
    import lancedb
    from lancedb.pydantic import LanceModel, Vector
    from pydantic import Field

    db = await lancedb.connect_async(temp_db_path)

    class SettingsRecord(LanceModel):
        id: str = Field(default="settings")
        settings: str = Field(default="{}")

    class DocumentRecord(LanceModel):
        id: str
        content: str

    class ChunkRecord(LanceModel):
        id: str
        document_id: str
        content: str
        vector: Vector(3)  # type: ignore

    settings_tbl = await db.create_table("settings", schema=SettingsRecord)
    docs_tbl = await db.create_table("documents", schema=DocumentRecord)
    chunks_tbl = await db.create_table("chunks", schema=ChunkRecord)
    await db.create_table("document_items", schema=DocumentItemRecord)

    # Insert one of each - using the new config format
    await settings_tbl.add(
        [
            SettingsRecord(
                id="settings",
                settings=json.dumps(
                    {
                        "version": "1.2.3",
                        "embeddings": {
                            "model": {
                                "provider": "openai",
                                "name": "text-embedding-3-small",
                                "vector_dim": 3,
                            }
                        },
                    }
                ),
            )
        ]
    )
    await docs_tbl.add([DocumentRecord(id="doc-1", content="hello")])
    await chunks_tbl.add(
        [ChunkRecord(id="c1", document_id="doc-1", content="c", vector=[0.1, 0.2, 0.3])]
    )

    app = HaikuRAGApp(db_path=temp_db_path)
    await app.info()

    out = capsys.readouterr().out
    # Validate expected content substrings
    # Note: Rich console may wrap long paths to new lines, so check separately
    assert "path:" in out
    # Rich may wrap long paths across lines — check with newlines stripped
    out_no_wrap = out.replace("\n", "")
    assert str(temp_db_path) in out_no_wrap
    assert "haiku.rag version (db):" in out
    assert "embeddings: openai/text-embedding-3-small (dim: 3)" in out
    assert "documents: 1" in out
    assert "chunks: 1" in out

    # Vector index should not exist (only 1 chunk, need 256)
    assert "vector index: ✗ not created" in out
    assert "need 255 more chunks" in out

    # Table versions should be shown
    assert "versions (documents):" in out
    assert "versions (chunks):" in out

    # Package versions section
    assert "lancedb:" in out
    assert "haiku.rag:" in out
    assert "docling-document schema:" in out
    assert "pydantic-ai:" in out


@pytest.mark.asyncio
async def test_app_info_with_vector_index(temp_db_path, capsys):
    # Build a database with enough chunks to create a vector index
    import lancedb
    from lancedb.index import IvfPq
    from lancedb.pydantic import LanceModel, Vector
    from pydantic import Field

    db = await lancedb.connect_async(temp_db_path)

    class SettingsRecord(LanceModel):
        id: str = Field(default="settings")
        settings: str = Field(default="{}")

    class DocumentRecord(LanceModel):
        id: str
        content: str

    class ChunkRecord(LanceModel):
        id: str
        document_id: str
        content: str
        vector: Vector(3)  # type: ignore

    settings_tbl = await db.create_table("settings", schema=SettingsRecord)
    docs_tbl = await db.create_table("documents", schema=DocumentRecord)
    chunks_tbl = await db.create_table("chunks", schema=ChunkRecord)
    await db.create_table("document_items", schema=DocumentItemRecord)

    # Insert settings
    await settings_tbl.add(
        [
            SettingsRecord(
                id="settings",
                settings='{"version": "1.0.0", "embeddings": {"model": {"provider": "ollama", "name": "test", "vector_dim": 3}}}',
            )
        ]
    )

    # Insert document
    await docs_tbl.add([DocumentRecord(id="doc-1", content="test")])

    # Insert 512 chunks to allow index creation (PQ needs more than 256 for training)
    chunks = [
        ChunkRecord(
            id=f"chunk-{i}",
            document_id="doc-1",
            content=f"content {i}",
            vector=[0.1 * i, 0.2 * i, 0.3 * i],
        )
        for i in range(512)
    ]
    await chunks_tbl.add(chunks)

    # Create vector index
    await chunks_tbl.create_index("vector", config=IvfPq(distance_type="cosine"))

    app = HaikuRAGApp(db_path=temp_db_path)
    await app.info()

    out = capsys.readouterr().out

    # Check vector index exists
    assert "vector index: ✓ exists" in out
    assert "indexed chunks: 512" in out
    assert "unindexed chunks: 0" in out

    # Check basic info still present
    assert "documents: 1" in out
    assert "chunks: 512" in out


@pytest.mark.asyncio
async def test_app_info_uses_connect_lancedb_for_remote(tmp_path):
    """info() should use connect_lancedb() instead of direct lancedb.connect() for remote URIs."""
    nonexistent = tmp_path / "does_not_exist" / "db.lancedb"
    config = AppConfig(
        lancedb=LanceDBConfig(
            uri="s3://bucket/path",
            storage_options={"endpoint": "http://localhost:9000"},
        )
    )
    app = HaikuRAGApp(db_path=nonexistent, config=config)

    with patch(
        "haiku.rag.store.engine.connect_lancedb", new_callable=AsyncMock
    ) as mock_connect:
        # Empty DB triggers the early-return path - enough to prove connect_lancedb was used
        mock_db = mock_connect.return_value
        mock_list_result = MagicMock()
        mock_list_result.tables = []
        mock_db.list_tables = AsyncMock(return_value=mock_list_result)
        await app.info()

    mock_connect.assert_called_once_with(config, nonexistent)


@pytest.mark.asyncio
async def test_app_info_with_missing_document_items_table(temp_db_path, capsys):
    """info() should still output database info and report pending migrations
    when a required table is absent (as for a DB created before 0.40.0)."""
    import lancedb
    from lancedb.pydantic import LanceModel, Vector
    from pydantic import Field

    db = await lancedb.connect_async(temp_db_path)

    class SettingsRecord(LanceModel):
        id: str = Field(default="settings")
        settings: str = Field(default="{}")

    class DocumentRecord(LanceModel):
        id: str
        content: str

    class ChunkRecord(LanceModel):
        id: str
        document_id: str
        content: str
        vector: Vector(3)  # type: ignore

    settings_tbl = await db.create_table("settings", schema=SettingsRecord)
    docs_tbl = await db.create_table("documents", schema=DocumentRecord)
    chunks_tbl = await db.create_table("chunks", schema=ChunkRecord)
    # Intentionally omit document_items (added in 0.40.0)

    await settings_tbl.add(
        [
            SettingsRecord(
                id="settings",
                settings=json.dumps(
                    {
                        "version": "0.39.0",
                        "embeddings": {
                            "model": {
                                "provider": "openai",
                                "name": "text-embedding-3-small",
                                "vector_dim": 3,
                            }
                        },
                    }
                ),
            )
        ]
    )
    await docs_tbl.add([DocumentRecord(id="doc-1", content="hello")])
    await chunks_tbl.add(
        [ChunkRecord(id="c1", document_id="doc-1", content="c", vector=[0.1, 0.2, 0.3])]
    )

    app = HaikuRAGApp(db_path=temp_db_path)
    await app.info()

    out = capsys.readouterr().out

    # Core stats should still be reported
    assert "haiku.rag version (db): 0.39.0" in out
    assert "documents: 1" in out
    assert "chunks: 1" in out

    # Missing table should be flagged, not cause a crash
    assert "document_items: absent" in out

    # Migration status should be surfaced
    assert "migration(s) pending" in out
    assert "haiku-rag migrate" in out


@pytest.mark.asyncio
async def test_app_info_reports_up_to_date(temp_db_path, capsys):
    """info() should report the database is up to date when no migrations
    are pending."""
    from importlib import metadata

    import lancedb
    from lancedb.pydantic import LanceModel, Vector
    from pydantic import Field

    db = await lancedb.connect_async(temp_db_path)

    class SettingsRecord(LanceModel):
        id: str = Field(default="settings")
        settings: str = Field(default="{}")

    class DocumentRecord(LanceModel):
        id: str
        content: str

    class ChunkRecord(LanceModel):
        id: str
        document_id: str
        content: str
        vector: Vector(3)  # type: ignore

    settings_tbl = await db.create_table("settings", schema=SettingsRecord)
    await db.create_table("documents", schema=DocumentRecord)
    await db.create_table("chunks", schema=ChunkRecord)
    await db.create_table("document_items", schema=DocumentItemRecord)

    current_version = metadata.version("haiku.rag-slim")
    await settings_tbl.add(
        [
            SettingsRecord(
                id="settings",
                settings=json.dumps(
                    {
                        "version": current_version,
                        "embeddings": {
                            "model": {
                                "provider": "openai",
                                "name": "text-embedding-3-small",
                                "vector_dim": 3,
                            }
                        },
                    }
                ),
            )
        ]
    )

    app = HaikuRAGApp(db_path=temp_db_path)
    await app.info()

    out = capsys.readouterr().out
    assert "Database is up to date." in out
    assert "migration(s) pending" not in out


@pytest.mark.asyncio
async def test_app_init_skips_exists_check_for_remote(tmp_path):
    """init() should not check db_path.exists() for remote URIs."""
    nonexistent = tmp_path / "does_not_exist" / "db.lancedb"
    config = AppConfig(
        lancedb=LanceDBConfig(
            uri="s3://bucket/path",
            storage_options={"endpoint": "http://localhost:9000"},
        )
    )
    app = HaikuRAGApp(db_path=nonexistent, config=config)

    with patch("haiku.rag.app.HaikuRAG") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        await app.init()
        # Should have called HaikuRAG to create, not returned early
        mock_client_cls.assert_called_once()


@pytest.mark.asyncio
async def test_app_history_skips_exists_check_for_remote(tmp_path):
    """history() should not check db_path.exists() for remote URIs."""
    nonexistent = tmp_path / "does_not_exist" / "db.lancedb"
    config = AppConfig(
        lancedb=LanceDBConfig(
            uri="s3://bucket/path",
            storage_options={"endpoint": "http://localhost:9000"},
        )
    )
    app = HaikuRAGApp(db_path=nonexistent, config=config)

    with patch("haiku.rag.store.engine.Store") as mock_store_cls:
        mock_store = AsyncMock()
        mock_store.list_table_versions = AsyncMock(return_value=[])
        mock_store_cls.return_value.__aenter__ = AsyncMock(return_value=mock_store)
        mock_store_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        await app.history()
        mock_store_cls.assert_called_once()
