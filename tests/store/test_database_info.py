import json

import pytest
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field

from haiku.rag.config.models import AppConfig
from haiku.rag.store.info import gather_database_info
from haiku.rag.store.schema import DocumentItemRecord


class _SettingsRecord(LanceModel):
    id: str = Field(default="settings")
    settings: str = Field(default="{}")


class _DocumentRecord(LanceModel):
    id: str
    content: str


class _ChunkRecord(LanceModel):
    id: str
    document_id: str
    content: str
    vector: Vector(3)  # type: ignore


async def _seed(temp_db_path, *, version: str, with_items: bool = True):
    import lancedb

    db = await lancedb.connect_async(temp_db_path)
    settings_tbl = await db.create_table("settings", schema=_SettingsRecord)
    docs_tbl = await db.create_table("documents", schema=_DocumentRecord)
    chunks_tbl = await db.create_table("chunks", schema=_ChunkRecord)
    if with_items:
        await db.create_table("document_items", schema=DocumentItemRecord)

    await settings_tbl.add(
        [
            _SettingsRecord(
                id="settings",
                settings=json.dumps(
                    {
                        "version": version,
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
    await docs_tbl.add([_DocumentRecord(id="doc-1", content="hello")])
    await chunks_tbl.add(
        [
            _ChunkRecord(
                id="c1", document_id="doc-1", content="c", vector=[0.1, 0.2, 0.3]
            )
        ]
    )


@pytest.mark.asyncio
async def test_gather_database_info_reports_tables_and_settings(temp_db_path):
    await _seed(temp_db_path, version="1.2.3")

    info = await gather_database_info(temp_db_path, AppConfig())

    assert info.exists is True
    assert info.path == str(temp_db_path)
    assert info.stored_version == "1.2.3"
    assert info.embeddings.provider == "openai"
    assert info.embeddings.name == "text-embedding-3-small"
    assert info.embeddings.vector_dim == 3

    tables = {t.name: t for t in info.tables}
    assert tables["documents"].exists and tables["documents"].num_rows == 1
    assert tables["chunks"].exists and tables["chunks"].num_rows == 1
    assert tables["document_items"].exists
    assert tables["documents"].num_versions >= 1
    assert tables["chunks"].num_versions >= 1

    # Only one chunk: no vector index.
    assert info.vector_index.exists is False

    assert "haiku_rag" in info.packages
    assert "lancedb" in info.packages


@pytest.mark.asyncio
async def test_gather_database_info_flags_missing_table_and_pending_migrations(
    temp_db_path,
):
    await _seed(temp_db_path, version="0.39.0", with_items=False)

    info = await gather_database_info(temp_db_path, AppConfig())

    tables = {t.name: t for t in info.tables}
    assert tables["document_items"].exists is False
    assert info.pending_migrations  # 0.39.0 is behind current schema
    assert all(m.version and m.description is not None for m in info.pending_migrations)


@pytest.mark.asyncio
async def test_gather_database_info_empty_database(temp_db_path):
    import lancedb

    await lancedb.connect_async(temp_db_path)  # creates the dir, no tables

    info = await gather_database_info(temp_db_path, AppConfig())

    assert info.exists is False
    assert info.path == str(temp_db_path)


@pytest.mark.asyncio
async def test_gather_database_info_connects_to_the_location_it_is_given():
    """A remote location is passed to the connection as is and reported back
    as the path; the configuration's own `uri` plays no part."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from haiku.rag.config.models import LanceDBConfig

    config = AppConfig(lancedb=LanceDBConfig(uri="s3://elsewhere/other.lancedb"))
    with patch(
        "haiku.rag.store.info.connect_lancedb", new_callable=AsyncMock
    ) as mock_connect:
        listing = MagicMock()
        listing.tables = []
        mock_connect.return_value.list_tables = AsyncMock(return_value=listing)

        info = await gather_database_info("s3://bucket/papers.lancedb", config)

    assert mock_connect.call_args.args[0] == "s3://bucket/papers.lancedb"
    assert info.path == "s3://bucket/papers.lancedb"
    assert info.exists is False
