from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from haiku.rag.app import HaikuRAGApp
from haiku.rag.config.models import AppConfig, LanceDBConfig
from tests.conftest import for_path
from tests.store.minimal_database import openai_settings, seed_database


async def test_app_info_outputs(temp_db_path, capsys):
    await seed_database(
        temp_db_path,
        openai_settings("1.2.3", query_prefix="Q: ", document_prefix="D: "),
    )

    app = HaikuRAGApp(scope=for_path(temp_db_path))
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
    assert 'query prefix: "Q: "' in out
    assert 'document prefix: "D: "' in out
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


async def test_app_info_with_vector_index(temp_db_path, capsys):
    from lancedb.index import IvfPq

    # PQ needs more than 256 chunks for training
    chunks_tbl = await seed_database(
        temp_db_path,
        {
            "version": "1.0.0",
            "embeddings": {
                "model": {"provider": "ollama", "name": "test", "vector_dim": 3}
            },
        },
        chunks=512,
    )

    # Create vector index
    await chunks_tbl.create_index("vector", config=IvfPq(distance_type="cosine"))

    app = HaikuRAGApp(scope=for_path(temp_db_path))
    await app.info()

    out = capsys.readouterr().out

    # Check vector index exists
    assert "vector index: ✓ exists" in out
    assert "indexed chunks: 512" in out
    assert "unindexed chunks: 0" in out

    # Check basic info still present
    assert "documents: 1" in out
    assert "chunks: 512" in out


@pytest.mark.parametrize("database_name", ["papers", None])
async def test_app_info_opens_a_named_remote_database(database_name):
    """A database named in `lancedb.databases`, selected by name or as the sole
    entry, can sit behind a URI; info opens that URI."""
    from haiku.rag.client.scope import DatabaseScope

    config = AppConfig(
        lancedb=LanceDBConfig(
            databases={"papers": "s3://bucket/papers.lancedb"},
            storage_options={"endpoint": "http://localhost:9000"},
        )
    )
    scope = DatabaseScope.resolve(config, database_name=database_name)
    app = HaikuRAGApp(scope=scope, config=config)

    assert app._is_local is False
    assert app._location == "s3://bucket/papers.lancedb"

    with patch(
        "haiku.rag.store.info.connect_lancedb", new_callable=AsyncMock
    ) as mock_connect:
        mock_db = mock_connect.return_value
        mock_list_result = MagicMock()
        mock_list_result.tables = []
        mock_db.list_tables = AsyncMock(return_value=mock_list_result)
        await app.info()

    mock_connect.assert_called_once()
    assert mock_connect.call_args.args[0] == "s3://bucket/papers.lancedb"


async def test_app_doctor_opens_a_named_remote_database():
    """`run_doctor` is handed the database's location, not the configuration
    naming the set."""
    from haiku.rag.client.scope import DatabaseScope

    config = AppConfig(
        lancedb=LanceDBConfig(databases={"papers": "s3://bucket/papers.lancedb"})
    )
    app = HaikuRAGApp(scope=DatabaseScope.resolve(config, database_name="papers"))

    with patch("haiku.rag.doctor.run_doctor", new_callable=AsyncMock) as run:
        run.return_value = MagicMock(checks=[], ok=True, duplicates=None)
        await app.doctor()

    assert run.call_args.args[1] == "s3://bucket/papers.lancedb"


async def test_app_info_with_missing_document_items_table(temp_db_path, capsys):
    """info() should still output database info and report pending migrations
    when a required table is absent (as for a DB created before 0.40.0)."""
    await seed_database(temp_db_path, openai_settings("0.39.0"), with_items=False)

    app = HaikuRAGApp(scope=for_path(temp_db_path))
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


async def test_app_info_reports_up_to_date(temp_db_path, capsys):
    """info() should report the database is up to date when no migrations
    are pending."""
    from importlib import metadata

    await seed_database(
        temp_db_path,
        openai_settings(metadata.version("haiku.rag-slim")),
        with_document=False,
        chunks=0,
    )

    app = HaikuRAGApp(scope=for_path(temp_db_path))
    await app.info()

    out = capsys.readouterr().out
    assert "Database is up to date." in out
    assert "migration(s) pending" not in out


async def test_app_init_skips_exists_check_for_remote(tmp_path):
    """init() should not check db_path.exists() for remote URIs."""
    config = AppConfig(
        lancedb=LanceDBConfig(
            databases={"path": "s3://bucket/path"},
            storage_options={"endpoint": "http://localhost:9000"},
        )
    )
    app = HaikuRAGApp(scope=for_path(None, config), config=config)

    with patch("haiku.rag.app.HaikuRAG") as mock_client_cls:
        mock_client = AsyncMock()
        covering = mock_client_cls._covering.return_value
        covering.__aenter__ = AsyncMock(return_value=mock_client)
        covering.__aexit__ = AsyncMock(return_value=False)
        await app.init()
        # A missing local path is opened to create, not returned early on.
        mock_client_cls._covering.assert_called_once()


async def test_app_history_skips_exists_check_for_remote(tmp_path):
    """history() should not check db_path.exists() for remote URIs."""
    config = AppConfig(
        lancedb=LanceDBConfig(
            databases={"path": "s3://bucket/path"},
            storage_options={"endpoint": "http://localhost:9000"},
        )
    )
    app = HaikuRAGApp(scope=for_path(None, config), config=config)

    with patch("haiku.rag.store.engine.Store") as mock_store_cls:
        mock_store = AsyncMock()
        mock_store.list_table_versions = AsyncMock(return_value=[])
        mock_store.list_tags = AsyncMock(return_value={})
        mock_store_cls.return_value.__aenter__ = AsyncMock(return_value=mock_store)
        mock_store_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        await app.history()
        mock_store_cls.assert_called_once()


async def test_app_tag_rendering_escapes_markup(tmp_path):
    """lance forbids markup characters in ref names, but externally created
    tags are rendered defensively: markup-looking names must come out as
    literal text in tag list and history, not be interpreted by Rich."""
    from rich.console import Console

    from haiku.rag.store.engine import TagInfo

    config = AppConfig(
        lancedb=LanceDBConfig(
            databases={"path": "s3://bucket/path"},
            storage_options={"endpoint": "http://localhost:9000"},
        )
    )
    app = HaikuRAGApp(scope=for_path(None, config), config=config)
    app.console = Console(record=True, width=200)

    hostile = "[red]release[/red]"
    tags = {hostile: TagInfo(tables={"documents": 1}, missing_tables=[])}

    with patch("haiku.rag.store.engine.Store") as mock_store_cls:
        mock_store = AsyncMock()
        mock_store.list_tags = AsyncMock(return_value=tags)
        mock_store.list_table_versions = AsyncMock(
            return_value=[{"version": 1, "timestamp": "2026-07-14 10:00:00"}]
        )
        mock_store_cls.return_value.__aenter__ = AsyncMock(return_value=mock_store)
        mock_store_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        await app.list_tags()
        await app.history(table="documents")

    output = app.console.export_text()
    assert output.count(hostile) == 2


async def test_app_history_survives_tag_annotation_failure(tmp_path):
    """history degrades to version history without annotations, with a
    warning, when aggregate tag loading fails."""
    from rich.console import Console

    config = AppConfig(
        lancedb=LanceDBConfig(
            databases={"path": "s3://bucket/path"},
            storage_options={"endpoint": "http://localhost:9000"},
        )
    )
    app = HaikuRAGApp(scope=for_path(None, config), config=config)
    app.console = Console(record=True, width=200)

    with patch("haiku.rag.store.engine.Store") as mock_store_cls:
        mock_store = AsyncMock()
        mock_store.list_tags = AsyncMock(side_effect=RuntimeError("tags boom"))
        mock_store.list_table_versions = AsyncMock(
            return_value=[{"version": 1, "timestamp": "2026-07-15 10:00:00"}]
        )
        mock_store_cls.return_value.__aenter__ = AsyncMock(return_value=mock_store)
        mock_store_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        await app.history(table="documents")

    output = app.console.export_text()
    assert "v1" in output
    assert "2026-07-15 10:00:00" in output
    assert "tags boom" in output
