from importlib import metadata

import pytest

from haiku.rag.config.models import AppConfig
from haiku.rag.store import MigrationRequiredError, TagError
from haiku.rag.store.engine import Store
from haiku.rag.store.info import gather_database_info
from haiku.rag.store.models import Document
from haiku.rag.store.repositories.document import DocumentRepository
from tests.store.minimal_database import openai_settings, seed_database


async def test_gather_database_info_reports_tables_and_settings(temp_db_path):
    await seed_database(temp_db_path, openai_settings("1.2.3"))

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
    assert tables["documents"].latest_version_at is not None

    # Only one chunk: no vector index.
    assert info.vector_index.exists is False

    assert "haiku_rag" in info.packages
    assert "lancedb" in info.packages


async def test_gather_database_info_flags_missing_table_and_pending_migrations(
    temp_db_path,
):
    await seed_database(temp_db_path, openai_settings("0.39.0"), with_items=False)

    info = await gather_database_info(temp_db_path, AppConfig())

    tables = {t.name: t for t in info.tables}
    assert tables["document_items"].exists is False
    assert info.pending_migrations  # 0.39.0 is behind current schema
    assert all(m.version and m.description is not None for m in info.pending_migrations)


@pytest.mark.parametrize("missing", ["table", "row"])
async def test_gather_database_info_without_settings_reports_unknown(
    temp_db_path, missing
):
    import lancedb

    await seed_database(temp_db_path, openai_settings("1.2.3"))
    db = await lancedb.connect_async(temp_db_path)
    if missing == "table":
        await db.drop_table("settings")
    else:
        await (await db.open_table("settings")).delete("id = 'settings'")

    info = await gather_database_info(temp_db_path, AppConfig())

    assert info.stored_version == "unknown"
    assert info.embeddings.provider == "unknown"
    tables = {t.name: t for t in info.tables}
    assert tables["documents"].num_rows == 1


async def test_gather_database_info_empty_database(temp_db_path):
    import lancedb

    await lancedb.connect_async(temp_db_path)  # creates the dir, no tables

    info = await gather_database_info(temp_db_path, AppConfig())

    assert info.exists is False
    assert info.path == str(temp_db_path)


async def test_gather_database_info_connects_to_the_location_it_is_given():
    """A remote location is passed to the connection as is and reported back
    as the path; the configuration's own `uri` plays no part."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from haiku.rag.config.models import LanceDBConfig

    config = AppConfig(
        lancedb=LanceDBConfig(databases={"other": "s3://elsewhere/other.lancedb"})
    )
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


async def test_the_latest_version_time_moves_with_a_write(temp_db_path):
    """A content update advances the latest version time and keeps the row count."""
    await seed_database(temp_db_path, openai_settings("1.2.3"))
    before = {
        t.name: t
        for t in (await gather_database_info(temp_db_path, AppConfig())).tables
    }

    import lancedb

    db = await lancedb.connect_async(temp_db_path)
    docs_tbl = await db.open_table("documents")
    await docs_tbl.update(where="id = 'doc-1'", updates={"content": "rewritten"})

    after = {
        t.name: t
        for t in (await gather_database_info(temp_db_path, AppConfig())).tables
    }

    assert after["documents"].num_rows == before["documents"].num_rows
    assert after["documents"].latest_version_at is not None
    assert before["documents"].latest_version_at is not None
    assert after["documents"].latest_version_at > before["documents"].latest_version_at


async def _tag_between(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="before the tag"))
        await store.create_tag("before")
        await repo.create(Document(content="after the tag"))


async def test_info_at_a_tag_reports_the_tagged_state(temp_db_path):
    await _tag_between(temp_db_path)

    live = await gather_database_info(temp_db_path, AppConfig())
    tagged = await gather_database_info(temp_db_path, AppConfig(), tag="before")

    rows = {t.name: t.num_rows for t in tagged.tables}
    assert rows["documents"] == 1
    assert {t.name: t.num_rows for t in live.tables}["documents"] == 2
    times = {t.name: t.latest_version_at for t in tagged.tables}
    live_times = {t.name: t.latest_version_at for t in live.tables}
    assert times["documents"] is not None and live_times["documents"] is not None
    assert times["documents"] < live_times["documents"]


async def test_info_at_a_tag_needing_a_migration_is_refused(temp_db_path):
    current = metadata.version("haiku.rag-slim")
    async with Store(temp_db_path, create=True) as store:
        await store.set_haiku_version("0.88.0")
        await store.create_tag("old")
        await store.set_haiku_version(current)

    with pytest.raises(MigrationRequiredError, match="Tag 'old' was taken at 0.88.0"):
        await gather_database_info(temp_db_path, AppConfig(), tag="old")
    assert (await gather_database_info(temp_db_path, AppConfig())).exists


async def test_info_at_a_tag_with_no_recorded_version_is_refused(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        await store.settings_table.update({"settings": "{}"}, where="id = 'settings'")
        await store.create_tag("versionless")

    with pytest.raises(MigrationRequiredError, match="taken at 0.0.0"):
        await gather_database_info(temp_db_path, AppConfig(), tag="versionless")


async def test_info_at_a_missing_tag_raises(temp_db_path):
    await _tag_between(temp_db_path)

    with pytest.raises(TagError, match="Tag 'nope' does not exist"):
        await gather_database_info(temp_db_path, AppConfig(), tag="nope")
