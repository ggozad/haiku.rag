import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig


async def test_database_not_created_without_create_flag(tmp_path):
    """Test that database is not created without create=True."""
    db_path = tmp_path / "test.lancedb"

    config = AppConfig()

    with pytest.raises(FileNotFoundError, match="Database does not exist"):
        async with HaikuRAG(db_path=db_path, config=config):
            pass


async def test_database_created_with_create_flag(tmp_path):
    """Test that database is created with create=True."""
    db_path = tmp_path / "test.lancedb"

    config = AppConfig()

    async with HaikuRAG(db_path=db_path, config=config, create=True):
        assert db_path.exists()


@pytest.mark.vcr()
async def test_operations_work_after_database_created(tmp_path):
    """Test that operations work after DB is created."""
    db_path = tmp_path / "test.lancedb"

    config = AppConfig()

    # First, create DB with create=True and add document
    async with HaikuRAG(db_path=db_path, config=config, create=True) as client:
        await client.create_document("Test content", uri="test://doc1")

    # Re-open without create flag and verify we can read the document
    async with HaikuRAG(db_path=db_path, config=config) as client:
        docs = await client.list_documents()
        assert len(docs) == 1
        doc = await client.get_document_by_id(docs[0].id)
        assert doc is not None
        assert doc.content == "Test content"


def test_default_db_path_comes_from_storage_data_dir(tmp_path):
    """Omitting db_path places the database under the configured data dir."""
    from haiku.rag.client import HaikuRAG
    from haiku.rag.config import AppConfig

    config = AppConfig()
    config.storage.data_dir = tmp_path

    [ref] = HaikuRAG(config=config)._resolve_scope().databases

    assert ref.db_path == tmp_path / "haiku.rag.lancedb"


@pytest.mark.asyncio
async def test_vacuum_optimizes_tables_without_losing_rows(temp_db_path):
    """The public vacuum() runs the store's optimize pass over real rows."""
    from haiku.rag.store.models.document import Document
    from haiku.rag.store.repositories.document import DocumentRepository

    async with HaikuRAG(temp_db_path, create=True) as client:
        repo = DocumentRepository(client.store)
        for i in range(3):
            await repo.create(Document(content=f"body {i}", uri=f"test://doc{i}"))
        before = len(await client.store.documents_table.list_versions())

        await client.vacuum()

        # Optimize compacts the per-document fragments into new versions; an
        # unchanged count would mean nothing reached the tables.
        assert len(await client.store.documents_table.list_versions()) > before
        assert await client.count_documents() == 3
