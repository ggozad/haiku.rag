import tempfile
from pathlib import Path

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig


def test_database_not_created_without_create_flag():
    """Test that database is not created without create=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.lancedb"

        config = AppConfig()

        with pytest.raises(FileNotFoundError, match="Database does not exist"):
            HaikuRAG(db_path=db_path, config=config)


def test_database_created_with_create_flag():
    """Test that database is created with create=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.lancedb"

        config = AppConfig()

        client = HaikuRAG(db_path=db_path, config=config, create=True)
        assert db_path.exists()
        client.close()


@pytest.mark.asyncio
async def test_operations_work_after_database_created():
    """Test that operations work after DB is created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.lancedb"

        config = AppConfig()

        # First, create DB with create=True and add document
        async with HaikuRAG(db_path=db_path, config=config, create=True) as client:
            await client.create_document("Test content", uri="test://doc1")

        # Re-open without create flag and verify we can read the document
        async with HaikuRAG(db_path=db_path, config=config) as client:
            docs = await client.list_documents()
            assert len(docs) == 1
            assert docs[0].content == "Test content"
