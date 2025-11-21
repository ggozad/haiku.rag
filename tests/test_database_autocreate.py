import tempfile
from pathlib import Path

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig


def test_read_operations_do_not_create_database():
    """Test that read operations fail if database doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.lancedb"

        config = AppConfig()

        # Read operation with read_only=True should fail
        with pytest.raises(
            FileNotFoundError,
            match="Database does not exist.*Use a write operation",
        ):
            HaikuRAG(db_path=db_path, config=config, read_only=True)


def test_write_operations_create_database():
    """Test that write operations create the database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.lancedb"

        config = AppConfig()

        # Write operation with read_only=False (default) should succeed
        client = HaikuRAG(db_path=db_path, config=config, read_only=False)
        assert db_path.exists()
        client.close()


async def test_add_document_creates_database():
    """Test that add operations create the database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.lancedb"

        config = AppConfig()

        # Create a document (write operation) should work and create DB
        async with HaikuRAG(db_path=db_path, config=config, read_only=False) as client:
            doc = await client.create_document("Test content")
            assert doc.id is not None
            assert doc.content == "Test content"
            assert db_path.exists()


async def test_search_fails_if_database_does_not_exist():
    """Test that search operations fail if DB doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.lancedb"

        config = AppConfig()

        # Read operation (search) should fail if DB doesn't exist
        with pytest.raises(
            FileNotFoundError,
            match="Database does not exist.*Use a write operation",
        ):
            async with HaikuRAG(
                db_path=db_path, config=config, read_only=True
            ) as client:
                await client.search("test query")


async def test_read_operations_work_after_database_created():
    """Test that read operations work after DB is created via write operation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.lancedb"

        config = AppConfig()

        # First, create DB via write operation
        async with HaikuRAG(db_path=db_path, config=config, read_only=False) as client:
            await client.create_document("Test content", uri="test://doc1")

        # Now read operations should work since DB exists
        async with HaikuRAG(db_path=db_path, config=config, read_only=True) as client:
            docs = await client.list_documents()
            assert len(docs) == 1
            assert docs[0].content == "Test content"


def test_default_read_only_is_false():
    """Test that read_only defaults to False for backward compatibility."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.lancedb"

        config = AppConfig()

        # Without specifying read_only, it should default to False (allow creation)
        client = HaikuRAG(db_path=db_path, config=config)
        assert db_path.exists()
        client.close()
