from unittest.mock import patch

import pytest

from haiku.rag.config import Config
from haiku.rag.store.engine import Store
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository


@pytest.mark.asyncio
async def test_lancedb_cloud_skips_optimization(temp_db_path):
    """Test that optimization is skipped when using LanceDB Cloud (db:// URI)."""
    # Create a store
    store = Store(temp_db_path)
    chunk_repo = ChunkRepository(store)
    doc_repo = DocumentRepository(store)

    # Create a document
    document = Document(content="Test document content", metadata={})
    created_document = await doc_repo.create(document)
    document_id = created_document.id

    # Mock LANCEDB_URI to simulate LanceDB Cloud usage
    with patch.object(Config, "LANCEDB_URI", "db://test-database"):
        # Mock the optimize method to track if it's called
        with patch.object(store.chunks_table, "optimize") as mock_optimize:
            # Create a chunk - this should trigger optimization logic
            chunk = Chunk(
                document_id=document_id,
                content="Test chunk content",
                metadata={"test": "value"},
            )

            created_chunk = await chunk_repo.create(chunk)
            assert created_chunk.id is not None

            # Wait a moment to ensure any async optimization would complete
            import asyncio

            await asyncio.sleep(0.1)

            # The optimize method should NOT have been called for LanceDB Cloud
            mock_optimize.assert_not_called()

    store.close()


@pytest.mark.asyncio
async def test_local_storage_calls_optimization(temp_db_path):
    """Test that optimization is called for local storage."""
    # Create a store
    store = Store(temp_db_path)
    chunk_repo = ChunkRepository(store)
    doc_repo = DocumentRepository(store)

    # Create a document
    document = Document(content="Test document content", metadata={})
    created_document = await doc_repo.create(document)
    document_id = created_document.id

    # Ensure LANCEDB_URI is empty (local storage)
    with patch.object(Config, "LANCEDB_URI", ""):
        # Mock the optimize method to track if it's called
        with patch.object(store.chunks_table, "optimize") as mock_optimize:
            # Create a chunk - this should trigger optimization logic
            chunk = Chunk(
                document_id=document_id,
                content="Test chunk content",
                metadata={"test": "value"},
            )

            created_chunk = await chunk_repo.create(chunk)
            assert created_chunk.id is not None

            # Wait a moment to ensure async optimization completes
            import asyncio

            await asyncio.sleep(0.1)

            # The optimize method SHOULD have been called for local storage
            mock_optimize.assert_called()

    store.close()
