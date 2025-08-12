import pytest
from datasets import Dataset

from haiku.rag.store.engine import Store
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.utils import text_to_docling_document


@pytest.mark.asyncio
async def test_chunk_repository_operations(qa_corpus: Dataset):
    """Test ChunkRepository operations."""
    # Create an in-memory store and repositories
    store = Store(":memory:")
    doc_repo = DocumentRepository(store)
    chunk_repo = ChunkRepository(store)

    # Get the first document from the corpus
    first_doc = qa_corpus[0]
    document_text = first_doc["document_extracted"]

    # Create a document first
    document = Document(content=document_text, metadata={"source": "test"})
    created_document = await doc_repo.create(document)
    assert created_document.id is not None

    # Test getting chunks by document ID
    chunks = await chunk_repo.get_by_document_id(created_document.id)
    assert len(chunks) > 0
    assert all(chunk.document_id == created_document.id for chunk in chunks)

    # Test chunk search
    results = await chunk_repo.search_chunks("election", limit=2)
    assert len(results) <= 2
    assert all(hasattr(chunk, "content") for chunk, _ in results)

    # Test deleting chunks by document ID
    deleted = await chunk_repo.delete_by_document_id(created_document.id)
    assert deleted is True

    # Verify chunks are gone
    chunks_after_delete = await chunk_repo.get_by_document_id(created_document.id)
    assert len(chunks_after_delete) == 0

    store.close()


@pytest.mark.asyncio
async def test_create_chunks_for_document(qa_corpus: Dataset):
    """Test creating chunks for a document."""
    # Create an in-memory store and repositories
    store = Store(":memory:")
    chunk_repo = ChunkRepository(store)

    # Get the first document from the corpus
    first_doc = qa_corpus[0]
    document_text = first_doc["document_extracted"]

    # Create a document first (without chunks)
    document = Document(content=document_text, metadata={"source": "test"})

    # Insert document manually to test chunk creation independently
    document_id = None
    if store._connection is not None:
        cursor = store._connection.cursor()
        cursor.execute(
            """
            INSERT INTO documents (content, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (document.content, "{}", document.created_at, document.updated_at),
        )
        document_id = cursor.lastrowid
        document.id = document_id
        store._connection.commit()

    assert document_id is not None, "Document ID should not be None"

    # Convert text to DoclingDocument
    docling_document = text_to_docling_document(document_text, name="test.md")

    # Test creating chunks for the document
    chunks = await chunk_repo.create_chunks_for_document(document_id, docling_document)

    # Verify chunks were created
    assert len(chunks) > 0
    assert all(chunk.document_id == document_id for chunk in chunks)
    assert all(chunk.id is not None for chunk in chunks)

    # Verify chunk order metadata
    for i, chunk in enumerate(chunks):
        assert chunk.metadata.get("order") == i

    # Verify chunks exist in database
    db_chunks = await chunk_repo.get_by_document_id(document_id)
    assert len(db_chunks) == len(chunks)

    store.close()


@pytest.mark.asyncio
async def test_chunk_repository_crud():
    """Test basic CRUD operations in ChunkRepository."""
    # Create an in-memory store
    store = Store(":memory:")
    chunk_repo = ChunkRepository(store)

    # First create a document to reference
    document_id = None
    if store._connection is not None:
        cursor = store._connection.cursor()
        cursor.execute(
            """
            INSERT INTO documents (content, metadata, created_at, updated_at)
            VALUES (?, ?, datetime('now'), datetime('now'))
            """,
            ("Test document content", "{}"),
        )
        document_id = cursor.lastrowid
        store._connection.commit()

    assert document_id is not None, "Document ID should not be None"

    # Test create chunk manually
    chunk = Chunk(
        document_id=document_id,
        content="Test chunk content",
        metadata={"test": "value"},
    )

    created_chunk = await chunk_repo.create(chunk)
    assert created_chunk.id is not None
    assert created_chunk.content == "Test chunk content"

    # Test get by ID
    retrieved_chunk = await chunk_repo.get_by_id(created_chunk.id)
    assert retrieved_chunk is not None
    assert retrieved_chunk.content == "Test chunk content"
    assert retrieved_chunk.metadata["test"] == "value"

    # Test update
    retrieved_chunk.content = "Updated chunk content"
    updated_chunk = await chunk_repo.update(retrieved_chunk)
    assert updated_chunk.content == "Updated chunk content"

    # Test list all
    all_chunks = await chunk_repo.list_all()
    assert len(all_chunks) >= 1
    assert any(chunk.id == created_chunk.id for chunk in all_chunks)

    # Test delete
    deleted = await chunk_repo.delete(created_chunk.id)
    assert deleted is True

    # Verify chunk is gone
    retrieved_chunk = await chunk_repo.get_by_id(created_chunk.id)
    assert retrieved_chunk is None

    store.close()


@pytest.mark.asyncio
async def test_adjacent_chunks():
    """Test the get_adjacent_chunks repository method."""
    store = Store(":memory:")
    doc_repo = DocumentRepository(store)
    chunk_repo = ChunkRepository(store)

    # Create a simple document first
    document_content = "Test document for chunking"
    document = Document(content=document_content)
    created_document = await doc_repo.create(document)

    # Manually create multiple chunks with order metadata
    chunks_data = [
        ("First chunk content", 0),
        ("Second chunk content", 1),
        ("Third chunk content", 2),
        ("Fourth chunk content", 3),
        ("Fifth chunk content", 4),
    ]

    created_chunks = []
    for content, order in chunks_data:
        chunk = Chunk(
            document_id=created_document.id, content=content, metadata={"order": order}
        )
        created_chunk = await chunk_repo.create(chunk)
        created_chunks.append(created_chunk)

    # Test with the middle chunk (index 2, order 2)
    middle_chunk = created_chunks[2]

    # Get adjacent chunks (1 before and after)
    adjacent_chunks = await chunk_repo.get_adjacent_chunks(middle_chunk, 1)

    # Should have 2 chunks (one before, one after)
    assert len(adjacent_chunks) == 2

    # Should not include the original chunk
    assert middle_chunk.id not in [chunk.id for chunk in adjacent_chunks]

    # Should include chunks with order 1 and 3
    orders = [chunk.metadata.get("order") for chunk in adjacent_chunks]
    assert 1 in orders
    assert 3 in orders

    # All adjacent chunks should be from the same document
    for chunk in adjacent_chunks:
        assert chunk.document_id == created_document.id

    store.close()
