import pytest
from datasets import Dataset

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.store.engine import Store
from haiku.rag.store.models.chunk import Chunk, ChunkMetadata
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository


@pytest.mark.asyncio
async def test_chunk_repository_operations(qa_corpus: Dataset, temp_db_path):
    """Test ChunkRepository operations."""
    # Create client
    client = HaikuRAG(db_path=temp_db_path, config=Config, create=True)

    # Get the first document from the corpus
    first_doc = qa_corpus[0]
    document_text = first_doc["document_extracted"]

    # Create a document first with chunks
    created_document = await client.create_document(
        content=document_text, metadata={"source": "test"}
    )
    assert created_document.id is not None

    # Test getting chunks by document ID
    chunks = await client.chunk_repository.get_by_document_id(created_document.id)
    assert len(chunks) > 0
    assert all(chunk.document_id == created_document.id for chunk in chunks)

    # Test chunk search
    results = await client.chunk_repository.search(
        "election", limit=2, search_type="vector"
    )
    assert len(results) <= 2
    assert all(hasattr(chunk, "content") for chunk, _ in results)

    # Test deleting chunks by document ID
    deleted = await client.chunk_repository.delete_by_document_id(created_document.id)
    assert deleted is True

    # Verify chunks are gone
    chunks_after_delete = await client.chunk_repository.get_by_document_id(
        created_document.id
    )
    assert len(chunks_after_delete) == 0

    client.close()


@pytest.mark.asyncio
async def test_chunking_pipeline(qa_corpus: Dataset, temp_db_path):
    """Test document chunking using client primitives."""
    from haiku.rag.client import HaikuRAG
    from haiku.rag.embeddings import embed_chunks

    async with HaikuRAG(db_path=temp_db_path, create=True) as client:
        # Get the first document from the corpus
        first_doc = qa_corpus[0]
        document_text = first_doc["document_extracted"]

        # Use client primitives: convert → chunk → embed
        docling_document = await client.convert(document_text)
        chunks = await client.chunk(docling_document)
        embedded_chunks = await embed_chunks(chunks)

        # Verify chunks were created with embeddings
        assert len(chunks) > 0
        assert all(chunk.embedding is None for chunk in chunks)  # Before embedding
        assert all(chunk.embedding is not None for chunk in embedded_chunks)  # After

        # Verify chunk order
        for i, chunk in enumerate(chunks):
            assert chunk.order == i


@pytest.mark.asyncio
async def test_chunk_repository_crud(temp_db_path):
    """Test basic CRUD operations in ChunkRepository."""
    # Create a store
    store = Store(temp_db_path, create=True)
    chunk_repo = ChunkRepository(store)
    doc_repo = DocumentRepository(store)

    # First create a document to reference
    document = Document(content="Test document content", metadata={})
    created_document = await doc_repo.create(document)
    document_id = created_document.id

    assert document_id is not None, "Document ID should not be None"

    # Test create chunk manually
    chunk = Chunk(
        document_id=document_id,
        content="Test chunk content",
        metadata={"test": "value"},
    )

    created_chunk = await chunk_repo.create(chunk)
    assert isinstance(created_chunk, Chunk)
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
async def test_adjacent_chunks(temp_db_path):
    """Test the get_adjacent_chunks repository method."""
    store = Store(temp_db_path, create=True)
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
        chunk = Chunk(document_id=created_document.id, content=content, order=order)
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
    orders = [chunk.order for chunk in adjacent_chunks]
    assert 1 in orders
    assert 3 in orders

    # All adjacent chunks should be from the same document
    for chunk in adjacent_chunks:
        assert chunk.document_id == created_document.id

    store.close()


def test_chunk_metadata_parsing():
    """Test ChunkMetadata parsing from chunk metadata dict."""
    metadata_dict = {
        "doc_item_refs": ["#/texts/0", "#/texts/1", "#/tables/0"],
        "headings": ["Chapter 1", "Section 1.1"],
        "labels": ["paragraph", "paragraph", "table"],
        "page_numbers": [1, 1, 2],
    }

    chunk = Chunk(
        content="Test content",
        metadata=metadata_dict,
    )

    chunk_meta = chunk.get_chunk_metadata()

    assert isinstance(chunk_meta, ChunkMetadata)
    assert chunk_meta.doc_item_refs == ["#/texts/0", "#/texts/1", "#/tables/0"]
    assert chunk_meta.headings == ["Chapter 1", "Section 1.1"]
    assert chunk_meta.labels == ["paragraph", "paragraph", "table"]
    assert chunk_meta.page_numbers == [1, 1, 2]


def test_chunk_metadata_defaults():
    """Test ChunkMetadata with empty/default values."""
    chunk = Chunk(content="Test content", metadata={})
    chunk_meta = chunk.get_chunk_metadata()

    assert chunk_meta.doc_item_refs == []
    assert chunk_meta.headings is None
    assert chunk_meta.labels == []
    assert chunk_meta.page_numbers == []


def test_chunk_metadata_resolve_doc_items():
    """Test resolving doc_item_refs to actual DocItem objects."""
    from docling_core.types.doc.document import DoclingDocument

    # Create a minimal DoclingDocument with some text items
    doc_json = {
        "name": "test_doc",
        "texts": [
            {
                "self_ref": "#/texts/0",
                "text": "First text",
                "orig": "First text",
                "label": "paragraph",
            },
            {
                "self_ref": "#/texts/1",
                "text": "Second text",
                "orig": "Second text",
                "label": "title",
            },
        ],
        "tables": [],
        "pictures": [],
        "groups": [],
        "body": {"self_ref": "#/body", "children": []},
        "furniture": {"self_ref": "#/furniture", "children": []},
    }
    docling_doc = DoclingDocument.model_validate(doc_json)

    # Create chunk metadata with refs
    chunk_meta = ChunkMetadata(
        doc_item_refs=["#/texts/0", "#/texts/1"],
        labels=["paragraph", "title"],
    )

    # Resolve refs
    doc_items = chunk_meta.resolve_doc_items(docling_doc)

    assert len(doc_items) == 2
    assert getattr(doc_items[0], "text") == "First text"
    assert getattr(doc_items[1], "text") == "Second text"


def test_chunk_metadata_resolve_doc_items_graceful_degradation():
    """Test that invalid refs are skipped gracefully."""
    from docling_core.types.doc.document import DoclingDocument

    doc_json = {
        "name": "test_doc",
        "texts": [
            {
                "self_ref": "#/texts/0",
                "text": "Only text",
                "orig": "Only text",
                "label": "paragraph",
            },
        ],
        "tables": [],
        "pictures": [],
        "groups": [],
        "body": {"self_ref": "#/body", "children": []},
        "furniture": {"self_ref": "#/furniture", "children": []},
    }
    docling_doc = DoclingDocument.model_validate(doc_json)

    # Create chunk metadata with one valid and one invalid ref
    chunk_meta = ChunkMetadata(
        doc_item_refs=["#/texts/0", "#/texts/999", "#/invalid/path"],
    )

    # Resolve refs - invalid ones should be skipped
    doc_items = chunk_meta.resolve_doc_items(docling_doc)

    assert len(doc_items) == 1
    assert getattr(doc_items[0], "text") == "Only text"


def test_chunk_metadata_resolve_empty_refs():
    """Test resolving with no refs returns empty list."""
    from docling_core.types.doc.document import DoclingDocument

    doc_json = {
        "name": "test_doc",
        "texts": [],
        "tables": [],
        "pictures": [],
        "groups": [],
        "body": {"self_ref": "#/body", "children": []},
        "furniture": {"self_ref": "#/furniture", "children": []},
    }
    docling_doc = DoclingDocument.model_validate(doc_json)

    chunk_meta = ChunkMetadata()
    doc_items = chunk_meta.resolve_doc_items(docling_doc)

    assert doc_items == []
