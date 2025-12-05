import pytest
from datasets import Dataset

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.store.models.chunk import Chunk, ChunkMetadata


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
