import pytest
from datasets import Dataset

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.store.engine import Store
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.document import DocumentRepository


@pytest.mark.asyncio
async def test_create_document_with_chunks(qa_corpus: Dataset, temp_db_path):
    """Test creating a document with chunks from the qa_corpus using repository."""
    # Create client
    client = HaikuRAG(db_path=temp_db_path, config=Config, create=True)

    # Get the first document from the corpus
    first_doc = qa_corpus[0]
    document_text = first_doc["document_extracted"]

    # Create the document with chunks in the database
    created_document = await client.create_document(
        content=document_text,
        metadata={"source": "qa_corpus", "topic": first_doc.get("document_topic", "")},
    )

    # Verify the document was created
    assert created_document.id is not None
    assert created_document.content == document_text

    # Check that chunks were created using repository
    chunks = await client.chunk_repository.get_by_document_id(created_document.id)

    assert len(chunks) > 0

    # Verify chunk order is set correctly
    for i, chunk in enumerate(chunks):
        assert chunk.order == i

    client.close()


@pytest.mark.asyncio
async def test_document_repository_crud(qa_corpus: Dataset, temp_db_path):
    """Test CRUD operations in DocumentRepository."""
    # Create a store and repository
    store = Store(temp_db_path, create=True)
    doc_repo = DocumentRepository(store)

    # Get the first document from the corpus
    first_doc = qa_corpus[0]
    document_text = first_doc["document_extracted"]

    # Create a document with URI
    test_uri = "file:///path/to/test.txt"
    document = Document(
        content=document_text, uri=test_uri, metadata={"source": "test"}
    )
    created_document = await doc_repo.create(document)

    # Test get_by_id
    assert created_document.id is not None
    retrieved_document = await doc_repo.get_by_id(created_document.id)
    assert retrieved_document is not None
    assert retrieved_document.content == document_text
    assert retrieved_document.uri == test_uri

    # Test get_by_uri
    retrieved_by_uri = await doc_repo.get_by_uri(test_uri)
    assert retrieved_by_uri is not None
    assert retrieved_by_uri.id == created_document.id
    assert retrieved_by_uri.content == document_text
    assert retrieved_by_uri.uri == test_uri

    # Test get_by_uri with non-existent URI
    non_existent = await doc_repo.get_by_uri("file:///non/existent.txt")
    assert non_existent is None

    # Test update (should regenerate chunks)
    retrieved_document.content = "Updated content for testing"
    updated_document = await doc_repo.update(retrieved_document)
    assert updated_document.content == "Updated content for testing"

    # Test list_all
    all_documents = await doc_repo.list_all()
    assert len(all_documents) == 1
    assert all_documents[0].id == created_document.id

    # Test delete
    deleted = await doc_repo.delete(created_document.id)
    assert deleted is True

    # Verify document is gone
    retrieved_document = await doc_repo.get_by_id(created_document.id)
    assert retrieved_document is None

    store.close()


@pytest.mark.asyncio
async def test_document_list_with_filter(qa_corpus: Dataset, temp_db_path):
    """Test listing documents with filter clause."""
    store = Store(temp_db_path, create=True)
    doc_repo = DocumentRepository(store)

    first_doc = qa_corpus[0]
    document_text = first_doc["document_extracted"]

    doc1 = Document(
        content=document_text,
        uri="https://example.com/doc1.txt",
        metadata={"source": "test", "category": "A"},
    )
    doc2 = Document(
        content=document_text,
        uri="https://arxiv.org/paper.pdf",
        metadata={"source": "test", "category": "B"},
    )
    doc3 = Document(
        content=document_text,
        uri="https://example.com/doc3.txt",
        metadata={"source": "test", "category": "A"},
    )

    created_doc1 = await doc_repo.create(doc1)
    created_doc2 = await doc_repo.create(doc2)
    created_doc3 = await doc_repo.create(doc3)

    all_documents = await doc_repo.list_all()
    assert len(all_documents) == 3

    arxiv_documents = await doc_repo.list_all(filter="uri LIKE '%arxiv%'")
    assert len(arxiv_documents) == 1
    assert arxiv_documents[0].id == created_doc2.id

    example_documents = await doc_repo.list_all(filter="uri LIKE '%example.com%'")
    assert len(example_documents) == 2
    assert {doc.id for doc in example_documents} == {created_doc1.id, created_doc3.id}

    store.close()


def test_document_get_docling_document():
    """Test parsing stored DoclingDocument JSON."""
    doc_json = {
        "name": "test_doc",
        "texts": [
            {
                "self_ref": "#/texts/0",
                "text": "Test text",
                "orig": "Test text",
                "label": "paragraph",
            },
        ],
        "tables": [],
        "pictures": [],
        "groups": [],
        "body": {"self_ref": "#/body", "children": []},
        "furniture": {"self_ref": "#/furniture", "children": []},
    }

    import json

    document = Document(
        content="Test content",
        docling_document_json=json.dumps(doc_json),
        docling_version="1.3.0",
    )

    docling_doc = document.get_docling_document()

    assert docling_doc is not None
    assert docling_doc.name == "test_doc"
    assert len(docling_doc.texts) == 1
    assert docling_doc.texts[0].text == "Test text"


def test_document_get_docling_document_none():
    """Test get_docling_document returns None when not stored."""
    document = Document(content="Test content")

    assert document.docling_document_json is None
    assert document.get_docling_document() is None


def test_document_get_docling_document_caching():
    """Test that get_docling_document uses LRU cache keyed by document ID."""
    from haiku.rag.store.models.document import (
        _docling_document_cache,
        invalidate_docling_document_cache,
    )

    doc_json = {
        "name": "test_doc",
        "texts": [
            {
                "self_ref": "#/texts/0",
                "text": "Test text",
                "orig": "Test text",
                "label": "paragraph",
            },
        ],
        "tables": [],
        "pictures": [],
        "groups": [],
        "body": {"self_ref": "#/body", "children": []},
        "furniture": {"self_ref": "#/furniture", "children": []},
    }

    import json

    json_str = json.dumps(doc_json)

    # Clear cache to get clean state
    _docling_document_cache.clear()

    document = Document(
        id="test-doc-id", content="Test content", docling_document_json=json_str
    )

    # First call - not in cache
    assert "test-doc-id" not in _docling_document_cache
    doc1 = document.get_docling_document()
    assert "test-doc-id" in _docling_document_cache

    # Second call - cache hit, same object
    doc2 = document.get_docling_document()
    assert doc1 is doc2

    # Invalidation removes from cache
    invalidate_docling_document_cache("test-doc-id")
    assert "test-doc-id" not in _docling_document_cache


def test_document_get_docling_document_no_id_no_cache():
    """Test that documents without ID don't use cache."""
    from haiku.rag.store.models.document import _docling_document_cache

    doc_json = {
        "name": "test_doc",
        "texts": [],
        "tables": [],
        "pictures": [],
        "groups": [],
        "body": {"self_ref": "#/body", "children": []},
        "furniture": {"self_ref": "#/furniture", "children": []},
    }

    import json

    json_str = json.dumps(doc_json)

    # Clear cache
    _docling_document_cache.clear()

    # Document without ID
    document = Document(content="Test content", docling_document_json=json_str)

    doc1 = document.get_docling_document()
    doc2 = document.get_docling_document()

    # Cache should remain empty (no ID to cache by)
    assert len(_docling_document_cache) == 0

    # Each call parses fresh (different objects)
    assert doc1 is not doc2
