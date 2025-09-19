import pytest
from datasets import Dataset

from haiku.rag.store.engine import Store
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.document import DocumentRepository


@pytest.mark.asyncio
async def test_create_document_with_chunks(qa_corpus: Dataset, temp_db_path):
    """Test creating a document with chunks from the qa_corpus using repository."""
    # Create a store and repository
    store = Store(temp_db_path)
    doc_repo = DocumentRepository(store)

    # Get the first document from the corpus
    first_doc = qa_corpus[0]
    document_text = first_doc["document_extracted"]

    # Create a Document instance
    document = Document(
        content=document_text,
        metadata={"source": "qa_corpus", "topic": first_doc.get("document_topic", "")},
    )

    # Convert text to DoclingDocument for chunk creation
    from haiku.rag.utils import text_to_docling_document

    docling_document = text_to_docling_document(document_text, name="test.md")

    # Create the document with chunks in the database
    created_document = await doc_repo._create_with_docling(document, docling_document)

    # Verify the document was created
    assert created_document.id is not None
    assert created_document.content == document_text

    # Check that chunks were created using repository
    from haiku.rag.store.repositories.chunk import ChunkRepository

    chunk_repo = ChunkRepository(store)
    chunks = await chunk_repo.get_by_document_id(created_document.id)

    assert len(chunks) > 0

    # Verify chunk order is set correctly
    for i, chunk in enumerate(chunks):
        assert chunk.order == i

    store.close()


@pytest.mark.asyncio
async def test_document_repository_crud(qa_corpus: Dataset, temp_db_path):
    """Test CRUD operations in DocumentRepository."""
    # Create a store and repository
    store = Store(temp_db_path)
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
