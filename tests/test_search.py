import pytest
from datasets import Dataset

from haiku.rag.store.engine import Store
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository


@pytest.mark.asyncio
async def test_search_qa_corpus(qa_corpus: Dataset, temp_db_path):
    """Test that documents can be found by searching with their associated questions."""
    # Create a store and repositories
    store = Store(temp_db_path)
    doc_repo = DocumentRepository(store)
    chunk_repo = ChunkRepository(store)

    # Load first 20 documents with embeddings (reduced for faster testing)
    num_documents = 20
    documents = []
    for i in range(num_documents):
        doc_data = qa_corpus[i]
        document_text = doc_data["document_extracted"]

        # Create a Document instance
        document = Document(
            content=document_text,
            metadata={
                "source": "qa_corpus",
                "topic": doc_data.get("document_topic", ""),
                "document_id": doc_data.get("document_id", ""),
                "question": doc_data["question"],
            },
        )

        # Create the document with chunks and embeddings
        from haiku.rag.utils import text_to_docling_document

        docling_document = text_to_docling_document(document_text, name="test.md")
        created_document = await doc_repo._create_with_docling(
            document, docling_document
        )
        documents.append((created_document, doc_data))

    for i in range(3):  # Test with first few documents
        target_document, doc_data = documents[i]
        question = doc_data["question"]

        # Test vector search
        vector_results = await chunk_repo.search(
            question, limit=5, search_type="vector"
        )
        target_document_ids = {chunk.document_id for chunk, _ in vector_results}
        assert target_document.id in target_document_ids

        # Test FTS search
        fts_results = await chunk_repo.search(question, limit=5, search_type="fts")
        target_document_ids = {chunk.document_id for chunk, _ in fts_results}
        assert target_document.id in target_document_ids

        # Test hybrid search
        hybrid_results = await chunk_repo.search(
            question, limit=5, search_type="hybrid"
        )
        target_document_ids = {chunk.document_id for chunk, _ in hybrid_results}
        assert target_document.id in target_document_ids

    store.close()


@pytest.mark.asyncio
async def test_chunks_include_document_info(temp_db_path):
    """Test that search results include document URI and metadata."""
    store = Store(temp_db_path)
    doc_repo = DocumentRepository(store)
    chunk_repo = ChunkRepository(store)

    # Create a document with URI and metadata
    document = Document(
        content="This is a test document with some content for searching.",
        uri="https://example.com/test.html",
        metadata={"title": "Test Document", "author": "Test Author"},
    )

    # Create the document with chunks
    from haiku.rag.utils import text_to_docling_document

    docling_document = text_to_docling_document(document.content, name="test.md")
    created_document = await doc_repo._create_with_docling(document, docling_document)

    # Search for chunks
    results = await chunk_repo.search("test document", limit=1, search_type="hybrid")

    assert len(results) > 0
    chunk, _ = results[0]

    # Verify the chunk includes document information
    assert chunk.document_uri == "https://example.com/test.html"
    assert chunk.document_meta == {"title": "Test Document", "author": "Test Author"}
    assert chunk.document_id == created_document.id

    store.close()
