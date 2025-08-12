import pytest
from datasets import Dataset

from haiku.rag.store.engine import Store
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository


@pytest.mark.asyncio
async def test_search_qa_corpus(qa_corpus: Dataset):
    """Test that documents can be found by searching with their associated questions."""
    # Create an in-memory store and repositories
    store = Store(":memory:")
    doc_repo = DocumentRepository(store)
    chunk_repo = ChunkRepository(store)
    num_documents = 20
    # Load first 10 documents with embeddings (reduced for faster testing)
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
        created_document = await doc_repo.create(document)
        documents.append((created_document, doc_data))

    for i in range(5):  # Test with first few documents
        target_document, doc_data = documents[i]
        question = doc_data["question"]

        # Test vector search
        vector_results = await chunk_repo.search_chunks(question, limit=5)
        target_document_ids = {chunk.document_id for chunk, _ in vector_results}
        assert target_document.id in target_document_ids

        # Test FTS search
        fts_results = await chunk_repo.search_chunks_fts(question, limit=5)
        target_document_ids = {chunk.document_id for chunk, _ in fts_results}
        assert target_document.id in target_document_ids

    for i in range(num_documents):  # Test with first few documents
        target_document, doc_data = documents[i]
        question = doc_data["question"]

        # Test hybrid search
        hybrid_results = await chunk_repo.search_chunks_hybrid(question, limit=5)
        target_document_ids = {chunk.document_id for chunk, _ in hybrid_results}
        assert target_document.id in target_document_ids

    store.close()


@pytest.mark.asyncio
async def test_chunks_include_document_info():
    """Test that search results include document URI and metadata."""
    store = Store(":memory:")
    doc_repo = DocumentRepository(store)
    chunk_repo = ChunkRepository(store)

    # Create a document with URI and metadata
    document = Document(
        content="This is a test document with some content for searching.",
        uri="https://example.com/test.html",
        metadata={"title": "Test Document", "author": "Test Author"},
    )

    created_document = await doc_repo.create(document)

    # Search for chunks
    results = await chunk_repo.search_chunks_hybrid("test document", limit=1)

    assert len(results) > 0
    chunk, _ = results[0]

    # Verify the chunk includes document information
    assert chunk.document_uri == "https://example.com/test.html"
    assert chunk.document_meta == {"title": "Test Document", "author": "Test Author"}
    assert chunk.document_id == created_document.id

    store.close()
