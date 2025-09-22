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

    # Load unique documents (limited to 10)
    seen_documents = set()
    documents = []

    for doc_data in qa_corpus:
        if len(seen_documents) >= 10:
            break
        document_text = doc_data["document_extracted"]  # type: ignore
        document_id = doc_data.get("document_id", "")  # type: ignore

        if document_id in seen_documents:
            continue
        seen_documents.add(document_id)

        # Create a Document instance
        document = Document(content=document_text)

        # Create the document with chunks and embeddings
        from haiku.rag.utils import text_to_docling_document

        docling_document = text_to_docling_document(document_text, name="test.md")
        created_document = await doc_repo._create_with_docling(
            document, docling_document
        )
        documents.append((created_document, doc_data))

    # Test with first few unique documents

    for target_document, doc_data in documents:
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
    chunk, score = results[0]

    # Test that score is valid
    assert isinstance(score, int | float), f"Score should be numeric, got {type(score)}"
    assert score >= 0, f"Score should be non-negative, got {score}"

    # Verify the chunk includes document information
    assert chunk.document_uri == "https://example.com/test.html"
    assert chunk.document_meta == {"title": "Test Document", "author": "Test Author"}
    assert chunk.document_id == created_document.id

    store.close()


@pytest.mark.asyncio
async def test_chunks_include_document_title(temp_db_path):
    """Test that search results include the parent document title when present."""
    store = Store(temp_db_path)
    doc_repo = DocumentRepository(store)
    chunk_repo = ChunkRepository(store)

    # Create a document with URI and title
    document = Document(
        content="This is a test document with a custom title to verify enrichment.",
        uri="file:///tmp/title-test.md",
        title="My Custom Title",
    )

    # Create the document with chunks
    from haiku.rag.utils import text_to_docling_document

    dl = text_to_docling_document(document.content, name="title-test.md")
    await doc_repo._create_with_docling(document, dl)

    # Perform a search that should find this document
    results = await chunk_repo.search("custom title", limit=3, search_type="hybrid")

    assert results, "Expected at least one search result"
    for chunk, _ in results:
        # All returned chunks for this doc should carry the document title
        if chunk.document_uri == "file:///tmp/title-test.md":
            assert chunk.document_title == "My Custom Title"

    store.close()


@pytest.mark.asyncio
async def test_search_score_types(temp_db_path):
    """Test that different search types return appropriate score ranges."""
    store = Store(temp_db_path)
    doc_repo = DocumentRepository(store)
    chunk_repo = ChunkRepository(store)

    # Create multiple documents with different content
    documents_content = [
        "Machine learning algorithms are powerful tools for data analysis and pattern recognition.",
        "Deep learning neural networks can process complex datasets and identify hidden patterns.",
        "Natural language processing enables computers to understand and generate human text.",
        "Computer vision systems can interpret and analyze visual information from images.",
    ]

    for content in documents_content:
        document = Document(content=content)
        from haiku.rag.utils import text_to_docling_document

        docling_document = text_to_docling_document(content, name="test.md")
        await doc_repo._create_with_docling(document, docling_document)

    query = "machine learning"

    # Test vector search scores (should be converted from distances)
    vector_results = await chunk_repo.search(query, limit=3, search_type="vector")
    assert len(vector_results) > 0
    vector_scores = [score for _, score in vector_results]

    # Test FTS search scores (should be native LanceDB FTS scores)
    fts_results = await chunk_repo.search(query, limit=3, search_type="fts")
    assert len(fts_results) > 0
    fts_scores = [score for _, score in fts_results]

    # Test hybrid search scores (should be native LanceDB relevance scores)
    hybrid_results = await chunk_repo.search(query, limit=3, search_type="hybrid")
    assert len(hybrid_results) > 0
    hybrid_scores = [score for _, score in hybrid_results]

    # All scores should be numeric and non-negative
    for scores, search_type in [
        (vector_scores, "vector"),
        (fts_scores, "fts"),
        (hybrid_scores, "hybrid"),
    ]:
        for score in scores:
            assert isinstance(score, int | float), (
                f"{search_type} score should be numeric"
            )
            assert score >= 0, f"{search_type} score should be non-negative"

    # Vector scores should typically be small (0-1 range due to distance conversion)
    assert all(0 <= score <= 1 for score in vector_scores), (
        "Vector scores should be in 0-1 range"
    )

    # Scores should be sorted in descending order (most relevant first)
    for scores, search_type in [
        (vector_scores, "vector"),
        (fts_scores, "fts"),
        (hybrid_scores, "hybrid"),
    ]:
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"{search_type} results should be sorted by score descending"
            )

    store.close()
