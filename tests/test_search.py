import pytest
from datasets import Dataset

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.store.models import SearchResult


@pytest.mark.vcr()
async def test_search_qa_corpus(qa_corpus: Dataset, temp_db_path):
    """Test that documents can be found by searching with their associated questions."""
    # Create client
    client = HaikuRAG(db_path=temp_db_path, config=Config, create=True)

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

        # Create the document with chunks and embeddings
        created_document = await client.create_document(content=document_text)
        documents.append((created_document, doc_data))

    # Test with first few unique documents

    for target_document, doc_data in documents:
        question = doc_data["question"]

        # Test vector search (limit=10 to accommodate different embedding models)
        vector_results = await client.chunk_repository.search(
            question, limit=10, search_type="vector"
        )
        target_document_ids = {chunk.document_id for chunk, _ in vector_results}
        assert target_document.id in target_document_ids

        # Test FTS search
        fts_results = await client.chunk_repository.search(
            question, limit=10, search_type="fts"
        )
        target_document_ids = {chunk.document_id for chunk, _ in fts_results}
        assert target_document.id in target_document_ids

        # Test hybrid search
        hybrid_results = await client.chunk_repository.search(
            question, limit=10, search_type="hybrid"
        )
        target_document_ids = {chunk.document_id for chunk, _ in hybrid_results}
        assert target_document.id in target_document_ids

    client.close()


@pytest.mark.vcr()
async def test_chunks_include_document_info(temp_db_path):
    """Test that search results include document URI and metadata."""
    client = HaikuRAG(db_path=temp_db_path, config=Config, create=True)

    # Create a document with URI and metadata
    created_document = await client.create_document(
        content="This is a test document with some content for searching.",
        uri="https://example.com/test.html",
        metadata={"title": "Test Document", "author": "Test Author"},
    )

    # Search for chunks
    results = await client.chunk_repository.search(
        "test document", limit=1, search_type="hybrid"
    )

    assert len(results) > 0
    chunk, score = results[0]

    # Test that score is valid
    assert isinstance(score, int | float), f"Score should be numeric, got {type(score)}"
    assert score >= 0, f"Score should be non-negative, got {score}"

    # Verify the chunk includes document information
    assert chunk.document_uri == "https://example.com/test.html"
    assert chunk.document_meta == {"title": "Test Document", "author": "Test Author"}
    assert chunk.document_id == created_document.id

    client.close()


@pytest.mark.vcr()
async def test_chunks_include_document_title(temp_db_path):
    """Test that search results include the parent document title when present."""
    client = HaikuRAG(db_path=temp_db_path, config=Config, create=True)

    # Create a document with URI and title
    await client.create_document(
        content="This is a test document with a custom title to verify enrichment.",
        uri="file:///tmp/title-test.md",
        title="My Custom Title",
    )

    # Perform a search that should find this document
    results = await client.chunk_repository.search(
        "custom title", limit=3, search_type="hybrid"
    )

    assert results, "Expected at least one search result"
    for chunk, _ in results:
        # All returned chunks for this doc should carry the document title
        if chunk.document_uri == "file:///tmp/title-test.md":
            assert chunk.document_title == "My Custom Title"

    client.close()


@pytest.mark.vcr()
async def test_search_score_types(temp_db_path):
    """Test that different search types return appropriate score ranges."""
    client = HaikuRAG(db_path=temp_db_path, config=Config, create=True)

    # Create multiple documents with different content
    documents_content = [
        "Machine learning algorithms are powerful tools for data analysis and pattern recognition.",
        "Deep learning neural networks can process complex datasets and identify hidden patterns.",
        "Natural language processing enables computers to understand and generate human text.",
        "Computer vision systems can interpret and analyze visual information from images.",
    ]

    for content in documents_content:
        await client.create_document(content=content)

    query = "machine learning"

    # Test vector search scores (should be converted from distances)
    vector_results = await client.chunk_repository.search(
        query, limit=3, search_type="vector"
    )
    assert len(vector_results) > 0
    vector_scores = [score for _, score in vector_results]

    # Test FTS search scores (should be native LanceDB FTS scores)
    fts_results = await client.chunk_repository.search(
        query, limit=3, search_type="fts"
    )
    assert len(fts_results) > 0
    fts_scores = [score for _, score in fts_results]

    # Test hybrid search scores (should be native LanceDB relevance scores)
    hybrid_results = await client.chunk_repository.search(
        query, limit=3, search_type="hybrid"
    )
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

    client.close()


@pytest.mark.vcr()
async def test_search_returns_search_result(temp_db_path):
    """Test that client.search() returns SearchResult with provenance info."""
    client = HaikuRAG(db_path=temp_db_path, config=Config, create=True)

    await client.create_document(
        content="Machine learning models can classify images with high accuracy.",
        uri="https://example.com/ml.html",
        title="ML Guide",
    )

    results = await client.search("machine learning", limit=3)

    assert len(results) > 0
    result = results[0]
    assert isinstance(result, SearchResult)
    assert result.content
    assert result.score > 0
    assert result.document_uri == "https://example.com/ml.html"
    assert result.document_title == "ML Guide"
    # page_numbers and headings come from chunk metadata
    assert isinstance(result.page_numbers, list)
    assert isinstance(result.labels, list)

    client.close()


@pytest.mark.vcr()
async def test_search_graceful_degradation(temp_db_path):
    """Test search works when docling data is unavailable."""
    from haiku.rag.store.models import Chunk

    client = HaikuRAG(db_path=temp_db_path, config=Config, create=True)

    # Import document with custom chunks (no docling document)
    custom_chunks = [
        Chunk(content="Custom chunk without docling metadata", metadata={}),
    ]
    docling_doc = await client.convert("Document with custom chunks")
    await client.import_document(
        docling_document=docling_doc,
        chunks=custom_chunks,
        uri="https://example.com/custom.html",
    )

    results = await client.search("custom chunk", limit=3)

    assert len(results) > 0
    result = results[0]
    assert isinstance(result, SearchResult)
    assert result.content
    # Metadata defaults should still work
    assert result.page_numbers == []
    assert result.labels == []

    client.close()


@pytest.mark.vcr()
async def test_search_result_format_includes_metadata(temp_db_path):
    """Test that formatted search results include document metadata."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        await client.create_document(
            content="Important information about machine learning algorithms.",
            title="ML Guide",
            uri="https://example.com/ml-guide",
        )

        results = await client.search("machine learning", limit=1)
        assert len(results) > 0

        formatted = results[0].format_for_agent()

        # Should include chunk ID and score
        assert "[" in formatted and "]" in formatted
        assert "score:" in formatted

        # Should include document title in Source
        assert "ML Guide" in formatted
        assert "Source:" in formatted

        # Should include content
        assert "Content:" in formatted
        assert "machine learning" in formatted.lower()
