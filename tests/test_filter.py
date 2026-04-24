import pytest

from haiku.rag.client import HaikuRAG


@pytest.mark.vcr()
async def test_search_with_uri_filter(temp_db_path):
    """Test filtering by document URI."""
    async with HaikuRAG(db_path=temp_db_path, create=True) as client:
        # Add multiple test documents
        await client.create_document(
            content="Python tutorial content",
            uri="https://example.com/python.html",
            title="Python Guide",
        )
        await client.create_document(
            content=" Java tutorial content",
            uri="https://other.com/java.html",
            title="Java Guide",
        )

        # Filter by URI pattern
        results = await client.search(
            "tutorial", limit=5, filter="uri LIKE '%example.com%'"
        )
        assert len(results) > 0
        for result in results:
            assert result.document_uri is not None
            assert "example.com" in result.document_uri

        # Filter by exact URI
        results = await client.search(
            "tutorial", limit=5, filter="uri = 'https://other.com/java.html'"
        )
        assert len(results) > 0
        for result in results:
            assert result.document_uri == "https://other.com/java.html"


@pytest.mark.vcr()
async def test_search_with_title_filter(temp_db_path):
    """Test filtering by document title."""
    async with HaikuRAG(db_path=temp_db_path, create=True) as client:
        # Add test documents
        await client.create_document(
            content="Programming content",
            uri="https://example.com/doc1.html",
            title="Python Programming",
        )
        await client.create_document(
            content="Programming content",
            uri="https://example.com/doc2.html",
            title="Java Programming",
        )

        # Filter by title pattern
        results = await client.search(
            "programming", limit=5, filter="title LIKE '%Python%'"
        )
        assert len(results) > 0
        for result in results:
            assert result.document_title is not None
            assert "Python" in result.document_title


@pytest.mark.vcr()
async def test_search_with_combined_filters(temp_db_path):
    """Test filtering with AND/OR conditions."""
    async with HaikuRAG(db_path=temp_db_path, create=True) as client:
        # Add test documents
        await client.create_document(
            content="Content about AI",
            uri="https://arxiv.org/paper1.pdf",
            title="Machine Learning Paper",
        )
        await client.create_document(
            content="Content about AI",
            uri="https://example.com/tutorial.html",
            title="AI Tutorial",
        )
        await client.create_document(
            content="Content about AI",
            uri="https://arxiv.org/paper2.pdf",
            title="Deep Learning Paper",
        )

        # Filter with AND condition
        results = await client.search(
            "AI", limit=5, filter="uri LIKE '%arxiv%' AND title LIKE '%Machine%'"
        )
        assert len(results) > 0
        for result in results:
            assert result.document_uri is not None
            assert result.document_title is not None
            assert "arxiv" in result.document_uri
            assert "Machine" in result.document_title

        # Filter with OR condition (if supported)
        results = await client.search(
            "AI", limit=5, filter="title LIKE '%Tutorial%' OR title LIKE '%Deep%'"
        )
        assert len(results) > 0


@pytest.mark.vcr()
async def test_search_with_no_matching_filter(temp_db_path):
    """Test that search returns empty results when filter matches no documents."""
    async with HaikuRAG(db_path=temp_db_path, create=True) as client:
        # Add a test document
        await client.create_document(
            content="Test content",
            uri="https://example.com/test.html",
            title="Test Document",
        )

        # Search with non-matching filter
        results = await client.search(
            "test", limit=5, filter="uri = 'https://nonexistent.com/doc.html'"
        )
        assert len(results) == 0


@pytest.mark.vcr()
async def test_search_with_invalid_filter(temp_db_path):
    """Test that invalid filter syntax raises an appropriate error."""
    async with HaikuRAG(db_path=temp_db_path, create=True) as client:
        # Add a test document
        await client.create_document(
            content="Test content",
            uri="https://example.com/test.html",
            title="Test Document",
        )

        # Invalid filter should raise RuntimeError
        with pytest.raises(RuntimeError, match="No field named invalid"):
            await client.search("test", limit=5, filter="invalid = 'value'")


@pytest.mark.vcr()
async def test_search_filter_with_all_search_types(temp_db_path):
    """Test that filtering works with all search types (vector, fts, hybrid)."""
    async with HaikuRAG(db_path=temp_db_path, create=True) as client:
        await client.create_document(
            content="Machine learning is a subset of artificial intelligence",
            uri="https://ai.example.com/ml.html",
            title="ML Guide",
        )
        await client.create_document(
            content="Deep learning uses neural networks",
            uri="https://other.com/dl.html",
            title="DL Guide",
        )

        # Test vector search with filter
        results = await client.search(
            "machine learning",
            limit=5,
            search_type="vector",
            filter="uri LIKE '%ai.example%'",
        )
        assert len(results) > 0
        for result in results:
            assert result.document_uri is not None
            assert "ai.example" in result.document_uri

        # Test FTS search with filter
        results = await client.search(
            "learning", limit=5, search_type="fts", filter="title = 'ML Guide'"
        )
        assert all(r.document_title == "ML Guide" for r in results)

        # Test hybrid search with filter (default)
        results = await client.search(
            "neural networks",
            limit=5,
            search_type="hybrid",
            filter="uri LIKE '%other.com%'",
        )
        for result in results:
            assert result.document_uri is not None
            assert "other.com" in result.document_uri


@pytest.mark.vcr()
async def test_search_with_filter_returns_full_limit(temp_db_path):
    """Regression: filter + limit must return up to `limit` matching chunks
    even when non-matching chunks would dominate the top-N window.

    Previously the filter path materialized LanceDB's default top-N window
    (~10), filtered to matching document_ids in pandas, then took `head(limit)`.
    If the top-N window was dominated by non-matching chunks, the caller got
    silently fewer results than requested — even when plenty of matching
    chunks existed further down the ranking. This test puts the target
    document behind many distractor documents and asserts we still get the
    requested count back.
    """
    async with HaikuRAG(db_path=temp_db_path, create=True) as client:
        for i in range(12):
            await client.create_document(
                content=(
                    "machine learning neural network deep learning model "
                    "machine learning neural network deep learning model "
                    "machine learning neural network deep learning model"
                ),
                uri=f"https://distractor.com/doc{i}.html",
                title=f"Distractor {i}",
            )

        await client.create_document(
            content="one passing mention of machine learning here",
            uri="https://target.com/one.html",
            title="Target One",
        )

        results = await client.search(
            "machine learning",
            limit=5,
            search_type="fts",
            filter="uri LIKE '%target.com%'",
        )

        assert len(results) == 1
        assert results[0].document_uri == "https://target.com/one.html"
