import pytest
from datasets import Dataset

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.store.models import SearchResult


@pytest.mark.vcr()
async def test_search_qa_corpus(qa_corpus: Dataset, temp_db_path):
    """Test that documents can be found by searching with their associated questions."""
    async with HaikuRAG(db_path=temp_db_path, config=Config, create=True) as client:
        # Load unique documents (limited to 10)
        seen_documents = set()
        documents = []

        for doc_data in qa_corpus:
            if len(seen_documents) >= 10:
                break
            document_text = doc_data["document_extracted"]
            document_id = doc_data.get("document_id", "")

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


@pytest.mark.vcr()
async def test_search_chunk_includes_document_provenance(temp_db_path):
    """Test that raw chunk search results include document URI, metadata, and ID."""
    async with HaikuRAG(db_path=temp_db_path, config=Config, create=True) as client:
        # Create a document with URI and metadata but no title
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
        assert isinstance(score, int | float), (
            f"Score should be numeric, got {type(score)}"
        )
        assert score >= 0, f"Score should be non-negative, got {score}"

        # Verify the chunk includes document information
        assert chunk.document_uri == "https://example.com/test.html"
        assert chunk.document_meta == {
            "title": "Test Document",
            "author": "Test Author",
        }
        assert chunk.document_id == created_document.id
        assert chunk.document_title is None


@pytest.mark.vcr()
async def test_search_score_types(temp_db_path):
    """Test that different search types return appropriate score ranges."""
    async with HaikuRAG(db_path=temp_db_path, config=Config, create=True) as client:
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


@pytest.mark.vcr()
async def test_search_returns_search_result(temp_db_path):
    """Test that client.search() returns SearchResult with provenance info."""
    async with HaikuRAG(db_path=temp_db_path, config=Config, create=True) as client:
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
        assert result.chunk_id is not None
        assert result.document_id is not None
        # page_numbers and headings come from chunk metadata
        assert isinstance(result.page_numbers, list)
        assert isinstance(result.labels, list)
        assert len(result.labels) > 0


@pytest.mark.vcr()
async def test_search_graceful_degradation(temp_db_path):
    """Test search works when docling data is unavailable."""
    from haiku.rag.store.models import Chunk

    async with HaikuRAG(db_path=temp_db_path, config=Config, create=True) as client:
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

        # Format with rank (the way agents use it)
        formatted = results[0].format_for_agent(rank=1, total=1)

        # Should include chunk ID and rank
        assert "[" in formatted and "]" in formatted
        assert "[rank 1 of 1]" in formatted

        # Should include document title in Source
        assert "ML Guide" in formatted
        assert "Source:" in formatted

        # Should include content
        assert "Content:" in formatted
        assert "machine learning" in formatted.lower()


@pytest.mark.vcr()
async def test_fts_search_targets_content_fts_column(temp_db_path):
    """FTS search must target the content_fts column (where the FTS index
    lives and where contextualized heading prefixes end up) — not the raw
    content column. Regression guard against upstream default-column changes
    in LanceDB's nearest_to_text().
    """
    from haiku.rag.store.models.chunk import Chunk

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content="seed", uri="test://doc")
        assert doc.id is not None

        # Heading-only term — contextualization will prepend headings to the
        # body when populating content_fts, so this word ends up ONLY in
        # content_fts, not in the content column.
        heading_only_term = "zxqvjfoowizardry"
        chunk = Chunk(
            content="unrelated body text",
            document_id=doc.id,
            metadata={"headings": [heading_only_term]},
            embedding=[0.0] * client.store.embedder._vector_dim,
        )
        await client.chunk_repository.create(chunk)

        # FTS on the heading-only word must match via content_fts.
        results = await client.chunk_repository.search(
            heading_only_term, limit=5, search_type="fts"
        )
        assert any(c.content == "unrelated body text" for c, _ in results), (
            "FTS did not match a heading-only term — nearest_to_text is not "
            "targeting the content_fts column"
        )


def test_search_result_primary_label_prioritizes_structural_types():
    """Test _get_primary_label prioritizes structural labels correctly."""
    # Table should be prioritized
    result = SearchResult(
        content="test",
        score=0.5,
        chunk_id="c1",
        document_id="d1",
        labels=["paragraph", "table", "text"],
    )
    assert result._get_primary_label() == "table"

    # Code should be prioritized over paragraph
    result = SearchResult(
        content="test",
        score=0.5,
        chunk_id="c2",
        document_id="d2",
        labels=["paragraph", "code"],
    )
    assert result._get_primary_label() == "code"

    # list_item should be prioritized
    result = SearchResult(
        content="test",
        score=0.5,
        chunk_id="c3",
        document_id="d3",
        labels=["text", "list_item"],
    )
    assert result._get_primary_label() == "list_item"

    # Returns first label when no priority match
    result = SearchResult(
        content="test",
        score=0.5,
        chunk_id="c4",
        document_id="d4",
        labels=["paragraph", "text"],
    )
    assert result._get_primary_label() == "paragraph"

    # Returns None for empty labels
    result = SearchResult(
        content="test",
        score=0.5,
        chunk_id="c5",
        document_id="d5",
        labels=[],
    )
    assert result._get_primary_label() is None


# Image queries (bytes / PIL.Image)


@pytest.mark.asyncio
async def test_search_with_bytes_query_uses_multimodal_embedder(
    temp_db_path, monkeypatch
):
    """``client.search(bytes)`` embeds via ``embed_image`` and dispatches
    to vector-only chunk search (skipping FTS and reranker)."""
    from haiku.rag.embeddings import EmbedderWrapper
    from haiku.rag.store.models.chunk import Chunk

    image_calls: list[bytes] = []

    class StubMultimodal(EmbedderWrapper):
        supports_images = True

        def __init__(self):
            super().__init__(embedder=None, vector_dim=4)

        async def embed_image(self, image):
            image_calls.append(image)
            return [0.5, 0.5, 0.5, 0.5]

    monkeypatch.setattr(
        "haiku.rag.store.engine.get_embedder",
        lambda *a, **kw: StubMultimodal(),
    )

    received_kwargs: dict = {}

    async def fake_chunk_search(
        query="", limit=5, search_type="hybrid", filter=None, query_vector=None
    ):
        received_kwargs.update(
            {
                "query": query,
                "limit": limit,
                "search_type": search_type,
                "filter": filter,
                "query_vector": query_vector,
            }
        )
        return [
            (
                Chunk(
                    content="figure 1",
                    metadata={"labels": ["picture"], "doc_item_refs": ["#/pictures/0"]},
                ),
                0.91,
            )
        ]

    async with HaikuRAG(temp_db_path, create=True) as rag:
        rag.chunk_repository.search = fake_chunk_search  # type: ignore[method-assign]
        results = await rag.search(b"\x89PNG\r\n\x1a\n", limit=3, include_images=False)

    assert len(results) == 1
    assert results[0].score == 0.91
    # The bytes were sent through the image embedder once.
    assert image_calls == [b"\x89PNG\r\n\x1a\n"]
    # The chunk repo received a pre-computed vector and an empty text query.
    assert received_kwargs["query_vector"] == [0.5, 0.5, 0.5, 0.5]
    assert received_kwargs["query"] == ""


@pytest.mark.asyncio
async def test_reranker_built_once_across_searches(temp_db_path, monkeypatch):
    """The reranker is constructed once per client and reused across searches,
    rather than rebuilt (reloading model weights) on every query."""
    from haiku.rag.store.models.chunk import Chunk

    build_count = 0

    class StubReranker:
        async def rerank(self, query, chunks, top_n):
            return [(chunk, 1.0) for chunk in chunks][:top_n]

    def fake_get_reranker(config):
        nonlocal build_count
        build_count += 1
        return StubReranker()

    monkeypatch.setattr("haiku.rag.client.get_reranker", fake_get_reranker)

    async def fake_chunk_search(query, limit, search_type, filter):
        return [(Chunk(content="x", metadata={}), 0.5)]

    async with HaikuRAG(temp_db_path, create=True) as rag:
        rag.chunk_repository.search = fake_chunk_search  # type: ignore[method-assign]
        await rag.search("first", include_images=False)
        await rag.search("second", include_images=False)
        await rag.search("third", include_images=False)

    assert build_count == 1


@pytest.mark.asyncio
async def test_search_with_pil_image_works_like_bytes(temp_db_path, monkeypatch):
    from PIL import Image as PILImageModule

    from haiku.rag.embeddings import EmbedderWrapper
    from haiku.rag.store.models.chunk import Chunk

    seen_types: list[type] = []

    class StubMultimodal(EmbedderWrapper):
        supports_images = True

        def __init__(self):
            super().__init__(embedder=None, vector_dim=4)

        async def embed_image(self, image):
            seen_types.append(type(image))
            return [0.1] * 4

    monkeypatch.setattr(
        "haiku.rag.store.engine.get_embedder",
        lambda *a, **kw: StubMultimodal(),
    )

    async def fake_chunk_search(**kwargs):
        return [(Chunk(content="x", metadata={}), 1.0)]

    async with HaikuRAG(temp_db_path, create=True) as rag:
        rag.chunk_repository.search = fake_chunk_search  # type: ignore[method-assign]
        img = PILImageModule.new("RGB", (8, 8), "red")
        results = await rag.search(img, include_images=False)

    assert len(results) == 1
    assert seen_types == [PILImageModule.Image]


@pytest.mark.asyncio
async def test_search_with_bytes_query_raises_for_text_only_embedder(
    temp_db_path,
):
    """A text-only embedder configured for QA must reject image queries
    with a clear error rather than silently degrading."""
    async with HaikuRAG(temp_db_path, create=True) as rag:
        with pytest.raises(ValueError, match="multimodal embedder"):
            await rag.search(b"\x89PNG\r\n\x1a\n")


def _picture_only_result(
    self_ref: str, score: float, document_id: str = "doc-1"
) -> SearchResult:
    return SearchResult(
        content="x",
        score=score,
        chunk_id=f"chunk-{self_ref}-{score}",
        document_id=document_id,
        doc_item_refs=[self_ref],
        labels=["picture"],
    )


def test_dedup_keeps_higher_scoring_picture_chunk():
    """Two results referencing the same single picture self_ref collapse
    to the one with the higher score."""
    from haiku.rag.client.search import _dedup_picture_chunks

    text_chunk = _picture_only_result("#/pictures/0", score=0.7)
    pic_chunk = _picture_only_result("#/pictures/0", score=0.9)
    other = _picture_only_result("#/pictures/1", score=0.6)

    deduped = _dedup_picture_chunks([text_chunk, pic_chunk, other])

    assert len(deduped) == 2
    chosen = next(r for r in deduped if r.doc_item_refs == ["#/pictures/0"])
    assert chosen.score == 0.9
    assert any(r.doc_item_refs == ["#/pictures/1"] for r in deduped)


def test_dedup_preserves_wider_chunks_referencing_same_picture():
    """A wider chunk that contains the picture plus surrounding items
    is independent signal — keep it alongside a picture-only chunk."""
    from haiku.rag.client.search import _dedup_picture_chunks

    pic_only = _picture_only_result("#/pictures/0", score=0.9)
    wider = SearchResult(
        content="surrounding paragraph text and a figure",
        score=0.7,
        chunk_id="wider",
        document_id="doc-1",
        doc_item_refs=["#/texts/3", "#/pictures/0", "#/texts/4"],
        labels=["text", "picture", "text"],
    )

    deduped = _dedup_picture_chunks([pic_only, wider])
    assert len(deduped) == 2


def test_dedup_does_not_collapse_across_documents():
    """Same self_ref in different documents is different content."""
    from haiku.rag.client.search import _dedup_picture_chunks

    a = _picture_only_result("#/pictures/0", score=0.5, document_id="doc-1")
    b = _picture_only_result("#/pictures/0", score=0.9, document_id="doc-2")

    deduped = _dedup_picture_chunks([a, b])
    assert len(deduped) == 2
