from haiku.rag.agents.research.models import Citation, SearchAnswer, resolve_citations
from haiku.rag.store.models import SearchResult


class TestCitation:
    """Tests for unified Citation class."""

    def test_citation_without_index(self):
        """Test Citation can be created without index (research graph use case)."""
        citation = Citation(
            document_id="doc-1",
            chunk_id="chunk-1",
            document_uri="test.md",
            document_title="Test Document",
            page_numbers=[1, 2],
            headings=["Introduction"],
            content="Test content",
        )
        assert citation.document_id == "doc-1"
        assert citation.chunk_id == "chunk-1"
        assert citation.document_uri == "test.md"
        assert citation.document_title == "Test Document"
        assert citation.page_numbers == [1, 2]
        assert citation.headings == ["Introduction"]
        assert citation.content == "Test content"
        assert citation.index is None

    def test_citation_with_index(self):
        """Test Citation can be created with index (chat use case)."""
        citation = Citation(
            index=1,
            document_id="doc-1",
            chunk_id="chunk-1",
            document_uri="test.md",
            content="Test content",
        )
        assert citation.index == 1
        assert citation.document_id == "doc-1"

    def test_citation_index_defaults_to_none(self):
        """Test Citation index defaults to None."""
        citation = Citation(
            document_id="doc-1",
            chunk_id="chunk-1",
            document_uri="test.md",
            content="Test content",
        )
        assert citation.index is None

    def test_citation_serialization_includes_index_when_set(self):
        """Test Citation serialization includes index when set."""
        citation = Citation(
            index=2,
            document_id="doc-1",
            chunk_id="chunk-1",
            document_uri="test.md",
            content="Test content",
        )
        data = citation.model_dump()
        assert data["index"] == 2

    def test_citation_deserialization_from_dict_with_index(self):
        """Test Citation can be deserialized from dict with index (AG-UI state sync)."""
        data = {
            "index": 1,
            "document_id": "doc-1",
            "chunk_id": "chunk-1",
            "document_uri": "test.md",
            "document_title": "Test Doc",
            "page_numbers": [1, 2],
            "headings": ["Intro"],
            "content": "Test content",
        }
        citation = Citation.model_validate(data)
        assert citation.index == 1
        assert citation.document_id == "doc-1"


class TestSearchAnswerPrimarySource:
    """Tests for SearchAnswer.primary_source property."""

    def test_primary_source_returns_title_when_available(self):
        """Test primary_source returns first citation's title."""
        answer = SearchAnswer(
            query="test query",
            answer="test answer",
            citations=[
                Citation(
                    document_id="doc-1",
                    chunk_id="chunk-1",
                    document_uri="test.md",
                    document_title="Test Document",
                    content="content",
                ),
            ],
        )
        assert answer.primary_source == "Test Document"

    def test_primary_source_returns_uri_when_no_title(self):
        """Test primary_source returns URI when title is None."""
        answer = SearchAnswer(
            query="test query",
            answer="test answer",
            citations=[
                Citation(
                    document_id="doc-1",
                    chunk_id="chunk-1",
                    document_uri="test.md",
                    document_title=None,
                    content="content",
                ),
            ],
        )
        assert answer.primary_source == "test.md"

    def test_primary_source_returns_none_when_no_citations(self):
        """Test primary_source returns None when no citations."""
        answer = SearchAnswer(
            query="test query",
            answer="test answer",
            citations=[],
        )
        assert answer.primary_source is None


class TestResolveCitations:
    """Tests for resolve_citations function."""

    def _make_result(self, chunk_id: str) -> SearchResult:
        return SearchResult(
            content="test content",
            score=1.0,
            chunk_id=chunk_id,
            document_id="doc-1",
            document_uri="test.md",
            document_title="Test Doc",
        )

    def test_resolves_exact_ids(self):
        results = [self._make_result("abc123")]
        citations = resolve_citations(["abc123"], results)
        assert len(citations) == 1
        assert citations[0].chunk_id == "abc123"

    def test_strips_brackets_from_ids(self):
        results = [self._make_result("abc123")]
        citations = resolve_citations(["[abc123]"], results)
        assert len(citations) == 1
        assert citations[0].chunk_id == "abc123"

    def test_skips_unmatched_ids(self):
        results = [self._make_result("abc123")]
        citations = resolve_citations(["nonexistent"], results)
        assert len(citations) == 0

    def test_empty_cited_chunks(self):
        results = [self._make_result("abc123")]
        citations = resolve_citations([], results)
        assert len(citations) == 0
