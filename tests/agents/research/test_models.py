from haiku.rag.agents.research.models import Citation, SearchAnswer


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
        """Test Citation can be created with index (chat agent use case)."""
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


class TestFormatContextMerged:
    """Tests for merged format_context_for_prompt function."""

    def test_format_context_includes_pending_questions_by_default(self):
        """Test format_context_for_prompt includes pending_questions by default."""
        from haiku.rag.agents.research.dependencies import ResearchContext
        from haiku.rag.agents.research.graph import format_context_for_prompt

        context = ResearchContext(
            original_question="What is X?",
            sub_questions=["What is A?", "What is B?"],
        )
        result = format_context_for_prompt(context)
        assert "<pending_questions>" in result
        assert "What is A?" in result
        assert "What is B?" in result

    def test_format_context_excludes_pending_questions_when_flag_false(self):
        """Test format_context_for_prompt excludes pending_questions when flag is False."""
        from haiku.rag.agents.research.dependencies import ResearchContext
        from haiku.rag.agents.research.graph import format_context_for_prompt

        context = ResearchContext(
            original_question="What is X?",
            sub_questions=["What is A?", "What is B?"],
        )
        result = format_context_for_prompt(context, include_pending_questions=False)
        assert "<pending_questions>" not in result
        assert "What is A?" not in result

    def test_format_context_uses_primary_source_helper(self):
        """Test format_context_for_prompt uses primary_source from SearchAnswer."""
        from haiku.rag.agents.research.dependencies import ResearchContext
        from haiku.rag.agents.research.graph import format_context_for_prompt

        context = ResearchContext(
            original_question="What is X?",
        )
        # Add a QA response with citation
        answer = SearchAnswer(
            query="What is A?",
            answer="A is...",
            confidence=0.9,
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
        context.add_qa_response(answer)

        result = format_context_for_prompt(context)
        assert "Test Document" in result
