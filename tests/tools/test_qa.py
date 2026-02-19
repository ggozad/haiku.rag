from haiku.rag.agents.research.models import Citation
from haiku.rag.tools.qa import PRIOR_ANSWER_RELEVANCE_THRESHOLD, QAHistoryEntry


class TestQAHistoryEntry:
    """Tests for QAHistoryEntry model."""

    def test_defaults(self):
        """QAHistoryEntry has sensible defaults."""
        entry = QAHistoryEntry(question="What is X?", answer="X is Y.")
        assert entry.confidence == 0.9
        assert entry.citations == []
        assert entry.question_embedding is None

    def test_sources_property(self):
        """sources returns unique document titles."""
        citations = [
            Citation(
                document_id="d1",
                chunk_id="c1",
                document_uri="doc1.md",
                document_title="Document One",
                content="Content 1",
            ),
            Citation(
                document_id="d1",
                chunk_id="c2",
                document_uri="doc1.md",
                document_title="Document One",
                content="Content 2",
            ),
            Citation(
                document_id="d2",
                chunk_id="c3",
                document_uri="doc2.md",
                document_title="Document Two",
                content="Content 3",
            ),
        ]
        entry = QAHistoryEntry(question="Q", answer="A", citations=citations)
        sources = entry.sources
        assert len(sources) == 2
        assert "Document One" in sources
        assert "Document Two" in sources

    def test_sources_uses_uri_as_fallback(self):
        """sources uses uri when title is None."""
        citations = [
            Citation(
                document_id="d1",
                chunk_id="c1",
                document_uri="test.md",
                document_title=None,
                content="Content",
            ),
        ]
        entry = QAHistoryEntry(question="Q", answer="A", citations=citations)
        assert entry.sources == ["test.md"]

    def test_to_search_answer(self):
        """to_search_answer converts to SearchAnswer."""
        citation = Citation(
            document_id="d1",
            chunk_id="c1",
            document_uri="doc1.md",
            document_title="Doc",
            content="Content",
        )
        entry = QAHistoryEntry(
            question="What is X?",
            answer="X is Y.",
            confidence=0.85,
            citations=[citation],
        )
        sa = entry.to_search_answer()
        assert sa.query == "What is X?"
        assert sa.answer == "X is Y."
        assert sa.confidence == 0.85
        assert sa.cited_chunks == ["c1"]
        assert len(sa.citations) == 1

    def test_question_embedding_excluded_from_serialization(self):
        """question_embedding is excluded from model_dump."""
        entry = QAHistoryEntry(
            question="Q",
            answer="A",
            question_embedding=[0.1, 0.2],
        )
        data = entry.model_dump()
        assert "question_embedding" not in data


def test_prior_answer_relevance_threshold():
    """PRIOR_ANSWER_RELEVANCE_THRESHOLD is a sensible value."""
    assert 0 < PRIOR_ANSWER_RELEVANCE_THRESHOLD < 1
