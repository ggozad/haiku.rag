from haiku.rag.agents.research.models import Citation
from haiku.rag.tools.models import AnalysisResult, QAResult


def test_qa_result_defaults():
    """Test QAResult has sensible defaults."""
    result = QAResult(question="What is X?", answer="X is Y.")
    assert result.confidence == 1.0
    assert result.citations == []


def test_qa_result_with_citations():
    """Test QAResult with citations."""
    citation = Citation(
        index=1,
        document_id="doc-1",
        chunk_id="chunk-1",
        document_uri="test.md",
        document_title="Test Doc",
        content="Citation content",
    )
    result = QAResult(
        question="What is X?",
        answer="X is Y.",
        confidence=0.9,
        citations=[citation],
    )
    assert result.confidence == 0.9
    assert len(result.citations) == 1
    assert result.citations[0].document_title == "Test Doc"


def test_qa_result_sources_property():
    """Test QAResult.sources returns unique source names."""
    citations = [
        Citation(
            document_id="doc-1",
            chunk_id="chunk-1",
            document_uri="doc1.md",
            document_title="Document One",
            content="Content 1",
        ),
        Citation(
            document_id="doc-1",
            chunk_id="chunk-2",
            document_uri="doc1.md",
            document_title="Document One",
            content="Content 2",
        ),
        Citation(
            document_id="doc-2",
            chunk_id="chunk-3",
            document_uri="doc2.md",
            document_title="Document Two",
            content="Content 3",
        ),
    ]
    result = QAResult(question="Q", answer="A", citations=citations)

    sources = result.sources
    assert len(sources) == 2
    assert "Document One" in sources
    assert "Document Two" in sources


def test_qa_result_sources_uses_uri_as_fallback():
    """Test QAResult.sources uses uri when title is None."""
    citations = [
        Citation(
            document_id="doc-1",
            chunk_id="chunk-1",
            document_uri="test.md",
            document_title=None,
            content="Content",
        ),
    ]
    result = QAResult(question="Q", answer="A", citations=citations)

    sources = result.sources
    assert sources == ["test.md"]


def test_analysis_result_defaults():
    """Test AnalysisResult has sensible defaults."""
    result = AnalysisResult(answer="The result is 42")
    assert result.code_executed is True
    assert result.execution_count == 0


def test_analysis_result_with_values():
    """Test AnalysisResult with explicit values."""
    result = AnalysisResult(
        answer="The result is 42",
        code_executed=True,
        execution_count=3,
    )
    assert result.answer == "The result is 42"
    assert result.code_executed is True
    assert result.execution_count == 3
