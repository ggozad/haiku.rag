from pathlib import Path

import pytest

from haiku.rag.agents.chat.state import (
    MAX_QA_HISTORY,
    CitationInfo,
    QAResponse,
    _embedding_cache,
    _qa_cache_key,
    build_document_filter,
    format_conversation_context,
    rank_qa_history_by_similarity,
)
from haiku.rag.client import HaikuRAG


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent.parent / "cassettes" / "test_chat_state")


@pytest.mark.asyncio
async def test_rank_qa_history_empty():
    """Test empty history returns empty list."""
    # Create a mock embedder - we won't actually call it
    result = await rank_qa_history_by_similarity(
        current_question="What is this?",
        qa_history=[],
        embedder=None,  # type: ignore - won't be called for empty list
        top_k=5,
    )
    assert result == []


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_rank_qa_history_small_list(temp_db_path, allow_model_requests):
    """Test with history smaller than top_k returns all entries."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        embedder = client.chunk_repository.embedder

        # Create 3 Q&A pairs (less than top_k=5)
        qa_history = [
            QAResponse(question="What is Python?", answer="A programming language"),
            QAResponse(question="What is Java?", answer="Another programming language"),
            QAResponse(
                question="What is Rust?", answer="A systems programming language"
            ),
        ]

        result = await rank_qa_history_by_similarity(
            current_question="Tell me about Python",
            qa_history=qa_history,
            embedder=embedder,
            top_k=5,
        )

        # Should return all 3 entries since history < top_k
        assert len(result) == 3
        # All original entries should be present
        assert set(qa.question for qa in result) == {
            "What is Python?",
            "What is Java?",
            "What is Rust?",
        }


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_rank_qa_history_returns_top_k(temp_db_path, allow_model_requests):
    """Test ranking returns top-K most similar entries and populates cache."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        embedder = client.chunk_repository.embedder

        # Create 10 Q&A pairs on different topics
        qa_history = [
            QAResponse(
                question="What are the 11 class labels in DocLayNet?",
                answer="Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, and Title",
            ),
            QAResponse(
                question="How was the annotation process organized?",
                answer="The process had 4 phases with 40 dedicated annotators",
            ),
            QAResponse(
                question="What data sources were used?",
                answer="arXiv, government offices, company websites, financial reports and patents",
            ),
            QAResponse(
                question="How were pages selected?",
                answer="By selective subsampling with bias towards pages with figures or tables",
            ),
            QAResponse(
                question="What is the inter-annotator agreement?",
                answer="Computed as mAP@0.5-0.95 metric between pairwise annotations",
            ),
            QAResponse(
                question="What is machine learning?",
                answer="A field of AI that enables systems to learn from data",
            ),
            QAResponse(
                question="How does neural network training work?",
                answer="Through backpropagation and gradient descent",
            ),
            QAResponse(
                question="What is deep learning?",
                answer="A subset of ML using neural networks with many layers",
            ),
        ]

        # Clear cache to verify it gets populated
        _embedding_cache.clear()

        # Ask a question related to class labels (Q1)
        result = await rank_qa_history_by_similarity(
            current_question="Which class label has the highest count in DocLayNet?",
            qa_history=qa_history,
            embedder=embedder,
            top_k=5,
        )

        # Should return exactly 5 entries
        assert len(result) == 5

        # The class labels Q&A should be in the top 5 (it's most semantically similar)
        result_questions = [qa.question for qa in result]
        assert "What are the 11 class labels in DocLayNet?" in result_questions

        # Verify cache is populated for all Q/A pairs
        for qa in qa_history:
            cache_key = _qa_cache_key(qa.question, qa.answer)
            assert cache_key in _embedding_cache
            assert len(_embedding_cache[cache_key]) > 0


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_rank_qa_history_preserves_order(temp_db_path, allow_model_requests):
    """Test that ranking preserves original order among selected items."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        embedder = client.chunk_repository.embedder

        # Create Q&A pairs where multiple are similar
        qa_history = [
            QAResponse(
                question="What is Python?",
                answer="A programming language",
                citations=[
                    CitationInfo(
                        index=1,
                        document_id="doc1",
                        chunk_id="chunk1",
                        document_uri="python.md",
                        content="Python content",
                    )
                ],
            ),
            QAResponse(
                question="What is Java?",
                answer="Another programming language",
            ),
            QAResponse(
                question="How to use Python for data science?",
                answer="Use pandas, numpy, and scikit-learn",
            ),
        ]

        result = await rank_qa_history_by_similarity(
            current_question="Tell me about Python programming",
            qa_history=qa_history,
            embedder=embedder,
            top_k=3,
        )

        # All should be returned
        assert len(result) == 3

        # The two Python-related questions should be in the results
        result_questions = [qa.question for qa in result]
        assert "What is Python?" in result_questions
        assert "How to use Python for data science?" in result_questions


def test_format_conversation_context_empty():
    """Test format_conversation_context with empty history."""
    result = format_conversation_context([])
    assert result == ""


def test_format_conversation_context_with_history():
    """Test format_conversation_context formats qa_history as XML."""
    citation = CitationInfo(
        index=1,
        document_id="doc-123",
        chunk_id="chunk-456",
        document_uri="test.md",
        document_title="Test Document",
        content="Test content",
    )
    qa_history = [
        QAResponse(
            question="What is Python?",
            answer="A programming language",
            citations=[citation],
        ),
        QAResponse(
            question="What is Java?",
            answer="Another programming language",
        ),
    ]

    result = format_conversation_context(qa_history)

    assert "<conversation_context>" in result
    assert "previous_qa" in result
    assert "What is Python?" in result
    assert "A programming language" in result
    assert "What is Java?" in result
    assert "Another programming language" in result
    assert "Test Document" in result  # source from first citation


def test_build_document_filter_simple():
    """Test build_document_filter with simple name."""
    result = build_document_filter("mytest")
    assert "LOWER(uri) LIKE LOWER('%mytest%')" in result
    assert "LOWER(title) LIKE LOWER('%mytest%')" in result


def test_build_document_filter_with_spaces():
    """Test build_document_filter handles spaces correctly."""
    result = build_document_filter("TB MED 593")
    # Should include both the original (with spaces) and without spaces
    assert "LOWER(uri) LIKE LOWER('%TB MED 593%')" in result
    assert "LOWER(uri) LIKE LOWER('%TBMED593%')" in result
    assert "LOWER(title) LIKE LOWER('%TB MED 593%')" in result
    assert "LOWER(title) LIKE LOWER('%TBMED593%')" in result


def test_build_document_filter_escapes_quotes():
    """Test build_document_filter escapes single quotes."""
    result = build_document_filter("O'Reilly")
    # Single quotes should be doubled for SQL escaping
    assert "O''Reilly" in result


def test_max_qa_history_constant():
    """Test MAX_QA_HISTORY constant value."""
    assert MAX_QA_HISTORY == 50
