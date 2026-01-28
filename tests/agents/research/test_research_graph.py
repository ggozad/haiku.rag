from pathlib import Path

import pytest

from haiku.rag.agents.research.dependencies import ResearchContext
from haiku.rag.agents.research.graph import build_research_graph
from haiku.rag.agents.research.models import ResearchReport
from haiku.rag.agents.research.state import ResearchDeps, ResearchState
from haiku.rag.client import HaikuRAG


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(
        Path(__file__).parent.parent.parent / "cassettes" / "test_research_graph"
    )


@pytest.mark.vcr()
async def test_graph_end_to_end(allow_model_requests, temp_db_path, qa_corpus):
    """Test research graph with real LLM calls recorded via VCR."""
    graph = build_research_graph()

    client = HaikuRAG(temp_db_path, create=True)
    doc = qa_corpus[0]
    await client.create_document(
        content=doc["document_extracted"], uri=doc["document_id"]
    )

    state = ResearchState(
        context=ResearchContext(original_question=doc["question"]),
        max_iterations=1,
        confidence_threshold=0.5,
        max_concurrency=1,
    )

    deps = ResearchDeps(client=client)

    result = await graph.run(state=state, deps=deps)

    assert result is not None
    assert isinstance(result, ResearchReport)
    assert result.title
    assert result.executive_summary

    client.close()


def test_research_plan_allows_empty_sub_questions():
    """Test ResearchPlan accepts empty sub_questions when context is sufficient."""
    from haiku.rag.agents.research.models import ResearchPlan

    plan = ResearchPlan(sub_questions=[])
    assert plan.sub_questions == []


def test_research_plan_rejects_too_many_sub_questions():
    """Test ResearchPlan rejects more than 12 sub_questions."""
    from pydantic import ValidationError

    from haiku.rag.agents.research.models import ResearchPlan

    with pytest.raises(ValidationError, match="Cannot have more than 12"):
        ResearchPlan(sub_questions=[f"q{i}" for i in range(13)])


# =============================================================================
# Conversational Graph Tests
# =============================================================================


def test_build_conversational_graph_returns_graph():
    """Test build_conversational_graph returns a valid Graph instance."""
    from pydantic_graph.beta import Graph

    from haiku.rag.agents.research.graph import build_conversational_graph

    graph = build_conversational_graph()
    assert graph is not None
    assert isinstance(graph, Graph)


def test_conversational_answer_model():
    """Test ConversationalAnswer model can be created with all fields."""
    from haiku.rag.agents.research.models import Citation, ConversationalAnswer

    citation = Citation(
        index=1,
        document_id="doc-1",
        chunk_id="chunk-1",
        document_uri="test.md",
        document_title="Test Doc",
        content="Test content",
    )

    answer = ConversationalAnswer(
        answer="The answer is 42.",
        citations=[citation],
        confidence=0.95,
    )

    assert answer.answer == "The answer is 42."
    assert len(answer.citations) == 1
    assert answer.confidence == 0.95


def test_conversational_answer_default_values():
    """Test ConversationalAnswer uses correct default values."""
    from haiku.rag.agents.research.models import ConversationalAnswer

    answer = ConversationalAnswer(answer="Just the answer.")

    assert answer.answer == "Just the answer."
    assert answer.citations == []
    assert answer.confidence == 1.0


def test_format_context_for_prompt_basic():
    """Test format_context_for_prompt with basic context."""
    from haiku.rag.agents.research.dependencies import ResearchContext
    from haiku.rag.agents.research.graph import format_context_for_prompt

    context = ResearchContext(original_question="What is X?")
    result = format_context_for_prompt(context)

    assert "<context>" in result
    assert "What is X?" in result


def test_format_context_for_prompt_with_session_context():
    """Test format_context_for_prompt includes session_context as background."""
    from haiku.rag.agents.research.dependencies import ResearchContext
    from haiku.rag.agents.research.graph import format_context_for_prompt

    context = ResearchContext(
        original_question="What is Y?",
        session_context="Previous discussion about topic Z.",
    )
    result = format_context_for_prompt(context)

    assert "<background>" in result
    assert "Previous discussion" in result
    assert "What is Y?" in result


def test_format_context_for_prompt_excludes_pending_questions():
    """Test format_context_for_prompt can exclude pending questions."""
    from haiku.rag.agents.research.dependencies import ResearchContext
    from haiku.rag.agents.research.graph import format_context_for_prompt

    context = ResearchContext(
        original_question="Main question?",
        sub_questions=["Sub Q1?", "Sub Q2?"],
    )

    # With pending questions (default)
    with_pending = format_context_for_prompt(context, include_pending_questions=True)
    assert "Sub Q1?" in with_pending

    # Without pending questions (for synthesis)
    without_pending = format_context_for_prompt(
        context, include_pending_questions=False
    )
    assert "Sub Q1?" not in without_pending


def test_format_context_for_prompt_with_prior_answers():
    """Test format_context_for_prompt includes prior_answers."""
    from haiku.rag.agents.research.dependencies import ResearchContext
    from haiku.rag.agents.research.graph import format_context_for_prompt
    from haiku.rag.agents.research.models import SearchAnswer

    context = ResearchContext(original_question="Main question?")
    context.add_qa_response(
        SearchAnswer(
            query="Sub question?",
            answer="The answer is here.",
            confidence=0.9,
        )
    )

    result = format_context_for_prompt(context)

    assert "<prior_answers>" in result
    assert "Sub question?" in result
    assert "The answer is here." in result


def test_format_context_for_prompt_with_citations():
    """Test format_context_for_prompt includes available_citations when requested."""
    from haiku.rag.agents.research.dependencies import ResearchContext
    from haiku.rag.agents.research.graph import format_context_for_prompt
    from haiku.rag.agents.research.models import Citation, SearchAnswer

    context = ResearchContext(original_question="Main question?")
    context.add_qa_response(
        SearchAnswer(
            query="Sub question?",
            answer="The answer is here.",
            confidence=0.9,
            cited_chunks=["chunk-123", "chunk-456"],
            citations=[
                Citation(
                    document_id="doc-1",
                    chunk_id="chunk-123",
                    document_uri="test://doc1",
                    document_title="Test Document",
                    content="This is the chunk content.",
                ),
                Citation(
                    document_id="doc-1",
                    chunk_id="chunk-456",
                    document_uri="test://doc1",
                    document_title="Test Document",
                    content="More chunk content here.",
                ),
            ],
        )
    )

    result_without = format_context_for_prompt(context, include_citations=False)
    assert "<available_citations>" not in result_without

    result_with = format_context_for_prompt(context, include_citations=True)
    assert "<available_citations>" in result_with
    assert "chunk-123" in result_with
    assert "chunk-456" in result_with
    assert "Test Document" in result_with
    assert "This is the chunk content." in result_with
