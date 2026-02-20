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
        max_concurrency=1,
    )

    deps = ResearchDeps(client=client)

    result = await graph.run(state=state, deps=deps)

    assert result is not None
    assert isinstance(result, ResearchReport)
    assert result.title
    assert result.executive_summary

    client.close()


def test_iterative_plan_result_model():
    """Test IterativePlanResult model validation."""
    from haiku.rag.agents.research.models import IterativePlanResult

    # Test complete state
    complete = IterativePlanResult(
        is_complete=True,
        next_question=None,
        reasoning="All aspects covered.",
    )
    assert complete.is_complete is True
    assert complete.next_question is None

    # Test continue state
    continue_result = IterativePlanResult(
        is_complete=False,
        next_question="What are the specific requirements?",
        reasoning="Need more details.",
    )
    assert continue_result.is_complete is False
    assert continue_result.next_question == "What are the specific requirements?"


def test_build_research_graph_returns_graph():
    """Test build_research_graph returns a valid Graph instance."""
    from pydantic_graph.beta import Graph

    graph = build_research_graph()
    assert graph is not None
    assert isinstance(graph, Graph)


def test_format_context_for_prompt_basic():
    """Test format_context_for_prompt with basic context."""
    from haiku.rag.agents.research.dependencies import ResearchContext
    from haiku.rag.agents.research.graph import format_context_for_prompt

    context = ResearchContext(original_question="What is X?")
    result = format_context_for_prompt(context)

    assert "<context>" in result
    assert "What is X?" in result


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
