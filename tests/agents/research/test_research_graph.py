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
