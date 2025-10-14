import pytest
from pydantic_ai.models.test import TestModel

from haiku.rag.client import HaikuRAG
from haiku.rag.graph.nodes.plan import PlanNode
from haiku.rag.research.dependencies import ResearchContext
from haiku.rag.research.graph import (
    ResearchDeps,
    ResearchState,
    build_research_graph,
)
from haiku.rag.research.models import ResearchReport
from haiku.rag.research.stream import stream_research_graph


@pytest.mark.asyncio
async def test_graph_end_to_end_with_test_model(monkeypatch, temp_db_path):
    """Test research graph with mocked LLM using TestModel."""
    graph = build_research_graph()

    state = ResearchState(
        context=ResearchContext(original_question="What is haiku.rag?"),
        max_iterations=1,
        confidence_threshold=0.5,
        max_concurrency=2,
    )

    # Use real client but with TestModel for LLM calls
    client = HaikuRAG(temp_db_path)
    deps = ResearchDeps(client=client, console=None)

    # Mock get_model to return TestModel which generates valid schema-compliant data
    # Need to patch in all modules that import it
    def test_model_factory(provider, model):
        return TestModel()

    monkeypatch.setattr("haiku.rag.graph.common.get_model", test_model_factory)
    monkeypatch.setattr("haiku.rag.graph.nodes.plan.get_model", test_model_factory)
    monkeypatch.setattr("haiku.rag.graph.nodes.search.get_model", test_model_factory)
    monkeypatch.setattr("haiku.rag.graph.nodes.analysis.get_model", test_model_factory)
    monkeypatch.setattr(
        "haiku.rag.graph.nodes.synthesize.get_model", test_model_factory
    )

    start = PlanNode(provider="test", model="test")

    collected = []
    report = None
    async for event in stream_research_graph(graph, start, state, deps):
        collected.append(event)
        if event.type == "report":
            report = event.report
            break
        elif event.type == "error":
            pytest.fail(f"Graph execution failed: {event.error}")

    # TestModel will generate valid structured output for each node
    assert report is not None, (
        f"No report generated. Events collected: {[e.type for e in collected]}"
    )
    assert isinstance(report, ResearchReport)
    assert report.title is not None
    assert isinstance(report.title, str)
    assert any(evt.type == "log" for evt in collected)

    client.close()
