import asyncio

import pytest
from pydantic_ai.models.test import TestModel

from haiku.rag.client import HaikuRAG
from haiku.rag.graph.agui.stream import stream_graph
from haiku.rag.graph.research.dependencies import ResearchContext
from haiku.rag.graph.research.graph import build_research_graph
from haiku.rag.graph.research.state import ResearchDeps, ResearchState


def test_build_graph_and_state():
    graph = build_research_graph()
    assert graph is not None

    state = ResearchState(
        context=ResearchContext(
            original_question="What are the key features of haiku.rag?"
        ),
        max_iterations=1,
        confidence_threshold=0.8,
    )
    assert state.iterations == 0
    assert state.context.sub_questions == []


def test_async_loop_available():
    # Ensure an event loop can be created in test env
    loop = asyncio.new_event_loop()
    loop.close()


@pytest.mark.asyncio
async def test_graph_end_to_end_with_test_model(monkeypatch, temp_db_path):
    """Test research graph with mocked LLM using AG-UI events."""

    # Mock get_model to return TestModel which generates valid schema-compliant data
    def test_model_factory(_provider, _model, _config=None):
        return TestModel()

    monkeypatch.setattr("haiku.rag.graph.common.utils.get_model", test_model_factory)
    monkeypatch.setattr("haiku.rag.graph.research.graph.get_model", test_model_factory)

    graph = build_research_graph()

    state = ResearchState(
        context=ResearchContext(original_question="What is haiku.rag?"),
        max_iterations=1,
        confidence_threshold=0.5,
        max_concurrency=2,
    )

    # Use real client but with TestModel for LLM calls
    client = HaikuRAG(temp_db_path)
    deps = ResearchDeps(client=client)

    events = []
    result = None
    async for event in stream_graph(graph, state, deps):
        events.append(event)
        if event["type"] == "RUN_FINISHED":
            result = event["result"]
        elif event["type"] == "RUN_ERROR":
            pytest.fail(f"Graph execution failed: {event['message']}")

    # TestModel will generate valid structured output for each node
    assert result is not None, (
        f"No result. Events collected: {[e['type'] for e in events]}"
    )
    # Result is serialized as dict in AG-UI events
    assert isinstance(result, dict)
    assert "title" in result
    assert isinstance(result["title"], str)
    assert "executive_summary" in result
    assert "main_findings" in result

    # Verify AG-UI events were emitted
    event_types = [e["type"] for e in events]
    assert "RUN_STARTED" in event_types
    assert "RUN_FINISHED" in event_types
    assert "STATE_SNAPSHOT" in event_types
    assert "STEP_STARTED" in event_types

    client.close()
