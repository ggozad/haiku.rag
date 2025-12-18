import asyncio

import pytest
from pydantic_ai.models.test import TestModel

from haiku.rag.client import HaikuRAG
from haiku.rag.graph.agui.stream import stream_graph
from haiku.rag.graph.research.dependencies import ResearchContext
from haiku.rag.graph.research.graph import build_research_graph
from haiku.rag.graph.research.state import HumanDecision, ResearchDeps, ResearchState


@pytest.mark.asyncio
async def test_graph_end_to_end_with_test_model(monkeypatch, temp_db_path):
    """Test research graph with mocked LLM using AG-UI events."""

    # Mock get_model to return TestModel which generates valid schema-compliant data
    def test_model_factory(_provider, _model, _config=None):
        return TestModel()

    # Patch all locations where get_model is imported
    monkeypatch.setattr("haiku.rag.utils.get_model", test_model_factory)
    monkeypatch.setattr("haiku.rag.graph.research.graph.get_model", test_model_factory)

    graph = build_research_graph()

    state = ResearchState(
        context=ResearchContext(original_question="What is haiku.rag?"),
        max_iterations=1,
        confidence_threshold=0.5,
        max_concurrency=2,
    )

    # Use real client but with TestModel for LLM calls
    client = HaikuRAG(temp_db_path, create=True)
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


@pytest.mark.asyncio
async def test_interactive_graph_with_human_decision(monkeypatch, temp_db_path):
    """Test interactive research graph pauses and resumes with human decisions."""

    # Mock get_model to return TestModel
    def test_model_factory(_provider, _model, _config=None):
        return TestModel()

    monkeypatch.setattr("haiku.rag.utils.get_model", test_model_factory)
    monkeypatch.setattr("haiku.rag.graph.research.graph.get_model", test_model_factory)

    # Build interactive graph
    graph = build_research_graph(interactive=True)

    state = ResearchState(
        context=ResearchContext(original_question="What is haiku.rag?"),
        max_iterations=1,
        confidence_threshold=0.5,
        max_concurrency=2,
    )

    # Create human input queue
    human_input_queue: asyncio.Queue[HumanDecision] = asyncio.Queue()

    client = HaikuRAG(temp_db_path, create=True)
    deps = ResearchDeps(
        client=client,
        human_input_queue=human_input_queue,
        interactive=True,
    )

    events = []
    tool_call_received = asyncio.Event()
    result = None

    async def run_graph():
        nonlocal result
        async for event in stream_graph(graph, state, deps):
            events.append(event)
            if event["type"] == "TOOL_CALL_START":
                tool_name = event.get("toolCallName")
                if tool_name == "human_decision":
                    tool_call_received.set()
            elif event["type"] == "RUN_FINISHED":
                result = event["result"]
            elif event["type"] == "RUN_ERROR":
                pytest.fail(f"Graph execution failed: {event['message']}")

    async def send_decisions():
        # Wait for first tool call (after planning)
        await asyncio.wait_for(tool_call_received.wait(), timeout=30)
        tool_call_received.clear()

        # Send search decision
        await human_input_queue.put(HumanDecision(action="search"))

        # Wait for second tool call (after search cycle)
        await asyncio.wait_for(tool_call_received.wait(), timeout=30)

        # Send synthesize decision
        await human_input_queue.put(HumanDecision(action="synthesize"))

    # Run graph and decision sender concurrently
    await asyncio.gather(run_graph(), send_decisions())

    # Verify result
    assert result is not None, (
        f"No result. Events collected: {[e['type'] for e in events]}"
    )
    assert isinstance(result, dict)
    assert "title" in result

    # Verify human_decision tool calls were emitted
    event_types = [e["type"] for e in events]
    assert "TOOL_CALL_START" in event_types
    assert "TOOL_CALL_END" in event_types

    client.close()
