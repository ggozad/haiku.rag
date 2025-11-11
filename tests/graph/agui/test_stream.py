"""Tests for stream_graph function."""

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from haiku.rag.graph.agui.emitter import AGUIEmitter
from haiku.rag.graph.agui.stream import stream_graph


class TestState(BaseModel):
    """Test state model."""

    value: int


@dataclass
class TestDeps:
    """Test dependencies."""

    agui_emitter: AGUIEmitter | None = None


class MockGraph:
    """Mock graph for testing."""

    def __init__(self, result: str | dict[str, str | int] = "done"):
        self.result = result
        self.run_called = False

    async def run(self, state, deps):  # type: ignore[no-untyped-def]
        """Mock run method."""
        self.run_called = True

        # Emit some events through the emitter
        if deps.agui_emitter:
            deps.agui_emitter.start_step("mock_step")
            deps.agui_emitter.update_activity("working", "Doing work")
            deps.agui_emitter.finish_step()

        return self.result


@pytest.mark.asyncio
async def test_stream_graph_basic():
    """Test basic graph streaming."""
    graph = MockGraph(result="success")
    state = TestState(value=1)
    deps = TestDeps()

    events = []
    async for event in stream_graph(graph, state, deps):
        events.append(event)

    # Should have collected events
    assert len(events) > 0

    # Should have RUN_STARTED and RUN_FINISHED
    event_types = [e["type"] for e in events]
    assert "RUN_STARTED" in event_types
    assert "RUN_FINISHED" in event_types

    # Graph should have been executed
    assert graph.run_called


@pytest.mark.asyncio
async def test_stream_graph_emits_initial_state():
    """Test that initial state is emitted."""
    graph = MockGraph()
    state = TestState(value=42)
    deps = TestDeps()

    events = []
    async for event in stream_graph(graph, state, deps):
        events.append(event)

    # Should have initial state snapshot
    state_snapshots = [e for e in events if e["type"] == "STATE_SNAPSHOT"]
    assert len(state_snapshots) > 0
    # First snapshot should be initial state
    assert state_snapshots[0]["snapshot"] == {"value": 42}


@pytest.mark.asyncio
async def test_stream_graph_emits_step_events():
    """Test that step events from graph are emitted."""
    graph = MockGraph()
    state = TestState(value=1)
    deps = TestDeps()

    events = []
    async for event in stream_graph(graph, state, deps):
        events.append(event)

    # Should have step events from MockGraph
    step_started = [e for e in events if e["type"] == "STEP_STARTED"]
    assert len(step_started) > 0
    assert step_started[0]["stepName"] == "mock_step"


@pytest.mark.asyncio
async def test_stream_graph_emits_activity():
    """Test that activity events from graph are emitted."""
    graph = MockGraph()
    state = TestState(value=1)
    deps = TestDeps()

    events = []
    async for event in stream_graph(graph, state, deps):
        events.append(event)

    # Should have activity events from MockGraph
    activities = [e for e in events if e["type"] == "ACTIVITY_SNAPSHOT"]
    assert len(activities) > 0
    assert activities[0]["content"] == "Doing work"


@pytest.mark.asyncio
async def test_stream_graph_handles_error():
    """Test that graph errors are captured and emitted."""

    class ErrorGraph:
        async def run(self, state, deps):
            raise ValueError("Test error")

    graph = ErrorGraph()
    state = TestState(value=1)
    deps = TestDeps()

    events = []
    async for event in stream_graph(graph, state, deps):
        events.append(event)

    # Should have error event
    errors = [e for e in events if e["type"] == "RUN_ERROR"]
    assert len(errors) > 0
    assert "Test error" in errors[0]["message"]


@pytest.mark.asyncio
async def test_stream_graph_closes_emitter():
    """Test that emitter is properly closed."""

    class NeverReturnsGraph:
        async def run(self, state, deps):
            # Don't return anything
            pass

    graph = NeverReturnsGraph()
    state = TestState(value=1)
    deps = TestDeps()

    events = []
    try:
        async for event in stream_graph(graph, state, deps):
            events.append(event)
    except RuntimeError:
        # Expected - graph didn't return a result
        pass

    # Should have error event about no result
    errors = [e for e in events if e["type"] == "RUN_ERROR"]
    assert len(errors) > 0


@pytest.mark.asyncio
async def test_stream_graph_without_emitter_support():
    """Test error when deps doesn't support agui_emitter."""

    @dataclass
    class BadDeps:
        pass

    graph = MockGraph()
    state = TestState(value=1)
    deps = BadDeps()

    with pytest.raises(TypeError, match="agui_emitter"):
        async for _ in stream_graph(graph, state, deps):  # type: ignore[arg-type]
            pass


@pytest.mark.asyncio
async def test_stream_graph_result_in_finish_event():
    """Test that graph result is included in RUN_FINISHED event."""
    graph = MockGraph(result={"status": "complete", "count": 42})
    state = TestState(value=1)
    deps = TestDeps()

    events = []
    async for event in stream_graph(graph, state, deps):
        events.append(event)

    # Find RUN_FINISHED event
    finished = [e for e in events if e["type"] == "RUN_FINISHED"]
    assert len(finished) == 1
    assert finished[0]["result"] == {"status": "complete", "count": 42}
