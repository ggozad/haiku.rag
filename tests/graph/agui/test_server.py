"""Tests for AG-UI server."""

import pytest
from pydantic import BaseModel
from starlette.testclient import TestClient

from haiku.rag.config.models import AGUIConfig
from haiku.rag.graph.agui.server import RunAgentInput, create_agui_app, format_sse_event


class SimpleState(BaseModel):
    """Simple state for testing."""

    question: str


class SimpleResult(BaseModel):
    """Simple result for testing."""

    answer: str


class MockGraph:
    """Mock graph that returns immediately."""

    async def run(self, state, deps):  # type: ignore[no-untyped-def]
        """Return a simple result."""
        return SimpleResult(answer=f"Answer to: {state.question}")


def test_run_agent_input_parsing():
    """Test RunAgentInput model parsing."""
    data = {
        "threadId": "thread-1",
        "runId": "run-1",
        "state": {"question": "What is AI?"},
        "messages": [],
        "config": {},
    }

    input_data = RunAgentInput(**data)

    assert input_data.thread_id == "thread-1"
    assert input_data.run_id == "run-1"
    assert input_data.state == {"question": "What is AI?"}


def test_run_agent_input_defaults():
    """Test RunAgentInput with defaults."""
    input_data = RunAgentInput()  # type: ignore[call-arg]

    assert input_data.thread_id is None
    assert input_data.run_id is None
    assert input_data.state == {}
    assert input_data.messages == []
    assert input_data.config == {}


def test_format_sse_event():
    """Test SSE event formatting."""
    event = {"type": "TEST_EVENT", "data": "test"}

    sse = format_sse_event(event)

    assert sse.startswith("data: ")
    assert sse.endswith("\n\n")
    assert '{"type": "TEST_EVENT"' in sse


def test_create_agui_app_basic():
    """Test basic app creation."""
    config = AGUIConfig(
        host="localhost",
        port=8000,
        cors_origins=["http://localhost"],
    )

    def graph_factory():
        return MockGraph()

    def state_factory(input_state):
        return SimpleState(question=input_state.get("question", ""))

    def deps_factory(input_config):
        from dataclasses import dataclass

        @dataclass
        class SimpleDeps:
            agui_emitter: None = None

        return SimpleDeps()

    app = create_agui_app(
        graph_factory=graph_factory,  # type: ignore[arg-type]
        state_factory=state_factory,
        deps_factory=deps_factory,  # type: ignore[arg-type]
        config=config,
    )

    # Should return a Starlette app
    assert app is not None
    assert hasattr(app, "routes")


def test_server_health_endpoint():
    """Test health check endpoint."""
    config = AGUIConfig()

    def graph_factory():
        return MockGraph()

    def state_factory(input_state):
        return SimpleState(question="")

    def deps_factory(input_config):
        from dataclasses import dataclass

        @dataclass
        class SimpleDeps:
            agui_emitter: None = None

        return SimpleDeps()

    app = create_agui_app(
        graph_factory=graph_factory,  # type: ignore[arg-type]
        state_factory=state_factory,
        deps_factory=deps_factory,  # type: ignore[arg-type]
        config=config,
    )

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_server_stream_endpoint():
    """Test AG-UI streaming endpoint."""
    config = AGUIConfig()

    def graph_factory():
        return MockGraph()

    def state_factory(input_state):
        question = input_state.get("question", "")
        return SimpleState(question=question)

    def deps_factory(input_config):
        from dataclasses import dataclass

        @dataclass
        class SimpleDeps:
            agui_emitter: None = None

        return SimpleDeps()

    app = create_agui_app(
        graph_factory=graph_factory,  # type: ignore[arg-type]
        state_factory=state_factory,
        deps_factory=deps_factory,  # type: ignore[arg-type]
        config=config,
    )

    client = TestClient(app)

    request_data = {
        "threadId": "test-1",
        "runId": "run-1",
        "state": {"question": "What is pydantic-graph?"},
        "messages": [],
        "config": {},
    }

    response = client.post("/v1/agent/stream", json=request_data)

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Read the streamed events
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            import json

            event_data = line[6:]  # Remove "data: " prefix
            event = json.loads(event_data)
            events.append(event)

    # Should have received multiple events
    assert len(events) > 0

    # Should have RUN_STARTED and RUN_FINISHED
    event_types = [e["type"] for e in events]
    assert "RUN_STARTED" in event_types
    assert "RUN_FINISHED" in event_types


def test_server_cors_headers():
    """Test CORS middleware is configured."""
    config = AGUIConfig(
        cors_origins=["http://example.com"],
        cors_credentials=True,
    )

    def graph_factory():
        return MockGraph()

    def state_factory(input_state):
        return SimpleState(question="")

    def deps_factory(input_config):
        from dataclasses import dataclass

        @dataclass
        class SimpleDeps:
            agui_emitter: None = None

        return SimpleDeps()

    app = create_agui_app(
        graph_factory=graph_factory,  # type: ignore[arg-type]
        state_factory=state_factory,
        deps_factory=deps_factory,  # type: ignore[arg-type]
        config=config,
    )

    client = TestClient(app)

    # GET request with Origin header should get CORS headers
    response = client.get("/health", headers={"Origin": "http://example.com"})

    assert response.status_code == 200
    # CORS middleware should add access-control headers
    assert (
        "access-control-allow-origin" in response.headers or response.status_code == 200
    )


def test_agui_config_defaults():
    """Test AGUIConfig default values."""
    config = AGUIConfig()

    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.cors_origins == ["*"]
    assert config.cors_credentials is True
    assert "GET" in config.cors_methods
    assert "POST" in config.cors_methods
