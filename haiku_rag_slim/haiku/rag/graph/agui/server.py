"""AG-UI HTTP server implementation for graph execution."""

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from haiku.rag.config.models import AppConfig

from pydantic import BaseModel, Field
from pydantic_graph.beta import Graph
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from haiku.rag.config.models import AGUIConfig
from haiku.rag.graph.agui.emitter import AGUIEmitter
from haiku.rag.graph.agui.events import AGUIEvent
from haiku.rag.graph.agui.stream import stream_graph


class GraphDeps(Protocol):
    """Protocol for graph dependencies that support AG-UI emission."""

    agui_emitter: AGUIEmitter[Any, Any] | None


class RunAgentInput(BaseModel):
    """AG-UI protocol run agent input.

    See: https://docs.ag-ui.com/concepts/agents#runagentinput
    """

    thread_id: str | None = Field(None, alias="threadId")
    run_id: str | None = Field(None, alias="runId")
    state: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)


def create_agui_app(
    graph_factory: Callable[[], Graph],
    state_factory: Callable[[dict[str, Any]], BaseModel],
    deps_factory: Callable[[dict[str, Any]], GraphDeps],
    config: AGUIConfig,
) -> Starlette:
    """Create Starlette app with AG-UI endpoint.

    Args:
        graph_factory: Factory function to create graph instance
        state_factory: Factory to create initial state from input
        deps_factory: Factory to create graph dependencies
        config: AG-UI server configuration

    Returns:
        Starlette application with AG-UI endpoints
    """

    async def event_stream(
        input_data: RunAgentInput,
    ) -> AsyncIterator[str]:
        """Generate SSE event stream from graph execution.

        Yields:
            Server-Sent Events formatted strings
        """
        # Create graph, state, and dependencies
        graph = graph_factory()

        # Create initial state from input
        initial_state = state_factory(input_data.state)

        # Create dependencies (may use config from input)
        deps = deps_factory(input_data.config)

        # Execute graph and stream events
        async for event in stream_graph(graph, initial_state, deps):
            # Format as SSE event
            event_data = format_sse_event(event)
            yield event_data

    async def stream_agent(request: Request) -> StreamingResponse:
        """AG-UI agent stream endpoint.

        Accepts AG-UI RunAgentInput and streams events via SSE.
        """
        # Parse request body
        body = await request.json()
        input_data = RunAgentInput(**body)

        # Return SSE stream
        return StreamingResponse(
            event_stream(input_data),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable buffering in nginx
            },
        )

    async def health_check(_: Request) -> JSONResponse:
        """Health check endpoint."""
        return JSONResponse({"status": "healthy"})

    # Define routes
    routes = [
        Route("/v1/agent/stream", stream_agent, methods=["POST"]),
        Route("/health", health_check, methods=["GET"]),
    ]

    # Configure CORS middleware
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=config.cors_credentials,
            allow_methods=config.cors_methods,
            allow_headers=config.cors_headers,
        )
    ]

    # Create Starlette app
    app = Starlette(
        routes=routes,
        middleware=middleware,
        debug=False,
    )

    return app


def format_sse_event(event: AGUIEvent) -> str:
    """Format AG-UI event as Server-Sent Event.

    Args:
        event: AG-UI event dictionary

    Returns:
        SSE formatted string with event data
    """
    # Convert event to JSON
    event_json = json.dumps(event, ensure_ascii=False)

    # Format as SSE
    # Each event is: data: <json>\n\n
    return f"data: {event_json}\n\n"


def create_agui_server(config: "AppConfig", db_path: Path | None = None) -> Starlette:
    """Create AG-UI server with both research and deep ask endpoints.

    Args:
        config: Application config with research and qa settings
        db_path: Optional database path override

    Returns:
        Starlette app with research and deep ask endpoints
    """
    from haiku.rag.client import HaikuRAG
    from haiku.rag.graph.deep_qa.dependencies import DeepQAContext
    from haiku.rag.graph.deep_qa.graph import build_deep_qa_graph
    from haiku.rag.graph.deep_qa.state import DeepQADeps, DeepQAState
    from haiku.rag.graph.research.dependencies import ResearchContext
    from haiku.rag.graph.research.graph import build_research_graph
    from haiku.rag.graph.research.state import ResearchDeps, ResearchState

    # Store client reference for proper lifecycle management
    _client_cache: dict[str, HaikuRAG] = {}

    def get_client(effective_db_path: Path) -> HaikuRAG:
        """Get or create cached client."""
        path_key = str(effective_db_path)
        if path_key not in _client_cache:
            _client_cache[path_key] = HaikuRAG(db_path=effective_db_path, config=config)
        return _client_cache[path_key]

    # Research graph factories
    def research_graph_factory() -> Graph:
        return build_research_graph(config)

    def research_state_factory(input_state: dict[str, Any]) -> ResearchState:
        question = input_state.get("question", "")
        if not question:
            messages = input_state.get("messages", [])
            if messages:
                question = messages[0].get("content", "")
        context = ResearchContext(original_question=question)
        return ResearchState.from_config(context=context, config=config)

    def research_deps_factory(input_config: dict[str, Any]) -> ResearchDeps:
        effective_db_path = (
            db_path
            or input_config.get("db_path")
            or config.storage.data_dir / "haiku.rag.lancedb"
        )
        return ResearchDeps(client=get_client(effective_db_path))

    # Deep ask graph factories
    def deep_ask_graph_factory() -> Graph:
        return build_deep_qa_graph(config)

    def deep_ask_state_factory(input_state: dict[str, Any]) -> DeepQAState:
        question = input_state.get("question", "")
        if not question:
            messages = input_state.get("messages", [])
            if messages:
                question = messages[0].get("content", "")
        use_citations = input_state.get("use_citations", False)
        context = DeepQAContext(original_question=question, use_citations=use_citations)
        return DeepQAState.from_config(context=context, config=config)

    def deep_ask_deps_factory(input_config: dict[str, Any]) -> DeepQADeps:
        effective_db_path = (
            db_path
            or input_config.get("db_path")
            or config.storage.data_dir / "haiku.rag.lancedb"
        )
        return DeepQADeps(client=get_client(effective_db_path))

    # Create event stream functions for each graph type
    async def research_event_stream(
        input_data: RunAgentInput,
    ) -> AsyncIterator[str]:
        """Generate SSE event stream from research graph execution."""
        graph = research_graph_factory()
        initial_state = research_state_factory(input_data.state)
        deps = research_deps_factory(input_data.config)

        async for event in stream_graph(graph, initial_state, deps):
            event_data = format_sse_event(event)
            yield event_data

    async def deep_ask_event_stream(
        input_data: RunAgentInput,
    ) -> AsyncIterator[str]:
        """Generate SSE event stream from deep ask graph execution."""
        graph = deep_ask_graph_factory()
        initial_state = deep_ask_state_factory(input_data.state)
        deps = deep_ask_deps_factory(input_data.config)

        async for event in stream_graph(graph, initial_state, deps):
            event_data = format_sse_event(event)
            yield event_data

    # Endpoint handlers
    async def stream_research(request: Request) -> StreamingResponse:
        """Research graph streaming endpoint."""
        body = await request.json()
        input_data = RunAgentInput(**body)

        return StreamingResponse(
            research_event_stream(input_data),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def stream_deep_ask(request: Request) -> StreamingResponse:
        """Deep ask graph streaming endpoint."""
        body = await request.json()
        input_data = RunAgentInput(**body)

        return StreamingResponse(
            deep_ask_event_stream(input_data),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def health_check(_: Request) -> JSONResponse:
        """Health check endpoint."""
        return JSONResponse({"status": "healthy"})

    # Define routes
    routes = [
        Route("/v1/research/stream", stream_research, methods=["POST"]),
        Route("/v1/deep-ask/stream", stream_deep_ask, methods=["POST"]),
        Route("/health", health_check, methods=["GET"]),
    ]

    # Configure CORS middleware
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=config.agui.cors_origins,
            allow_credentials=config.agui.cors_credentials,
            allow_methods=config.agui.cors_methods,
            allow_headers=config.agui.cors_headers,
        )
    ]

    # Create Starlette app
    app = Starlette(
        routes=routes,
        middleware=middleware,
        debug=False,
    )

    return app
