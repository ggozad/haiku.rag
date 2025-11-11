"""AG-UI HTTP server implementation for graph execution."""

import json
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field
from pydantic_graph.beta import Graph
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from haiku.rag.agui.events import AGUIEvent
from haiku.rag.agui.stream import stream_graph
from haiku.rag.config.models import AGUIConfig


class GraphDeps(Protocol):
    """Protocol for graph dependencies that support AG-UI emission."""

    agui_emitter: Any | None


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


def create_research_server(config: Any, db_path: Any | None = None) -> Starlette:
    """Create AG-UI server for research graph.

    This is a convenience function specifically for the research graph.

    Args:
        config: Application config with research settings
        db_path: Optional database path override

    Returns:
        Starlette app configured for research graph
    """
    from haiku.rag.client import HaikuRAG
    from haiku.rag.research.dependencies import ResearchContext
    from haiku.rag.research.graph import build_research_graph
    from haiku.rag.research.state import ResearchDeps, ResearchState

    # Store client reference for proper lifecycle management
    _client_cache: dict[str, HaikuRAG] = {}

    def graph_factory() -> Graph:
        """Create research graph instance."""
        return build_research_graph(config)

    def state_factory(input_state: dict[str, Any]) -> ResearchState:
        """Create research state from input."""
        # Extract question from input state or messages
        question = input_state.get("question", "")
        if not question:
            # Try to get from first message if available
            messages = input_state.get("messages", [])
            if messages:
                question = messages[0].get("content", "")

        # Create context and state
        context = ResearchContext(original_question=question)
        return ResearchState.from_config(context=context, config=config)

    def deps_factory(input_config: dict[str, Any]) -> ResearchDeps:
        """Create research dependencies."""
        # Use provided db_path or fallback to config
        effective_db_path = (
            db_path
            or input_config.get("db_path")
            or config.storage.data_dir / "haiku.rag.lancedb"
        )

        # Reuse existing client if available
        path_key = str(effective_db_path)
        if path_key not in _client_cache:
            _client_cache[path_key] = HaikuRAG(db_path=effective_db_path, config=config)

        return ResearchDeps(client=_client_cache[path_key])

    # Use AG-UI config from app config
    return create_agui_app(
        graph_factory=graph_factory,
        state_factory=state_factory,
        deps_factory=deps_factory,
        config=config.agui,
    )
