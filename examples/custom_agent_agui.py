"""Custom agent with AG-UI streaming.

A Starlette app that composes haiku.rag toolsets into an AG-UI compatible
agent. Multi-session support via ToolContextCache.

Requirements:
    - An Ollama instance running locally (default embedder)
    - An Anthropic API key (for the QA model) or adjust the model below

Usage:
    DB_PATH=/path/to/db.lancedb uv run uvicorn examples.custom_agent_agui:app --reload --port 8000
"""

import os
import sys

from pydantic_ai import Agent
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.tools import (
    AgentDeps,
    ToolContextCache,
    create_qa_toolset,
    create_search_toolset,
    prepare_context,
)

db_path = os.environ.get("DB_PATH")
if not db_path:
    print(
        "Set DB_PATH environment variable to your haiku.rag database", file=sys.stderr
    )
    sys.exit(1)

AGUI_STATE_KEY = "my_app"

config = AppConfig()

# ToolContextCache maintains per-thread state across requests
context_cache = ToolContextCache()

# Singleton client
_client: HaikuRAG | None = None


def get_client() -> HaikuRAG:
    global _client
    if _client is None:
        _client = HaikuRAG(db_path=db_path)
    return _client


# Create the agent once at module level
agent = Agent(
    "anthropic:claude-haiku-4-5-20251001",
    deps_type=AgentDeps,
    output_type=str,
    instructions=(
        "You are a helpful assistant with access to a knowledge base. "
        "Use the search and ask tools to answer questions."
    ),
    toolsets=[
        create_search_toolset(config),
        create_qa_toolset(config),
    ],
)


async def stream_chat(request: Request) -> Response:
    body = await request.body()
    accept = request.headers.get("accept", SSE_CONTENT_TYPE)
    run_input = AGUIAdapter.build_run_input(body)

    thread_id = getattr(run_input, "thread_id", None) or "default"
    context, is_new = context_cache.get_or_create(thread_id)
    if is_new:
        prepare_context(
            context,
            features=["search", "qa"],
            state_key=AGUI_STATE_KEY,
        )

    deps = AgentDeps(client=get_client(), tool_context=context)

    adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)
    event_stream = adapter.run_stream(deps=deps)
    sse_event_stream = adapter.encode_stream(event_stream)

    return StreamingResponse(
        sse_event_stream,
        media_type=accept,
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def health_check(_: Request) -> JSONResponse:
    return JSONResponse({"status": "healthy"})


app = Starlette(
    routes=[
        Route("/v1/chat/stream", stream_chat, methods=["POST"]),
        Route("/health", health_check, methods=["GET"]),
    ],
)
