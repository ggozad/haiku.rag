"""Custom agent with AG-UI streaming.

A Starlette app that serves an AG-UI streaming endpoint using the
haiku.rag's native Pydantic AI RAG capability.

Requirements:
    - An Ollama instance running locally (default embedder)
    - An Anthropic API key (for the QA model) or adjust the model below

Usage:
    DB_PATH=/path/to/db.lancedb uv run uvicorn examples.custom_agent_agui:app --reload --port 8000
"""

import os
import sys
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from haiku.rag.capabilities.rag import create_capability

db_path = os.environ.get("DB_PATH")
if not db_path:
    print(
        "Set DB_PATH environment variable to your haiku.rag database", file=sys.stderr
    )
    sys.exit(1)

capability = create_capability(db_path=Path(db_path))

agent = Agent(
    "anthropic:claude-haiku-4-5-20251001",
    capabilities=[capability],
)


async def stream_chat(request: Request) -> Response:
    body = await request.body()
    accept = request.headers.get("accept", SSE_CONTENT_TYPE)
    run_input = AGUIAdapter.build_run_input(body)

    adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)

    async def event_stream():
        async for chunk in adapter.encode_stream(adapter.run_stream()):
            yield chunk

    return StreamingResponse(
        event_stream(),
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
