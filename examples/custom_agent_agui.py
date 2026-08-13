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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ag_ui.core import EventType, StateSnapshotEvent
from pydantic_ai import Agent
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from haiku.rag.capabilities.compaction import create_capability as compaction
from haiku.rag.capabilities.policy import create_capability as citation_policy
from haiku.rag.capabilities.rag import RAGState, create_capability

db_path = os.environ.get("DB_PATH")
if not db_path:
    print(
        "Set DB_PATH environment variable to your haiku.rag database", file=sys.stderr
    )
    sys.exit(1)

capability = create_capability(db_path=Path(db_path), defer_loading=False)


@dataclass
class AppDeps:
    state: dict[str, Any] = field(default_factory=dict)


agent = Agent(
    "anthropic:claude-haiku-4-5-20251001",
    # The client returns the state snapshot with every run, so earlier questions are
    # reduced to the evidence they cited and every answer declares its grounding.
    capabilities=[capability, compaction(), citation_policy()],
    deps_type=AppDeps,
)


async def stream_chat(request: Request) -> Response:
    body = await request.body()
    accept = request.headers.get("accept", SSE_CONTENT_TYPE)
    run_input = AGUIAdapter.build_run_input(body)

    adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)

    incoming_state = run_input.state if isinstance(run_input.state, dict) else {}
    incoming_state.setdefault("rag", RAGState().model_dump(mode="json"))
    deps = AppDeps(state=incoming_state)

    async def event_stream():
        async def with_final_state():
            async for event in adapter.run_stream(deps=deps):
                if getattr(event, "type", None) == EventType.RUN_FINISHED:
                    yield StateSnapshotEvent(
                        type=EventType.STATE_SNAPSHOT,
                        snapshot=deps.state,
                    )
                yield event

        async for chunk in adapter.encode_stream(with_final_state()):
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
