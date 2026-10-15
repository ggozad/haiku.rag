import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from ag_ui.core import BaseEvent, EventType, StateSnapshotEvent
from PIL import Image
from pydantic_ai import Agent
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from haiku.rag.capabilities.compaction import create_capability as create_compaction
from haiku.rag.capabilities.policy import create_capability as create_citation_policy
from haiku.rag.capabilities.rag import RAGState, create_capability
from haiku.rag.config import AppConfig
from haiku.rag.utils.models import get_model


@dataclass
class ChatDeps:
    state: dict[str, Any] = field(default_factory=dict)


def build_agent(config: AppConfig) -> Agent[ChatDeps, str]:
    return Agent(
        get_model(config.qa.model, config),
        deps_type=ChatDeps,
        # Earlier questions are reduced to the evidence they cited, and every
        # answer declares what grounds it so the UI can show its citations.
        capabilities=[
            create_capability(config=config),
            create_compaction(),
            create_citation_policy(),
        ],
    )


async def chat_events(
    adapter: AGUIAdapter[ChatDeps, str], deps: ChatDeps
) -> AsyncIterator[BaseEvent]:
    """The run's events, with the whole state as a snapshot before it ends."""
    async for event in adapter.run_stream(deps=deps):
        if event.type == EventType.RUN_FINISHED:
            yield StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=deps.state)
        yield event


async def stream_chat(request: Request) -> Response:
    """The AG-UI chat stream."""
    agent: Agent[ChatDeps, str] = request.app.state.agent
    accept = request.headers.get("accept", SSE_CONTENT_TYPE)
    run_input = AGUIAdapter.build_run_input(await request.body())
    adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)

    state = run_input.state if isinstance(run_input.state, dict) else {}
    state.setdefault("rag", RAGState().model_dump(mode="json"))
    return StreamingResponse(
        adapter.encode_stream(chat_events(adapter, ChatDeps(state=state))),
        media_type=accept,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def list_documents(request: Request) -> JSONResponse:
    """Documents of one database, or of every database with failures named."""
    databases = request.app.state.databases
    source = request.query_params.get("source")
    names = [source] if source is not None else list(databases.names)
    documents: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for name in names:
        try:
            owner = await databases.reader(name)
            listed = await owner.list_documents()
        except Exception as error:
            if not isinstance(error, HTTPException):
                error = HTTPException(503, str(error))
            if source is not None:
                raise error
            errors[name] = error.detail
            continue
        documents.extend(
            {
                "id": doc.id,
                "title": doc.title,
                "uri": doc.uri,
                "source": name,
                "metadata": doc.metadata,
            }
            for doc in listed
        )
    return JSONResponse({"documents": documents, "errors": errors})


async def picture(request: Request) -> Response:
    """A picture's stored bytes, with the media type read from them."""
    params = request.query_params
    document_id, ref = params.get("document_id"), params.get("ref")
    if not document_id or not ref:
        raise HTTPException(400, "document_id and ref are required")
    owner = await request.app.state.databases.reader(params.get("source"))
    data = await owner.get_picture_bytes(document_id, ref)
    if data is None:
        raise HTTPException(404, f"no picture {ref!r} in document {document_id!r}")
    try:
        with Image.open(BytesIO(data)) as image:
            media_type = Image.MIME.get(image.format or "")
    except OSError:
        media_type = None
    if media_type is None:
        raise HTTPException(415, "the stored picture is in no known image format")
    return Response(data, media_type=media_type)


async def visualize(request: Request) -> JSONResponse:
    """Page images highlighting chunks, as base64 PNGs.

    `chunk_ids` is comma-separated, the constituents of a merged citation, and
    `refs` the JSON list of the items the model saw, drawn instead of
    re-expanding.
    """
    owner = await request.app.state.databases.reader(request.query_params.get("source"))
    refs: list[str] | None = None
    if refs_param := request.query_params.get("refs"):
        try:
            parsed = json.loads(refs_param)
        except ValueError as error:
            raise HTTPException(400, "refs must be a JSON list") from error
        if not isinstance(parsed, list):
            raise HTTPException(400, "refs must be a JSON list")
        refs = [str(ref) for ref in parsed]

    chunks = []
    for chunk_id in request.path_params["chunk_ids"].split(","):
        chunk = await owner.chunk_repository.get_by_id(chunk_id)
        if chunk is not None:
            chunks.append(chunk)
    if not chunks:
        raise HTTPException(404, "chunk not found")

    images = []
    for image in await owner.visualize_chunk(chunks, refs):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        images.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    return JSONResponse(
        {
            "images": images,
            "message": None if images else "No visual grounding available",
        }
    )


routes = [
    Route("/v1/chat/stream", stream_chat, methods=["POST"]),
    Route("/api/documents", list_documents, methods=["GET"]),
    Route("/api/pictures", picture, methods=["GET"]),
    Route("/api/visualize/{chunk_ids}", visualize, methods=["GET"]),
]
