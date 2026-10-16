import asyncio
import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from ag_ui.core import BaseEvent, CustomEvent, EventType, Message, StateSnapshotEvent
from PIL import Image
from pydantic import TypeAdapter
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from haiku.rag.capabilities.compaction import create_capability as create_compaction
from haiku.rag.capabilities.ledger import citation_status
from haiku.rag.capabilities.policy import STATE_NAMESPACE as POLICY_NAMESPACE
from haiku.rag.capabilities.policy import CitationPolicyState
from haiku.rag.capabilities.policy import create_capability as create_citation_policy
from haiku.rag.capabilities.rag import RAGState, create_capability
from haiku.rag.config import AppConfig
from haiku.rag.utils.models import get_model
from haiku.rag.utils.sql import build_document_id_filter
from haiku_rag_app.sessions import Sessions


@dataclass
class ChatDeps:
    state: dict[str, Any] = field(default_factory=dict)


async def agent_for(app: Starlette) -> Agent[ChatDeps, str]:
    """The chat agent, built on first use over the app's client.

    The capability borrows the client, so searches, citations and the sandbox
    read through the connections the rest of the app holds.
    """
    if app.state.agent is None:
        async with app.state.agent_lock:
            if app.state.agent is None:
                config: AppConfig = app.state.config
                client = await app.state.databases.covering()
                app.state.agent = Agent(
                    get_model(config.qa.model, config),
                    deps_type=ChatDeps,
                    # Earlier questions are reduced to the evidence they cited,
                    # and every answer declares what grounds it so the UI can
                    # show its citations.
                    capabilities=[
                        create_capability(config=config, rag=client),
                        create_compaction(),
                        create_citation_policy(),
                    ],
                )
    return app.state.agent


def _scope_to_selection(
    state: dict[str, Any], documents: list[dict[str, str]], covers_multiple: bool
) -> None:
    """Point the run's search at the documents the session selected.

    The filter is one id list applied in every selected database; across a
    set, `sources` narrows the search to the databases the selection names. An
    empty selection searches everything.
    """
    rag = RAGState.model_validate(state.get("rag") or {})
    rag.document_filter = build_document_id_filter(
        sorted({doc["id"] for doc in documents})
    )
    rag.sources = (
        sorted({doc["source"] for doc in documents})
        if covers_multiple and documents
        else None
    )
    state["rag"] = rag.model_dump(mode="json")


def run_summary(state: dict[str, Any]) -> dict[str, Any]:
    """How the question was grounded, and whether the policy recorded a breach."""
    record = RAGState.model_validate(state.get("rag") or {}).evidence
    question = record.question
    # A completed run has begun its question, so it always has an identity.
    assert question is not None
    violations = CitationPolicyState.model_validate(
        state.get(POLICY_NAMESPACE) or {}
    ).violations
    return {
        "citation_status": citation_status([record], question=question),
        "violation": question in violations,
    }


def run_evidence(state: dict[str, Any]) -> dict[str, Any]:
    """The citations and the last successful program of the question just answered."""
    rag = RAGState.model_validate(state.get("rag") or {})
    successful = [entry for entry in rag.executions if entry.success]
    return {
        "citations": [
            rag.citation_index[chunk_id].model_dump(mode="json")
            for chunk_id in rag.citations
            if chunk_id in rag.citation_index
        ],
        "program": successful[-1].code if successful else None,
        "summary": run_summary(state),
    }


def _dumped(message: Message) -> dict[str, Any]:
    return message.model_dump(mode="json", by_alias=True, exclude_none=True)


async def chat_events(
    adapter: AGUIAdapter[ChatDeps, str],
    deps: ChatDeps,
    sessions: Sessions,
    session_id: str,
    run_id: str,
    user_message: dict[str, Any],
) -> AsyncIterator[BaseEvent]:
    """The run's events, with the turn committed once the run completes.

    The commit is the durable boundary: it runs before the summary, the state
    snapshot and `RUN_FINISHED` go out, and it is shielded, so a client leaving
    during or after it does not undo a completed turn. A run cancelled before it
    completes stores nothing. The session is held until the commit lands.
    """
    commit: asyncio.Future | None = None

    async def on_complete(result: AgentRunResult[Any]) -> AsyncIterator[BaseEvent]:
        nonlocal commit
        evidence = run_evidence(deps.state)
        commit = sessions.start_commit(
            session_id,
            run_id,
            sessions.commit_turn(
                session_id,
                run_id,
                user_message,
                [_dumped(m) for m in AGUIAdapter.dump_messages(result.new_messages())],
                deps.state,
                evidence,
            ),
        )
        await asyncio.shield(commit)
        yield CustomEvent(
            type=EventType.CUSTOM, name="haiku.run_summary", value=evidence["summary"]
        )

    try:
        async for event in adapter.run_stream(deps=deps, on_complete=on_complete):
            if event.type == EventType.RUN_FINISHED:
                yield StateSnapshotEvent(
                    type=EventType.STATE_SNAPSHOT, snapshot=deps.state
                )
            yield event
    finally:
        # A started commit releases the session when it lands.
        if commit is None:
            sessions.release(session_id, run_id)


async def stream_chat(request: Request) -> Response:
    """The AG-UI chat stream over a stored session.

    The client sends the new user message; the history, the state and the
    selection come from the stored session, whatever else the client sends.
    """
    accept = request.headers.get("accept", SSE_CONTENT_TYPE)
    run_input = AGUIAdapter.build_run_input(await request.body())
    sessions: Sessions = request.app.state.sessions
    session_id, run_id = run_input.thread_id, run_input.run_id
    sessions.admit(session_id, run_id)
    try:
        session = await sessions.get(session_id)
        stored_ids = {message["id"] for message in session["messages"]}
        new = [m for m in run_input.messages if m.id not in stored_ids]
        if (
            len(new) != 1
            or new[0] is not run_input.messages[-1]
            or new[0].role != "user"
        ):
            raise HTTPException(400, "the input must end in one new user message")
        if run_id in session["runs"]:
            raise HTTPException(400, f"run {run_id!r} is already stored")
        user_message = _dumped(new[0])

        state = session["state"]
        _scope_to_selection(
            state,
            session["selection"],
            request.app.state.databases.scope.covers_multiple,
        )
        authoritative = run_input.model_copy(
            update={
                "messages": [
                    *TypeAdapter(list[Message]).validate_python(session["messages"]),
                    new[0],
                ],
                "state": state,
                "tools": [],
                "context": [],
                "forwarded_props": {},
                # A newer protocol's interrupt resumption, kept as an extra
                # field by this one.
                "resume": None,
            }
        )
        agent = await agent_for(request.app)
        adapter = AGUIAdapter(agent=agent, run_input=authoritative, accept=accept)
        events = chat_events(
            adapter, ChatDeps(state=state), sessions, session_id, run_id, user_message
        )
        sessions.tie(session_id, run_id, events)
    except BaseException:
        sessions.release(session_id, run_id)
        raise
    return StreamingResponse(
        adapter.encode_stream(events),
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
