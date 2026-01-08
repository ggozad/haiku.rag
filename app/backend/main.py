import logging
import os
from pathlib import Path

from agent import ChatDeps, ChatSessionState, QAResponse, create_chat_agent
from anyio import (
    EndOfStream,
    create_memory_object_stream,
    create_task_group,
    move_on_after,
)
from anyio.streams.memory import MemoryObjectSendStream
from dotenv import load_dotenv
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from haiku.rag.client import HaikuRAG
from haiku.rag.config import load_yaml_config
from haiku.rag.config.models import AppConfig
from haiku.rag.graph.agui.emitter import AGUIEmitter
from haiku.rag.graph.agui.server import RunAgentInput, format_sse_event


def convert_messages_to_history(
    messages: list[dict[str, str]],
) -> list[ModelMessage]:
    """Convert AG-UI/CopilotKit messages to pydantic-ai message history.

    Skips the last message since it will be passed as user_prompt to agent.run().
    """
    history: list[ModelMessage] = []

    # Skip the last message - it will be the current user prompt
    for msg in messages[:-1]:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        elif role == "assistant":
            history.append(ModelResponse(parts=[TextPart(content=content)]))
        # Skip other roles (system, tool, etc.) for now

    return history


load_dotenv()

# Configure logfire (only sends data if LOGFIRE_TOKEN is present)
try:
    import logfire

    logfire.configure(send_to_logfire="if-token-present", console=False)
    logfire.instrument_pydantic_ai()
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load config
config_path = Path("/app/haiku.rag.yaml")
if config_path.exists():
    yaml_data = load_yaml_config(config_path)
    Config = AppConfig.model_validate(yaml_data)
else:
    Config = AppConfig()

# Get DB path from environment
db_path_str = os.getenv("DB_PATH", "haiku_rag.lancedb")
db_path = Path(db_path_str)

logger.info(f"Database path: {db_path}")
logger.info(f"QA Provider: {Config.qa.model.provider}, Model: {Config.qa.model.name}")

# Create the chat agent
chat_agent = create_chat_agent(Config)

# Client cache for proper lifecycle
_client_cache: dict[str, HaikuRAG] = {}


def get_client(effective_db_path: Path) -> HaikuRAG:
    """Get or create cached client."""
    path_key = str(effective_db_path)
    if path_key not in _client_cache:
        _client_cache[path_key] = HaikuRAG(
            db_path=effective_db_path, config=Config, create=True
        )
    return _client_cache[path_key]


async def stream_chat(request: Request) -> StreamingResponse:
    """Chat streaming endpoint with AG-UI protocol."""
    body = await request.json()
    logger.info(f"Received request: {list(body.keys())}")
    input_data = RunAgentInput(**body)

    user_message = ""
    message_history: list[ModelMessage] = []
    if input_data.messages:
        user_message = input_data.messages[-1].get("content", "")
        message_history = convert_messages_to_history(input_data.messages)

    send_stream, receive_stream = create_memory_object_stream[str]()

    async def run_agent_with_streaming(
        send_stream: MemoryObjectSendStream[str],
    ) -> None:
        """Execute agent and forward events to stream."""
        async with send_stream:
            try:
                # Create emitter for streaming
                emitter: AGUIEmitter = AGUIEmitter(
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                    use_deltas=True,
                )

                # Get client
                effective_db_path = db_path
                if input_data.config and input_data.config.get("db_path"):
                    effective_db_path = Path(input_data.config["db_path"])
                client = get_client(effective_db_path)

                # Parse incoming state to restore qa_history
                initial_qa_history: list[QAResponse] = []
                if input_data.state and "qa_history" in input_data.state:
                    initial_qa_history = [
                        QAResponse(**qa)
                        for qa in input_data.state.get("qa_history", [])
                    ]

                # Create initial state with restored history
                initial_state = ChatSessionState(
                    session_id=input_data.thread_id or "",
                    qa_history=initial_qa_history,
                )

                # Create deps with session state
                deps = ChatDeps(
                    client=client,
                    config=Config,
                    agui_emitter=emitter,
                    session_state=initial_state,
                )
                emitter.start_run(initial_state=initial_state)

                # Forward events
                async def forward_events():
                    async for event in emitter:
                        event_type = event.get("type")
                        logger.debug(f"AG-UI event: {event_type}")
                        await send_stream.send(format_sse_event(event))

                # Run agent and forward concurrently
                async with create_task_group() as tg:
                    tg.start_soon(forward_events)

                    result = await chat_agent.run(
                        user_message, deps=deps, message_history=message_history
                    )
                    emitter.log(result.output)
                    emitter.finish_run(result.output)
                    await emitter.close()

            except Exception as e:
                logger.exception("Error executing agent")
                try:
                    await send_stream.send(
                        format_sse_event({"type": "RUN_ERROR", "message": str(e)})
                    )
                except Exception:
                    pass

    async def event_generator():
        """Generate SSE events with heartbeat to keep connection alive."""
        async with create_task_group() as tg:
            tg.start_soon(run_agent_with_streaming, send_stream)
            async with receive_stream:
                while True:
                    try:
                        # Wait for event with timeout, send heartbeat if nothing received
                        with move_on_after(15):  # 15 second timeout
                            event_str = await receive_stream.receive()
                            yield event_str
                            continue
                        # No event received within timeout - send SSE comment as heartbeat
                        yield ": heartbeat\n\n"
                    except EndOfStream:
                        break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def health_check(_: Request) -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse(
        {
            "status": "healthy",
            "agent_model": str(chat_agent.model),
            "qa_provider": Config.qa.model.provider,
            "qa_model": Config.qa.model.name,
            "db_path": str(db_path),
            "db_exists": db_path.exists(),
        }
    )


async def list_documents(_: Request) -> JSONResponse:
    """List all documents in the database."""
    if not db_path.exists():
        return JSONResponse({"documents": [], "error": "Database not found"})

    client = get_client(db_path)
    docs = await client.document_repository.list_all()
    return JSONResponse(
        {
            "documents": [
                {"id": doc.id, "title": doc.title, "uri": doc.uri} for doc in docs
            ]
        }
    )


async def db_info(_: Request) -> JSONResponse:
    """Get database info and statistics."""
    if not db_path.exists():
        return JSONResponse(
            {
                "exists": False,
                "path": str(db_path),
                "documents": 0,
                "chunks": 0,
            }
        )

    client = get_client(db_path)
    stats = client.store.get_stats()

    return JSONResponse(
        {
            "exists": True,
            "path": str(db_path),
            "documents": stats.get("documents", {}).get("num_rows", 0),
            "chunks": stats.get("chunks", {}).get("num_rows", 0),
            "documents_bytes": stats.get("documents", {}).get("total_bytes", 0),
            "chunks_bytes": stats.get("chunks", {}).get("total_bytes", 0),
            "has_vector_index": stats.get("chunks", {}).get("has_vector_index", False),
        }
    )


async def visualize_chunk(request: Request) -> JSONResponse:
    """Return visual grounding images for a chunk as base64."""
    import base64
    from io import BytesIO

    chunk_id = request.path_params["chunk_id"]

    if not db_path.exists():
        return JSONResponse({"error": "Database not found"}, status_code=404)

    client = get_client(db_path)

    chunk = await client.chunk_repository.get_by_id(chunk_id)
    if not chunk:
        return JSONResponse({"error": "Chunk not found"}, status_code=404)

    images = await client.visualize_chunk(chunk)
    if not images:
        return JSONResponse({"images": [], "message": "No visual grounding available"})

    base64_images = []
    for img in images:
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        base64_images.append(base64.b64encode(buffer.read()).decode("utf-8"))

    return JSONResponse(
        {
            "images": base64_images,
            "chunk_id": chunk_id,
            "document_uri": chunk.document_uri,
        }
    )


# Create Starlette app
app = Starlette(
    routes=[
        Route("/v1/chat/stream", stream_chat, methods=["POST"]),
        Route("/api/documents", list_documents, methods=["GET"]),
        Route("/api/info", db_info, methods=["GET"]),
        Route("/api/visualize/{chunk_id}", visualize_chunk, methods=["GET"]),
        Route("/health", health_check, methods=["GET"]),
    ],
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "http://frontend:3000"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
