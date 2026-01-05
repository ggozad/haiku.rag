import logging
import os
from pathlib import Path

from agent import ChatDeps, ChatSessionState, create_chat_agent
from anyio import create_memory_object_stream, create_task_group
from anyio.streams.memory import MemoryObjectSendStream
from dotenv import load_dotenv
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

load_dotenv()

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
    if input_data.messages:
        user_message = input_data.messages[-1].get("content", "")

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

                # Create deps
                deps = ChatDeps(
                    client=client,
                    config=Config,
                    agui_emitter=emitter,
                )

                # Start run with empty state
                initial_state = ChatSessionState(
                    session_id=input_data.thread_id or "",
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

                    result = await chat_agent.run(user_message, deps=deps)
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
        """Generate SSE events."""
        async with create_task_group() as tg:
            tg.start_soon(run_agent_with_streaming, send_stream)
            async with receive_stream:
                async for event_str in receive_stream:
                    yield event_str

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


# Create Starlette app
app = Starlette(
    routes=[
        Route("/v1/chat/stream", stream_chat, methods=["POST"]),
        Route("/api/documents", list_documents, methods=["GET"]),
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
