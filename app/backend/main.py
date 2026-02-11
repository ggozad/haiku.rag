import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from haiku.rag.agents.chat import (
    AGUI_STATE_KEY,
    ChatDeps,
    create_chat_agent,
)
from haiku.rag.client import HaikuRAG
from haiku.rag.config import load_yaml_config
from haiku.rag.config.models import AppConfig
from haiku.rag.tools import ToolContext

load_dotenv(find_dotenv(usecwd=True))

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

# Only HaikuRAG client is a singleton (expensive to create)
_client: HaikuRAG | None = None


def get_client() -> HaikuRAG:
    """Get or create cached client."""
    global _client
    if _client is None:
        _client = HaikuRAG(db_path=db_path, config=Config, create=True)
    return _client


async def stream_chat(request: Request) -> Response:
    """Chat streaming endpoint with AG-UI protocol.

    This endpoint is stateless - all state flows via AG-UI protocol:
    - Fresh ToolContext created per request
    - AGUIAdapter restores state via ChatDeps.state setter
    - ChatDeps generates session_id if not provided
    - Ask tool triggers background summarization internally
    - ChatDeps.state getter emits final state in response
    """
    body = await request.body()
    accept = request.headers.get("accept", SSE_CONTENT_TYPE)
    run_input = AGUIAdapter.build_run_input(body)

    # Fresh context per request - state restored by AGUIAdapter via ChatDeps.state setter
    context = ToolContext()
    agent = create_chat_agent(Config, get_client(), context)

    deps = ChatDeps(
        config=Config,
        tool_context=context,
        state_key=AGUI_STATE_KEY,
    )

    # Use AGUIAdapter for streaming
    # State restoration happens automatically via ChatDeps.state setter
    # Background summarization triggered by ask() tool internally
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
    """Health check endpoint."""
    return JSONResponse(
        {
            "status": "healthy",
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

    client = get_client()
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

    client = get_client()
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

    client = get_client()

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
            CORSMiddleware,  # type: ignore[invalid-argument-type]
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
