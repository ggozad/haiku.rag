import logging
import os
from pathlib import Path

from agent import AgentDeps, agent
from anyio import create_memory_object_stream, create_task_group
from anyio.streams.memory import MemoryObjectSendStream
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
from haiku.rag.graph.research.dependencies import ResearchContext
from haiku.rag.graph.research.models import ResearchReport
from haiku.rag.graph.research.state import ResearchState

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load config from mounted haiku.rag.yaml
config_path = Path("/app/haiku.rag.yaml")
if config_path.exists():
    yaml_data = load_yaml_config(config_path)
    Config = AppConfig.model_validate(yaml_data)
else:
    # Fallback to default config
    Config = AppConfig()


# Get DB path from environment
db_path_str = os.getenv("DB_PATH", "haiku_rag.lancedb")
db_path = Path(db_path_str)

if not db_path.exists():
    logger.error(f"Database not found at {db_path}")
    logger.error("Run: haiku-rag add <path-to-documents>")
    raise RuntimeError(f"Database not found: {db_path}")

logger.info(f"Initializing research assistant with database: {db_path}")
logger.info(
    f"Research Provider: {Config.research.provider}, Model: {Config.research.model}"
)

# Store client reference for proper lifecycle management
_client_cache: dict[str, HaikuRAG] = {}


def get_client(effective_db_path: Path) -> HaikuRAG:
    """Get or create cached client."""
    path_key = str(effective_db_path)
    if path_key not in _client_cache:
        _client_cache[path_key] = HaikuRAG(db_path=effective_db_path, config=Config)
    return _client_cache[path_key]


async def stream_research_agent(request: Request) -> StreamingResponse:
    """Agent streaming endpoint with research graph integration."""
    body = await request.json()
    input_data = RunAgentInput(**body)

    user_message = ""
    if input_data.messages:
        user_message = input_data.messages[-1].get("content", "")

    send_stream, receive_stream = create_memory_object_stream[str]()

    async def run_agent_with_streaming(
        send_stream: MemoryObjectSendStream[str],
    ) -> None:
        """Execute agent and forward emitter events to memory stream."""
        async with send_stream:
            try:
                # Create shared emitter
                emitter: AGUIEmitter[ResearchState, ResearchReport] = AGUIEmitter(
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                    use_deltas=False,
                )

                # Get client
                effective_db_path = input_data.config.get("db_path") or db_path
                if isinstance(effective_db_path, str):
                    effective_db_path = Path(effective_db_path)
                client = get_client(effective_db_path)

                # Create agent dependencies with shared emitter
                agent_deps = AgentDeps(client=client, agui_emitter=emitter)

                # Start run with empty initial state
                emitter.start_run(
                    initial_state=ResearchState.from_config(
                        context=ResearchContext(original_question=""),
                        config=Config,
                    )
                )

                # Forward emitter events to stream
                async def forward_events():
                    async for event in emitter:
                        # Filter out ACTIVITY_SNAPSHOT - not supported by CopilotKit
                        if event.get("type") == "ACTIVITY_SNAPSHOT":
                            continue
                        await send_stream.send(format_sse_event(event))

                # Run agent and event forwarding concurrently
                async with create_task_group() as tg:
                    tg.start_soon(forward_events)

                    result = await agent.run(user_message, deps=agent_deps)
                    emitter.log(result.output)
                    await emitter.close()

            except Exception as e:
                logger.exception("Error executing agent")
                try:
                    await send_stream.send(
                        format_sse_event({"type": "error", "error": str(e)})
                    )
                except Exception:
                    pass

    async def event_generator():
        """Generate SSE events from memory stream."""
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
    """Health check endpoint with configuration info."""
    return JSONResponse(
        {
            "status": "healthy",
            "agent_model": str(agent.model),
            "research_provider": Config.research.provider,
            "research_model": Config.research.model,
            "db_path": str(db_path),
            "db_exists": db_path.exists(),
        }
    )


# Create Starlette app
app = Starlette(
    routes=[
        Route("/v1/research/stream", stream_research_agent, methods=["POST"]),
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
