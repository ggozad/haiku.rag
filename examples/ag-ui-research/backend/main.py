"""Main entry point for the haiku.rag AG-UI research assistant backend."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from agent import ResearchDeps, ResearchState, create_agent
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config

logger = logging.getLogger(__name__)

# Global client instance
client: HaikuRAG | None = None


@asynccontextmanager
async def lifespan(app):
    """Manage HaikuRAG client lifecycle."""
    global client

    # Get database path from environment or use default
    db_path_str = os.getenv("DB_PATH", "haiku_rag.lancedb")
    db_path = Path(db_path_str)

    if not db_path.exists():
        logger.error(
            f"Database not found at {db_path}. Please initialize haiku.rag first."
        )
        logger.error("Run: haiku-rag add <path-to-documents>")
        raise RuntimeError(f"Database not found: {db_path}")

    logger.info(f"Initializing HaikuRAG client with database: {db_path}")
    client = HaikuRAG(db_path)

    logger.info("Research assistant backend ready")
    logger.info(f"QA Provider: {Config.QA_PROVIDER}")
    logger.info(f"QA Model: {Config.QA_MODEL}")

    yield

    # Cleanup
    if client:
        logger.info("Closing HaikuRAG client")
        client.close()


# Create research agent instance
agent = create_agent()


async def health(request):
    """Health check endpoint."""
    db_path_str = os.getenv("DB_PATH", "haiku_rag.lancedb")
    return JSONResponse(
        {
            "status": "healthy",
            "agent_model": str(agent.model),
            "qa_provider": Config.QA_PROVIDER,
            "qa_model": Config.QA_MODEL,
            "ollama_base_url": Config.OLLAMA_BASE_URL,
            "db_path": db_path_str,
            "db_exists": Path(db_path_str).exists(),
        }
    )


# Create AG-UI app once with the client
# State will be managed per-session by AG-UI
ag_ui_app = None


def get_ag_ui_app():
    """Get or create AG-UI app."""
    global ag_ui_app
    if ag_ui_app is None and client is not None:
        if client is None:
            raise RuntimeError("Client not initialized")

        # Create deps with shared client but new state per session
        research_deps = ResearchDeps(client=client, state=ResearchState())
        logger.info("Creating AG-UI app with initial state")
        ag_ui_app = agent.to_ag_ui(deps=research_deps)
    return ag_ui_app


# Initialize AG-UI app after client is ready in lifespan
async def agent_endpoint(scope, receive, send):
    """Proxy requests to AG-UI app."""
    app = get_ag_ui_app()
    if app is None:
        response = JSONResponse({"error": "Client not initialized"}, status_code=503)
        await response(scope, receive, send)
        return
    await app(scope, receive, send)


# Mount the AG-UI app at /agent and add health endpoint
app = Starlette(
    routes=[
        Route("/health", health),
        Mount("/agent", agent_endpoint),
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
    lifespan=lifespan,
)

if __name__ == "__main__":
    import uvicorn

    print("Starting haiku.rag research assistant backend...")
    print(f"Agent model: {agent.model}")
    print(f"QA provider: {Config.QA_PROVIDER}")
    print(f"QA model: {Config.QA_MODEL}")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
