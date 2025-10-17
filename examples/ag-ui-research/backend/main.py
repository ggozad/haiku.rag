"""Main entry point for the haiku.rag AG-UI research assistant backend."""

from agent import ResearchState, create_agent
from pydantic_ai.ag_ui import StateDeps
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from haiku.rag.config import Config

# Create research agent instance using haiku.rag config
agent = create_agent()


async def health(request):
    """Health check endpoint."""
    return JSONResponse(
        {
            "status": "healthy",
            "agent_model": str(agent.model),
            "qa_provider": Config.QA_PROVIDER,
            "qa_model": Config.QA_MODEL,
            "ollama_base_url": Config.OLLAMA_BASE_URL,
        }
    )


# Convert PydanticAI agent to AG-UI compatible ASGI app
ag_ui_app = agent.to_ag_ui(deps=StateDeps(ResearchState()))  # type: ignore[arg-type]

# Mount the AG-UI app at /agent and add health endpoint
app = Starlette(
    routes=[
        Route("/health", health),
        Mount("/agent", ag_ui_app),
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
