from dataclasses import dataclass
from typing import TYPE_CHECKING

import logfire
from fastapi import Depends, FastAPI, Request

from haiku.rag.ingester.api.auth import require_auth

if TYPE_CHECKING:
    from haiku.rag.config import AppConfig
    from haiku.rag.ingester.pollers.manager import PollerManager
    from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo
    from haiku.rag.ingester.workers.pool import WorkerPool


@dataclass
class APIState:
    """Everything the API handlers need to read or act on. Pool/pollers are
    optional so the FastAPI app can be tested in isolation."""

    config: "AppConfig"
    job_repo: "JobRepo"
    sync_repo: "SyncStateRepo"
    pool: "WorkerPool | None" = None
    pollers: "PollerManager | None" = None


def get_state(request: Request) -> APIState:
    return request.app.state.api_state


def build_app(
    state: APIState,
    *,
    auth_token: str | None = None,
) -> FastAPI:
    """Construct the ingester's FastAPI control plane."""
    from haiku.rag.ingester.api.routes import dlq, health, jobs, sources

    app = FastAPI(
        title="haiku-ingester",
        description="Control plane for the haiku.rag production ingester.",
        version="1",
    )
    app.state.api_state = state
    app.state.auth_token = auth_token

    auth_dep = [Depends(require_auth)]
    app.include_router(health.router)  # /health is unauthenticated by design
    app.include_router(jobs.router, dependencies=auth_dep)
    app.include_router(sources.router, dependencies=auth_dep)
    app.include_router(dlq.router, dependencies=auth_dep)

    # Every request becomes a span when logfire is configured; no-op otherwise.
    logfire.instrument_fastapi(app)
    return app
