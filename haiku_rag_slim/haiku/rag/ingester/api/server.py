from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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
    db_path: Path | None = None


def get_state(request: Request) -> APIState:
    return request.app.state.api_state


def build_app(
    state: APIState,
    *,
    auth_token: str | None = None,
    root_path: str = "",
) -> FastAPI:
    """Construct the ingester's FastAPI control plane.

    ``root_path`` serves the control plane under a base path when it is
    reverse-proxied behind a sub-path (e.g. ``/ingester``). It is forwarded to
    FastAPI so generated URLs (OpenAPI/docs) and the dashboard's ``<base href>``
    are prefix-aware; empty serves at the root.
    """
    from haiku.rag.ingester.api.routes import (
        config as config_route,
    )
    from haiku.rag.ingester.api.routes import (
        dashboard,
        database,
        dlq,
        health,
        jobs,
        providers,
        sources,
        stats,
    )

    app = FastAPI(
        title="haiku-ingester",
        description="Control plane for the haiku.rag production ingester.",
        version="1",
        root_path=root_path,
    )
    app.state.api_state = state
    app.state.auth_token = auth_token

    auth_dep = [Depends(require_auth)]
    app.include_router(health.router)  # /health is unauthenticated by design
    # Dashboard route is markup-only; the JS it serves attaches the bearer
    # token to its own fetches. Keeping the route unauthenticated lets an
    # operator open it in a browser and paste the token on demand.
    app.include_router(dashboard.router)
    app.include_router(jobs.router, dependencies=auth_dep)
    app.include_router(sources.router, dependencies=auth_dep)
    app.include_router(dlq.router, dependencies=auth_dep)
    app.include_router(stats.router, dependencies=auth_dep)
    app.include_router(providers.router, dependencies=auth_dep)
    app.include_router(database.router, dependencies=auth_dep)
    app.include_router(config_route.router, dependencies=auth_dep)

    return app
