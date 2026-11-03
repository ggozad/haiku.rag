import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import logfire
from fastapi import Depends, FastAPI, Request

from haiku.rag.ingester.api.auth import require_auth

if TYPE_CHECKING:
    from haiku.rag.client.scope import DatabaseRef
    from haiku.rag.config import AppConfig
    from haiku.rag.ingester.app import ReloadReport
    from haiku.rag.ingester.pollers.manager import PollerManager
    from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo
    from haiku.rag.ingester.workers.pool import WorkerPool
    from haiku.rag.service.search import SearchService
    from haiku.rag.service.throttle import Throttle


@dataclass
class APIState:
    """Everything the API handlers need to read or act on. Pool/pollers are
    optional so the FastAPI app can be tested in isolation. `databases` holds
    the ingested databases by name, `failed_databases` the configured ones
    that failed to start, with their errors, and `source_databases` maps a
    served source id to the name of its database. A retry holds `claim_lock`, the
    pool's, so it cannot land between a removal's unserve and its delete.
    `reload` is the running app's, and `request_reload` queues one without
    waiting for it. `search` answers the search and document endpoints, and
    `throttle` is what holds conversions back, when the app runs one.
    `last_request_at` is the monotonic time of the latest request."""

    config: "AppConfig"
    job_repo: "JobRepo"
    sync_repo: "SyncStateRepo"
    pool: "WorkerPool | None" = None
    pollers: "PollerManager | None" = None
    databases: "Mapping[str, DatabaseRef]" = field(default_factory=dict)
    failed_databases: Mapping[str, str] = field(default_factory=dict)
    source_databases: Mapping[str, str] = field(default_factory=dict)
    claim_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    reload: "Callable[[], Awaitable[ReloadReport]] | None" = None
    request_reload: "Callable[[], None] | None" = None
    search: "SearchService | None" = None
    throttle: "Throttle | None" = None
    last_request_at: float = field(default_factory=time.monotonic)


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
        folders,
        health,
        jobs,
        providers,
        reload,
        service,
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

    @app.middleware("http")
    async def _note_activity(request: Request, call_next):
        state.last_request_at = time.monotonic()
        return await call_next(request)

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
    app.include_router(reload.router, dependencies=auth_dep)
    app.include_router(folders.router, dependencies=auth_dep)
    app.include_router(service.router, dependencies=auth_dep)
    # The dashboards poll these every few seconds.
    polled = "|".join(("health", "sources", "providers", "stats", "jobs"))
    logfire.instrument_fastapi(
        app,
        excluded_urls=[f"^[^:]+://[^/]+(?:{re.escape(root_path)})?/(?:{polled})$"],
    )

    return app
