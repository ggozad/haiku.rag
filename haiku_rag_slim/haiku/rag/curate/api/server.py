from dataclasses import dataclass, field

from fastapi import Depends, FastAPI, Request

from haiku.rag.client.scope import DatabaseScope
from haiku.rag.config import AppConfig
from haiku.rag.curate.store.repository import CurateRepository
from haiku.rag.curate.sweep import curated_scope
from haiku.rag.ingester.api.auth import require_auth


@dataclass
class APIState:
    config: AppConfig
    repository: CurateRepository
    scope: DatabaseScope = field(init=False)

    def __post_init__(self) -> None:
        self.scope = curated_scope(self.config)


def get_state(request: Request) -> APIState:
    return request.app.state.api_state


def build_app(
    state: APIState, *, auth_token: str | None = None, root_path: str = ""
) -> FastAPI:
    """haiku-curate's HTTP API; everything but /health needs the token when one is set."""
    from haiku.rag.curate.api import routes

    app = FastAPI(
        title="haiku-curate",
        description="Curation of haiku.rag databases.",
        version="1",
        root_path=root_path,
    )
    app.state.api_state = state
    app.state.auth_token = auth_token
    app.include_router(routes.public)
    app.include_router(routes.router, dependencies=[Depends(require_auth)])
    return app
