import asyncio
from dataclasses import dataclass, field

from fastapi import Depends, FastAPI, Request

from haiku.rag.client.scope import DatabaseScope
from haiku.rag.config import AppConfig
from haiku.rag.curate.projection import project
from haiku.rag.curate.store.repository import CurateRepository
from haiku.rag.curate.sweep import curated_scope
from haiku.rag.ingester.api.auth import require_auth


@dataclass
class APIState:
    config: AppConfig
    repository: CurateRepository
    scope: DatabaseScope = field(init=False)
    # Set by `serve` while its own sweep runs; a sweep in another process is not seen.
    sweeping: bool = field(init=False, default=False)
    _maps: dict[str, tuple[int | None, dict[str, tuple[float, float]]]] = field(
        init=False, default_factory=dict
    )
    # One lock for every database's map: a second request waits for the first.
    _map_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self.scope = curated_scope(self.config)

    async def map(self, database: str) -> dict[str, tuple[float, float]]:
        """Map positions of `database`'s documents, computed once per changed sweep."""
        async with self._map_lock:
            last = await self.repository.last_ok_sweep(database)
            key = last.id if last is not None else None
            cached = self._maps.get(database)
            if cached is None or cached[0] != key:
                centroids = await self.repository.current_centroids(database)
                cached = (key, await asyncio.to_thread(project, centroids))
                self._maps[database] = cached
            return cached[1]


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
