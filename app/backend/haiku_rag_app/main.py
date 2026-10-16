import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from haiku.rag.client import HaikuRAG
from haiku.rag.client.scope import DatabaseScope
from haiku.rag.config import AppConfig, get_config
from haiku_rag_app import chat, doctor, inspect, sessions
from haiku_rag_app.sessions import Sessions

logger = logging.getLogger(__name__)


class Databases:
    """The configured databases, opened read-only on first use.

    A database that cannot be opened is retried on the next request, so the app
    starts and serves whatever the state of the databases.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.scope = DatabaseScope.resolve(config)
        self._client: HaikuRAG | None = None
        self._lock = asyncio.Lock()

    @property
    def names(self) -> tuple[str, ...]:
        return self.scope.names

    async def covering(self) -> HaikuRAG:
        """The client covering every configured database."""
        try:
            return await self._covering_client()
        except Exception as error:
            raise HTTPException(503, str(error)) from error

    async def _covering_client(self) -> HaikuRAG:
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    client = HaikuRAG._covering(self.scope, self.config, read_only=True)
                    await client.__aenter__()
                    self._client = client
        return self._client

    async def reader(self, name: str | None) -> HaikuRAG:
        """The client reading database `name`; `None` names the only one."""
        if name is None and self.scope.covers_multiple:
            raise HTTPException(400, f"name a database, one of {', '.join(self.names)}")
        if name is not None and name not in self.names:
            raise HTTPException(404, f"unknown database {name!r}")
        try:
            client = await self._covering_client()
            owner = await client.reader_for(name)
        except Exception as error:
            raise HTTPException(503, str(error)) from error
        assert owner is not None
        return owner

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None


async def health(request: Request) -> JSONResponse:
    config: AppConfig = request.app.state.config
    return JSONResponse(
        {
            "status": "ok",
            "qa_provider": config.qa.model.provider,
            "qa_model": config.qa.model.name,
            "vision": config.qa.model.vision,
            "databases": list(request.app.state.databases.names),
        }
    )


async def list_databases(request: Request) -> JSONResponse:
    """Each configured database with its availability and size."""
    from haiku.rag.store.info import get_database_stats

    databases: Databases = request.app.state.databases
    entries = []
    for name in databases.names:
        try:
            owner = await databases.reader(name)
            stats = await get_database_stats(owner.store.db, tag=owner.store.tag)
        except Exception as error:
            detail = error.detail if isinstance(error, HTTPException) else str(error)
            entries.append({"name": name, "available": False, "error": detail})
            continue
        entries.append(
            {
                "name": name,
                "available": True,
                "error": None,
                "tag": owner.store.tag,
                "documents": stats["documents"].get("num_rows", 0),
                "chunks": stats["chunks"].get("num_rows", 0),
                "has_vector_index": stats["chunks"].get("has_vector_index", False),
                "supports_image_search": owner.store.embedder.supports_images,
            }
        )
    return JSONResponse(
        {"vision": request.app.state.config.qa.model.vision, "databases": entries}
    )


async def http_error(_: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, HTTPException)
    return JSONResponse({"error": error.detail}, status_code=error.status_code)


def create_app(config: AppConfig, sessions_path: Path | None = None) -> Starlette:
    databases = Databases(config)
    session_store = Sessions(sessions_path or sessions.default_path())

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        try:
            yield
        finally:
            await databases.aclose()
            await session_store.aclose()

    app = Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/api/databases", list_databases, methods=["GET"]),
            *chat.routes,
            *sessions.routes,
            *inspect.routes,
            *doctor.routes,
        ],
        exception_handlers={HTTPException: http_error},
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.databases = databases
    app.state.sessions = session_store
    app.state.jobs = doctor.Jobs()
    app.state.retained = doctor.Retained()
    app.state.agent = None
    app.state.agent_lock = asyncio.Lock()
    logger.info(
        "Databases: %s; QA model: %s/%s",
        ", ".join(databases.names),
        config.qa.model.provider,
        config.qa.model.name,
    )
    return app


def app() -> Starlette:
    """The application as uvicorn serves it, configured from the environment."""
    from dotenv import find_dotenv, load_dotenv

    from haiku.rag.telemetry import configure as configure_telemetry

    load_dotenv(find_dotenv(usecwd=True))
    configure_telemetry(service_name="haiku-rag-app")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return create_app(get_config())
