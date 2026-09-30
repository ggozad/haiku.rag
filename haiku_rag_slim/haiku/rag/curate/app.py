import asyncio
import logging
import signal
from typing import TYPE_CHECKING

from haiku.rag.config import AppConfig
from haiku.rag.config.models import CurateStoreConfig
from haiku.rag.curate.store.migrations import open_store
from haiku.rag.curate.store.models import DatabaseSweep, SweepStatus
from haiku.rag.curate.store.repository import CurateRepository
from haiku.rag.curate.sweep import sweep

if TYPE_CHECKING:
    from haiku.rag.curate.api.server import APIState

logger = logging.getLogger(__name__)


class APIServerStopped(Exception):
    """The API server ended while haiku-curate serve was running."""


def describe(result: DatabaseSweep) -> str:
    """One line for a database's sweep."""
    if result.status is SweepStatus.OK:
        return (
            f"{result.database}: ok, {result.documents} documents, "
            f"{len(result.new)} new, {len(result.deleted)} deleted"
            + (", embedder changed" if result.rebaseline else "")
        )
    if result.status is SweepStatus.ERROR:
        return f"{result.database}: error: {result.error}"
    return f"{result.database}: {result.status}"


async def run_sweep(config: AppConfig) -> list[DatabaseSweep]:
    """Open the store, sweep every configured database once, close the store."""
    engine = await open_store(config.curate.store)
    try:
        return await sweep(config, CurateRepository(engine))
    finally:
        await engine.dispose()


async def ensure_store(store: CurateStoreConfig) -> None:
    """Create the store if missing and bring its schema up to date."""
    engine = await open_store(store)
    await engine.dispose()


async def serve(
    config: AppConfig, *, sweeping: bool = True, stop: asyncio.Event | None = None
) -> None:
    """Sweep every `curate.sweep_interval_s` and serve the API until `stop` or a signal."""
    stop = stop or asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # pragma: no cover - Windows only
            pass

    from haiku.rag.curate.api.server import APIState

    engine = await open_store(config.curate.store)
    state = APIState(config, CurateRepository(engine))
    stopped = asyncio.create_task(stop.wait())
    server = task = None
    try:
        if config.curate.api.enabled:
            server = _api_server(config, state)
            task = asyncio.create_task(server.serve())
        while not stop.is_set():
            if sweeping:
                state.sweeping = True
                try:
                    results = await asyncio.to_thread(_sweep_on_own_loop, config)
                finally:
                    state.sweeping = False
                for result in results:
                    level = (
                        logging.WARNING
                        if result.status is SweepStatus.ERROR
                        else logging.INFO
                    )
                    logger.log(level, describe(result))
            waiters = {stopped} if task is None else {stopped, task}
            done, _ = await asyncio.wait(
                waiters,
                timeout=config.curate.sweep_interval_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if task in done and not stop.is_set():
                task.result()
                raise APIServerStopped("the API server stopped")
    finally:
        stopped.cancel()
        if server is not None and task is not None and not task.done():
            server.should_exit = True
            await task
        await engine.dispose()


def _sweep_on_own_loop(config: AppConfig) -> list[DatabaseSweep]:
    """A sweep on its own event loop, run in a worker thread so the API's loop never waits on it."""
    return asyncio.run(run_sweep(config))


def _api_server(config: AppConfig, state: "APIState"):
    import uvicorn

    from haiku.rag.curate.api.server import build_app

    api = config.curate.api
    if api.auth_token is None:
        logger.warning("curate.api.auth_token is unset: the API is unauthenticated")
    app = build_app(
        state,
        auth_token=api.auth_token,
        root_path=api.root_path,
    )
    return uvicorn.Server(
        uvicorn.Config(
            app,
            host=api.host,
            port=api.port,
            root_path=api.root_path,
            log_level="warning",
            lifespan="off",
        )
    )
