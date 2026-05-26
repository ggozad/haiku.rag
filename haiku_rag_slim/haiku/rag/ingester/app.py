import asyncio
import logging
import signal
from pathlib import Path
from typing import TYPE_CHECKING

from haiku.rag.config import AppConfig
from haiku.rag.ingester.pollers.manager import PollerManager
from haiku.rag.ingester.queue.migrations import open_queue
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo
from haiku.rag.ingester.workers.pool import WorkerPool
from haiku.rag.ingester.workers.retry import RetryPolicy

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)


class IngesterApp:
    """Top-level lifecycle for the production ingester.

    Owns: SQLite queue connection, JobRepo/SyncStateRepo, PollerManager,
    WorkerPool, and a HaikuRAG client for the worker pool to ingest through.
    """

    def __init__(self, *, config: AppConfig, db_path: Path):
        self._config = config
        self._db_path = db_path
        self._queue_conn: aiosqlite.Connection | None = None
        self._jobs: JobRepo | None = None
        self._sync: SyncStateRepo | None = None
        self._pool: WorkerPool | None = None
        self._pollers: PollerManager | None = None
        self._client = None

    async def serve(self, *, api: bool = True) -> None:
        """Run pollers + workers (and the HTTP API when enabled) until a
        SIGINT/SIGTERM is received. Drains in-flight work on shutdown."""
        from haiku.rag.client import HaikuRAG
        from haiku.rag.converters import get_converter

        ingester_cfg = self._config.ingester
        self._queue_conn = await open_queue(ingester_cfg.queue.path)
        try:
            self._jobs = JobRepo(self._queue_conn)
            self._sync = SyncStateRepo(self._queue_conn)

            supported_extensions = get_converter(self._config).supported_extensions
            retry = RetryPolicy(
                max_attempts=ingester_cfg.workers.retry.max_attempts,
                base_delay_s=ingester_cfg.workers.retry.base_delay_s,
                max_delay_s=ingester_cfg.workers.retry.max_delay_s,
                jitter=ingester_cfg.workers.retry.jitter,
            )

            # The ingester is the sole writer for its LanceDB target;
            # create on first start so docker-compose / fresh deployments
            # don't require a manual `haiku-rag init`.
            async with HaikuRAG(
                self._db_path, config=self._config, create=True
            ) as client:
                self._client = client
                self._pollers = PollerManager(
                    configs=ingester_cfg.sources,
                    job_repo=self._jobs,
                    sync_repo=self._sync,
                    supported_extensions=supported_extensions,
                    default_max_attempts=ingester_cfg.workers.retry.max_attempts,
                )
                self._pool = WorkerPool(
                    client=client,
                    job_repo=self._jobs,
                    sync_repo=self._sync,
                    worker_count=ingester_cfg.workers.worker_count,
                    max_concurrent=ingester_cfg.workers.max_concurrent,
                    retry_policy=retry,
                    poll_idle_interval_s=ingester_cfg.workers.poll_idle_interval_s,
                    claim_timeout_s=ingester_cfg.workers.claim_timeout_s,
                    reaper_interval_s=ingester_cfg.workers.reaper_interval_s,
                    # Same Source instances the pollers discover with —
                    # workers resolve URIs through them so authenticated
                    # HTTP / WebDAV / S3 fetches reuse credentials.
                    sources=self._pollers.sources,
                )

                stop_event = asyncio.Event()
                loop = asyncio.get_running_loop()
                for sig in (signal.SIGINT, signal.SIGTERM):
                    try:
                        loop.add_signal_handler(sig, stop_event.set)
                    except NotImplementedError:
                        # Windows; signal handlers unavailable in asyncio.
                        pass

                await self._pollers.start()
                await self._pool.start()
                # Log the docling-serve fleet size when relevant so the
                # operator can eyeball the worker/instance ratio. The convert
                # phase is usually the throughput ceiling.
                proc = self._config.processing
                uses_docling_serve = (
                    proc.converter == "docling-serve" or proc.chunker == "docling-serve"
                )
                if uses_docling_serve:
                    logger.info(
                        "Ingester running: %d worker(s), %d source(s), "
                        "%d docling-serve instance(s)",
                        ingester_cfg.workers.worker_count,
                        len(ingester_cfg.sources),
                        len(self._config.providers.docling_serve.base_urls),
                    )
                else:
                    logger.info(
                        "Ingester running: %d worker(s), %d source(s)",
                        ingester_cfg.workers.worker_count,
                        len(ingester_cfg.sources),
                    )

                api_task, api_server = await self._maybe_start_api(api)

                try:
                    await stop_event.wait()
                finally:
                    logger.info("Shutting down ingester")
                    if api_server is not None:
                        api_server.should_exit = True
                    if api_task is not None:
                        await asyncio.gather(api_task, return_exceptions=True)
                    await self._pollers.stop()
                    grace_s = ingester_cfg.workers.shutdown_grace_s
                    try:
                        await asyncio.wait_for(self._pool.stop(), timeout=grace_s)
                    except TimeoutError:
                        # In-flight jobs stay 'claimed'; the reaper resets
                        # them after claim_timeout_s on the next start.
                        logger.warning(
                            "Shutdown grace of %.1fs elapsed with jobs still "
                            "in flight; cancelling — they'll be reclaimed after "
                            "claim_timeout_s on next start",
                            grace_s,
                        )
        finally:
            # Close the queue connection unconditionally. aiosqlite runs the
            # underlying sqlite3 in a background thread; leaving it open holds
            # the event loop alive and blocks process exit on early failures
            # (e.g. HaikuRAG raising MigrationRequiredError).
            if self._queue_conn is not None:
                await self._queue_conn.close()
                self._queue_conn = None

    async def _maybe_start_api(self, api: bool):
        """Spin up the FastAPI control plane on an asyncio task. Returns
        (task, server) or (None, None) when the API is disabled."""
        ingester_cfg = self._config.ingester
        if not (api and ingester_cfg.api.enabled):
            return None, None

        import uvicorn

        from haiku.rag.ingester.api.server import APIState, build_app

        assert self._jobs is not None and self._sync is not None
        state = APIState(
            config=self._config,
            job_repo=self._jobs,
            sync_repo=self._sync,
            pool=self._pool,
            pollers=self._pollers,
        )
        if ingester_cfg.api.auth_token is None:
            logger.warning("API auth_token is unset — control plane is unauthenticated")
        app = build_app(state, auth_token=ingester_cfg.api.auth_token)
        config = uvicorn.Config(
            app,
            host=ingester_cfg.api.host,
            port=ingester_cfg.api.port,
            log_level="info",
            lifespan="off",
        )
        server = uvicorn.Server(config)
        logger.info(
            "API listening on %s:%d", ingester_cfg.api.host, ingester_cfg.api.port
        )
        return asyncio.create_task(server.serve()), server
