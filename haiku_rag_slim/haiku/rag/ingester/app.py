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
        self._jobs = JobRepo(self._queue_conn)
        self._sync = SyncStateRepo(self._queue_conn)

        supported_extensions = get_converter(self._config).supported_extensions
        retry = RetryPolicy(
            max_attempts=ingester_cfg.workers.retry.max_attempts,
            base_delay_s=ingester_cfg.workers.retry.base_delay_s,
            max_delay_s=ingester_cfg.workers.retry.max_delay_s,
            jitter=ingester_cfg.workers.retry.jitter,
        )

        async with HaikuRAG(self._db_path, config=self._config) as client:
            self._client = client
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
            )
            self._pollers = PollerManager(
                configs=ingester_cfg.sources,
                job_repo=self._jobs,
                sync_repo=self._sync,
                supported_extensions=supported_extensions,
                default_max_attempts=ingester_cfg.workers.retry.max_attempts,
            )

            stop_event = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, stop_event.set)
                except NotImplementedError:
                    # Windows; signal handlers unavailable in asyncio.
                    pass

            await self._pool.start()
            await self._pollers.start()
            logger.info(
                "Ingester running: %d worker(s), %d source(s)",
                ingester_cfg.workers.worker_count,
                len(ingester_cfg.sources),
            )

            if api:
                # HTTP control plane lands in a follow-up; for now this branch
                # is a no-op so callers can still pass api=True without error.
                logger.info(
                    "HTTP API not yet implemented; running pollers + workers only"
                )

            try:
                await stop_event.wait()
            finally:
                logger.info("Shutting down ingester")
                await self._pollers.stop()
                await self._pool.stop()

        if self._queue_conn is not None:
            await self._queue_conn.close()
            self._queue_conn = None
