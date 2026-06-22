import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from haiku.rag.config import AppConfig
from haiku.rag.ingester.batch import BatchDryRunReport, BatchManifest
from haiku.rag.ingester.metadata import build_providers, load_metadata_providers
from haiku.rag.ingester.pollers.manager import PollerManager
from haiku.rag.ingester.queue.migrations import open_queue
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo
from haiku.rag.ingester.workers.pool import WorkerPool
from haiku.rag.ingester.workers.retry import RetryPolicy

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)


def _api_access_log_enabled() -> bool:
    """Per-request access logging only when the haiku.rag logger is at DEBUG.
    The dashboard polls the control plane every few seconds, so at the normal
    INFO level the access log is pure noise."""
    return logging.getLogger("haiku.rag").isEnabledFor(logging.DEBUG)


class BatchReport(BaseModel):
    """Outcome of a one-shot batch run: terminal job counts after the queue
    drained, plus any sources whose discovery sweep did not complete."""

    succeeded: int = 0
    dead: int = 0
    failed_sweeps: list[str] = []


class IngesterApp:
    """Top-level lifecycle for the production ingester.

    Owns: queue engine, JobRepo/SyncStateRepo, PollerManager,
    WorkerPool, and a HaikuRAG client for the worker pool to ingest through.
    """

    def __init__(self, *, config: AppConfig, db_path: Path):
        self._config = config
        self._db_path = db_path
        self._engine: AsyncEngine | None = None
        self._jobs: JobRepo | None = None
        self._sync: SyncStateRepo | None = None
        self._pool: WorkerPool | None = None
        self._pollers: PollerManager | None = None

    @asynccontextmanager
    async def _resources(self):
        """Open the queue engine and construct the repos, client, pollers
        and worker pool. Yields with everything built but nothing started —
        callers own the start/stop lifecycle. Closes the client and queue
        engine on exit."""
        from haiku.rag.client import HaikuRAG
        from haiku.rag.converters import get_converter

        ingester_cfg = self._config.ingester
        self._engine = await open_queue(ingester_cfg.queue)
        try:
            self._jobs = JobRepo(self._engine)
            self._sync = SyncStateRepo(self._engine)

            supported_extensions = get_converter(self._config).supported_extensions
            retry = RetryPolicy(
                max_attempts=ingester_cfg.workers.retry.max_attempts,
                base_delay_s=ingester_cfg.workers.retry.base_delay_s,
                max_delay_s=ingester_cfg.workers.retry.max_delay_s,
                jitter=ingester_cfg.workers.retry.jitter,
            )

            # The ingester is the sole writer for its LanceDB target; create on
            # first start so docker-compose / fresh deployments don't require a
            # manual `haiku-rag init`.
            async with HaikuRAG(
                self._db_path, config=self._config, create=True
            ) as client:
                self._pollers = PollerManager(
                    configs=ingester_cfg.sources,
                    job_repo=self._jobs,
                    sync_repo=self._sync,
                    supported_extensions=supported_extensions,
                    default_max_attempts=ingester_cfg.workers.retry.max_attempts,
                )
                metadata_providers = build_providers(
                    [
                        (source.source_id, cfg.metadata_provider)
                        for cfg, source in zip(
                            ingester_cfg.sources, self._pollers.sources
                        )
                    ],
                    load_metadata_providers(),
                )
                self._pool = WorkerPool(
                    client=client,
                    job_repo=self._jobs,
                    sync_repo=self._sync,
                    worker_count=ingester_cfg.workers.worker_count,
                    retry_policy=retry,
                    poll_idle_interval_s=ingester_cfg.workers.poll_idle_interval_s,
                    claim_timeout_s=ingester_cfg.workers.claim_timeout_s,
                    reaper_interval_s=ingester_cfg.workers.reaper_interval_s,
                    retention_s=(
                        ingester_cfg.queue.retention_days * 86400
                        if ingester_cfg.queue.retention_days is not None
                        else None
                    ),
                    # Same Source instances the pollers discover with —
                    # workers resolve URIs through them so authenticated
                    # HTTP / WebDAV / S3 fetches reuse credentials.
                    sources=self._pollers.sources,
                    metadata_providers=metadata_providers,
                )
                yield
        finally:
            # Dispose the engine unconditionally. aiosqlite runs the underlying
            # sqlite3 in a background thread; leaving the pool open holds the
            # event loop alive and blocks process exit on early failures (e.g.
            # HaikuRAG raising MigrationRequiredError).
            if self._engine is not None:
                await self._engine.dispose()
                self._engine = None

    @asynccontextmanager
    async def _discovery_resources(self):
        """Open only the queue and source pollers needed for discovery.
        Dry-runs must not create/open the LanceDB document store or worker
        pool because they are upstream checks only."""
        from haiku.rag.converters import get_converter

        ingester_cfg = self._config.ingester
        self._engine = await open_queue(ingester_cfg.queue)
        try:
            self._jobs = JobRepo(self._engine)
            self._sync = SyncStateRepo(self._engine)
            supported_extensions = get_converter(self._config).supported_extensions
            self._pollers = PollerManager(
                configs=ingester_cfg.sources,
                job_repo=self._jobs,
                sync_repo=self._sync,
                supported_extensions=supported_extensions,
                default_max_attempts=ingester_cfg.workers.retry.max_attempts,
            )
            yield
        finally:
            if self._pollers is not None:
                await self._pollers.close_sources()
                self._pollers = None
            if self._engine is not None:
                await self._engine.dispose()
                self._engine = None

    async def _stop_pool(self) -> None:
        """Stop the worker pool, honouring the shutdown grace, then drain any
        cancel-cleanup release tasks before the queue connection closes."""
        assert self._pool is not None
        grace_s = self._config.ingester.workers.shutdown_grace_s
        try:
            await asyncio.wait_for(self._pool.stop(), timeout=grace_s)
        except TimeoutError:
            # In-flight jobs stay 'claimed'; the reaper resets them after
            # claim_timeout_s on the next start.
            logger.warning(
                "Shutdown grace of %.1fs elapsed with jobs still in flight; "
                "cancelling — they'll be reclaimed after claim_timeout_s on "
                "next start",
                grace_s,
            )
        landed = await self._pool.drain_pending_releases(timeout=2.0)
        if landed:
            logger.info("Drained %d cancel-cleanup release(s) before close", landed)

    async def _drain_batch(self, started_at: datetime) -> BatchReport:
        assert self._pool is not None and self._jobs is not None
        while True:
            counts = await self._jobs.counts_by_status()
            if not counts.get("queued") and not counts.get("claimed"):
                break
            if self._pool.live_workers == 0:
                outstanding = counts.get("queued", 0) + counts.get("claimed", 0)
                logger.error(
                    "All workers have died with %d outstanding job(s) "
                    "— aborting batch; stranded jobs will be reaped "
                    "on next start",
                    outstanding,
                )
                break
            await asyncio.sleep(0.1)
        completed = await self._jobs.counts_by_status_since(started_at)
        return BatchReport(
            succeeded=completed.get("succeeded", 0),
            dead=completed.get("dead", 0),
        )

    async def serve(self, *, api: bool = True) -> None:
        """Run pollers + workers (and the HTTP API when enabled) until a
        SIGINT/SIGTERM is received. Drains in-flight work on shutdown."""
        ingester_cfg = self._config.ingester
        async with self._resources():
            assert self._pollers is not None and self._pool is not None
            await self._pollers.start()
            await self._pool.start()
            # Log the docling-serve fleet size when relevant so the operator
            # can eyeball the worker/instance ratio. The convert phase is
            # usually the throughput ceiling.
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

            stop_event = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, stop_event.set)
                except NotImplementedError:  # pragma: no cover - Windows only
                    # Windows; signal handlers unavailable in asyncio.
                    pass

            try:
                await stop_event.wait()
            finally:
                logger.info("Shutting down ingester")
                if api_server is not None:
                    api_server.should_exit = True
                if api_task is not None:
                    await asyncio.gather(api_task, return_exceptions=True)
                await self._pollers.stop()
                await self._stop_pool()
                await self._pollers.close_sources()

    async def run_batch(self) -> BatchReport:
        """Run one discover() sweep across every configured source, drain the
        queue to completion, then stop. Unlike `serve`, the periodic poller
        loops never start — discovery is driven explicitly, so the run is
        deterministic and exits as soon as the queue is empty."""
        async with self._resources():
            assert (
                self._pollers is not None
                and self._pool is not None
                and self._jobs is not None
            )
            # A persisted queue carries terminal rows from previous runs, and
            # a recovered URI's dead row gets pruned mid-run, so the report
            # counts only jobs that completed at or after this start instant.
            started_at = datetime.now(UTC)
            await self._pool.start()
            try:
                failed_sweeps = await self._pollers.sweep_all()
                report = await self._drain_batch(started_at)
                report.failed_sweeps = failed_sweeps
                return report
            finally:
                await self._stop_pool()
                await self._pollers.close_sources()

    async def run_batch_dry_run(self) -> BatchDryRunReport:
        """Run one discover() sweep across every configured source and return
        the jobs that would be enqueued, without mutating jobs or sync_state."""
        async with self._discovery_resources():
            assert self._pollers is not None
            manifest, failed_sweeps = await self._pollers.dry_run_manifest()
            return BatchDryRunReport(manifest=manifest, failed_sweeps=failed_sweeps)

    async def run_batch_from_manifest(self, manifest: BatchManifest) -> BatchReport:
        """Enqueue and drain a dry-run manifest without running a fresh
        discovery sweep."""
        if manifest.version != 1:
            raise ValueError(f"Unsupported manifest version: {manifest.version}")
        async with self._resources():
            assert (
                self._pollers is not None
                and self._pool is not None
                and self._jobs is not None
            )
            configured = {source.source_id for source in self._pollers.sources}
            manifest_sources = {change.source_id for change in manifest.changes}
            missing = sorted(manifest_sources - configured)
            if missing:
                await self._pollers.close_sources()
                raise ValueError(
                    "Manifest references unconfigured source(s): " + ", ".join(missing)
                )

            counts = await self._jobs.counts_by_status()
            pending = counts.get("queued", 0) + counts.get("claimed", 0)
            if pending:
                await self._pollers.close_sources()
                raise ValueError(
                    "Cannot replay manifest while the queue has pending work: "
                    f"{pending} queued/claimed job(s)"
                )

            seen: set[tuple[str, str]] = set()
            duplicates: set[tuple[str, str]] = set()
            for change in manifest.changes:
                key = (change.source_id, change.uri)
                if key in seen:
                    duplicates.add(key)
                seen.add(key)
            if duplicates:
                await self._pollers.close_sources()
                rendered = ", ".join(
                    f"{source_id}:{uri}" for source_id, uri in duplicates
                )
                raise ValueError(f"Manifest contains duplicate change(s): {rendered}")

            default_max_attempts = self._config.ingester.workers.retry.max_attempts
            max_attempts_by_source = {
                poller.source_id: (
                    poller.config.retry.max_attempts
                    if poller.config.retry is not None
                    else default_max_attempts
                )
                for poller in self._pollers.pollers
            }
            for change in manifest.changes:
                job = await self._jobs.enqueue(
                    change.source_id,
                    change.uri,
                    op=change.op,
                    revision=change.revision,
                    max_attempts=max_attempts_by_source[change.source_id],
                    extra={
                        "_manifest": {
                            "version": manifest.version,
                            "generated_at": manifest.generated_at.isoformat(),
                            "discovered_at": change.discovered_at.isoformat(),
                        }
                    },
                )
                if job is None:
                    await self._pollers.close_sources()
                    raise ValueError(
                        "Cannot replay manifest because a live job already exists "
                        f"for {change.source_id}:{change.uri}"
                    )

            started_at = datetime.now(UTC)
            await self._pool.start()
            try:
                return await self._drain_batch(started_at)
            finally:
                await self._stop_pool()
                await self._pollers.close_sources()

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
            db_path=self._db_path,
        )
        if ingester_cfg.api.auth_token is None:
            logger.warning("API auth_token is unset — control plane is unauthenticated")
        app = build_app(
            state,
            auth_token=ingester_cfg.api.auth_token,
            root_path=ingester_cfg.api.root_path,
        )
        config = uvicorn.Config(
            app,
            host=ingester_cfg.api.host,
            port=ingester_cfg.api.port,
            root_path=ingester_cfg.api.root_path,
            log_level="info",
            access_log=_api_access_log_enabled(),
            lifespan="off",
        )
        server = uvicorn.Server(config)
        logger.info(
            "API listening on %s:%d%s",
            ingester_cfg.api.host,
            ingester_cfg.api.port,
            ingester_cfg.api.root_path,
        )
        return asyncio.create_task(server.serve()), server
