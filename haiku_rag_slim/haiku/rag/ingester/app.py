import asyncio
import logging
import signal
from collections.abc import Callable, Collection
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel
from sqlalchemy.engine import make_url

from haiku.rag.config import AppConfig, SourceConfig
from haiku.rag.ingester.batch import BatchChange, BatchDryRunReport, BatchManifest
from haiku.rag.ingester.exceptions import (
    DuplicateSourceError,
    NothingToIngestError,
    QueueError,
)
from haiku.rag.ingester.metadata import build_providers, load_metadata_providers
from haiku.rag.ingester.pollers.base import max_attempts
from haiku.rag.ingester.pollers.manager import PollerManager
from haiku.rag.ingester.queue.migrations import open_queue
from haiku.rag.ingester.queue.models import Job, JobStatus
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo
from haiku.rag.ingester.reconcile import reconcile
from haiku.rag.ingester.workers.pool import WorkerPool
from haiku.rag.ingester.workers.retry import RetryPolicy
from haiku.rag.uri import is_local_uri, uri_to_path

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from haiku.rag.client import HaikuRAG
    from haiku.rag.client.scope import DatabaseRef

logger = logging.getLogger(__name__)

_MANIFEST_EXTRA_KEY = "_manifest"


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
    rejected: int = 0
    retried: int = 0
    failed_sweeps: list[str] = []


class BatchProgress(BaseModel):
    """Snapshot emitted while a one-shot batch drains queued work."""

    total: int = 0
    completed: int = 0
    succeeded: int = 0
    dead: int = 0
    rejected: int = 0
    queued: int = 0
    claimed: int = 0


BatchProgressCallback = Callable[[BatchProgress], None]


def _manifest_change_key(change: BatchChange) -> tuple[str, str, str, str | None]:
    return (change.source_id, change.uri, change.op.value, change.revision)


def _manifest_job_key(job: Job) -> tuple[str, str, str, str | None]:
    return (job.source_id, job.uri, job.op.value, job.revision)


class IngesterApp:
    """Top-level lifecycle for the production ingester.

    Owns: queue engine, JobRepo/SyncStateRepo, PollerManager,
    WorkerPool, and a HaikuRAG client for the worker pool to ingest through.
    """

    def __init__(self, *, config: AppConfig, database_name: str | None = None):
        """The ingester over every database `ingester.databases` names, or
        over the one `database_name` selects, fed by its configured sources."""
        from haiku.rag.client.scope import DatabaseScope
        from haiku.rag.store.exceptions import UnknownDatabaseError

        self._config = config
        ingested = config.ingester.databases
        if not ingested:
            raise NothingToIngestError(
                "ingester.databases names no database. Map each database to "
                "the sources it is ingested from"
            )
        if database_name is not None and database_name not in ingested:
            raise UnknownDatabaseError(
                f"unknown database {database_name!r}: ingester.databases names "
                f"{', '.join(ingested) or 'none'}"
            )
        databases = [
            (
                DatabaseScope.resolve(config, database_name=name).databases[0],
                entry.sources,
            )
            for name, entry in ingested.items()
        ]
        _refuse_shared_locations([ref for ref, _ in databases])
        self._databases: list[tuple[DatabaseRef, list[SourceConfig]]] = [
            (ref, sources)
            for ref, sources in databases
            if database_name in (None, ref.name)
        ]
        self._unselected_sources = [
            source
            for ref, sources in databases
            if database_name not in (None, ref.name)
            for source in sources
        ]
        self._configured_source_ids: set[str] = set()
        queue = config.ingester.queue
        if (
            queue.dburi
            and not queue.namespace.strip()
            and make_url(queue.dburi).get_backend_name() != "sqlite"
        ):
            raise QueueError(
                "ingester.queue.namespace is required when dburi names a server: "
                "it tells this ingester's rows from other ingesters' in the "
                "shared queue"
            )
        self._engine: AsyncEngine | None = None
        self._jobs: JobRepo | None = None
        self._sync: SyncStateRepo | None = None
        self._pool: WorkerPool | None = None
        self._pollers: PollerManager | None = None

    @asynccontextmanager
    async def _resources(self):
        """Open the queue engine and build the ingestion around it. Yields with
        everything built but nothing started: callers own the start/stop
        lifecycle. Closes the clients and the queue engine on exit."""
        self._engine = await open_queue(self._config.ingester.queue)
        try:
            self._jobs = JobRepo(self._engine, self._config.ingester.queue.namespace)
            self._sync = SyncStateRepo(
                self._engine, self._config.ingester.queue.namespace
            )
            async with self._ingestion():
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
    async def _ingestion(self):
        """The clients, pollers and worker pool built from the configuration,
        with each database reconciled against its sources."""
        assert self._jobs is not None and self._sync is not None
        ingester_cfg = self._config.ingester
        self._pollers = await self._build_pollers()
        source_databases = self._source_databases()
        async with AsyncExitStack() as stack:
            try:
                clients = {
                    ref.name: await self._open(stack, ref) for ref, _ in self._databases
                }
                # Every local database exists once open, so comparing directory
                # identities catches two names the filesystem treats as one.
                _refuse_shared_locations([ref for ref, _ in self._databases])
            except BaseException:
                await self._pollers.close_sources()
                raise
            cancelled = await self._jobs.cancel_unconfigured(
                self._configured_source_ids
            )
            if cancelled:
                logger.info(
                    "Deleted %d queued or claimed job(s) of sources no longer "
                    "configured",
                    cancelled,
                )
            metadata_providers = build_providers(
                [
                    (poller.source_id, poller.config.metadata_provider)
                    for poller in self._pollers.pollers
                ],
                load_metadata_providers(),
            )
            self._pool = WorkerPool(
                clients={
                    source_id: clients[database]
                    for source_id, database in source_databases.items()
                },
                job_repo=self._jobs,
                sync_repo=self._sync,
                worker_count=ingester_cfg.workers.worker_count,
                retry_policy=RetryPolicy(**ingester_cfg.workers.retry.model_dump()),
                source_retry_policies={
                    poller.source_id: RetryPolicy(**poller.config.retry.model_dump())
                    for poller in self._pollers.pollers
                    if poller.config.retry is not None
                },
                poll_idle_interval_s=ingester_cfg.workers.poll_idle_interval_s,
                lease_ttl_s=ingester_cfg.workers.lease_ttl_s,
                heartbeat_interval_s=ingester_cfg.workers.heartbeat_interval_s,
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
            # Between process lifetimes either database can be restored,
            # rebuilt or lost on its own, so the two are brought back in
            # step before any sweep reads sync_state.
            for ref, _ in self._databases:
                await reconcile(
                    clients[ref.name],
                    self._sync,
                    [
                        source_id
                        for source_id, database in source_databases.items()
                        if database == ref.name
                    ],
                )
            yield

    async def _open(self, stack: AsyncExitStack, ref: "DatabaseRef") -> "HaikuRAG":
        """A writable client over `ref`, created on first start and closed
        with `stack`. An error opening it names the database."""
        from haiku.rag.client import HaikuRAG
        from haiku.rag.client.scope import DatabaseScope
        from haiku.rag.store.exceptions import (
            ConfigMismatchError,
            MigrationRequiredError,
        )

        try:
            return await stack.enter_async_context(
                HaikuRAG._covering(DatabaseScope((ref,)), self._config, create=True)
            )
        except (ConfigMismatchError, MigrationRequiredError) as error:
            raise type(error)(f"database {ref.name!r}: {error}") from error

    async def _build_pollers(self) -> PollerManager:
        """One poller per source of the selected databases, together. Records
        the id of every configured source, selected or not, and raises
        DuplicateSourceError when two share one."""
        from haiku.rag.converters import get_converter
        from haiku.rag.ingester.pollers.factory import build_source

        assert self._jobs is not None and self._sync is not None
        supported_extensions = get_converter(self._config).supported_extensions
        manager = PollerManager(
            configs=[source for _, sources in self._databases for source in sources],
            job_repo=self._jobs,
            sync_repo=self._sync,
            supported_extensions=supported_extensions,
            default_max_attempts=self._config.ingester.workers.retry.max_attempts,
        )
        ids = [poller.source_id for poller in manager.pollers]
        # An unselected source's id is known only once its adapter is built.
        try:
            for config in self._unselected_sources:
                source = build_source(config, supported_extensions=supported_extensions)
                ids.append(source.source_id)
                await source.aclose()
        except BaseException:
            await manager.close_sources()
            raise
        self._configured_source_ids = set(ids)
        duplicates = sorted(
            {source_id for source_id in ids if ids.count(source_id) > 1}
        )
        if duplicates:
            await manager.close_sources()
            raise DuplicateSourceError(
                f"source id(s) {', '.join(duplicates)} name several configured "
                "sources. Give each source its own id"
            )
        return manager

    def _source_databases(self) -> dict[str, str]:
        """Source id to the name of the database it is ingested into."""
        assert self._pollers is not None
        databases = [ref.name for ref, sources in self._databases for _ in sources]
        return {
            poller.source_id: database
            for poller, database in zip(self._pollers.pollers, databases)
        }

    @asynccontextmanager
    async def _discovery_resources(self):
        """Open only the queue and source pollers needed for discovery.
        Dry-runs must not create/open the LanceDB document store or worker
        pool because they are upstream checks only."""
        ingester_cfg = self._config.ingester
        self._engine = await open_queue(ingester_cfg.queue)
        try:
            self._jobs = JobRepo(self._engine, ingester_cfg.queue.namespace)
            self._sync = SyncStateRepo(self._engine, ingester_cfg.queue.namespace)
            self._pollers = await self._build_pollers()
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
            # Cancelling the workers triggers their cancel-cleanup, which
            # releases each in-flight job back to `queued` (drained just below).
            # Any release that doesn't land has its lease stop being renewed, so
            # the reaper reclaims it after lease_ttl_s.
            logger.warning(
                "Shutdown grace of %.1fs elapsed with jobs still in flight; "
                "cancelling — they'll be released back to the queue (or "
                "reclaimed after lease_ttl_s if release doesn't land)",
                grace_s,
            )
        landed = await self._pool.drain_pending_releases(timeout=2.0)
        if landed:
            logger.info("Drained %d cancel-cleanup release(s) before close", landed)

    async def _drain_batch(
        self,
        started_at: datetime,
        *,
        progress_callback: BatchProgressCallback | None = None,
    ) -> BatchReport:
        assert self._pool is not None and self._jobs is not None
        total = 0
        while True:
            counts = await self._jobs.batch_progress_counts_since(
                started_at, source_ids=self._source_databases()
            )
            queued = counts.get("queued", 0)
            claimed = counts.get("claimed", 0)
            outstanding = queued + claimed
            succeeded = counts.get("succeeded", 0)
            dead = counts.get("dead", 0)
            rejected = counts.get("rejected", 0)
            completed_count = succeeded + dead + rejected
            total = max(total, outstanding + completed_count)
            if progress_callback is not None:
                progress_callback(
                    BatchProgress(
                        total=total,
                        completed=min(completed_count, total),
                        succeeded=succeeded,
                        dead=dead,
                        rejected=rejected,
                        queued=queued,
                        claimed=claimed,
                    )
                )
            if not outstanding:
                break
            if self._pool.live_workers == 0:
                logger.error(
                    "All workers have died with %d outstanding job(s) "
                    "— aborting batch; stranded jobs will be reaped "
                    "on next start",
                    outstanding,
                )
                break
            await asyncio.sleep(0.1)
        completed = await self._jobs.counts_by_status_since(
            started_at, source_ids=self._source_databases()
        )
        return BatchReport(
            succeeded=completed.get("succeeded", 0),
            dead=completed.get("dead", 0),
            rejected=completed.get("rejected", 0),
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
            if uses_docling_serve:  # pragma: no cover
                logger.info(
                    "Ingester running: %d worker(s), %d database(s), %d "
                    "source(s), %d docling-serve instance(s)",
                    ingester_cfg.workers.worker_count,
                    len(self._databases),
                    len(self._pollers.pollers),
                    len(self._config.providers.docling_serve.base_urls),
                )
            else:
                logger.info(
                    "Ingester running: %d worker(s), %d database(s), %d source(s)",
                    ingester_cfg.workers.worker_count,
                    len(self._databases),
                    len(self._pollers.pollers),
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

    async def run_batch(
        self,
        *,
        retry: Collection[JobStatus] = (),
        progress_callback: BatchProgressCallback | None = None,
    ) -> BatchReport:
        """Run one discover() sweep across every configured source, drain the
        queue to completion, then stop. Unlike `serve`, the periodic poller
        loops never start — discovery is driven explicitly, so the run is
        deterministic and exits as soon as the queue is empty.

        `retry` re-queues the latest failure of each URI in those statuses
        after the sweep: before it, the re-queued jobs would make each
        poller skip its source as one with pending work."""
        async with self._resources():
            assert (
                self._pollers is not None
                and self._pool is not None
                and self._jobs is not None
            )
            # A persisted queue carries terminal rows from previous runs, and
            # a successful DELETE prunes a URI's dead rows mid-run, so the report
            # counts only jobs that completed at or after this start instant.
            started_at = datetime.now(UTC)
            await self._pool.start()
            try:
                failed_sweeps = await self._pollers.sweep_all()
                retried = await self._jobs.retry_failed(
                    retry, source_ids=self._source_databases()
                )
                report = await self._drain_batch(
                    started_at, progress_callback=progress_callback
                )
                report.failed_sweeps = failed_sweeps
                report.retried = retried
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

    async def run_batch_from_manifest(
        self,
        manifest: BatchManifest,
        *,
        progress_callback: BatchProgressCallback | None = None,
    ) -> BatchReport:
        """Enqueue and drain a dry-run manifest without running a fresh
        discovery sweep."""
        if manifest.version != 1:  # pragma: no cover
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
                    "Manifest names source(s) this run does not ingest: "
                    + ", ".join(missing)
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

            manifest_key = manifest.generated_at.isoformat()
            manifest_change_keys = {
                _manifest_change_key(change) for change in manifest.changes
            }
            live_jobs = await self._jobs.live_jobs(configured)
            stale_jobs: list[Job] = []
            live_manifest_keys: set[tuple[str, str, str, str | None]] = set()
            for job in live_jobs:
                extra = job.extra or {}
                job_manifest = extra.get(_MANIFEST_EXTRA_KEY) or {}
                key = _manifest_job_key(job)
                if (
                    job_manifest.get("generated_at") != manifest_key
                    or key not in manifest_change_keys
                ):
                    stale_jobs.append(job)
                    continue
                live_manifest_keys.add(key)
            if stale_jobs:
                await self._pollers.close_sources()
                raise ValueError(
                    "Cannot replay manifest while the queue has non-manifest "
                    f"pending work: {len(stale_jobs)} queued/claimed job(s)"
                )

            default_max_attempts = self._config.ingester.workers.retry.max_attempts
            max_attempts_by_source = {
                poller.source_id: max_attempts(poller.config, default_max_attempts)
                for poller in self._pollers.pollers
            }
            for change in manifest.changes:
                if _manifest_change_key(change) in live_manifest_keys:
                    continue
                job = await self._jobs.enqueue(
                    change.source_id,
                    change.uri,
                    op=change.op,
                    revision=change.revision,
                    max_attempts=max_attempts_by_source[change.source_id],
                    extra={
                        _MANIFEST_EXTRA_KEY: {
                            "version": manifest.version,
                            "generated_at": manifest_key,
                            "discovered_at": change.discovered_at.isoformat(),
                        }
                    },
                )
                if job is None:  # pragma: no cover
                    await self._pollers.close_sources()
                    raise ValueError(
                        "Cannot replay manifest because a live job already exists "
                        f"for {change.source_id}:{change.uri}"
                    )

            started_at = datetime.now(UTC)
            await self._pool.start()
            try:
                return await self._drain_batch(
                    started_at, progress_callback=progress_callback
                )
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
            databases={ref.name: ref for ref, _ in self._databases},
            source_databases=self._source_databases(),
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


def _refuse_shared_locations(refs: "list[DatabaseRef]") -> None:
    """Raise when two databases sit at one location: each would be a writer
    of the same tables."""
    from haiku.rag.store.exceptions import AmbiguousDatabaseError

    placed: dict[str | tuple[int, int], str] = {}
    for ref in refs:
        location = _location_identity(ref.location)
        if location in placed:
            raise AmbiguousDatabaseError(
                f"ingester.databases names {placed[location]!r} and {ref.name!r} "
                "at one location, and a database has one writer"
            )
        placed[location] = ref.name


def _location_identity(location: "str | Path") -> str | tuple[int, int]:
    """The identity of a database location: device and inode for an existing
    local directory, the resolved path for a missing one, the URI for a
    remote one."""
    if isinstance(location, str) and not is_local_uri(location):
        return location.rstrip("/")
    path = (location if isinstance(location, Path) else uri_to_path(location)).resolve()
    try:
        stat = path.stat()
    except FileNotFoundError:
        return str(path)
    return (stat.st_dev, stat.st_ino)
