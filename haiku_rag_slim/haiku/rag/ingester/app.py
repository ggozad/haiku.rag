import asyncio
import logging
import signal
import sys
from collections.abc import Awaitable, Callable, Collection, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel
from sqlalchemy.engine import make_url

from haiku.rag.config import AppConfig, SourceConfig, load_yaml_config
from haiku.rag.ingester.batch import BatchChange, BatchDryRunReport, BatchManifest
from haiku.rag.ingester.exceptions import (
    DuplicateSourceError,
    InvalidConfigError,
    NoDatabaseStartedError,
    NothingToIngestError,
    QueueError,
    ReloadCancelledError,
    ReloadRefusedError,
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
    from haiku.rag.ingester.api.server import APIState
    from haiku.rag.ingester.metadata import MetadataProvider
    from haiku.rag.ingester.pollers.base import BasePoller

logger = logging.getLogger(__name__)

_MANIFEST_EXTRA_KEY = "_manifest"


def _api_access_log_enabled() -> bool:
    """Per-request access logging only when the haiku.rag logger is at DEBUG.
    The dashboard polls the control plane every few seconds, so at the normal
    INFO level the access log is pure noise."""
    return logging.getLogger("haiku.rag").isEnabledFor(logging.DEBUG)


class FailedDatabase(BaseModel):
    """A configured database that is not running, and why."""

    database: str
    error: str


class BatchReport(BaseModel):
    """Outcome of a one-shot batch run: terminal job counts after the queue
    drained, plus any sources whose discovery sweep did not complete and any
    databases that failed to start."""

    succeeded: int = 0
    dead: int = 0
    rejected: int = 0
    retried: int = 0
    failed_sweeps: list[str] = []
    failed_databases: list[FailedDatabase] = []


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


class DatabaseChange(BaseModel):
    """What a reload did to one database."""

    database: str
    change: Literal["added", "removed", "failed"]
    error: str | None = None


class ReloadReport(BaseModel):
    """A reload's changes. `failed` means a database failed and nothing was
    added or removed, `partial` that some changes were applied and some
    failed."""

    outcome: Literal["applied", "failed", "partial"] = "applied"
    databases: list[DatabaseChange] = []

    def record(self, database: str, change: Literal["added", "removed"]) -> None:
        self._append(DatabaseChange(database=database, change=change))

    def fail(self, database: str, error: BaseException) -> None:
        self._append(
            DatabaseChange(database=database, change="failed", error=_message(error))
        )

    def _append(self, change: DatabaseChange) -> None:
        self.databases.append(change)
        failed = any(change.change == "failed" for change in self.databases)
        changed = any(change.change != "failed" for change in self.databases)
        if failed:
            self.outcome = "partial" if changed else "failed"


async def _close_each(closes: "list[Callable[[], Awaitable[object]]]") -> None:
    """Run every close, then raise the first failure, a cancellation or
    interrupt before any error."""
    failures: list[BaseException] = []
    for close in closes:
        try:
            await close()
        except BaseException as failure:
            failures.append(failure)
    for failure in failures:
        if not isinstance(failure, Exception):
            raise failure
    if failures:
        raise failures[0]


def failed_databases(failed: Mapping[str, str]) -> list[FailedDatabase]:
    return [
        FailedDatabase(database=name, error=error) for name, error in failed.items()
    ]


def _message(error: BaseException) -> str:
    return str(error) or type(error).__name__


def _manifest_change_key(change: BatchChange) -> tuple[str, str, str, str | None]:
    return (change.source_id, change.uri, change.op.value, change.revision)


def _manifest_job_key(job: Job) -> tuple[str, str, str, str | None]:
    return (job.source_id, job.uri, job.op.value, job.revision)


@dataclass
class _Database:
    """A database the app ingests into, with its sources' pollers. `opened`
    is what `_covering` returned, and `client` is what entering it returned.
    Closing exits `opened`, which retries a close that failed."""

    ref: "DatabaseRef"
    opened: "HaikuRAG"
    client: "HaikuRAG"
    pollers: "list[BasePoller]"
    metadata_providers: "dict[str, MetadataProvider]"
    removing: bool = False

    async def close_client(
        self,
        exc_type: type[BaseException] | None = None,
        exc: BaseException | None = None,
        tb: TracebackType | None = None,
    ) -> None:
        """Close the client, telling its hooks which exception is unwound."""
        await self.opened.__aexit__(exc_type, exc, tb)

    @property
    def source_ids(self) -> list[str]:
        return [poller.source_id for poller in self.pollers]


class IngesterApp:
    """Top-level lifecycle for the production ingester.

    Owns: queue engine, JobRepo/SyncStateRepo, PollerManager,
    WorkerPool, and a HaikuRAG client for the worker pool to ingest through.
    """

    def __init__(
        self,
        *,
        config: AppConfig,
        database_name: str | None = None,
        config_path: Path | None = None,
        api_overrides: "Mapping[str, Any] | None" = None,
    ):
        """The ingester over every database `ingester.databases` names, or
        over the one `database_name` selects, fed by its configured sources.

        `config_path` is the file `config` was read from, and `api_overrides`
        the `ingester.api` fields set over it: a reload reads the file again
        and sets them again. It excludes `database_name`."""
        from haiku.rag.store.exceptions import UnknownDatabaseError

        if database_name is not None and config_path is not None:
            raise ValueError(
                "database_name and config_path are exclusive: a reload applies "
                "every database the file names"
            )
        self._config = config
        self._config_path = config_path
        self._api_overrides = dict(api_overrides or {})
        ingested = config.ingester.databases
        if database_name is not None and database_name not in ingested:
            raise UnknownDatabaseError(
                f"unknown database {database_name!r}: ingester.databases names "
                f"{', '.join(ingested) or 'none'}"
            )
        databases = _ingested_databases(config)
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
        self._running: dict[str, _Database] = {}
        self._ingested: dict[str, DatabaseRef] = {}
        self._source_databases: dict[str, str] = {}
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
        self._api: APIState | None = None
        self._reload_lock = asyncio.Lock()
        self._reloads: set[asyncio.Task] = set()
        self._signal_reload_queued = False
        self._stopping = False
        # A reload's prepared additions, each kept until it is applied or
        # closed: the next reload and shutdown close what is left.
        self._prepared: list[_Database] = []
        # Configured databases that failed to start, by name, with the error.
        self._failed: dict[str, str] = {}

    def _require_databases(self) -> None:
        if not self._databases:
            raise NothingToIngestError(
                "ingester.databases names no database. Map each database to "
                "the sources it is ingested from"
            )

    @asynccontextmanager
    async def _resources(self):
        """Open the queue engine and build the ingestion around it. Yields with
        everything built but nothing started: callers own the start/stop
        lifecycle. Closes the clients and the queue engine on exit."""
        engine = self._engine = await open_queue(self._config.ingester.queue)
        try:
            self._jobs = JobRepo(engine, self._config.ingester.queue.namespace)
            self._sync = SyncStateRepo(engine, self._config.ingester.queue.namespace)
            async with self._ingestion():
                yield
        finally:
            # Dispose the engine unconditionally. aiosqlite runs the underlying
            # sqlite3 in a background thread; leaving the pool open holds the
            # event loop alive and blocks process exit on early failures (e.g.
            # HaikuRAG raising MigrationRequiredError).
            await engine.dispose()
            self._engine = None

    @asynccontextmanager
    async def _ingestion(self):
        """The clients, pollers and worker pool built from the configuration,
        with each database reconciled against its sources."""
        assert self._jobs is not None and self._sync is not None
        ingester_cfg = self._config.ingester
        self._pollers = await self._build_pollers()
        pollers = iter(self._pollers.pollers)
        prepared: list[_Database] = []
        try:
            for ref, sources in self._databases:
                database_pollers = [next(pollers) for _ in sources]
                try:
                    prepared.append(await self._prepare(ref, database_pollers))
                except Exception as error:
                    self._fail(ref.name, error)
                    await self._pollers.remove(
                        [poller.source_id for poller in database_pollers]
                    )
                    await self._close_adapters(ref.name, database_pollers)
            # Every local database exists once open, so comparing directory
            # identities catches two names the filesystem treats as one.
            _refuse_shared_locations([database.ref for database in prepared])
            if self._databases and not prepared:
                raise self._nothing_started()
        except BaseException:
            async with AsyncExitStack() as closing:
                closing.push_async_callback(self._pollers.close_sources)
                for database in prepared:
                    closing.push_async_exit(database.close_client)
                raise
        unapplied = list(prepared)
        try:
            self._pool = WorkerPool(
                clients={},
                job_repo=self._jobs,
                sync_repo=self._sync,
                worker_count=ingester_cfg.workers.worker_count,
                retry_policy=RetryPolicy(**ingester_cfg.workers.retry.model_dump()),
                poll_idle_interval_s=ingester_cfg.workers.poll_idle_interval_s,
                lease_ttl_s=ingester_cfg.workers.lease_ttl_s,
                heartbeat_interval_s=ingester_cfg.workers.heartbeat_interval_s,
                reaper_interval_s=ingester_cfg.workers.reaper_interval_s,
                retention_s=(
                    ingester_cfg.queue.retention_days * 86400
                    if ingester_cfg.queue.retention_days is not None
                    else None
                ),
            )
            for database in prepared:
                try:
                    await self._apply(database)
                except Exception as error:
                    await self._pollers.remove(database.source_ids)
                    await self._discard(database, error)
                unapplied.remove(database)
            if self._databases and not self._running:
                raise self._nothing_started()
            cancelled = await self._jobs.cancel_unconfigured(
                self._configured_source_ids
            )
            if not self._databases:
                logger.warning(
                    "ingester.databases names no database: nothing is ingested "
                    "until a reload adds one. Deleted %d queued or claimed "
                    "job(s) of this queue namespace",
                    cancelled,
                )
            elif cancelled:
                logger.info(
                    "Deleted %d queued or claimed job(s) of sources no longer "
                    "configured",
                    cancelled,
                )
            yield
        except BaseException:
            async with self._closing(unapplied):
                raise
        async with self._closing(unapplied):
            pass

    def _closing(self, unapplied: list[_Database]) -> AsyncExitStack:
        """Closes every database on exit, whichever close fails, passing on
        the exception it exits with."""
        closing = AsyncExitStack()
        for database in self._prepared:
            closing.push_async_exit(partial(self._close, database))
        for database in [*unapplied, *self._running.values()]:
            # A removal detaches its pollers from the manager, so its
            # adapters are closed here.
            if database.removing:
                closing.push_async_exit(partial(self._close, database))
            else:
                closing.push_async_exit(database.close_client)
        return closing

    def _fail(self, name: str, error: BaseException) -> None:
        """Record that the configured database `name` failed to start."""
        logger.error("Database %r failed to start and is skipped: %s", name, error)
        self._failed[name] = _message(error)

    def _nothing_started(self) -> NoDatabaseStartedError:
        failures = "; ".join(
            f"{name!r}: {error}" for name, error in self._failed.items()
        )
        return NoDatabaseStartedError(
            f"every configured database failed to start: {failures}"
        )

    async def _discard(self, database: _Database, error: BaseException) -> None:
        """Record `database` as failed with `error` and close it, logging a
        close that fails."""
        # A failed close leaks the client's handles until restart. The client
        # never reached the workers, so it cannot write.
        self._fail(database.ref.name, error)
        try:
            await self._close(database, type(error), error, error.__traceback__)
        except Exception as close_error:
            logger.error(
                "Database %r failed to close: %s", database.ref.name, close_error
            )

    async def _close_adapters(self, name: str, pollers: "list[BasePoller]") -> None:
        """Close the adapters of `pollers`, logging those that fail."""
        try:
            await _close_each([poller.source.aclose for poller in pollers])
        except Exception as error:
            logger.error("Database %r failed to close: %s", name, error)

    async def add_database(
        self, ref: "DatabaseRef", sources: list[SourceConfig]
    ) -> None:
        """Ingest into `ref` from `sources`, whether the app serves or not. A
        database that fails to open or reconcile leaves the app as it was.
        The caller checks that no running database has the name, the location
        or one of the source ids."""
        database = await self._prepare_addition(ref, sources)
        await self._apply_addition(database)

    async def _prepare_addition(
        self, ref: "DatabaseRef", sources: list[SourceConfig]
    ) -> _Database:
        """Build the adapters of `sources` and open `ref`."""
        assert self._pollers is not None
        pollers: list[BasePoller] = []
        try:
            for source in sources:
                pollers.append(self._pollers.build(source))
            return await self._prepare(ref, pollers)
        except BaseException:
            await self._close_adapters(ref.name, pollers)
            raise

    async def _apply_addition(self, database: _Database) -> None:
        """Apply a prepared database, closing it if that fails."""
        try:
            await self._apply(database)
        except BaseException:
            await self._close(database, *sys.exc_info())
            raise

    async def remove_database(self, name: str) -> None:
        """Stop ingesting into `name` once its jobs in flight finish, delete
        its queued jobs and `sync_state`, and close it. Its documents stay.

        Each step is idempotent. A removal that fails leaves the database
        `removing`, and the next removal of it starts again."""
        assert (
            self._pool is not None
            and self._pollers is not None
            and self._jobs is not None
            and self._sync is not None
        )
        database = self._running[name]
        database.removing = True
        source_ids = database.source_ids
        for source_id in source_ids:
            self._source_databases.pop(source_id, None)
        await self._pool.unserve(source_ids)
        await self._pollers.remove(source_ids)
        await self._pool.wait_idle(source_ids)
        await self._jobs.cancel_sources(source_ids)
        await self._sync.delete_sources(source_ids)
        self._pool.remove_sources(source_ids)
        await self._close(database)
        del self._running[name]
        del self._ingested[name]

    async def _prepare(
        self, ref: "DatabaseRef", pollers: "list[BasePoller]"
    ) -> _Database:
        """Build the metadata providers of `ref`'s sources and open it, writing
        nothing to the queue."""
        metadata_providers = build_providers(
            [(poller.source_id, poller.config.metadata_provider) for poller in pollers],
            load_metadata_providers(),
        )
        opened, client = await self._open(ref)
        return _Database(ref, opened, client, pollers, metadata_providers)

    async def _apply(self, database: _Database) -> None:
        """Reconcile `database` against its sources, then serve them. Between
        process lifetimes either side can be restored, rebuilt or lost on its
        own, so they are brought back in step before any sweep reads
        `sync_state`."""
        assert (
            self._pool is not None
            and self._pollers is not None
            and self._sync is not None
        )
        await reconcile(database.client, self._sync, database.source_ids)
        self._pool.add_sources(
            clients={source_id: database.client for source_id in database.source_ids},
            # The pollers' own adapters: workers fetch through them, so
            # authenticated HTTP / WebDAV / S3 fetches reuse credentials.
            sources=[poller.source for poller in database.pollers],
            source_retry_policies={
                poller.source_id: RetryPolicy(**poller.config.retry.model_dump())
                for poller in database.pollers
                if poller.config.retry is not None
            },
            metadata_providers=database.metadata_providers,
        )
        self._pollers.add(database.pollers)
        name = database.ref.name
        self._running[name] = database
        self._ingested[name] = database.ref
        self._source_databases.update(
            {source_id: name for source_id in database.source_ids}
        )

    async def _close(
        self,
        database: _Database,
        exc_type: type[BaseException] | None = None,
        exc: BaseException | None = None,
        tb: TracebackType | None = None,
    ) -> None:
        """Close the adapters, then the client, whichever close fails."""
        await _close_each(
            [
                *(poller.source.aclose for poller in database.pollers),
                partial(database.close_client, exc_type, exc, tb),
            ]
        )

    async def _open(self, ref: "DatabaseRef") -> "tuple[HaikuRAG, HaikuRAG]":
        """A writable client over `ref`, created on first start, as opened
        and as entered. An error opening it names the database."""
        from haiku.rag.client import HaikuRAG
        from haiku.rag.client.scope import DatabaseScope
        from haiku.rag.store.exceptions import (
            ConfigMismatchError,
            MigrationRequiredError,
            UpgradeRequiredError,
        )

        opened = HaikuRAG._covering(DatabaseScope((ref,)), self._config, create=True)
        try:
            return opened, await opened.__aenter__()
        except (
            ConfigMismatchError,
            MigrationRequiredError,
            UpgradeRequiredError,
        ) as error:
            raise type(error)(f"database {ref.name!r}: {error}") from error

    async def _build_pollers(self) -> PollerManager:
        """One poller per source of the selected databases, together. Records
        the id of every configured source, selected or not, and raises
        DuplicateSourceError when two share one."""
        from haiku.rag.converters import get_converter

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
        try:
            ids.extend(await self._source_ids(self._unselected_sources))
            _refuse_duplicate_ids(ids)
        except BaseException:
            await manager.close_sources()
            raise
        self._configured_source_ids = set(ids)
        return manager

    async def _source_ids(self, sources: list[SourceConfig]) -> list[str]:
        """The ids of `sources`. An id is known only once its adapter is
        built, so each is built and closed."""
        from haiku.rag.converters import get_converter
        from haiku.rag.ingester.pollers.factory import build_source

        supported_extensions = get_converter(self._config).supported_extensions
        ids = []
        for config in sources:
            source = build_source(config, supported_extensions=supported_extensions)
            ids.append(source.source_id)
            await source.aclose()
        return ids

    @asynccontextmanager
    async def _discovery_resources(self):
        """Open only the queue and source pollers needed for discovery.
        Dry-runs must not create/open the LanceDB document store or worker
        pool because they are upstream checks only."""
        ingester_cfg = self._config.ingester
        engine = self._engine = await open_queue(ingester_cfg.queue)
        try:
            self._jobs = JobRepo(engine, ingester_cfg.queue.namespace)
            self._sync = SyncStateRepo(engine, ingester_cfg.queue.namespace)
            self._pollers = await self._build_pollers()
            yield
        finally:
            if self._pollers is not None:
                await self._pollers.close_sources()
                self._pollers = None
            await engine.dispose()
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
                started_at, source_ids=self._source_databases
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
            started_at, source_ids=self._source_databases
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
                    len(self._running),
                    len(self._pollers.pollers),
                    len(self._config.providers.docling_serve.base_urls),
                )
            else:
                logger.info(
                    "Ingester running: %d worker(s), %d database(s), %d source(s)",
                    ingester_cfg.workers.worker_count,
                    len(self._running),
                    len(self._pollers.pollers),
                )

            api_task, api_server = await self._maybe_start_api(api)

            stop_event = asyncio.Event()
            loop = asyncio.get_running_loop()
            handlers: dict[int, Callable[[], None]] = {
                signal.SIGINT: stop_event.set,
                signal.SIGTERM: stop_event.set,
            }
            if sys.platform != "win32":  # pragma: no branch - Windows has no coverage
                handlers[signal.SIGHUP] = self._reload_on_signal
            installed = []
            for sig, handler in handlers.items():
                try:
                    loop.add_signal_handler(sig, handler)
                    installed.append(sig)
                except NotImplementedError:  # pragma: no cover - Windows only
                    # Windows; signal handlers unavailable in asyncio.
                    pass

            try:
                await stop_event.wait()
            finally:
                logger.info("Shutting down ingester")
                for sig in installed:
                    loop.remove_signal_handler(sig)
                # Cancelled first: a reload waiting on a removal's drain holds
                # the API's shutdown, and with it the pool's.
                self._stopping = True
                for task in list(self._reloads):
                    task.cancel()
                await asyncio.gather(*self._reloads, return_exceptions=True)
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
        self._require_databases()
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
                    retry, source_ids=self._source_databases
                )
                report = await self._drain_batch(
                    started_at, progress_callback=progress_callback
                )
                report.failed_sweeps = failed_sweeps
                report.retried = retried
                report.failed_databases = failed_databases(self._failed)
                return report
            finally:
                await self._stop_pool()
                await self._pollers.close_sources()

    async def run_batch_dry_run(self) -> BatchDryRunReport:
        """Run one discover() sweep across every configured source and return
        the jobs that would be enqueued, without mutating jobs or sync_state."""
        self._require_databases()
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
        self._require_databases()
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
                report = await self._drain_batch(
                    started_at, progress_callback=progress_callback
                )
                report.failed_databases = failed_databases(self._failed)
                return report
            finally:
                await self._stop_pool()
                await self._pollers.close_sources()

    async def reload(self) -> ReloadReport:
        """Read the configuration file again and add, remove or replace the
        databases that changed. Reloads run one at a time, and a caller that
        stops waiting does not stop the reload: only shutdown cancels it."""
        task = self._start_reload()
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                raise ReloadCancelledError(
                    "the ingester shut down before the reload finished"
                ) from None
            raise

    def _start_reload(self, on_start: Callable[[], None] | None = None) -> asyncio.Task:
        if self._stopping:
            raise ReloadCancelledError("the ingester is shutting down")

        async def _serialized() -> ReloadReport:
            async with self._reload_lock:
                if on_start is not None:
                    on_start()
                return await self._reload()

        task = asyncio.create_task(_serialized())
        self._reloads.add(task)
        task.add_done_callback(self._reloads.discard)
        return task

    def _reload_on_signal(self) -> None:
        """Queue a reload, unless one queued by a signal has not started."""
        if self._signal_reload_queued:
            return
        self._signal_reload_queued = True
        task = self._start_reload(on_start=self._signal_reload_started)
        task.add_done_callback(_log_signal_reload)

    def _signal_reload_started(self) -> None:
        self._signal_reload_queued = False

    async def _close_prepared(
        self, report: ReloadReport, failure: BaseException | None = None
    ) -> bool:
        """Close the prepared additions, telling their hooks the `failure`
        that ends the reload, and reporting those that fail. True when none is
        left."""
        exc_info = (
            (type(failure), failure, failure.__traceback__)
            if failure is not None
            else (None, None, None)
        )
        for database in list(self._prepared):
            try:
                await self._close(database, *exc_info)
            except Exception as error:
                report.fail(database.ref.name, error)
            else:
                self._prepared.remove(database)
        return not self._prepared

    async def _reload(self) -> ReloadReport:
        config, wanted = await self._read_configuration()
        report = ReloadReport()
        if not await self._close_prepared(report):
            return report
        for name in [name for name, db in self._running.items() if db.removing]:
            try:
                await self.remove_database(name)
            except Exception as error:
                report.fail(name, error)
                return report
            report.record(name, "removed")
        removals = [
            name
            for name, database in self._running.items()
            if name not in wanted
            or _placement(wanted[name][0]) != _placement(database.ref)
            or wanted[name][1] != [poller.config for poller in database.pollers]
        ]
        additions = [
            (ref, sources)
            for name, (ref, sources) in wanted.items()
            if name not in self._running or name in removals
        ]
        served = {
            _location_identity(database.ref.location)
            for database in self._running.values()
        }
        failure: BaseException | None = None
        completed = False
        try:
            # Additions sharing a running database's location are prepared
            # after the removals: a database has one writer.
            for ordered in (
                [
                    a
                    for a in additions
                    if _location_identity(a[0].location) not in served
                ],
                [a for a in additions if _location_identity(a[0].location) in served],
            ):
                for ref, sources in ordered:
                    try:
                        database = await self._prepare_addition(ref, sources)
                    except Exception as error:
                        report.fail(ref.name, error)
                        self._fail(ref.name, error)
                        continue
                    try:
                        # Every local database exists once open, so comparing
                        # directory identities catches two names for one.
                        _refuse_shared_locations(
                            [
                                d.ref
                                for d in [
                                    *self._running.values(),
                                    *self._prepared,
                                    database,
                                ]
                            ]
                        )
                    except Exception as error:
                        report.fail(ref.name, error)
                        await self._discard(database, error)
                        continue
                    self._prepared.append(database)
                for name in removals:
                    try:
                        await self.remove_database(name)
                    except Exception as error:
                        report.fail(name, error)
                        failure = error
                        return report
                    report.record(name, "removed")
                removals = []
            for database in list(self._prepared):
                try:
                    await self._apply(database)
                except Exception as error:
                    self._prepared.remove(database)
                    report.fail(database.ref.name, error)
                    await self._discard(database, error)
                    continue
                self._prepared.remove(database)
                report.record(database.ref.name, "added")
            completed = True
        except BaseException as error:
            failure = error
            raise
        finally:
            await self._close_prepared(report, failure)
            configured = wanted if completed else self._config.ingester.databases
            for name in [
                n for n in self._failed if n in self._running or n not in configured
            ]:
                del self._failed[name]
        self._config = config
        self._databases = list(wanted.values())
        if self._api is not None:
            self._api.config = config
        return report

    async def _read_configuration(
        self,
    ) -> "tuple[AppConfig, dict[str, tuple[DatabaseRef, list[SourceConfig]]]]":
        """The configuration file with the API overrides set again, and the
        databases it ingests. Raises when the running ingester cannot take it."""
        if self._config_path is None:
            raise ReloadRefusedError(
                "serve was started without a configuration file, so there is "
                "nothing to reload"
            )
        try:
            config = AppConfig.model_validate(load_yaml_config(self._config_path))
            for field, value in self._api_overrides.items():
                setattr(config.ingester.api, field, value)
        except Exception as error:
            raise InvalidConfigError(f"{self._config_path}: {error}") from error
        changed = _changed_sections(self._config, config)
        if changed:
            raise ReloadRefusedError(
                f"{', '.join(changed)} changed, and only lancedb.databases and "
                "ingester.databases are reloaded. Restart to apply the rest"
            )
        try:
            wanted = _ingested_databases(config)
            _refuse_duplicate_ids(
                await self._source_ids(
                    [source for _, sources in wanted for source in sources]
                )
            )
        except Exception as error:
            raise InvalidConfigError(str(error)) from error
        return config, {ref.name: (ref, sources) for ref, sources in wanted}

    def _api_state(self) -> "APIState":
        """The API's view of the running app: the maps it reads are the app's
        own, so they follow databases as they are added and removed."""
        from haiku.rag.ingester.api.server import APIState

        assert (
            self._jobs is not None and self._sync is not None and self._pool is not None
        )
        if self._api is None:
            self._api = APIState(
                config=self._config,
                job_repo=self._jobs,
                sync_repo=self._sync,
                pool=self._pool,
                pollers=self._pollers,
                databases=self._ingested,
                failed_databases=self._failed,
                source_databases=self._source_databases,
                claim_lock=self._pool.claim_lock,
                reload=self.reload,
            )
        return self._api

    async def _maybe_start_api(self, api: bool):
        """Spin up the FastAPI control plane on an asyncio task. Returns
        (task, server) or (None, None) when the API is disabled."""
        ingester_cfg = self._config.ingester
        if not (api and ingester_cfg.api.enabled):
            return None, None

        import uvicorn

        from haiku.rag.ingester.api.server import build_app

        if ingester_cfg.api.auth_token is None:
            logger.warning("API auth_token is unset — control plane is unauthenticated")
        app = build_app(
            self._api_state(),
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


def _ingested_databases(
    config: AppConfig,
) -> "list[tuple[DatabaseRef, list[SourceConfig]]]":
    """Each database named in `ingester.databases`, placed, with its sources.
    Raises when two sit at one location."""
    from haiku.rag.client.scope import DatabaseScope

    databases = [
        (
            DatabaseScope.resolve(config, database_name=name).databases[0],
            entry.sources,
        )
        for name, entry in config.ingester.databases.items()
    ]
    _refuse_shared_locations([ref for ref, _ in databases])
    return databases


def _refuse_duplicate_ids(ids: list[str]) -> None:
    duplicates = sorted({source_id for source_id in ids if ids.count(source_id) > 1})
    if duplicates:
        raise DuplicateSourceError(
            f"source id(s) {', '.join(duplicates)} name several configured "
            "sources. Give each source its own id"
        )


def _placement(ref: "DatabaseRef") -> tuple[str | tuple[int, int], str | None]:
    return _location_identity(ref.location), ref.tag


def _changed_sections(running: AppConfig, read: AppConfig) -> list[str]:
    """The sections of `read` that differ from `running`, besides the two
    database maps, named to their second level."""
    unloaded = {"lancedb": {"databases"}, "ingester": {"databases"}}
    before = running.model_dump(mode="json", exclude=unloaded)
    after = read.model_dump(mode="json", exclude=unloaded)
    changed = []
    for section, value in before.items():
        if value == after[section]:
            continue
        if isinstance(value, dict):
            changed.extend(
                f"{section}.{key}" for key in value if value[key] != after[section][key]
            )
        else:
            changed.append(section)
    return changed


def _log_signal_reload(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error("Reload on SIGHUP failed: %s", error)
        return
    report: ReloadReport = task.result()
    for change in report.databases:
        if change.change == "failed":
            logger.error(
                "Reload on SIGHUP failed for database %r: %s",
                change.database,
                change.error,
            )
        else:
            logger.info(
                "Reload on SIGHUP %s database %r", change.change, change.database
            )
    logger.info("Reload on SIGHUP: %s", report.outcome)


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
