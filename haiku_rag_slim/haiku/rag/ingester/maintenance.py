import asyncio
import logging
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from typing import TYPE_CHECKING

from haiku.rag.store.engine import ConnectionMode
from haiku.rag.telemetry import logfire

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG
    from haiku.rag.config import MaintenanceConfig
    from haiku.rag.ingester.workers.pool import WorkerPool
    from haiku.rag.store.engine import Store

logger = logging.getLogger(__name__)


class MaintenanceError(Exception):
    """A maintenance pass did not complete."""


@dataclass(frozen=True)
class MaintenanceResult:
    vacuumed: bool
    index_built: bool


async def has_vector_index(store: "Store") -> bool:
    """Whether the chunks table has an index on `vector`."""
    indices = await store.chunks_table.list_indices()
    return any("vector" in index.columns for index in indices)


async def maintain(client: "HaikuRAG") -> MaintenanceResult:
    """Vacuum, which keeps every existing index current, then build the vector
    index if there is none. Raises `MaintenanceError` when the vacuum does not
    complete. Does nothing on LanceDB Cloud."""
    store = client.store
    if ConnectionMode.of(store.location) == ConnectionMode.CLOUD:
        return MaintenanceResult(vacuumed=False, index_built=False)
    if not await store.vacuum():
        raise MaintenanceError(f"vacuum of {client.source!r} did not complete")
    built = not await has_vector_index(store) and await client.create_vector_index()
    return MaintenanceResult(vacuumed=True, index_built=built)


@dataclass(frozen=True)
class MaintenanceStatus:
    """The last pass over a database: when it ended, what it did, or why it
    failed."""

    finished_at: datetime | None = None
    result: MaintenanceResult | None = None
    error: str | None = None


class DatabaseMaintenance:
    """Maintains one database the ingester writes, with its sources' claims
    paused for the pass."""

    def __init__(
        self,
        name: str,
        client: "HaikuRAG",
        pool: "WorkerPool",
        source_ids: Collection[str],
        config: "MaintenanceConfig",
    ):
        self.name = name
        self.status = MaintenanceStatus()
        self._client = client
        self._session = client._single_session("maintenance")
        self._pool = pool
        self._source_ids = list(source_ids)
        self._config = config

    @property
    def due(self) -> bool:
        """Whether a pass is owed: the database was written since its last one,
        or `run` started."""
        return self._session.unmaintained_since is not None

    async def run_once(self) -> MaintenanceResult:
        """Pause the database's claims, maintain it, resume. Jobs in flight run
        on, and their writes wait for the pass. A failed pass leaves the
        database due."""
        session = self._session
        since: float | None = None
        await self._pool.pause(self._source_ids)
        try:
            with logfire.span("ingester.maintenance", database=self.name) as span:
                since, session.unmaintained_since = session.unmaintained_since, None
                result = await maintain(self._client)
                span.set_attribute("vacuumed", result.vacuumed)
                span.set_attribute("index_built", result.index_built)
        except BaseException as error:
            if session.unmaintained_since is None:
                session.unmaintained_since = since
            if isinstance(error, Exception):
                self.status = MaintenanceStatus(datetime.now(UTC), None, str(error))
            raise
        finally:
            self._pool.resume(self._source_ids)
        self.status = MaintenanceStatus(datetime.now(UTC), result, None)
        return result

    async def run(self) -> None:
        """Pass whenever the database is due: `quiet_s` after its last write
        once none of its jobs is in flight, or `max_wait_s` after the first
        write since the last pass. A start makes it due. A failed pass is
        logged and retried after `quiet_s`."""
        session = self._session
        if session.unmaintained_since is None:
            session.unmaintained_since = monotonic()
        while True:
            delay = self._due_in()
            if delay > 0:
                await asyncio.sleep(delay)
                continue
            max_wait = self._max_wait_in()
            if max_wait > 0 and self._pool.busy(self._source_ids):
                try:
                    await asyncio.wait_for(
                        self._pool.wait_idle(self._source_ids), timeout=max_wait
                    )
                except TimeoutError:
                    pass
                continue
            try:
                await self.run_once()
            except Exception as error:
                logger.error("Maintenance of database %r failed: %s", self.name, error)
                await asyncio.sleep(self._config.quiet_s)

    def _due_in(self) -> float:
        """Seconds until the database is due. While it has not been written,
        the shorter of `quiet_s` and `max_wait_s`, so a write is seen in time
        for either."""
        session = self._session
        since = session.unmaintained_since
        if since is None:
            return min(self._config.quiet_s, self._config.max_wait_s)
        last = max(session.last_write_at or since, since)
        return min(last + self._config.quiet_s - monotonic(), self._max_wait_in())

    def _max_wait_in(self) -> float:
        """Seconds until a due database passes whatever is in flight."""
        since = self._session.unmaintained_since
        assert since is not None, "only a due database has a max_wait deadline"
        return since + self._config.max_wait_s - monotonic()
