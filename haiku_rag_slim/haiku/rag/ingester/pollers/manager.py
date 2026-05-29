import asyncio
import logging
from typing import TYPE_CHECKING

from haiku.rag.config import FSSourceConfig, SourceConfig
from haiku.rag.ingester.pollers.base import BasePoller
from haiku.rag.ingester.pollers.circuit_breaker import CircuitBreaker
from haiku.rag.ingester.pollers.factory import build_source
from haiku.rag.ingester.pollers.fs import FSPoller
from haiku.rag.ingester.pollers.periodic import PeriodicPoller

if TYPE_CHECKING:
    from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo
    from haiku.rag.ingester.sources.base import Source

logger = logging.getLogger(__name__)


class PollerManager:
    """Owns one poller per configured source. Lifecycle: build → start →
    stop. Each poller runs as an independent asyncio task; failures in one
    don't affect the others."""

    def __init__(
        self,
        *,
        configs: list[SourceConfig],
        job_repo: "JobRepo",
        sync_repo: "SyncStateRepo",
        supported_extensions: list[str] | None = None,
        default_max_attempts: int = 5,
    ):
        self._jobs = job_repo
        self._sync = sync_repo
        self._supported_extensions = supported_extensions
        self._default_max_attempts = default_max_attempts
        # Build eagerly so `sources` is available before `start()` — any
        # downstream component that holds the configured Source list (e.g.
        # WorkerPool) can do so via plain construction order.
        self._pollers: list[BasePoller] = [self._build_poller(cfg) for cfg in configs]
        self._tasks: list[asyncio.Task] = []
        self._started = False

    def _build_poller(self, cfg: SourceConfig) -> BasePoller:
        source = build_source(cfg, supported_extensions=self._supported_extensions)
        breaker = CircuitBreaker(cfg.circuit_breaker)
        if isinstance(cfg, FSSourceConfig):
            from haiku.rag.ingester.sources.fs import FSSource

            assert isinstance(source, FSSource)
            return FSPoller(
                source=source,
                config=cfg,
                job_repo=self._jobs,
                sync_repo=self._sync,
                breaker=breaker,
                default_max_attempts=self._default_max_attempts,
            )
        return PeriodicPoller(
            source=source,
            config=cfg,
            job_repo=self._jobs,
            sync_repo=self._sync,
            breaker=breaker,
            default_max_attempts=self._default_max_attempts,
        )

    async def start(self) -> None:
        if self._started:
            raise RuntimeError("PollerManager already started")
        self._started = True
        for poller in self._pollers:
            # Reset the stop signal synchronously *before* scheduling the
            # task. If a poller is being restarted (stop() set the event
            # on the previous cycle) clearing inside run() would race with
            # any concurrent stop() and could deadlock.
            poller._stop.clear()
            self._tasks.append(asyncio.create_task(poller.run()))

    async def sweep_all(self) -> None:
        """Run one discover() sweep on every poller, sequentially. Used by
        one-shot batch runs that drive discovery explicitly rather than
        through the periodic loop."""
        for poller in self._pollers:
            await poller._sweep_once()

    async def stop(self) -> None:
        for poller in self._pollers:
            await poller.stop()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._started = False

    @property
    def pollers(self) -> list[BasePoller]:
        return list(self._pollers)

    @property
    def sources(self) -> list["Source"]:
        """Configured Source adapters, one per poller, in config order."""
        return [p.source for p in self._pollers]

    @property
    def live_pollers(self) -> int:
        """Poller tasks that are still running. Equal to len(pollers) under
        normal operation; less when a poller has crashed."""
        return sum(1 for t in self._tasks if not t.done())
