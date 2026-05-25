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
        self._configs = configs
        self._jobs = job_repo
        self._sync = sync_repo
        self._supported_extensions = supported_extensions
        self._default_max_attempts = default_max_attempts
        self._pollers: list[BasePoller] = []
        self._tasks: list[asyncio.Task] = []

    def build_pollers(self) -> list[BasePoller]:
        pollers: list[BasePoller] = []
        for cfg in self._configs:
            source = build_source(cfg, supported_extensions=self._supported_extensions)
            breaker = CircuitBreaker(cfg.circuit_breaker)
            if isinstance(cfg, FSSourceConfig):
                from haiku.rag.ingester.sources.fs import FSSource

                assert isinstance(source, FSSource)
                pollers.append(
                    FSPoller(
                        source=source,
                        config=cfg,
                        job_repo=self._jobs,
                        sync_repo=self._sync,
                        breaker=breaker,
                        default_max_attempts=self._default_max_attempts,
                    )
                )
            else:
                pollers.append(
                    PeriodicPoller(
                        source=source,
                        config=cfg,
                        job_repo=self._jobs,
                        sync_repo=self._sync,
                        breaker=breaker,
                        default_max_attempts=self._default_max_attempts,
                    )
                )
        return pollers

    async def start(self) -> None:
        if self._pollers:
            raise RuntimeError("PollerManager already started")
        self._pollers = self.build_pollers()
        for poller in self._pollers:
            self._tasks.append(asyncio.create_task(poller.run()))

    async def stop(self) -> None:
        for poller in self._pollers:
            await poller.stop()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._pollers.clear()

    @property
    def pollers(self) -> list[BasePoller]:
        return list(self._pollers)

    @property
    def live_pollers(self) -> int:
        """Poller tasks that are still running. Equal to len(pollers) under
        normal operation; less when a poller has crashed."""
        return sum(1 for t in self._tasks if not t.done())
