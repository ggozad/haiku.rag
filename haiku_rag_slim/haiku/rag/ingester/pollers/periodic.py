import asyncio
import logging
import random
from typing import TYPE_CHECKING

from haiku.rag.ingester.pollers.base import BasePoller

if TYPE_CHECKING:
    from haiku.rag.ingester.pollers.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class PeriodicPoller(BasePoller):
    """Runs `source.discover()` on a fixed interval. Used for HTTP, S3, WebDAV
    — sources that only know about changes when we ask them."""

    def __init__(
        self,
        *,
        source,
        config,
        job_repo,
        sync_repo,
        breaker: "CircuitBreaker | None" = None,
        default_max_attempts: int = 5,
    ):
        super().__init__(
            source=source,
            config=config,
            job_repo=job_repo,
            sync_repo=sync_repo,
            breaker=breaker,
            default_max_attempts=default_max_attempts,
        )

    async def run(self) -> None:  # pragma: no cover - event-loop glue
        # Initial sweep on startup so newly-configured sources are scanned
        # immediately instead of waiting one full interval. The sweep
        # behaviour itself is exercised via `_sweep_once()` unit tests.
        await self._sweep_once()
        # Stagger the first sleep so pollers that share the same interval
        # don't all wake up and sweep at exactly the same moment.
        interval = self.config.poll_interval_s
        jitter = random.uniform(0, interval * 0.25)
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=jitter)
            return
        except TimeoutError:
            pass
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=interval
                )
                return
            except TimeoutError:
                pass
            await self._sweep_once()
