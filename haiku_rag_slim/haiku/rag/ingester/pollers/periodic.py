import asyncio
import logging
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

    async def run(self) -> None:
        # Initial sweep on startup so newly-configured sources are scanned
        # immediately instead of waiting one full interval.
        await self._sweep_once()
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.config.poll_interval_s
                )
                return
            except TimeoutError:
                pass
            await self._sweep_once()
