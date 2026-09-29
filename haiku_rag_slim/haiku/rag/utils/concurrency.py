import asyncio
import logging
from collections.abc import Awaitable
from typing import Any

logger = logging.getLogger(__name__)


async def gather_all[T](*awaitables: Awaitable[T]) -> list[T]:
    """Run `awaitables` concurrently, leaving none of them running when one fails."""
    tasks = [asyncio.ensure_future(awaitable) for awaitable in awaitables]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def aclose_quietly(closeable: Any, what: str) -> None:
    """Close; a failure is logged, never raised."""
    try:
        await closeable.aclose()
    except Exception:
        logger.debug("Closing the %s failed on teardown", what, exc_info=True)
