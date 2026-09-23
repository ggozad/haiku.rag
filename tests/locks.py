import asyncio
from types import TracebackType
from typing import Any


class ObservedLock:
    """Expose when a task attempts to acquire a held asyncio lock."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.attempted = asyncio.Event()

    async def __aenter__(self) -> None:
        if self._lock.locked():
            self.attempted.set()
        await self._lock.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._lock.release()


async def assert_waiting_for_lock(task: asyncio.Task[Any], lock: ObservedLock) -> None:
    """Assert that a task reached the lock before it could complete."""
    attempted = asyncio.create_task(lock.attempted.wait())
    try:
        done, _ = await asyncio.wait(
            {task, attempted}, return_when=asyncio.FIRST_COMPLETED
        )
        assert attempted in done, "task completed without attempting the held lock"
        assert task not in done, "task completed while the observed lock was held"
    finally:
        if not attempted.done():
            attempted.cancel()
            await asyncio.gather(attempted, return_exceptions=True)
