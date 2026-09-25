import asyncio
from collections.abc import Awaitable


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
