"""Generic graph streaming with AG-UI events."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Protocol, TypeVar

from pydantic import BaseModel
from pydantic_graph.beta import Graph

from haiku.rag.graph.agui.emitter import AGUIEmitter
from haiku.rag.graph.agui.events import AGUIEvent

StateT = TypeVar("StateT", bound=BaseModel)
ResultT = TypeVar("ResultT")


class GraphDeps[StateT: BaseModel, ResultT](Protocol):
    """Protocol for graph dependencies that support AG-UI emission."""

    agui_emitter: AGUIEmitter[StateT, ResultT] | None


async def stream_graph[StateT: BaseModel, DepsT: GraphDeps, ResultT](
    graph: Graph[StateT, DepsT, None, ResultT],
    state: StateT,
    deps: DepsT,
    use_deltas: bool = True,
) -> AsyncIterator[AGUIEvent]:
    """Run a graph and yield AG-UI events as they occur.

    This is a generic streaming function that works with any pydantic-graph
    that follows the AG-UI pattern:
    - State must be a Pydantic BaseModel
    - Deps must have an optional agui_emitter attribute
    - Graph must be a pydantic-graph Graph instance

    Args:
        graph: The pydantic-graph Graph to execute
        state: Initial state (Pydantic BaseModel)
        deps: Graph dependencies with agui_emitter support
        use_deltas: Whether to emit state deltas instead of full snapshots (default: True)

    Yields:
        AG-UI event dictionaries

    Raises:
        TypeError: If deps doesn't support agui_emitter
        RuntimeError: If graph doesn't produce a result
    """
    if not hasattr(deps, "agui_emitter"):
        raise TypeError("deps must have an 'agui_emitter' attribute")

    # Create AG-UI emitter
    emitter: AGUIEmitter[StateT, ResultT] = AGUIEmitter(use_deltas=use_deltas)
    deps.agui_emitter = emitter  # type: ignore[assignment]

    async def _execute() -> None:
        try:
            # Start the run with initial state
            emitter.start_run(initial_state=state)

            # Execute the graph
            result = await graph.run(state=state, deps=deps)

            if result is None:
                raise RuntimeError("Graph did not produce a result")

            # Finish the run with the result
            emitter.finish_run(result)
        except Exception as exc:
            # Emit error event
            emitter.error(exc)
        finally:
            await emitter.close()

    runner = asyncio.create_task(_execute())

    try:
        async for event in emitter:
            yield event
    finally:
        if not runner.done():
            runner.cancel()
        with suppress(asyncio.CancelledError):
            await runner
