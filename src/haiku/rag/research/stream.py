import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from haiku.rag.research.models import ResearchReport

if TYPE_CHECKING:  # pragma: no cover
    from haiku.rag.research.state import ResearchState


@dataclass(slots=True)
class ResearchStateSnapshot:
    question: str
    sub_questions: list[str]
    iterations: int
    max_iterations: int
    max_concurrency: int
    confidence_threshold: float
    pending_sub_questions: int
    answered_questions: int
    insights: list[str]
    gaps: list[str]
    last_confidence: float | None
    last_sufficient: bool | None

    @classmethod
    def from_state(cls, state: "ResearchState") -> "ResearchStateSnapshot":
        context = state.context
        last_confidence: float | None = None
        last_sufficient: bool | None = None
        if state.last_eval:
            last_confidence = state.last_eval.confidence_score
            last_sufficient = state.last_eval.is_sufficient

        return cls(
            question=context.original_question,
            sub_questions=list(context.sub_questions),
            iterations=state.iterations,
            max_iterations=state.max_iterations,
            max_concurrency=state.max_concurrency,
            confidence_threshold=state.confidence_threshold,
            pending_sub_questions=len(context.sub_questions),
            answered_questions=len(context.qa_responses),
            insights=[
                f"{insight.status.value}:{insight.summary}"
                for insight in context.insights
            ],
            gaps=[
                f"{gap.severity.value}/{'resolved' if gap.resolved else 'open'}:{gap.description}"
                for gap in context.gaps
            ],
            last_confidence=last_confidence,
            last_sufficient=last_sufficient,
        )


@dataclass(slots=True)
class ResearchStreamEvent:
    type: Literal["log", "report", "error"]
    message: str | None = None
    state: ResearchStateSnapshot | None = None
    report: ResearchReport | None = None
    error: str | None = None


class ResearchStream:
    """Queue-backed stream for research graph events."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[ResearchStreamEvent | None] = asyncio.Queue()
        self._closed = False

    def _snapshot(self, state: "ResearchState | None") -> ResearchStateSnapshot | None:
        if state is None:
            return None
        return ResearchStateSnapshot.from_state(state)

    def log(self, message: str, state: "ResearchState | None" = None) -> None:
        if self._closed:
            return
        event = ResearchStreamEvent(
            type="log", message=message, state=self._snapshot(state)
        )
        self._queue.put_nowait(event)

    def report(self, report: ResearchReport, state: "ResearchState") -> None:
        if self._closed:
            return
        event = ResearchStreamEvent(
            type="report",
            report=report,
            state=self._snapshot(state),
        )
        self._queue.put_nowait(event)

    def error(self, error: Exception, state: "ResearchState | None" = None) -> None:
        if self._closed:
            return
        event = ResearchStreamEvent(
            type="error",
            message=str(error),
            error=str(error),
            state=self._snapshot(state),
        )
        self._queue.put_nowait(event)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._queue.put(None)

    def __aiter__(self) -> AsyncIterator[ResearchStreamEvent]:
        return self._iter_events()

    async def _iter_events(self) -> AsyncIterator[ResearchStreamEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event


async def stream_research_graph(
    graph,
    start,
    state: "ResearchState",
    deps,
) -> AsyncIterator[ResearchStreamEvent]:
    """Run the research graph and yield streaming events as they occur."""

    from contextlib import suppress

    from haiku.rag.research.state import ResearchDeps  # Local import to avoid cycle

    if not isinstance(deps, ResearchDeps):
        raise TypeError("deps must be an instance of ResearchDeps")

    stream = ResearchStream()
    deps.stream = stream

    async def _execute() -> None:
        try:
            report = None
            try:
                result = await graph.run(start, state=state, deps=deps)
                report = result.output
            except Exception:
                from pydantic_graph import End

                async with graph.iter(start, state=state, deps=deps) as run:
                    node = run.next_node
                    while not isinstance(node, End):
                        node = await run.next(node)
                    if run.result:
                        report = run.result.output

            if report is None:
                raise RuntimeError("Graph did not produce a report")

            stream.report(report, state)
        except Exception as exc:  # pragma: no cover - defensive path
            stream.error(exc, state)
        finally:
            await stream.close()

    runner = asyncio.create_task(_execute())

    try:
        async for event in stream:
            yield event
    finally:
        if not runner.done():
            runner.cancel()
        with suppress(asyncio.CancelledError):
            await runner
