import asyncio
import queue
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from starlette.exceptions import HTTPException

from haiku.rag.client.scope import DatabaseRef
from haiku.rag.telemetry import attach_context, get_context, logfire

Progress = Callable[[str], None]
Work = Callable[[Progress], Awaitable[Any]]


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Job:
    kind: str
    source: str
    status: str = "queued"
    progress: list[str] = field(default_factory=list)
    queued_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    result: Any = None
    error: str | None = None
    # Runs on the request loop when the job succeeds, turning its result into
    # the one the job keeps.
    on_done: Callable[[Any], Any] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status,
            "progress": list(self.progress),
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
        }


class Jobs:
    """Doctor runs and similarity jobs, one at a time, on one worker thread.

    Both read every chunk vector and compare the documents in numpy. They run
    on the thread's own event loop, off the request loop, each over the
    connection it opens. A job is not cancellable once started, and the
    thread is a daemon, so a job in flight at shutdown is lost. Results live in
    memory until the next job of the same kind on the same database, or a
    restart.
    """

    def __init__(self) -> None:
        self._jobs: dict[tuple[DatabaseRef, str], Job] = {}
        self._queue: queue.Queue[
            tuple[Job, Work, asyncio.AbstractEventLoop, Mapping[str, Any]]
        ] = queue.Queue()
        self._thread: threading.Thread | None = None

    def get(self, database: DatabaseRef, kind: str) -> Job | None:
        return self._jobs.get((database, kind))

    def submit(
        self,
        database: DatabaseRef,
        kind: str,
        work: Work,
        on_done: Callable[[Any], Any] | None = None,
    ) -> Job:
        current = self._jobs.get((database, kind))
        if current is not None and current.status in ("queued", "running"):
            raise HTTPException(
                409, f"a {kind} of {database.name!r} is already {current.status}"
            )
        job = Job(kind=kind, source=database.name, on_done=on_done)
        self._jobs[(database, kind)] = job
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._serve, name="haiku-rag-app-jobs", daemon=True
            )
            self._thread.start()
        self._queue.put((job, work, asyncio.get_running_loop(), get_context()))
        return job

    def _serve(self) -> None:
        loop = asyncio.new_event_loop()
        while True:
            job, work, requests, context = self._queue.get()

            def relay(label: str, job: Job = job) -> None:
                self._post(requests, job.progress.append, label)

            self._post(requests, self._start, job)
            try:
                with (
                    attach_context(context),
                    logfire.span("app.job", kind=job.kind, database=job.source),
                ):
                    result = loop.run_until_complete(work(relay))
            except Exception as error:
                self._post(requests, self._finish, job, None, str(error), context)
            else:
                self._post(requests, self._finish, job, result, None, context)
                # Held until the next job otherwise, outside the centroid budget.
                del result

    @staticmethod
    def _post(requests: asyncio.AbstractEventLoop, callback, *args) -> None:
        """Hands `callback` to the request loop, dropped once that loop closed."""
        try:
            requests.call_soon_threadsafe(callback, *args)
        except RuntimeError:
            if not requests.is_closed():
                raise

    @staticmethod
    def _start(job: Job) -> None:
        job.status = "running"
        job.started_at = _now()

    @staticmethod
    def _finish(
        job: Job, result: Any, error: str | None, context: Mapping[str, Any]
    ) -> None:
        if error is None and job.on_done is not None:
            try:
                result = job.on_done(result)
            except Exception as failure:
                with attach_context(context):
                    logfire.exception(
                        "app.job result failed", kind=job.kind, database=job.source
                    )
                result, error = None, str(failure)
        job.result = result
        job.error = error
        job.status = "failed" if error is not None else "done"
        job.finished_at = _now()
