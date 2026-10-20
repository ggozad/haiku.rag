import asyncio
import queue
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from starlette.exceptions import HTTPException

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
    """Doctor runs and projections, one at a time, on one worker thread.

    Both read every chunk vector and cluster or project them in numpy, which
    would stall the request loop: they run on the thread's own event loop, each
    over the connection it opens. A job is not cancellable once started, and the
    thread is a daemon, so a job in flight at shutdown is lost. Results live in
    memory until the next job of the same kind on the same database, or a
    restart.
    """

    def __init__(self) -> None:
        self._jobs: dict[tuple[str, str], Job] = {}
        self._queue: queue.Queue[tuple[Job, Work, asyncio.AbstractEventLoop]] = (
            queue.Queue()
        )
        self._thread: threading.Thread | None = None

    def get(self, source: str, kind: str) -> Job | None:
        return self._jobs.get((source, kind))

    def submit(
        self,
        source: str,
        kind: str,
        work: Work,
        on_done: Callable[[Any], Any] | None = None,
    ) -> Job:
        current = self._jobs.get((source, kind))
        if current is not None and current.status in ("queued", "running"):
            raise HTTPException(
                409, f"a {kind} of {source!r} is already {current.status}"
            )
        job = Job(kind=kind, source=source, on_done=on_done)
        self._jobs[(source, kind)] = job
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._serve, name="haiku-rag-app-jobs", daemon=True
            )
            self._thread.start()
        self._queue.put((job, work, asyncio.get_running_loop()))
        return job

    def _serve(self) -> None:
        loop = asyncio.new_event_loop()
        while True:
            job, work, requests = self._queue.get()

            def relay(label: str, job: Job = job) -> None:
                self._post(requests, job.progress.append, label)

            self._post(requests, self._start, job)
            try:
                result = loop.run_until_complete(work(relay))
            except Exception as error:
                self._post(requests, self._finish, job, None, str(error))
            else:
                self._post(requests, self._finish, job, result, None)
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
    def _finish(job: Job, result: Any, error: str | None) -> None:
        if error is None and job.on_done is not None:
            result = job.on_done(result)
        job.result = result
        job.error = error
        job.status = "failed" if error is not None else "done"
        job.finished_at = _now()
