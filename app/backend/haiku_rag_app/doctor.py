import asyncio
import os
import queue
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from haiku.rag.client.scope import DatabaseRef
from haiku.rag.doctor import run_doctor

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

    def submit(self, source: str, kind: str, work: Work) -> Job:
        current = self._jobs.get((source, kind))
        if current is not None and current.status in ("queued", "running"):
            raise HTTPException(
                409, f"a {kind} of {source!r} is already {current.status}"
            )
        job = Job(kind=kind, source=source)
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
        job.result = result
        job.error = error
        job.status = "failed" if error is not None else "done"
        job.finished_at = _now()


def configured_ref(request: Request) -> DatabaseRef:
    """The configured database the path names, refused at a tag: the doctor
    reads the live state only."""
    name = request.path_params["source"]
    scope = request.app.state.databases.scope
    ref = next((r for r in scope.databases if r.name == name), None)
    if ref is None:
        raise HTTPException(404, f"unknown database {name!r}")
    if ref.tag is not None:
        raise HTTPException(
            400, f"{name!r} is read at tag {ref.tag!r}; the doctor reads the live state"
        )
    return ref


async def start_run(request: Request) -> JSONResponse:
    ref = configured_ref(request)
    config = request.app.state.config
    providers = request.query_params.get("providers", "true") != "false"

    async def work(progress: Progress) -> dict[str, Any]:
        report = await run_doctor(
            config,
            ref.location,
            dict(os.environ),
            on_progress=progress,
            providers=providers,
        )
        return report.model_dump(mode="json")

    job = request.app.state.jobs.submit(ref.name, "doctor", work)
    return JSONResponse(job.summary(), status_code=202)


async def get_run(request: Request) -> JSONResponse:
    ref = configured_ref(request)
    job = request.app.state.jobs.get(ref.name, "doctor")
    return JSONResponse({"run": job.summary() if job else None})


routes = [
    Route("/api/doctor/{source}/run", start_run, methods=["POST"]),
    Route("/api/doctor/{source}", get_run, methods=["GET"]),
]
