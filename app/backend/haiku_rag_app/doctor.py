import asyncio
import os
import queue
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from haiku.rag.client.scope import DatabaseRef
from haiku.rag.doctor import duplicates_yaml, run_doctor, scan_vectors
from haiku.rag.store.engine import Store
from haiku_rag_app.projection import Projection, nearest, project

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


class Retained:
    """Unit centroids kept for neighbour lookups, within one memory budget
    across databases. A newer projection evicts the oldest others; one that
    does not fit alone is not kept."""

    def __init__(self, budget: int = 256 * 1024**2) -> None:
        self.budget = budget
        self._kept: dict[str, tuple[list[str], np.ndarray]] = {}

    def keep(self, source: str, ids: list[str], unit: np.ndarray) -> bool:
        self._kept.pop(source, None)
        if unit.nbytes > self.budget:
            return False
        while self._kept and self._used() + unit.nbytes > self.budget:
            del self._kept[next(iter(self._kept))]
        self._kept[source] = (ids, unit)
        return True

    def get(self, source: str) -> tuple[list[str], np.ndarray] | None:
        return self._kept.get(source)

    def _used(self) -> int:
        return sum(unit.nbytes for _, unit in self._kept.values())


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


def _label(point: dict[str, Any]) -> str:
    return point.get("uri") or point.get("title") or point["id"]


async def start_projection(request: Request) -> JSONResponse:
    ref = configured_ref(request)
    config = request.app.state.config
    retained: Retained = request.app.state.retained

    async def work(progress: Progress) -> Projection:
        progress("Reading chunk vectors")
        async with Store(
            ref.location,
            config=config,
            skip_validation=True,
            read_only=True,
            skip_migration_check=True,
        ) as store:
            versions = await store.current_table_versions()
            scan = await scan_vectors(store)
            rows = (
                await store.document_meta_table.query()
                .select(["id", "title", "uri", "metadata"])
                .to_list()
            )
        progress("Projecting documents")
        projection = project(
            scan, {row["id"]: row for row in rows}, config.doctor.duplicates
        )
        projection.versions = {
            table: versions.get(table) for table in ("chunks", "document_meta")
        }
        return projection

    def done(projection: Projection) -> dict[str, Any]:
        neighbours = retained.keep(ref.name, projection.ids, projection.unit)
        labels = {point["id"]: _label(point) for point in projection.points}
        return {
            "points": projection.points,
            "families": [
                {
                    **family.model_dump(),
                    "label": {m: labels.get(m, m) for m in family.members},
                }
                for family in projection.families
            ],
            "left_off": projection.left_off,
            "explained_variance": projection.explained_variance,
            "versions": projection.versions,
            "neighbours": neighbours,
            "yaml": duplicates_yaml(
                projection.families, lambda doc_id: labels.get(doc_id, doc_id)
            ),
        }

    job = request.app.state.jobs.submit(ref.name, "projection", work, on_done=done)
    return JSONResponse(job.summary(), status_code=202)


async def get_projection(request: Request) -> JSONResponse:
    """The latest projection, marked stale when the chunks or the documents'
    attributes changed since its scan started."""
    ref = configured_ref(request)
    job = request.app.state.jobs.get(ref.name, "projection")
    if job is None:
        return JSONResponse({"projection": None})
    summary = job.summary()
    if job.status == "done":
        summary["stale"] = await _stale(request, ref.name, job.result["versions"])
        # A later map of another database may have evicted these centroids.
        summary["result"] = {
            **job.result,
            "neighbours": request.app.state.retained.get(ref.name) is not None,
        }
    return JSONResponse({"projection": summary})


async def _stale(request: Request, name: str, versions: dict) -> bool | None:
    try:
        owner = await request.app.state.databases.reader(name)
        # The ingester writes from another process: read its latest versions.
        for table in versions:
            await getattr(owner.store, f"{table}_table").checkout_latest()
        current = await owner.store.current_table_versions()
    except Exception:
        # Unknown, whether the database is unavailable or its storage failed.
        return None
    return any(current.get(table) != version for table, version in versions.items())


async def neighbours(request: Request) -> JSONResponse:
    ref = configured_ref(request)
    kept = request.app.state.retained.get(ref.name)
    if kept is None:
        raise HTTPException(
            404,
            "neighbours are unavailable: compute the map again, within the memory budget",
        )
    ids, unit = kept
    doc_id = request.query_params.get("id", "")
    if doc_id not in ids:
        raise HTTPException(404, f"no document {doc_id!r} on the map")
    try:
        k = int(request.query_params.get("k", "8"))
    except ValueError as error:
        raise HTTPException(400, "k must be a whole number") from error
    return JSONResponse({"neighbours": nearest(unit, ids, doc_id, max(1, min(k, 50)))})


async def duplicates_export(request: Request) -> PlainTextResponse:
    ref = configured_ref(request)
    job = request.app.state.jobs.get(ref.name, "projection")
    if job is None or job.status != "done":
        raise HTTPException(404, "compute the map first")
    return PlainTextResponse(
        job.result["yaml"],
        media_type="application/yaml",
        headers={
            "content-disposition": f'attachment; filename="{ref.name}-duplicates.yaml"'
        },
    )


routes = [
    Route("/api/doctor/{source}/run", start_run, methods=["POST"]),
    Route("/api/doctor/{source}/projection", start_projection, methods=["POST"]),
    Route("/api/doctor/{source}/projection", get_projection, methods=["GET"]),
    Route("/api/doctor/{source}/projection/neighbours", neighbours, methods=["GET"]),
    Route("/api/doctor/{source}/duplicates.yaml", duplicates_export, methods=["GET"]),
    Route("/api/doctor/{source}", get_run, methods=["GET"]),
]
