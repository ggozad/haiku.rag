import os
import uuid
from typing import Any

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from haiku.rag.client.scope import DatabaseRef
from haiku.rag.doctor import duplicates_yaml, run_doctor, scan_vectors
from haiku.rag.store.engine import Store
from haiku_rag_app.jobs import Progress
from haiku_rag_app.projection import Projection, Retained, project, similarity


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
        generation = uuid.uuid4().hex
        neighbours = retained.keep(
            ref.name, generation, projection.ids, projection.unit
        )
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
            "generation": generation,
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
        kept = request.app.state.retained.get(ref.name)
        summary["result"] = {
            **job.result,
            "neighbours": kept is not None
            and kept.generation == job.result["generation"],
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


async def similarities(request: Request) -> JSONResponse:
    ref = configured_ref(request)
    kept = request.app.state.retained.get(ref.name)
    if kept is None:
        raise HTTPException(
            404,
            "similarity is unavailable: compute the map again, within the memory budget",
        )
    generation = request.query_params.get("generation")
    if not generation:
        raise HTTPException(400, "generation names the map the scores are for")
    if generation != kept.generation:
        raise HTTPException(409, "the map was computed again: reload it")
    doc_id = request.query_params.get("id", "")
    if doc_id not in kept.ids:
        raise HTTPException(404, f"no document {doc_id!r} on the map")
    return JSONResponse(
        {
            "generation": kept.generation,
            "similarity": similarity(kept.unit, kept.ids, doc_id),
        }
    )


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
    Route("/api/doctor/{source}/projection/similarity", similarities, methods=["GET"]),
    Route("/api/doctor/{source}/duplicates.yaml", duplicates_export, methods=["GET"]),
    Route("/api/doctor/{source}", get_run, methods=["GET"]),
]
