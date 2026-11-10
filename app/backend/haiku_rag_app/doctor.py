import os
import uuid
from typing import Any

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from haiku.rag.client.scope import DatabaseRef
from haiku.rag.doctor import (
    display_label,
    duplicates_yaml,
    run_doctor,
    scan_vectors,
)
from haiku.rag.store.engine import Store
from haiku_rag_app.jobs import Progress
from haiku_rag_app.similarity import Comparison, Retained, compare, similarity


def configured_ref(request: Request) -> DatabaseRef:
    """The configured database the path names, refused at a tag: the doctor
    reads the live state only."""
    name = request.path_params["source"]
    scope = request.state.databases.scope
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
            # The similarity job reports the near-duplicate families.
            duplicates=False,
        )
        return report.model_dump(mode="json")

    job = request.app.state.jobs.submit(ref, "doctor", work)
    return JSONResponse(job.summary(), status_code=202)


async def get_run(request: Request) -> JSONResponse:
    ref = configured_ref(request)
    job = request.app.state.jobs.get(ref, "doctor")
    return JSONResponse({"run": job.summary() if job else None})


async def start_similarity(request: Request) -> JSONResponse:
    ref = configured_ref(request)
    config = request.app.state.config
    retained: Retained = request.app.state.retained

    async def work(progress: Progress) -> Comparison:
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
                .select(["id", "title", "uri"])
                .to_list()
            )
        progress("Comparing documents")
        comparison = compare(
            scan, {row["id"]: row for row in rows}, config.doctor.duplicates
        )
        comparison.versions = {
            table: versions.get(table) for table in ("chunks", "document_meta")
        }
        return comparison

    def done(comparison: Comparison) -> dict[str, Any]:
        generation = uuid.uuid4().hex
        neighbours = retained.keep(ref, generation, comparison.ids, comparison.unit)
        labels = {
            d["id"]: d["uri"] or d["title"] or d["id"] for d in comparison.documents
        }
        shown = {
            d["id"]: display_label(d["uri"], d["title"], d["id"])
            for d in comparison.documents
        }
        return {
            "documents": comparison.documents,
            "families": [
                {
                    **family.model_dump(),
                    "label": {m: shown.get(m, m) for m in family.members},
                }
                for family in comparison.families
            ],
            "left_off": comparison.left_off,
            "generation": generation,
            "versions": comparison.versions,
            "neighbours": neighbours,
            "yaml": duplicates_yaml(
                comparison.families, lambda doc_id: labels.get(doc_id, doc_id)
            ),
        }

    job = request.app.state.jobs.submit(ref, "similarity", work, on_done=done)
    return JSONResponse(job.summary(), status_code=202)


async def get_similarity(request: Request) -> JSONResponse:
    """The latest similarity job, marked stale when the chunks or the
    documents' attributes changed since its scan started."""
    ref = configured_ref(request)
    job = request.app.state.jobs.get(ref, "similarity")
    if job is None:
        return JSONResponse({"similarity": None})
    summary = job.summary()
    if job.status == "done":
        summary["stale"] = await _stale(request, ref.name, job.result["versions"])
        # A later job on another database may have evicted these centroids.
        kept = request.app.state.retained.get(ref)
        summary["result"] = {
            **job.result,
            "neighbours": kept is not None
            and kept.generation == job.result["generation"],
        }
    return JSONResponse({"similarity": summary})


async def _stale(request: Request, name: str, versions: dict) -> bool | None:
    try:
        owner = await request.state.databases.reader(name)
        # The ingester writes from another process: read its latest versions.
        for table in versions:
            await getattr(owner.store, f"{table}_table").checkout_latest()
        current = await owner.store.current_table_versions()
    except Exception:
        # Unknown, whether the database is unavailable or its storage failed.
        return None
    return any(current.get(table) != version for table, version in versions.items())


async def document_similarity(request: Request) -> JSONResponse:
    ref = configured_ref(request)
    kept = request.app.state.retained.get(ref)
    if kept is None:
        raise HTTPException(
            404,
            "similarity is unavailable: compute it again, within the memory budget",
        )
    generation = request.query_params.get("generation")
    if not generation:
        raise HTTPException(400, "generation names the job the scores are for")
    if generation != kept.generation:
        raise HTTPException(409, "similarity was computed again: reload it")
    doc_id = request.query_params.get("id", "")
    if doc_id not in kept.ids:
        raise HTTPException(404, f"document {doc_id!r} was not compared")
    return JSONResponse(
        {
            "generation": kept.generation,
            "similarity": similarity(kept.unit, kept.ids, doc_id),
        }
    )


async def duplicates_export(request: Request) -> PlainTextResponse:
    ref = configured_ref(request)
    job = request.app.state.jobs.get(ref, "similarity")
    if job is None or job.status != "done":
        raise HTTPException(404, "compute similarity first")
    return PlainTextResponse(
        job.result["yaml"],
        media_type="application/yaml",
        headers={
            "content-disposition": f'attachment; filename="{ref.name}-duplicates.yaml"'
        },
    )


routes = [
    Route("/api/doctor/{source}/run", start_run, methods=["POST"]),
    Route("/api/doctor/{source}/similarity", start_similarity, methods=["POST"]),
    Route("/api/doctor/{source}/similarity", get_similarity, methods=["GET"]),
    Route(
        "/api/doctor/{source}/similarity/documents",
        document_similarity,
        methods=["GET"],
    ),
    Route("/api/doctor/{source}/duplicates.yaml", duplicates_export, methods=["GET"]),
    Route("/api/doctor/{source}", get_run, methods=["GET"]),
]
