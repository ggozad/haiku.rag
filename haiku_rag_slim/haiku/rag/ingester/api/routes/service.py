from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from haiku.rag.ingester.api.routes.jobs import retry_served
from haiku.rag.ingester.api.server import APIState, get_state
from haiku.rag.ingester.queue.models import JobStatus
from haiku.rag.service.search import (
    DocumentType,
    SearchResponse,
    SearchService,
    document_type,
    local_path,
)
from haiku.rag.store.exceptions import UnknownDatabaseError

router = APIRouter(tags=["search"])

SearchType = Literal["hybrid", "vector", "fts"]


class DocumentView(BaseModel):
    document_id: str
    database: str | None
    folder_id: str | None
    title: str | None
    path: str | None
    uri: str | None
    document_type: DocumentType
    content_type: str | None
    created_at: datetime
    updated_at: datetime


class OutlineNode(BaseModel):
    id: str
    title: str
    level: int
    page_numbers: list[int]
    children: list["OutlineNode"]


class Outline(BaseModel):
    document_id: str
    outline: list[OutlineNode]


class DatabaseCount(BaseModel):
    name: str
    documents: int


class ServiceStatus(BaseModel):
    indexing: bool
    queued: int
    in_flight: int
    failed: int
    paused: bool
    pause_reason: str | None
    embedder_loaded: bool
    databases: list[DatabaseCount]


class Failure(BaseModel):
    job_id: str
    folder_id: str
    database: str | None
    path: str | None
    uri: str
    reason: str | None
    attempts: int
    failed_at: datetime | None


class Failures(BaseModel):
    failures: list[Failure]


class Retried(BaseModel):
    retried: list[str]


def _search(state: APIState) -> SearchService:
    if state.search is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "search is not running"
        )
    return state.search


async def _run(service: SearchService, query: str, **kwargs: Any) -> SearchResponse:
    try:
        return await service.search(query, **kwargs)
    except UnknownDatabaseError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from None


@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(min_length=1),
    limit: int = Query(10, ge=1, le=50),
    type: SearchType = "hybrid",
    database: list[str] | None = Query(None),
    folder: list[str] | None = Query(None),
    context: bool = False,
    state: APIState = Depends(get_state),
) -> SearchResponse:
    """Search the indexed files: each hit locates its passage by file, page
    and boxes on the page."""
    return await _run(
        _search(state),
        q,
        limit=limit,
        search_type=type,
        databases=database,
        folders=folder,
        context=context,
    )


@router.get("/documents/{document_id}/search", response_model=SearchResponse)
async def search_document(
    document_id: str,
    q: str = Query(min_length=1),
    limit: int = Query(10, ge=1, le=50),
    type: SearchType = "hybrid",
    database: str | None = None,
    context: bool = False,
    state: APIState = Depends(get_state),
) -> SearchResponse:
    """Search within one document."""
    return await _run(
        _search(state),
        q,
        limit=limit,
        search_type=type,
        databases=[database] if database else None,
        document_id=document_id,
        context=context,
    )


async def _document(state: APIState, document_id: str, database: str | None):
    async with _search(state).client() as rag:
        try:
            document = await rag.get_document_by_id(document_id, database)
        except UnknownDatabaseError as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from None
        if document is None or document.id is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"No document with id {document_id!r}"
            )
        owner = await rag.reader_for(database or document.source)
        assert owner is not None, "a stored document names its database"
        return document, owner


@router.get("/documents/{document_id}", response_model=DocumentView)
async def get_document(
    document_id: str,
    database: str | None = None,
    state: APIState = Depends(get_state),
) -> DocumentView:
    document, _ = await _document(state, document_id, database)
    metadata = document.metadata or {}
    path = local_path(document.uri)
    return DocumentView(
        document_id=document_id,
        database=document.source,
        folder_id=metadata.get("source_id"),
        title=document.title,
        path=str(path) if path is not None else None,
        uri=document.uri,
        document_type=document_type(document.uri, metadata.get("content_type")),
        content_type=metadata.get("content_type"),
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def _node(toc: dict[str, Any]) -> OutlineNode:
    return OutlineNode(
        id=toc["self_ref"],
        title=toc["title"],
        level=toc["level"],
        page_numbers=toc["page_numbers"],
        children=[_node(child) for child in toc["children"]],
    )


@router.get("/documents/{document_id}/outline", response_model=Outline)
async def get_outline(
    document_id: str,
    database: str | None = None,
    state: APIState = Depends(get_state),
) -> Outline:
    from haiku.rag.context import build_toc

    _, owner = await _document(state, document_id, database)
    items = await owner.document_item_repository.get_all_items(document_id)
    return Outline(
        document_id=document_id,
        outline=[_node(toc) for toc in build_toc(items, {})],
    )


@router.get("/status", response_model=ServiceStatus)
async def get_status(state: APIState = Depends(get_state)) -> ServiceStatus:
    """What the service is doing: indexing, held back, and what it holds."""
    counts = await state.job_repo.counts_by_status()
    queued = counts.get(JobStatus.QUEUED.value, 0)
    in_flight = counts.get(JobStatus.CLAIMED.value, 0)
    databases = []
    service = _search(state)
    async with service.client() as rag:
        for name in rag.source_names:
            owner = await rag.reader_for(name)
            assert owner is not None
            databases.append(
                DatabaseCount(name=name, documents=await owner.count_documents())
            )
    throttle = state.throttle
    return ServiceStatus(
        indexing=bool(queued or in_flight),
        queued=queued,
        in_flight=in_flight,
        failed=counts.get(JobStatus.DEAD.value, 0),
        paused=throttle is not None and throttle.pause_reason is not None,
        pause_reason=throttle.pause_reason if throttle is not None else None,
        embedder_loaded=service.loaded,
        databases=databases,
    )


@router.get("/failures", response_model=Failures)
async def list_failures(
    folder: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    state: APIState = Depends(get_state),
) -> Failures:
    """Documents that failed to ingest, newest first."""
    jobs = await state.job_repo.list_jobs(
        status=JobStatus.DEAD, source_id=folder, limit=limit, offset=offset
    )
    failures = []
    for job in jobs:
        path = local_path(job.uri)
        failures.append(
            Failure(
                job_id=job.id,
                folder_id=job.source_id,
                database=state.source_databases.get(job.source_id),
                path=str(path) if path is not None else None,
                uri=job.uri,
                reason=job.last_error,
                attempts=job.attempts,
                failed_at=job.completed_at,
            )
        )
    return Failures(failures=failures)


@router.post("/failures/{job_id}/retry", response_model=Retried)
async def retry_failure(job_id: str, state: APIState = Depends(get_state)) -> Retried:
    await retry_served(job_id, state)
    return Retried(retried=[job_id])


@router.post("/failures/retry", response_model=Retried)
async def retry_failures(
    folder: str | None = None, state: APIState = Depends(get_state)
) -> Retried:
    """Queue every failed document again, or every one of `folder`."""
    retried = []
    jobs = await state.job_repo.list_jobs(
        status=JobStatus.DEAD, source_id=folder, limit=10_000
    )
    for job in jobs:
        try:
            await retry_served(job.id, state)
        except HTTPException:
            continue
        retried.append(job.id)
    return Retried(retried=retried)
