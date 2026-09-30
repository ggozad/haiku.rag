import dataclasses
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from haiku.rag.curate.api.server import APIState, get_state
from haiku.rag.curate.store.models import (
    Change,
    CurrentDocument,
    DatabaseSummary,
    Flag,
    FlagKind,
    FlagStatus,
    Health,
    Watch,
)
from haiku.rag.curate.sweep import find_chunk_text
from haiku.rag.store.exceptions import MigrationRequiredError, SourceUnavailableError

public = APIRouter()
router = APIRouter()

_DASHBOARD = (Path(__file__).resolve().parent / "static" / "index.html").read_text(
    encoding="utf-8"
)


class HealthResponse(BaseModel):
    status: str
    databases: list[DatabaseSummary]


class AcknowledgeRequest(BaseModel):
    note: str | None = None


class WatchRequest(BaseModel):
    database: str
    uri: str
    note: str | None = None


class TextResponse(BaseModel):
    text: str


def _known(state: APIState, database: str | None) -> None:
    if database is not None and database not in state.scope.names:
        raise HTTPException(
            status_code=404,
            detail=f"unknown database {database!r}; the databases are "
            f"{', '.join(state.scope.names)}",
        )


def _in_scope(state: APIState, flag: Flag) -> bool:
    """A cross-database flag is in scope only when every member's database is."""
    if flag.database is not None:
        return flag.database in state.scope.names
    return all(m["database"] in state.scope.names for m in flag.members or [])


async def _scoped_flag(state: APIState, flag_id: int) -> Flag:
    flag = await state.repository.flag(flag_id)
    if flag is None or not _in_scope(state, flag):
        raise HTTPException(status_code=404, detail=f"no flag {flag_id}")
    return flag


@public.get("/", include_in_schema=False)
async def dashboard(request: Request) -> HTMLResponse:
    """The dashboard; its script sends the token on its own requests."""
    root_path = request.scope.get("root_path", "")
    base_href = f"{root_path}/" if root_path else "/"
    return HTMLResponse(
        _DASHBOARD.replace("<head>", f'<head>\n    <base href="{base_href}" />', 1)
    )


@public.get("/health", response_model=HealthResponse)
async def health(state: APIState = Depends(get_state)) -> HealthResponse:
    """Liveness and the last sweep of each database; needs no token."""
    return HealthResponse(
        status="ok", databases=await state.repository.databases(state.scope.names)
    )


@router.get("/health/{database}", response_model=Health)
async def database_health(
    database: str, state: APIState = Depends(get_state)
) -> Health:
    """Doctor's checks of a database, from the last sweep that ran them."""
    _known(state, database)
    health = await state.repository.health(database)
    if health is None:
        raise HTTPException(
            status_code=404, detail=f"database {database!r} has not been checked yet"
        )
    return health


@router.get("/databases", response_model=list[DatabaseSummary])
async def databases(state: APIState = Depends(get_state)) -> list[DatabaseSummary]:
    return await state.repository.databases(state.scope.names)


@router.get("/flags", response_model=list[Flag])
async def list_flags(
    database: str | None = None,
    kind: FlagKind | None = None,
    status: FlagStatus | None = None,
    state: APIState = Depends(get_state),
) -> list[Flag]:
    _known(state, database)
    found = await state.repository.flags(database=database, kind=kind, status=status)
    return [flag for flag in found if _in_scope(state, flag)]


@router.post("/flags/{flag_id}/acknowledge", response_model=Flag)
async def acknowledge(
    flag_id: int, request: AcknowledgeRequest, state: APIState = Depends(get_state)
) -> Flag:
    await _scoped_flag(state, flag_id)
    await state.repository.acknowledge(flag_id, request.note)
    return await _scoped_flag(state, flag_id)


@router.post("/flags/{flag_id}/reopen", response_model=Flag)
async def reopen(flag_id: int, state: APIState = Depends(get_state)) -> Flag:
    await _scoped_flag(state, flag_id)
    if not await state.repository.reopen(flag_id):
        raise HTTPException(
            status_code=409, detail=f"flag {flag_id} is not acknowledged"
        )
    return await _scoped_flag(state, flag_id)


@router.get("/flags/{flag_id}/text", response_model=TextResponse)
async def repeated_text(
    flag_id: int, state: APIState = Depends(get_state)
) -> TextResponse:
    """The text a repeated_chunk flag stands for, read from its database."""
    flag = await _scoped_flag(state, flag_id)
    if flag.kind is not FlagKind.REPEATED_CHUNK:
        raise HTTPException(status_code=404, detail=f"no repeated_chunk flag {flag_id}")
    assert flag.database is not None and flag.subject is not None
    [ref] = state.scope.select([flag.database]).databases
    try:
        text = await find_chunk_text(
            ref,
            state.config,
            [member["document_id"] for member in flag.members or []],
            flag.subject,
        )
    except (SourceUnavailableError, MigrationRequiredError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    if text is None:
        raise HTTPException(status_code=404, detail="the text is no longer present")
    return TextResponse(text=text)


@router.get("/changes", response_model=list[Change])
async def changes(
    since: datetime | None = None,
    database: str | None = None,
    state: APIState = Depends(get_state),
) -> list[Change]:
    """Documents added, updated or deleted at or after `since`; a naive time is UTC."""
    _known(state, database)
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=UTC)
    names = [database] if database is not None else list(state.scope.names)
    return await state.repository.changes(since, names)


@router.get("/documents", response_model=list[CurrentDocument])
async def documents(
    database: str, state: APIState = Depends(get_state)
) -> list[CurrentDocument]:
    _known(state, database)
    return await state.repository.documents(database)


@router.get("/documents/{database}/{document_id}/history")
async def history(
    database: str, document_id: str, state: APIState = Depends(get_state)
) -> list[dict]:
    """Every fingerprint of a document, oldest first, without its centroid."""
    _known(state, database)
    fingerprints = await state.repository.history(database, document_id)
    if not fingerprints:
        raise HTTPException(status_code=404, detail=f"no document {document_id!r}")
    return [
        {k: v for k, v in dataclasses.asdict(f).items() if k != "centroid"}
        for f in fingerprints
    ]


@router.get("/watched", response_model=list[Watch])
async def watches(state: APIState = Depends(get_state)) -> list[Watch]:
    return [
        w for w in await state.repository.watches() if w.database in state.scope.names
    ]


@router.post("/watched", status_code=201)
async def watch(request: WatchRequest, state: APIState = Depends(get_state)) -> None:
    _known(state, request.database)
    await state.repository.watch(request.database, request.uri, request.note)


@router.delete("/watched", status_code=204)
async def unwatch(
    database: str, uri: str, state: APIState = Depends(get_state)
) -> Response:
    _known(state, database)
    if not await state.repository.unwatch(database, uri):
        raise HTTPException(status_code=404, detail=f"{uri!r} is not watched")
    return Response(status_code=204)
