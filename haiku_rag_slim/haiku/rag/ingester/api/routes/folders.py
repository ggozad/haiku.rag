import asyncio
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from haiku.rag.folders import (
    Folder,
    FolderConflictError,
    FolderRegister,
    FolderRegisterError,
    UnknownFolderError,
)
from haiku.rag.ingester.api.server import APIState, get_state
from haiku.rag.ingester.queue.models import JobStatus

router = APIRouter(prefix="/folders", tags=["folders"])


class FolderStatus(BaseModel):
    watching: bool
    documents: int
    queued: int
    failed: int
    last_scan_at: datetime | None = None


class FolderView(BaseModel):
    id: str
    database: str
    path: Path
    include: list[str]
    exclude: list[str]
    status: FolderStatus | None = None


class FolderList(BaseModel):
    folders: list[FolderView]


class NewFolder(BaseModel):
    path: str
    database: str | None = None
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    id: str | None = None


class FolderChange(BaseModel):
    include: list[str] | None = None
    exclude: list[str] | None = None


class Removed(BaseModel):
    removed: str


def _register(state: APIState) -> FolderRegister:
    try:
        return FolderRegister.of(state.config)
    except FolderRegisterError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from None


async def _view(folder: Folder, state: APIState) -> FolderView:
    if folder.id not in state.source_databases:
        return FolderView(**folder.model_dump(), status=None)
    active = await state.job_repo.counts_by_source(
        JobStatus.QUEUED.value, JobStatus.CLAIMED.value
    )
    dead = await state.job_repo.counts_by_source(JobStatus.DEAD.value)
    last_scan = None
    if state.pollers is not None:
        for poller in state.pollers.pollers:
            if poller.source_id == folder.id:
                last_scan = poller.last_polled_at
    return FolderView(
        **folder.model_dump(),
        status=FolderStatus(
            watching=True,
            documents=len(await state.sync_repo.list_ingested_uris(folder.id)),
            queued=active.get(folder.id, 0),
            failed=dead.get(folder.id, 0),
            last_scan_at=last_scan,
        ),
    )


def _applied(state: APIState) -> None:
    if state.request_reload is not None:
        state.request_reload()


@router.get("", response_model=FolderList)
async def list_folders(state: APIState = Depends(get_state)) -> FolderList:
    folders = await asyncio.to_thread(_register(state).folders)
    return FolderList(folders=[await _view(folder, state) for folder in folders])


@router.post("", response_model=FolderView, status_code=status.HTTP_201_CREATED)
async def add_folder(
    body: NewFolder, state: APIState = Depends(get_state)
) -> FolderView:
    register = _register(state)
    try:
        folder = await asyncio.to_thread(
            register.add,
            body.path,
            database=body.database,
            include=body.include,
            exclude=body.exclude,
            folder_id=body.id,
        )
    except FolderConflictError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e)) from None
    except FolderRegisterError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from None
    _applied(state)
    return FolderView(**folder.model_dump())


@router.patch("/{folder_id}", response_model=FolderView)
async def update_folder(
    folder_id: str, body: FolderChange, state: APIState = Depends(get_state)
) -> FolderView:
    register = _register(state)
    try:
        folder = await asyncio.to_thread(
            register.update, folder_id, include=body.include, exclude=body.exclude
        )
    except UnknownFolderError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from None
    _applied(state)
    return await _view(folder, state)


@router.delete("/{folder_id}", response_model=Removed)
async def remove_folder(
    folder_id: str,
    keep_documents: bool = False,
    state: APIState = Depends(get_state),
) -> Removed:
    register = _register(state)
    try:
        await asyncio.to_thread(
            register.remove, folder_id, keep_documents=keep_documents
        )
    except UnknownFolderError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from None
    _applied(state)
    return Removed(removed=folder_id)
