from fastapi import APIRouter, Depends, HTTPException, Query, status

from haiku.rag.ingester.api.schemas import CancelResponse
from haiku.rag.ingester.api.server import APIState, get_state
from haiku.rag.ingester.queue.models import Job, JobStatus

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=list[Job])
async def list_jobs(
    status: JobStatus | None = None,
    source_id: str | None = None,
    uri: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    state: APIState = Depends(get_state),
) -> list[Job]:
    return await state.job_repo.list_jobs(
        status=status, source_id=source_id, uri=uri, limit=limit, offset=offset
    )


@router.get("/{job_id}", response_model=Job)
async def get_job(job_id: str, state: APIState = Depends(get_state)) -> Job:
    job = await state.job_repo.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="job not found"
        )
    return job


@router.post("/{job_id}/retry", response_model=Job)
async def retry_job(job_id: str, state: APIState = Depends(get_state)) -> Job:
    """Force a dead or queued job back to queued with attempts=0."""
    try:
        return await state.job_repo.retry(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="job not found"
        ) from exc


@router.delete("/{job_id}", response_model=CancelResponse)
async def cancel_job(
    job_id: str, state: APIState = Depends(get_state)
) -> CancelResponse:
    cancelled = await state.job_repo.cancel(job_id)
    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="job not found or already terminal",
        )
    return CancelResponse(job_id=job_id, cancelled=True)
