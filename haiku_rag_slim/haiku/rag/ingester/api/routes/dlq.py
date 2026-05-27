from fastapi import APIRouter, Depends, HTTPException, Query, status

from haiku.rag.ingester.api.server import APIState, get_state
from haiku.rag.ingester.queue.models import Job, JobStatus

router = APIRouter(prefix="/dlq", tags=["dlq"])


@router.get("", response_model=list[Job])
async def list_dlq(
    source_id: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    state: APIState = Depends(get_state),
) -> list[Job]:
    """Jobs that exhausted retries or hit a permanent error."""
    return await state.job_repo.list_jobs(
        status=JobStatus.DEAD, source_id=source_id, limit=limit, offset=offset
    )


@router.post("/{job_id}/retry", response_model=Job)
async def retry_from_dlq(job_id: str, state: APIState = Depends(get_state)) -> Job:
    """Convenience alias for /jobs/{id}/retry — same effect, separate
    endpoint so operator tooling can wire DLQ rescues without coupling
    to the generic /jobs path."""
    try:
        return await state.job_repo.retry(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="job not found"
        ) from exc
