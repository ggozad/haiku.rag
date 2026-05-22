from fastapi import APIRouter, Depends

from haiku.rag.ingester.api.schemas import HealthResponse
from haiku.rag.ingester.api.server import APIState, get_state

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(state: APIState = Depends(get_state)) -> HealthResponse:
    """Liveness signal + queue/worker overview. Unauthenticated so load
    balancers and uptime monitors can hit it without a token."""
    counts = await state.job_repo.counts_by_status()
    worker_count = (
        state.config.ingester.workers.worker_count if state.pool is not None else 0
    )
    poller_count = len(state.pollers.pollers) if state.pollers is not None else 0
    return HealthResponse(
        status="ok",
        queue_counts=counts,
        worker_count=worker_count,
        poller_count=poller_count,
    )
