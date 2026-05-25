from fastapi import APIRouter, Depends

from haiku.rag.ingester.api.schemas import HealthResponse
from haiku.rag.ingester.api.server import APIState, get_state

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(state: APIState = Depends(get_state)) -> HealthResponse:
    """Liveness signal + queue/worker overview. Unauthenticated so load
    balancers and uptime monitors can hit it without a token.

    `status="degraded"` when at least one configured worker or poller task
    has died (workers_alive < worker_count or pollers_alive < poller_count).
    The DB query alone can't catch that — workers might be all dead and the
    queue would still report counts cheerfully — so this branch is what
    actually distinguishes "alive" from "alive but doing nothing useful".
    """
    counts = await state.job_repo.counts_by_status()
    worker_count = (
        state.config.ingester.workers.worker_count if state.pool is not None else 0
    )
    workers_alive = state.pool.live_workers if state.pool is not None else 0
    poller_count = len(state.pollers.pollers) if state.pollers is not None else 0
    pollers_alive = state.pollers.live_pollers if state.pollers is not None else 0
    degraded = workers_alive < worker_count or pollers_alive < poller_count
    return HealthResponse(
        status="degraded" if degraded else "ok",
        queue_counts=counts,
        worker_count=worker_count,
        poller_count=poller_count,
        workers_alive=workers_alive,
        pollers_alive=pollers_alive,
    )
