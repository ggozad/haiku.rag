from fastapi import APIRouter, Depends

from haiku.rag.ingester.api.schemas import (
    StatsResponse,
    ThroughputStats,
    WorkerStats,
)
from haiku.rag.ingester.api.server import APIState, get_state

router = APIRouter(tags=["stats"])


@router.get("/stats", response_model=StatsResponse)
async def stats(state: APIState = Depends(get_state)) -> StatsResponse:
    """Dashboard summary: rolling throughput, worker occupancy, backlog age,
    and per-source DLQ / queue depth. Each field is a single SQL aggregation
    against the queue file — cheap to call every few seconds."""
    jobs = state.job_repo

    counts = await jobs.counts_by_status()
    worker_total = (
        state.config.ingester.workers.worker_count if state.pool is not None else 0
    )

    return StatsResponse(
        throughput=ThroughputStats(
            succeeded_5m=await jobs.count_succeeded_since(300),
            succeeded_30m=await jobs.count_succeeded_since(1800),
            succeeded_1h=await jobs.count_succeeded_since(3600),
        ),
        workers=WorkerStats(
            busy=counts.get("claimed", 0),
            total=worker_total,
        ),
        oldest_queued_age_s=await jobs.oldest_queued_age_seconds(),
        dlq_by_source=await jobs.counts_by_source("dead"),
        queue_depth_by_source=await jobs.counts_by_source("queued", "claimed"),
    )
