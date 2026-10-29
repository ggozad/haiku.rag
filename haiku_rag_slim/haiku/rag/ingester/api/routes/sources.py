from fastapi import APIRouter, Depends, HTTPException, Query, status

from haiku.rag.ingester.api.schemas import (
    RefreshResponse,
    SourceRetryResponse,
    SourceSummary,
)
from haiku.rag.ingester.api.server import APIState, get_state
from haiku.rag.ingester.pollers.base import BasePoller
from haiku.rag.ingester.queue.models import RetrySelection

router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("", response_model=list[SourceSummary])
async def list_sources(
    state: APIState = Depends(get_state),
) -> list[SourceSummary]:
    if state.pollers is None:
        return []
    summaries: list[SourceSummary] = []
    for poller in state.pollers.pollers:
        summaries.append(
            SourceSummary(
                source_id=poller.source_id,
                database=state.source_databases.get(poller.source_id),
                type=poller.config.type,
                last_polled_at=poller.last_polled_at,
                circuit_breaker_open=poller.is_circuit_open,
                last_skip_reason=poller.last_skip_reason,
                last_error=poller.last_error,
                last_error_at=poller.last_error_at,
            )
        )
    return summaries


@router.post("/{source_id:path}/refresh", response_model=RefreshResponse)
async def refresh_source(
    source_id: str, state: APIState = Depends(get_state)
) -> RefreshResponse:
    """Out-of-band poll: forces an immediate `discover()` sweep."""
    ok = await _poller(source_id, state)._sweep_once()
    return RefreshResponse(source_id=source_id, refreshed=ok)


@router.post("/{source_id:path}/retry", response_model=SourceRetryResponse)
async def retry_source(
    source_id: str,
    selection: RetrySelection = Query(alias="status"),
    state: APIState = Depends(get_state),
) -> SourceRetryResponse:
    """Re-queue the latest failure of each of the source's documents not
    ingested since, while no discovery of the source runs."""
    poller = _poller(source_id, state)
    retried = None
    if source_id in state.source_databases:
        retried = await poller.retry_failed(selection.statuses)
    if retried is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"source {source_id!r} is not served, so no worker would "
            "claim its jobs",
        )
    return SourceRetryResponse(
        source_id=source_id,
        retried=retried,
        worker_breaker_open=state.pool is not None
        and state.pool.breaker_open_for(source_id),
    )


def _poller(source_id: str, state: APIState) -> BasePoller:
    if state.pollers is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="pollers not running",
        )
    for poller in state.pollers.pollers:
        if poller.source_id == source_id:
            return poller
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail="source not found"
    )
