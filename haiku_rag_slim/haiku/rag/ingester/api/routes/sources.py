from fastapi import APIRouter, Depends, HTTPException, status

from haiku.rag.ingester.api.schemas import RefreshResponse, SourceSummary
from haiku.rag.ingester.api.server import APIState, get_state

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
                type=type(poller.config).__name__,
                last_polled_at=poller.last_polled_at,
                circuit_breaker_open=poller.is_circuit_open,
                last_skip_reason=poller.last_skip_reason,
            )
        )
    return summaries


@router.post("/{source_id}/refresh", response_model=RefreshResponse)
async def refresh_source(
    source_id: str, state: APIState = Depends(get_state)
) -> RefreshResponse:
    """Out-of-band poll: forces an immediate `discover()` sweep."""
    if state.pollers is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="pollers not running",
        )
    for poller in state.pollers.pollers:
        if poller.source_id == source_id:
            ok = await poller._sweep_once()
            return RefreshResponse(source_id=source_id, refreshed=ok)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail="source not found"
    )
