import asyncio

import httpx
from fastapi import APIRouter, Depends

from haiku.rag.ingester.api.schemas import ProviderEndpoint, ProvidersResponse
from haiku.rag.ingester.api.server import APIState, get_state

router = APIRouter(tags=["providers"])

# Short timeout — operators care that the endpoint is reachable right now,
# not that it might respond if we wait. A docling-serve that takes longer
# than this to answer /health is effectively down for ingest purposes.
_PROBE_TIMEOUT_S = 2.0


async def _probe(client: httpx.AsyncClient, base_url: str) -> ProviderEndpoint:
    url = f"{base_url.rstrip('/')}/health"
    try:
        response = await client.get(url)
    except httpx.HTTPError as exc:
        return ProviderEndpoint(base_url=base_url, reachable=False, error=str(exc))
    return ProviderEndpoint(
        base_url=base_url,
        reachable=response.is_success,
        status_code=response.status_code,
    )


@router.get("/providers", response_model=ProvidersResponse)
async def providers(state: APIState = Depends(get_state)) -> ProvidersResponse:
    """Probe external providers actually in use by the current converter /
    chunker and return their reachability. Dashboard polls this to surface a
    downstream outage that the ingester itself can only see via worker job
    failures."""
    processing = state.config.processing
    uses_docling_serve = (
        processing.converter == "docling-serve" or processing.chunker == "docling-serve"
    )
    if not uses_docling_serve:
        return ProvidersResponse(docling_serve=[])
    base_urls = state.config.providers.docling_serve.base_urls
    async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S) as client:
        results = await asyncio.gather(*(_probe(client, u) for u in base_urls))
    return ProvidersResponse(docling_serve=list(results))
