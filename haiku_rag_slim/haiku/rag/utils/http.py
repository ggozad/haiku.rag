import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Below the 5s idle timeout of uvicorn, which serves vLLM.
KEEPALIVE_EXPIRY = 2.0

# The connection died under a request: the server closed it as the request
# reached it, or the network dropped it.
DROPPED_CONNECTION_ERRORS = (
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
)


def pooled_client(timeout: float | httpx.Timeout) -> httpx.AsyncClient:
    """A client whose idle connections expire before the server closes them."""
    return httpx.AsyncClient(
        timeout=timeout,
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=KEEPALIVE_EXPIRY,
        ),
    )


async def post_retrying_dropped_connection(
    client: httpx.AsyncClient, url: str, **kwargs: Any
) -> httpx.Response:
    """POST, sending once more if the connection drops. Only for idempotent requests."""
    try:
        return await client.post(url, **kwargs)
    except DROPPED_CONNECTION_ERRORS as e:
        logger.warning(
            "%s dropped the connection (%s: %s); retrying once",
            url,
            type(e).__name__,
            e,
        )
        return await client.post(url, **kwargs)
