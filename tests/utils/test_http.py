import logging

import httpx
import pytest
from tests.conftest import capture_logs

from haiku.rag.utils import http
from haiku.rag.utils.http import post_retrying_dropped_connection


def _client(failures: list[Exception], calls: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if failures:
            raise failures.pop(0)
        return httpx.Response(200, json={"ok": True})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    "error", [httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError]
)
async def test_dropped_connection_is_retried_once(error):
    """A POST whose connection drops is sent once more, with the same body."""
    calls: list[httpx.Request] = []
    client = _client([error("Server disconnected")], calls)

    with capture_logs(http.logger, logging.WARNING) as records:
        response = await post_retrying_dropped_connection(
            client, "http://server/v1/embeddings", json={"input": ["a"]}
        )

    assert response.json() == {"ok": True}
    assert [c.content for c in calls] == [b'{"input":["a"]}'] * 2
    assert len(records) == 1
    assert error.__name__ in records[0].getMessage()


async def test_second_dropped_connection_raises():
    calls: list[httpx.Request] = []
    client = _client([httpx.RemoteProtocolError("one"), httpx.ReadError("two")], calls)

    with capture_logs(http.logger, logging.WARNING):
        with pytest.raises(httpx.ReadError, match="two"):
            await post_retrying_dropped_connection(client, "http://server/v1/x")
    assert len(calls) == 2


@pytest.mark.parametrize(
    "error", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")]
)
async def test_other_transport_errors_are_not_retried(error):
    calls: list[httpx.Request] = []
    client = _client([error], calls)

    with pytest.raises(type(error)):
        await post_retrying_dropped_connection(client, "http://server/v1/x")
    assert len(calls) == 1
