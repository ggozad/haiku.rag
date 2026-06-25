"""DoclingServeClient round-robin distribution tests.

Each test here uses a unique base-URL set so the process-global cycle
map gives it a fresh itertools.cycle. Don't reuse URL strings across
tests — cycles persist for the lifetime of the process and would
resume mid-rotation, breaking specific-order assertions.
"""

import httpx
import pytest

from haiku.rag.providers.docling_serve import DoclingServeClient


def _scripted_transport(responses_by_path: dict[str, httpx.Response]):
    """MockTransport routing on (host, path) — lets us assert which URL got hit."""
    seen_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.url.host)
        key = request.url.path
        if key in responses_by_path:
            return responses_by_path[key]
        return httpx.Response(404)

    return httpx.MockTransport(handler), seen_hosts


def _success_routes(task_id: str, result: dict) -> dict[str, httpx.Response]:
    return {
        "/v1/convert/file/async": httpx.Response(200, json={"task_id": task_id}),
        f"/v1/status/poll/{task_id}": httpx.Response(
            200, json={"task_status": "success"}
        ),
        f"/v1/result/{task_id}": httpx.Response(200, json=result),
    }


def test_single_url_input_normalises_to_list():
    client = DoclingServeClient(base_urls="http://only:5001")
    assert client.base_urls == ["http://only:5001"]


def test_empty_list_raises():
    with pytest.raises(ValueError, match="at least one"):
        DoclingServeClient(base_urls=[])


def test_trailing_slashes_stripped():
    client = DoclingServeClient(base_urls=["http://a:5001/", "http://b:5001//"])
    assert client.base_urls == ["http://a:5001", "http://b:5001"]


@pytest.mark.asyncio
async def test_round_robin_across_three_urls():
    transport, seen = _scripted_transport(_success_routes("t", {"ok": True}))
    client = DoclingServeClient(
        base_urls=["http://a:5001", "http://b:5001", "http://c:5001"],
        transport=transport,
    )

    for _ in range(6):
        await client.submit_and_poll(
            endpoint="/v1/convert/file/async",
            files={"file": ("x.md", b"x", "text/markdown")},
            data={},
        )

    # 3 round-trips per call (POST + GET poll + GET result) * 6 calls = 18 requests.
    # Each call must stay on one host; calls rotate through a, b, c, a, b, c.
    per_call = [seen[i : i + 3] for i in range(0, 18, 3)]
    assert all(len(set(triple)) == 1 for triple in per_call), (
        "submit/poll/result split across hosts — task_id wouldn't resolve"
    )
    hosts_picked = [triple[0] for triple in per_call]
    assert hosts_picked == ["a", "b", "c", "a", "b", "c"]


@pytest.mark.asyncio
async def test_task_lifecycle_pinned_to_same_url():
    """A single submit/poll/result trio must all hit the same instance —
    task IDs are local to the instance that issued them."""
    transport, seen = _scripted_transport(_success_routes("task-42", {"r": 1}))
    client = DoclingServeClient(
        base_urls=["http://primary:5001", "http://secondary:5001"],
        transport=transport,
    )

    await client.submit_and_poll(
        endpoint="/v1/convert/file/async",
        files={"file": ("x.md", b"x", "text/markdown")},
        data={},
    )

    # All three requests must be on the same host.
    assert len(set(seen)) == 1


def test_round_robin_shared_across_fresh_clients():
    """get_converter / get_chunker build a NEW DoclingServeClient per job.
    The cycle has to live outside the instance so successive jobs (each
    with its own freshly-constructed client) actually rotate."""
    urls = ["http://x:5001", "http://y:5001", "http://z:5001"]
    c1 = DoclingServeClient(base_urls=urls)
    c2 = DoclingServeClient(base_urls=urls)
    c3 = DoclingServeClient(base_urls=urls)
    c4 = DoclingServeClient(base_urls=urls)

    picks = [c1._pick_url(), c2._pick_url(), c3._pick_url(), c4._pick_url()]
    assert picks == [urls[0], urls[1], urls[2], urls[0]]


def _health_transport(
    down: set[str], task_id: str, result: dict
) -> tuple[httpx.MockTransport, list[str]]:
    """MockTransport where requests to a host in the mutable ``down`` set raise
    ConnectError (a crashed instance); other hosts serve a normal trio. The
    caller can mutate ``down`` mid-test to flip an instance's health. Records
    every host attempted."""
    seen: list[str] = []
    routes = _success_routes(task_id, result)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host in down:
            raise httpx.ConnectError("connection refused", request=request)
        return routes.get(request.url.path, httpx.Response(404))

    return httpx.MockTransport(handler), seen


async def _poll(client: DoclingServeClient) -> None:
    """Issue one submit_and_poll, swallowing the transport error raised when the
    picked instance is down (this baseline has no in-request failover yet)."""
    try:
        await client.submit_and_poll(
            endpoint="/v1/convert/file/async",
            files={"file": ("x.md", b"x", "text/markdown")},
            data={},
        )
    except httpx.TransportError:
        pass


@pytest.mark.asyncio
async def test_open_breaker_skips_crashed_instance():
    """Once an instance has failed enough to open its breaker, _pick_url skips
    it — later requests route straight to a healthy instance without even
    attempting the dead one."""
    down = {"crash-x"}
    transport, seen = _health_transport(down, "t", {"ok": True})
    client = DoclingServeClient(
        base_urls=["http://crash-x:5001", "http://live-x:5001"],
        transport=transport,
        breaker_failure_threshold=2,
    )

    await _poll(client)  # picks crash-x → fail (failures=1)
    await _poll(client)  # picks live-x → ok
    await _poll(client)  # picks crash-x → fail (failures=2 → breaker opens)

    assert client._breaker_for("http://crash-x:5001").is_open

    seen.clear()
    await _poll(client)
    await _poll(client)
    assert "crash-x" not in seen
    assert set(seen) == {"live-x"}


@pytest.mark.asyncio
async def test_breaker_recovers_after_cooldown():
    """An open breaker auto-probes after its cooldown; once the instance is
    healthy again a successful request closes the breaker."""
    clock = [1000.0]
    down = {"flip-y"}
    transport, seen = _health_transport(down, "t", {"ok": True})
    flip = "http://flip-y:5001"
    client = DoclingServeClient(
        base_urls=[flip, "http://spare-y:5001"],
        transport=transport,
        breaker_failure_threshold=2,
        breaker_cooldown_s=30.0,
        now_fn=lambda: clock[0],
    )

    await _poll(client)  # flip-y → fail (1)
    await _poll(client)  # spare-y → ok
    await _poll(client)  # flip-y → fail (2 → open)
    assert client._breaker_for(flip).is_open

    # Instance recovers, but within the cooldown it's still treated as open.
    down.clear()
    assert client._breaker_for(flip).is_open

    # After the cooldown the breaker allows a probe again.
    clock[0] += 31.0
    assert not client._breaker_for(flip).is_open

    # Traffic can return to flip-y and success closes the breaker for good.
    seen.clear()
    for _ in range(2):
        await _poll(client)
    assert "flip-y" in seen
    assert not client._breaker_for(flip).is_open


@pytest.mark.asyncio
async def test_4xx_does_not_trip_breaker():
    """A 4xx is the caller's fault — it must not count against instance health,
    even with a 1-failure threshold."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        return httpx.Response(400, json={"detail": "bad request"})

    bad = "http://bad-z:5001"
    client = DoclingServeClient(
        base_urls=[bad],
        transport=httpx.MockTransport(handler),
        breaker_failure_threshold=1,
    )

    for _ in range(3):
        with pytest.raises(httpx.HTTPStatusError):
            await client.submit_and_poll(
                endpoint="/v1/convert/file/async",
                files={"file": ("x.md", b"x", "text/markdown")},
                data={},
            )

    assert not client._breaker_for(bad).is_open


@pytest.mark.asyncio
async def test_zip_endpoint_uses_round_robin_too():
    transport, seen = _scripted_transport(
        {
            "/v1/convert/file/async": httpx.Response(200, json={"task_id": "t"}),
            "/v1/status/poll/t": httpx.Response(200, json={"task_status": "success"}),
            "/v1/result/t": httpx.Response(200, content=b"zip-bytes"),
        }
    )
    client = DoclingServeClient(
        base_urls=["http://a:5001", "http://b:5001"], transport=transport
    )
    await client.submit_and_poll_zip(
        endpoint="/v1/convert/file/async",
        files={"file": ("x.md", b"x", "text/markdown")},
        data={},
    )
    await client.submit_and_poll_zip(
        endpoint="/v1/convert/file/async",
        files={"file": ("x.md", b"x", "text/markdown")},
        data={},
    )
    per_call = [seen[i : i + 3] for i in range(0, 6, 3)]
    assert [triple[0] for triple in per_call] == ["a", "b"]
