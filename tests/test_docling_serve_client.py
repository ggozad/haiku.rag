"""DoclingServeClient round-robin distribution tests.

Each test here uses a unique base-URL set so the process-global cycle
map gives it a fresh itertools.cycle. Don't reuse URL strings across
tests — cycles persist for the lifetime of the process and would
resume mid-rotation, breaking specific-order assertions.
"""

import httpx
import pytest

from haiku.rag.config import CircuitBreakerConfig
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


def _failover_transport(
    down_hosts: set[str], task_id: str, result: dict
) -> tuple[httpx.MockTransport, list[str]]:
    """MockTransport where any request to a host in ``down_hosts`` raises a
    ConnectError (a crashed instance); other hosts serve a normal
    submit/poll/result trio. Mutate ``down_hosts`` mid-test to flip an
    instance's health. Records every host attempted."""
    seen: list[str] = []
    routes = _success_routes(task_id, result)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host in down_hosts:
            raise httpx.ConnectError("connection refused", request=request)
        return routes.get(request.url.path, httpx.Response(404))

    return httpx.MockTransport(handler), seen


async def _poll(client: DoclingServeClient):
    return await client.submit_and_poll(
        endpoint="/v1/convert/file/async",
        files={"file": ("x.md", b"x", "text/markdown")},
        data={},
    )


@pytest.mark.asyncio
async def test_retry_fails_over_to_healthy_instance():
    """A crashed instance (connection error) is retried on another instance and
    the call succeeds without surfacing the failure."""
    transport, seen = _failover_transport({"down-a"}, "t", {"ok": True})
    client = DoclingServeClient(
        base_urls=["http://down-a:5001", "http://up-a:5001"],
        transport=transport,
        retry_base_delay=0.0,
    )

    result = await _poll(client)

    assert result == {"ok": True}
    assert seen[0] == "down-a"
    # The successful trio all landed on the healthy host.
    assert seen[-3:] == ["up-a", "up-a", "up-a"]


@pytest.mark.asyncio
async def test_retry_5xx_fails_over():
    """A 5xx from a struggling instance is retried elsewhere (status-based
    retryability, distinct from the transport-error path)."""
    seen: list[str] = []
    ok = _success_routes("t", {"ok": True})

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host == "sad-b":
            return httpx.Response(503)
        return ok.get(request.url.path, httpx.Response(404))

    client = DoclingServeClient(
        base_urls=["http://sad-b:5001", "http://ok-b:5001"],
        transport=httpx.MockTransport(handler),
        retry_base_delay=0.0,
    )

    result = await _poll(client)
    assert result == {"ok": True}
    assert seen[0] == "sad-b"
    assert "ok-b" in seen


@pytest.mark.asyncio
async def test_retry_429_fails_over():
    """A 429 (transient overload) is retried elsewhere — the status-set
    membership branch of retryability, distinct from 5xx."""
    seen: list[str] = []
    ok = _success_routes("t", {"ok": True})

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host == "busy-g":
            return httpx.Response(429)
        return ok.get(request.url.path, httpx.Response(404))

    client = DoclingServeClient(
        base_urls=["http://busy-g:5001", "http://free-g:5001"],
        transport=httpx.MockTransport(handler),
        retry_base_delay=0.0,
    )

    result = await _poll(client)
    assert result == {"ok": True}
    assert seen[0] == "busy-g"
    assert "free-g" in seen


@pytest.mark.asyncio
async def test_retry_exhausts_all_instances_then_raises():
    """When every instance is down, the call retries up to max_attempts and then
    surfaces the transport error."""
    transport, seen = _failover_transport({"down-c", "down-d"}, "t", {})
    client = DoclingServeClient(
        base_urls=["http://down-c:5001", "http://down-d:5001"],
        transport=transport,
        max_attempts=2,
        retry_base_delay=0.0,
    )

    with pytest.raises(httpx.ConnectError):
        await _poll(client)

    # Two attempts, each preferring a not-yet-failed instance.
    assert seen == ["down-c", "down-d"]


@pytest.mark.asyncio
async def test_4xx_is_not_retried():
    """A 4xx (other than 408/429) is the caller's fault — not retried on another
    instance; it propagates after a single attempt."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        return httpx.Response(400, json={"detail": "bad request"})

    client = DoclingServeClient(
        base_urls=["http://e:5001", "http://f:5001"],
        transport=httpx.MockTransport(handler),
        retry_base_delay=0.0,
    )

    with pytest.raises(httpx.HTTPStatusError):
        await _poll(client)

    assert seen == ["e"]


@pytest.mark.asyncio
async def test_task_failure_is_not_retried():
    """A docling-serve task 'failure' status raises ValueError and is NOT
    retried — a document problem, not an instance one."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.path == "/v1/convert/file/async":
            return httpx.Response(200, json={"task_id": "t"})
        if request.url.path == "/v1/status/poll/t":
            return httpx.Response(200, json={"task_status": "failure", "detail": "x"})
        return httpx.Response(404)

    client = DoclingServeClient(
        base_urls=["http://h:5001", "http://i:5001"],
        transport=httpx.MockTransport(handler),
        retry_base_delay=0.0,
    )

    with pytest.raises(ValueError, match="task failed"):
        await _poll(client)

    # Only the first instance was attempted (submit + poll) — no failover.
    assert set(seen) == {"h"}


@pytest.mark.asyncio
async def test_request_span_records_instance_per_attempt(monkeypatch):
    """Each attempt opens a docling_serve.request span tagged with the
    instance URL, so failover is traceable in Logfire."""
    from contextlib import nullcontext

    from haiku.rag.providers import docling_serve as ds_module

    spans: list[dict] = []

    def _fake_span(span_name, /, **attrs):
        spans.append({"span_name": span_name, **attrs})
        return nullcontext()

    monkeypatch.setattr(ds_module.logfire, "span", _fake_span)

    transport, _ = _failover_transport({"down-s"}, "t", {"ok": True})
    client = DoclingServeClient(
        base_urls=["http://down-s:5001", "http://up-s:5001"],
        transport=transport,
        retry_base_delay=0.0,
    )

    await _poll(client)

    requests = [s for s in spans if s["span_name"] == "docling_serve.request"]
    assert [s["url"] for s in requests] == ["http://down-s:5001", "http://up-s:5001"]
    assert [s["attempt"] for s in requests] == [0, 1]


def test_pick_url_skips_excluded_instances():
    """On retry, _pick_url advances past every excluded instance."""
    urls = ["http://p1:5001", "http://p2:5001", "http://p3:5001"]
    client = DoclingServeClient(base_urls=urls)
    assert client._pick_url(exclude=frozenset({urls[0]})) == urls[1]
    assert client._pick_url(exclude=frozenset({urls[1], urls[2]})) == urls[0]


@pytest.mark.asyncio
async def test_retryable_failure_fails_over_and_trips_breaker():
    """A retryable failure does both jobs at once: the request fails over to a
    healthy instance AND the failure counts against the crashed instance's
    breaker."""
    trip_a, trip_b = "http://trip-a:5001", "http://trip-b:5001"
    transport, seen = _failover_transport({"trip-a"}, "t", {"ok": True})
    client = DoclingServeClient(
        base_urls=[trip_a, trip_b],
        transport=transport,
        circuit_breaker=CircuitBreakerConfig(failure_threshold=1, cooldown_s=30.0),
        max_attempts=2,
        retry_base_delay=0.0,
    )

    result = await _poll(client)

    assert result == {"ok": True}
    assert seen[0] == "trip-a"
    assert client._breaker_for(trip_a).is_open
    assert not client._breaker_for(trip_b).is_open


@pytest.mark.asyncio
async def test_open_breaker_skips_crashed_instance():
    """Once an instance's breaker has opened, later requests route straight to a
    healthy instance without even attempting the dead one."""
    crash, live = "http://crash-x:5001", "http://live-x:5001"
    transport, seen = _failover_transport({"crash-x"}, "t", {"ok": True})
    client = DoclingServeClient(
        base_urls=[crash, live],
        transport=transport,
        circuit_breaker=CircuitBreakerConfig(failure_threshold=1, cooldown_s=30.0),
        max_attempts=2,
        retry_base_delay=0.0,
    )

    await _poll(client)  # crash-x fails, opens its breaker, fails over to live-x
    assert client._breaker_for(crash).is_open

    seen.clear()
    for _ in range(3):
        await _poll(client)
    assert "crash-x" not in seen
    assert set(seen) == {"live-x"}


@pytest.mark.asyncio
async def test_breaker_recovers_after_cooldown():
    """An open breaker auto-probes after its cooldown; once the instance is
    healthy again a successful request closes the breaker."""
    clock = [1000.0]
    down = {"flip-y"}
    flip, spare = "http://flip-y:5001", "http://spare-y:5001"
    transport, seen = _failover_transport(down, "t", {"ok": True})
    client = DoclingServeClient(
        base_urls=[flip, spare],
        transport=transport,
        circuit_breaker=CircuitBreakerConfig(failure_threshold=1, cooldown_s=30.0),
        max_attempts=2,
        retry_base_delay=0.0,
        now_fn=lambda: clock[0],
    )

    await _poll(client)  # flip-y fails, opens; fails over to spare-y
    assert client._breaker_for(flip).is_open

    # Instance recovers, but within the cooldown it's still treated as open.
    down.clear()
    assert client._breaker_for(flip).is_open

    # After the cooldown the breaker allows a probe; traffic returns and a
    # success closes it.
    clock[0] += 31.0
    seen.clear()
    await _poll(client)
    assert "flip-y" in seen
    assert not client._breaker_for(flip).is_open


@pytest.mark.asyncio
async def test_4xx_does_not_trip_breaker():
    """A 4xx is the caller's fault — it must not count against instance health,
    even at a 1-failure threshold."""
    bad = "http://bad-z:5001"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "bad request"})

    client = DoclingServeClient(
        base_urls=[bad],
        transport=httpx.MockTransport(handler),
        circuit_breaker=CircuitBreakerConfig(failure_threshold=1, cooldown_s=30.0),
        retry_base_delay=0.0,
    )

    for _ in range(3):
        with pytest.raises(httpx.HTTPStatusError):
            await _poll(client)

    assert not client._breaker_for(bad).is_open


def test_all_open_breakers_rotate_instead_of_pinning_one():
    """When every breaker is open, successive picks rotate across the fleet
    rather than pinning the first instance — an all-429 overload shouldn't pile
    every retry on one node."""
    urls = [
        "http://allopen-a:5001",
        "http://allopen-b:5001",
        "http://allopen-c:5001",
    ]
    client = DoclingServeClient(
        base_urls=urls,
        circuit_breaker=CircuitBreakerConfig(failure_threshold=1, cooldown_s=30.0),
    )
    for u in urls:
        client._breaker_for(u).record_failure()
        assert client._breaker_for(u).is_open

    picks = [client._pick_url() for _ in range(6)]
    assert set(picks) == set(urls)


def test_from_config_wires_retry_and_breaker():
    """Retry/breaker knobs set in DoclingServeConfig reach the client via
    get_converter / get_chunker (DoclingServeClient.from_config)."""
    from haiku.rag.chunkers.docling_serve import DoclingServeChunker
    from haiku.rag.config import AppConfig
    from haiku.rag.converters.docling_serve import DoclingServeConverter

    config = AppConfig()
    ds = config.providers.docling_serve
    ds.base_url = "http://cfg-n:5001"
    ds.max_attempts = 7
    ds.timeout = 42.0
    ds.circuit_breaker = CircuitBreakerConfig(failure_threshold=9, cooldown_s=90.0)

    for component in (DoclingServeConverter(config), DoclingServeChunker(config)):
        client = component.client
        assert client._max_attempts == 7
        assert client.timeout == 42.0
        assert client._breaker_config.failure_threshold == 9
        assert client._breaker_config.cooldown_s == 90.0


@pytest.mark.asyncio
async def test_submit_without_task_id_raises():
    """A 200 that carries no task_id is a protocol violation, not a silent pass."""
    import httpx

    from haiku.rag.providers.docling_serve import DoclingServeClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    client = DoclingServeClient(base_urls="http://docling:5001")
    files = {"files": ("doc.pdf", b"pdf", "application/octet-stream")}

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ValueError, match="did not return a task_id"):
            await client._submit_and_wait(
                http,
                "http://docling:5001",
                "/v1/convert/source/async",
                files,
                {},
                {},
                "doc.pdf",
            )
