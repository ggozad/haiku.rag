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


def _failover_transport(
    down_hosts: set[str], task_id: str, result: dict
) -> tuple[httpx.MockTransport, list[str]]:
    """MockTransport where any request to a host in ``down_hosts`` raises a
    ConnectError (simulating a crashed instance); other hosts serve a normal
    submit/poll/result trio. Records every host attempted."""
    seen_hosts: list[str] = []
    routes = _success_routes(task_id, result)

    def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.url.host)
        if request.url.host in down_hosts:
            raise httpx.ConnectError("connection refused", request=request)
        return routes.get(request.url.path, httpx.Response(404))

    return httpx.MockTransport(handler), seen_hosts


@pytest.mark.asyncio
async def test_retry_fails_over_to_healthy_instance():
    """A crashed instance (connection error) is retried on the next instance,
    and the call succeeds without surfacing the failure."""
    transport, seen = _failover_transport(
        down_hosts={"down-a"}, task_id="t", result={"ok": True}
    )
    client = DoclingServeClient(
        base_urls=["http://down-a:5001", "http://up-a:5001"],
        transport=transport,
        retry_base_delay=0.0,  # don't sleep in tests
    )

    result = await client.submit_and_poll(
        endpoint="/v1/convert/file/async",
        files={"file": ("x.md", b"x", "text/markdown")},
        data={},
    )

    assert result == {"ok": True}
    # First attempt hit the crashed instance; failover moved to the healthy one.
    assert seen[0] == "down-a"
    assert "up-a" in seen
    # The successful trio (submit/poll/result) all landed on the healthy host.
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

    result = await client.submit_and_poll(
        endpoint="/v1/convert/file/async",
        files={"file": ("x.md", b"x", "text/markdown")},
        data={},
    )
    assert result == {"ok": True}
    assert seen[0] == "sad-b"
    assert "ok-b" in seen


@pytest.mark.asyncio
async def test_retry_exhausts_all_instances_then_raises():
    """When every instance is down, the call retries up to max_attempts and
    then surfaces the transport error."""
    transport, seen = _failover_transport(
        down_hosts={"down-c", "down-d"}, task_id="t", result={}
    )
    client = DoclingServeClient(
        base_urls=["http://down-c:5001", "http://down-d:5001"],
        transport=transport,
        max_attempts=2,
        retry_base_delay=0.0,
    )

    with pytest.raises(httpx.ConnectError):
        await client.submit_and_poll(
            endpoint="/v1/convert/file/async",
            files={"file": ("x.md", b"x", "text/markdown")},
            data={},
        )

    # Two attempts, each preferring a not-yet-failed instance.
    assert seen == ["down-c", "down-d"]


@pytest.mark.asyncio
async def test_4xx_is_not_retried():
    """A 4xx (other than 408/429) is the caller's fault — it must not be
    retried on another instance; it propagates after a single attempt."""
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
        await client.submit_and_poll(
            endpoint="/v1/convert/file/async",
            files={"file": ("x.md", b"x", "text/markdown")},
            data={},
        )

    # Single attempt — no failover on a non-retryable status.
    assert seen == ["e"]


@pytest.mark.asyncio
async def test_retry_429_fails_over():
    """A 429 (transient overload) is retried elsewhere — exercises the
    status-set membership branch of retryability, distinct from 5xx."""
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

    result = await client.submit_and_poll(
        endpoint="/v1/convert/file/async",
        files={"file": ("x.md", b"x", "text/markdown")},
        data={},
    )
    assert result == {"ok": True}
    assert seen[0] == "busy-g"
    assert "free-g" in seen


@pytest.mark.asyncio
async def test_task_failure_is_not_retried():
    """A docling-serve task 'failure' status raises ValueError and is NOT
    retried on another instance — it's a document problem, not an instance one,
    so it propagates after a single attempt."""
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
        await client.submit_and_poll(
            endpoint="/v1/convert/file/async",
            files={"file": ("x.md", b"x", "text/markdown")},
            data={},
        )

    # Only the first instance was attempted (submit + poll) — no failover.
    assert set(seen) == {"h"}


def test_pick_url_skips_a_single_excluded_instance():
    """On retry, _pick_url prefers an instance not in the exclude set."""
    urls = ["http://p1:5001", "http://p2:5001", "http://p3:5001"]
    client = DoclingServeClient(base_urls=urls)
    # Cursor starts at p1; excluding it advances to p2.
    assert client._pick_url(exclude=frozenset({"http://p1:5001"})) == "http://p2:5001"


def test_pick_url_skips_multiple_excluded_instances():
    """_pick_url keeps advancing past every excluded instance."""
    urls = ["http://q1:5001", "http://q2:5001", "http://q3:5001"]
    client = DoclingServeClient(base_urls=urls)
    excluded = frozenset({"http://q1:5001", "http://q2:5001"})
    assert client._pick_url(exclude=excluded) == "http://q3:5001"


def test_pick_url_all_excluded_falls_back():
    """When every instance is excluded (e.g. a single-instance fleet that just
    failed), _pick_url returns one anyway so the retry still probes it."""
    client = DoclingServeClient(base_urls=["http://solo:5001"])
    picked = client._pick_url(exclude=frozenset({"http://solo:5001"}))
    assert picked == "http://solo:5001"


async def _convert(client: DoclingServeClient) -> dict:
    return await client.submit_and_poll(
        endpoint="/v1/convert/file/async",
        files={"file": ("x.md", b"x", "text/markdown")},
        data={},
    )


@pytest.mark.asyncio
async def test_breaker_opens_after_repeated_failures_then_skips_instance():
    """Repeated failures on an instance (each failing over to a healthy one)
    trip its breaker; once open, _pick_url skips it entirely."""
    down = {"crash-h"}
    transport, seen = _failover_transport(down, "t", {"ok": True})
    crash = "http://crash-h:5001"
    client = DoclingServeClient(
        base_urls=[crash, "http://live-h:5001"],
        transport=transport,
        breaker_failure_threshold=2,
        retry_base_delay=0.0,
    )

    # Each call fails on crash-h (recorded) then fails over to live-h.
    assert await _convert(client) == {"ok": True}
    assert await _convert(client) == {"ok": True}  # 2nd failure opens crash-h
    assert client._breaker_for(crash).is_open

    seen.clear()
    assert await _convert(client) == {"ok": True}
    assert "crash-h" not in seen  # breaker open → never attempted
    assert set(seen) == {"live-h"}


@pytest.mark.asyncio
async def test_breaker_recovers_after_cooldown():
    """An open breaker auto-probes after its cooldown; once the instance is
    healthy again a successful request closes it."""
    clock = [1000.0]
    down = {"flip-j"}
    transport, seen = _failover_transport(down, "t", {"ok": True})
    flip = "http://flip-j:5001"
    client = DoclingServeClient(
        base_urls=[flip, "http://spare-j:5001"],
        transport=transport,
        breaker_failure_threshold=2,
        breaker_cooldown_s=30.0,
        retry_base_delay=0.0,
        now_fn=lambda: clock[0],
    )

    await _convert(client)
    await _convert(client)  # opens flip-j's breaker
    assert client._breaker_for(flip).is_open

    down.clear()  # instance recovers
    assert client._breaker_for(flip).is_open  # still skipped within cooldown
    clock[0] += 31.0
    assert not client._breaker_for(flip).is_open  # cooldown elapsed → probe ok

    seen.clear()
    await _convert(client)
    await _convert(client)
    assert "flip-j" in seen  # traffic returned
    assert not client._breaker_for(flip).is_open  # success closed it


@pytest.mark.asyncio
async def test_4xx_does_not_trip_breaker():
    """A 4xx is the caller's fault — it must not count against instance health,
    even at a 1-failure threshold."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        return httpx.Response(400, json={"detail": "bad request"})

    bad = "http://bad-k:5001"
    client = DoclingServeClient(
        base_urls=[bad],
        transport=httpx.MockTransport(handler),
        breaker_failure_threshold=1,
        retry_base_delay=0.0,
    )

    for _ in range(3):
        with pytest.raises(httpx.HTTPStatusError):
            await _convert(client)

    assert not client._breaker_for(bad).is_open


@pytest.mark.asyncio
async def test_pick_url_falls_back_when_all_breakers_open():
    """When every instance's breaker is open (here a single-instance fleet that
    just failed), _pick_url returns one anyway so the request can still probe
    it rather than having nothing to pick."""
    lone = "http://lone-m:5001"
    transport, _ = _failover_transport({"lone-m"}, "t", {"ok": True})
    client = DoclingServeClient(
        base_urls=[lone],
        transport=transport,
        breaker_failure_threshold=1,
        max_attempts=1,
        retry_base_delay=0.0,
    )

    with pytest.raises(httpx.ConnectError):
        await _convert(client)
    assert client._breaker_for(lone).is_open

    # All (one) instances open → fall back to it.
    assert client._pick_url() == lone


def test_config_knobs_reach_the_client():
    """Retry/breaker knobs set in DoclingServeConfig must reach the client via
    get_converter / get_chunker (previously hardcoded constructor defaults)."""
    from haiku.rag.chunkers.docling_serve import DoclingServeChunker
    from haiku.rag.config import AppConfig
    from haiku.rag.converters.docling_serve import DoclingServeConverter

    config = AppConfig()
    ds = config.providers.docling_serve
    ds.base_url = "http://cfg-n:5001"
    ds.max_attempts = 7
    ds.retry_base_delay = 1.25
    ds.retry_max_delay = 20.0
    ds.breaker_failure_threshold = 9
    ds.breaker_cooldown_s = 90.0

    for component in (DoclingServeConverter(config), DoclingServeChunker(config)):
        client = component.client
        assert client._max_attempts == 7
        assert client._retry_base_delay == 1.25
        assert client._retry_max_delay == 20.0
        assert client._breaker_failure_threshold == 9
        assert client._breaker_cooldown_s == 90.0


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
