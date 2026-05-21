import hashlib

import httpx
import pytest

from haiku.rag.ingester.sources.base import SourceEventKind
from haiku.rag.ingester.sources.http import HTTPSource


def _transport(routes: dict[tuple[str, str], httpx.Response]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, str(request.url))
        if key not in routes:
            return httpx.Response(404)
        return routes[key]

    return httpx.MockTransport(handler)


def test_supports_http_and_https():
    src = HTTPSource(source_id="default")
    assert src.supports("http://example.com/a.pdf")
    assert src.supports("https://example.com/a.pdf")
    assert not src.supports("file:///tmp/a.pdf")
    assert not src.supports("s3://bucket/a.pdf")


def test_source_id_is_user_provided():
    assert HTTPSource(source_id="arxiv").source_id == "arxiv"


@pytest.mark.asyncio
async def test_head_returns_none():
    # HTTP has no cheap revision lookup in v1 — the pipeline always GETs.
    src = HTTPSource(source_id="default")
    assert await src.head("https://example.com/a.md") is None


@pytest.mark.asyncio
async def test_fetch_returns_bytes_and_md5_and_etag():
    body = b"hello world"
    transport = _transport(
        {
            ("GET", "https://example.com/a.md"): httpx.Response(
                200,
                content=body,
                headers={
                    "content-type": "text/markdown",
                    "etag": '"abc123"',
                    "last-modified": "Wed, 21 Oct 2025 07:28:00 GMT",
                },
            ),
        }
    )
    src = HTTPSource(source_id="default", transport=transport)
    result = await src.fetch("https://example.com/a.md")
    assert result.uri == "https://example.com/a.md"
    assert result.body == body
    assert result.content_hash == hashlib.md5(body, usedforsecurity=False).hexdigest()
    assert result.content_type == "text/markdown"
    # etag preferred over last-modified, surrounding quotes stripped
    assert result.revision == "abc123"
    assert result.extra_metadata["etag"] == "abc123"
    assert result.extra_metadata["last_modified"] == "Wed, 21 Oct 2025 07:28:00 GMT"


@pytest.mark.asyncio
async def test_fetch_falls_back_to_last_modified_when_no_etag():
    transport = _transport(
        {
            ("GET", "https://example.com/a"): httpx.Response(
                200,
                content=b"x",
                headers={
                    "content-type": "application/pdf",
                    "last-modified": "Wed, 21 Oct 2025 07:28:00 GMT",
                },
            ),
        }
    )
    src = HTTPSource(source_id="default", transport=transport)
    result = await src.fetch("https://example.com/a")
    assert result.revision == "Wed, 21 Oct 2025 07:28:00 GMT"


@pytest.mark.asyncio
async def test_fetch_no_revision_when_neither_header_present():
    transport = _transport(
        {
            ("GET", "https://example.com/a"): httpx.Response(
                200, content=b"x", headers={"content-type": "text/plain"}
            ),
        }
    )
    src = HTTPSource(source_id="default", transport=transport)
    result = await src.fetch("https://example.com/a")
    assert result.revision is None


@pytest.mark.asyncio
async def test_fetch_strips_content_type_parameters():
    transport = _transport(
        {
            ("GET", "https://example.com/a"): httpx.Response(
                200,
                content=b"x",
                headers={"content-type": "text/html; charset=utf-8"},
            ),
        }
    )
    src = HTTPSource(source_id="default", transport=transport)
    result = await src.fetch("https://example.com/a")
    assert result.content_type == "text/html"


@pytest.mark.asyncio
async def test_fetch_raises_on_error_status():
    src = HTTPSource(source_id="default", transport=_transport({}))
    with pytest.raises(httpx.HTTPStatusError):
        await src.fetch("https://example.com/missing")


@pytest.mark.asyncio
async def test_fetch_sends_configured_headers():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, content=b"x", headers={"content-type": "text/plain"})

    src = HTTPSource(
        source_id="default",
        headers={"Authorization": "Bearer abc"},
        transport=httpx.MockTransport(handler),
    )
    await src.fetch("https://example.com/a")
    assert seen.get("authorization") == "Bearer abc"


@pytest.mark.asyncio
async def test_discover_empty_when_no_urls_configured():
    src = HTTPSource(source_id="default")
    assert [e async for e in src.discover()] == []


@pytest.mark.asyncio
async def test_discover_yields_upsert_for_each_configured_url():
    transport = _transport(
        {
            ("HEAD", "https://example.com/a.md"): httpx.Response(
                200, headers={"etag": '"abc"'}
            ),
            ("HEAD", "https://example.com/b.md"): httpx.Response(
                200, headers={"etag": '"def"'}
            ),
        }
    )
    src = HTTPSource(
        source_id="x",
        urls=["https://example.com/a.md", "https://example.com/b.md"],
        transport=transport,
    )
    events = [e async for e in src.discover()]
    assert {e.uri for e in events} == {
        "https://example.com/a.md",
        "https://example.com/b.md",
    }
    assert all(e.kind is SourceEventKind.UPSERT for e in events)
    assert {e.revision for e in events} == {"abc", "def"}


@pytest.mark.asyncio
async def test_discover_unchanged_against_matching_snapshot():
    transport = _transport(
        {
            ("HEAD", "https://example.com/a.md"): httpx.Response(
                200, headers={"etag": '"abc"'}
            ),
        }
    )
    src = HTTPSource(
        source_id="x", urls=["https://example.com/a.md"], transport=transport
    )
    events = [e async for e in src.discover(since={"https://example.com/a.md": "abc"})]
    assert len(events) == 1
    assert events[0].kind is SourceEventKind.UNCHANGED


@pytest.mark.asyncio
async def test_discover_emits_delete_on_410_gone():
    transport = _transport(
        {
            ("HEAD", "https://example.com/retired.md"): httpx.Response(410),
        }
    )
    src = HTTPSource(
        source_id="x",
        urls=["https://example.com/retired.md"],
        transport=transport,
    )
    events = [e async for e in src.discover()]
    assert len(events) == 1
    assert events[0].kind is SourceEventKind.DELETE
    assert events[0].uri == "https://example.com/retired.md"
    assert events[0].revision is None


@pytest.mark.parametrize("status", [404, 401, 403, 405, 500, 502, 503])
@pytest.mark.asyncio
async def test_discover_treats_non_410_errors_as_upsert(status):
    transport = _transport(
        {
            ("HEAD", "https://example.com/a.md"): httpx.Response(status),
        }
    )
    src = HTTPSource(
        source_id="x", urls=["https://example.com/a.md"], transport=transport
    )
    events = [e async for e in src.discover()]
    assert len(events) == 1
    assert events[0].kind is SourceEventKind.UPSERT
    assert events[0].revision is None


@pytest.mark.asyncio
async def test_discover_treats_network_errors_as_upsert():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    src = HTTPSource(
        source_id="x",
        urls=["https://example.com/a.md"],
        transport=httpx.MockTransport(handler),
    )
    events = [e async for e in src.discover()]
    assert len(events) == 1
    assert events[0].kind is SourceEventKind.UPSERT
    assert events[0].revision is None


@pytest.mark.asyncio
async def test_discover_does_not_emit_delete_for_unconfigured_uri():
    """Config drift is the poller's job, not HTTPSource's. A URI that was in
    the snapshot but is no longer configured must not appear as a DELETE."""
    transport = _transport(
        {
            ("HEAD", "https://example.com/a.md"): httpx.Response(
                200, headers={"etag": '"abc"'}
            ),
        }
    )
    src = HTTPSource(
        source_id="x", urls=["https://example.com/a.md"], transport=transport
    )
    events = [
        e async for e in src.discover(since={"https://example.com/gone.md": "old"})
    ]
    assert {e.uri for e in events} == {"https://example.com/a.md"}
    assert all(e.kind is not SourceEventKind.DELETE for e in events)
