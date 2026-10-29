import hashlib

import httpx
import pytest

from haiku.rag.sources.base import FileTooLargeError, SourceEventKind
from haiku.rag.sources.http import HTTPSource, is_failure_marker


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


def test_configured_source_supports_only_its_urls():
    src = HTTPSource(source_id="docs", urls=["https://example.com/a.pdf"])
    assert src.supports("https://example.com/a.pdf")
    assert not src.supports("https://example.com/b.pdf")
    assert not src.supports("file:///tmp/a.pdf")


def test_configured_source_without_urls_supports_nothing():
    src = HTTPSource(source_id="docs", urls=[])
    assert not src.supports("https://example.com/a.pdf")


def test_source_id_is_user_provided():
    assert HTTPSource(source_id="arxiv").source_id == "arxiv"


async def test_head_returns_etag():
    transport = _transport(
        {
            ("HEAD", "https://example.com/a.md"): httpx.Response(
                200, headers={"etag": '"rev-7"'}
            ),
        }
    )
    src = HTTPSource(source_id="default", transport=transport)
    assert await src.head("https://example.com/a.md") == "rev-7"


async def test_head_falls_back_to_last_modified():
    transport = _transport(
        {
            ("HEAD", "https://example.com/a.md"): httpx.Response(
                200, headers={"last-modified": "Wed, 21 Oct 2025 07:28:00 GMT"}
            ),
        }
    )
    src = HTTPSource(source_id="default", transport=transport)
    assert await src.head("https://example.com/a.md") == "Wed, 21 Oct 2025 07:28:00 GMT"


async def test_head_returns_none_on_error_status():
    transport = _transport(
        {("HEAD", "https://example.com/missing"): httpx.Response(404)}
    )
    src = HTTPSource(source_id="default", transport=transport)
    assert await src.head("https://example.com/missing") is None


async def test_head_returns_none_when_no_revision_headers():
    transport = _transport({("HEAD", "https://example.com/a"): httpx.Response(200)})
    src = HTTPSource(source_id="default", transport=transport)
    assert await src.head("https://example.com/a") is None


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
    assert "etag" not in result.extra_metadata
    assert result.extra_metadata["last_modified"] == "Wed, 21 Oct 2025 07:28:00 GMT"


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


async def test_fetch_raises_on_error_status():
    src = HTTPSource(source_id="default", transport=_transport({}))
    with pytest.raises(httpx.HTTPStatusError):
        await src.fetch("https://example.com/missing")


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


async def test_discover_empty_when_no_urls_configured():
    src = HTTPSource(source_id="default")
    assert [e async for e in src.discover()] == []


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


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503])
async def test_discover_treats_transient_errors_as_upsert(status):
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


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_discover_marks_a_permanent_error_with_its_status(status):
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
    assert events[0].revision == f"http-{status}"
    assert is_failure_marker(events[0].revision)


async def test_discover_reports_an_unchanged_permanent_error_as_unchanged():
    url = "https://example.com/a.md"
    src = HTTPSource(
        source_id="x",
        urls=[url],
        transport=_transport({("HEAD", url): httpx.Response(403)}),
    )
    events = [e async for e in src.discover(since={url: "http-403"}, known_uris={url})]
    assert [(e.kind, e.revision) for e in events] == [
        (SourceEventKind.UNCHANGED, "http-403")
    ]


async def test_discover_deletes_a_url_that_answers_404_twice():
    url = "https://example.com/a.md"
    src = HTTPSource(
        source_id="x",
        urls=[url],
        transport=_transport({("HEAD", url): httpx.Response(404)}),
    )
    events = [e async for e in src.discover(since={url: "http-404"}, known_uris={url})]
    assert [(e.kind, e.revision) for e in events] == [
        (SourceEventKind.DELETE, "http-404")
    ]


async def test_discover_marks_a_404_after_another_failure():
    url = "https://example.com/a.md"
    src = HTTPSource(
        source_id="x",
        urls=[url],
        transport=_transport({("HEAD", url): httpx.Response(404)}),
    )
    events = [e async for e in src.discover(since={url: "http-403"}, known_uris={url})]
    assert [(e.kind, e.revision) for e in events] == [
        (SourceEventKind.UPSERT, "http-404")
    ]


class _TrackedBody(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.read = False

    async def __aiter__(self):
        self.read = True
        yield b"body"


@pytest.mark.parametrize("status", [405, 501])
async def test_discover_probes_with_get_when_head_is_refused(status):
    url = "https://example.com/a.md"
    body = _TrackedBody()
    transport = _transport(
        {
            ("HEAD", url): httpx.Response(status),
            ("GET", url): httpx.Response(200, headers={"etag": '"v1"'}, stream=body),
        }
    )
    src = HTTPSource(source_id="x", urls=[url], transport=transport)
    events = [e async for e in src.discover()]
    assert [(e.kind, e.revision) for e in events] == [(SourceEventKind.UPSERT, "v1")]
    assert not body.read


async def test_discover_marks_the_status_of_a_refused_head_probe():
    url = "https://example.com/a.md"
    transport = _transport(
        {
            ("HEAD", url): httpx.Response(405),
            ("GET", url): httpx.Response(404),
        }
    )
    src = HTTPSource(source_id="x", urls=[url], transport=transport)
    events = [e async for e in src.discover()]
    assert [(e.kind, e.revision) for e in events] == [
        (SourceEventKind.UPSERT, "http-404")
    ]


async def test_probe_revision_matches_the_fetched_revision():
    url = "https://example.com/a.md"
    headers = {"etag": 'W/"v1"', "last-modified": "Wed, 21 Oct 2025 07:28:00 GMT"}
    transport = _transport(
        {
            ("HEAD", url): httpx.Response(405),
            ("GET", url): httpx.Response(200, headers=headers, content=b"x"),
        }
    )
    src = HTTPSource(source_id="x", urls=[url], transport=transport)
    [event] = [e async for e in src.discover()]
    assert event.revision == (await src.fetch(url)).revision


async def test_head_probes_with_get_when_head_is_refused():
    url = "https://example.com/a.md"
    transport = _transport(
        {
            ("HEAD", url): httpx.Response(405),
            ("GET", url): httpx.Response(200, headers={"etag": '"v1"'}),
        }
    )
    src = HTTPSource(source_id="x", transport=transport)
    assert await src.head(url) == "v1"


@pytest.mark.parametrize(
    "headers",
    [
        {"etag": "http-404"},
        {"etag": '"http-404"'},
        {"etag": ' "http-404" '},
        {"etag": 'W/"http-404"'},
        {"last-modified": "http-404"},
        {"last-modified": " http-404 "},
    ],
)
async def test_no_header_value_reads_as_a_failure_marker(headers):
    url = "https://example.com/a.md"
    transport = _transport(
        {
            ("HEAD", url): httpx.Response(200, headers=headers),
            ("GET", url): httpx.Response(200, headers=headers, content=b"x"),
        }
    )
    src = HTTPSource(source_id="x", transport=transport)
    revision = await src.head(url)
    assert revision is not None
    assert not is_failure_marker(revision)
    assert (await src.fetch(url)).revision == revision


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


async def test_discover_emits_delete_for_removed_url():
    """A URI that was previously in config (and therefore in the snapshot)
    but is no longer configured emits DELETE so the poller can clean up
    when delete_orphans=True. Mirrors what FS/S3/WebDAV already do for
    items missing from a listing."""
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
        e
        async for e in src.discover(
            known_uris={"https://example.com/a.md", "https://example.com/gone.md"}
        )
    ]
    by_uri = {e.uri: e for e in events}
    assert by_uri["https://example.com/a.md"].kind is not SourceEventKind.DELETE
    assert by_uri["https://example.com/gone.md"].kind is SourceEventKind.DELETE
    assert by_uri["https://example.com/gone.md"].revision is None


async def test_discover_propagates_non_transport_errors():
    """Programming errors (TypeError, etc.) should propagate instead of
    being silently swallowed as UPSERT events."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise TypeError("unexpected bug")

    src = HTTPSource(
        source_id="x",
        urls=["https://example.com/a.md"],
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(TypeError, match="unexpected bug"):
        async for _ in src.discover():
            pass


@pytest.mark.parametrize(
    "known_uris, kind",
    [
        ({"https://example.com/a.md"}, SourceEventKind.UNCHANGED),
        (None, SourceEventKind.UPSERT),
    ],
)
async def test_discover_without_revision_upserts_only_unknown_urls(known_uris, kind):
    """A server that returns no ETag or Last-Modified should not cause
    re-ingestion every sweep once the URL has been ingested. A new URL still
    UPSERTs on first sight."""
    transport = _transport({("HEAD", "https://example.com/a.md"): httpx.Response(200)})
    src = HTTPSource(
        source_id="x", urls=["https://example.com/a.md"], transport=transport
    )
    events = [e async for e in src.discover(known_uris=known_uris)]
    assert [(e.kind, e.revision) for e in events] == [(kind, None)]


async def test_discover_upserts_a_url_that_recovers_without_validators():
    url = "https://example.com/a.md"
    src = HTTPSource(
        source_id="x",
        urls=[url],
        transport=_transport({("HEAD", url): httpx.Response(200)}),
    )
    events = [e async for e in src.discover(since={url: "http-404"}, known_uris={url})]
    assert [(e.kind, e.revision) for e in events] == [(SourceEventKind.UPSERT, None)]


async def test_fetch_rejects_file_exceeding_max_size():
    transport = _transport(
        {
            ("HEAD", "https://example.com/big.bin"): httpx.Response(
                200, headers={"content-length": "5000"}
            ),
        }
    )
    src = HTTPSource(source_id="default", transport=transport, max_file_size=1000)
    with pytest.raises(FileTooLargeError):
        await src.fetch("https://example.com/big.bin")


async def test_fetch_allows_file_within_max_size():
    body = b"small"
    transport = _transport(
        {
            ("HEAD", "https://example.com/a.md"): httpx.Response(
                200, headers={"content-length": str(len(body))}
            ),
            ("GET", "https://example.com/a.md"): httpx.Response(
                200, content=body, headers={"content-type": "text/markdown"}
            ),
        }
    )
    src = HTTPSource(source_id="default", transport=transport, max_file_size=1000)
    result = await src.fetch("https://example.com/a.md")
    assert result.body == body


async def test_fetch_skips_head_when_no_max_size():
    """When max_file_size is None, no HEAD request is made."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, content=b"ok")
        return httpx.Response(200)

    src = HTTPSource(
        source_id="default",
        transport=httpx.MockTransport(handler),
        max_file_size=None,
    )
    await src.fetch("https://example.com/a.md")
    assert calls == ["GET"]


async def test_aclose_closes_the_http_client():
    src = HTTPSource(
        source_id="urls",
        urls=[],
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
    )
    await src.aclose()
    assert src._http.is_closed
