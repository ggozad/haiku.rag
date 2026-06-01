import hashlib

import httpx
import pytest

from haiku.rag.ingester.sources.base import FileTooLargeError, SourceEventKind
from haiku.rag.ingester.sources.webdav import WebDAVSource, _strip_etag


def test_strip_etag_strong_quoted():
    assert _strip_etag('"abc123"') == "abc123"


def test_strip_etag_weak_marker():
    assert _strip_etag('W/"abc123"') == "abc123"


def test_strip_etag_unquoted():
    assert _strip_etag("abc123") == "abc123"


def test_strip_etag_whitespace():
    assert _strip_etag('  W/"abc"  ') == "abc"


def test_strip_etag_empty_returns_none():
    assert _strip_etag("") is None
    assert _strip_etag('""') is None
    assert _strip_etag(None) is None


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _multistatus(*entries: dict) -> bytes:
    """Build a <multistatus> response. Each entry is a dict like
    {'href': '/dav/x.md', 'collection': False, 'etag': '"abc"',
     'last_modified': 'Wed, ...', 'content_type': 'text/markdown'}."""
    body = ['<?xml version="1.0" encoding="utf-8"?>', '<d:multistatus xmlns:d="DAV:">']
    for e in entries:
        body.append("  <d:response>")
        body.append(f"    <d:href>{e['href']}</d:href>")
        body.append("    <d:propstat>")
        body.append("      <d:status>HTTP/1.1 200 OK</d:status>")
        body.append("      <d:prop>")
        body.append("        <d:resourcetype>")
        if e.get("collection"):
            body.append("          <d:collection/>")
        body.append("        </d:resourcetype>")
        if "etag" in e:
            body.append(f"        <d:getetag>{e['etag']}</d:getetag>")
        if "last_modified" in e:
            body.append(
                f"        <d:getlastmodified>{e['last_modified']}</d:getlastmodified>"
            )
        if "content_type" in e:
            body.append(
                f"        <d:getcontenttype>{e['content_type']}</d:getcontenttype>"
            )
        body.append("      </d:prop>")
        body.append("    </d:propstat>")
        body.append("  </d:response>")
    body.append("</d:multistatus>")
    return "\n".join(body).encode()


def test_supports_uri_under_base_url():
    src = WebDAVSource(source_id="nc", base_url="https://nc.example.com/dav/")
    assert src.supports("https://nc.example.com/dav/file.md")
    assert src.supports("https://nc.example.com/dav/sub/file.md")
    assert not src.supports("https://nc.example.com/other/file.md")
    assert not src.supports("https://other.example.com/dav/file.md")


def test_base_url_trailing_slash_normalised():
    """base_url without trailing slash mustn't break urljoin during discovery."""
    src = WebDAVSource(source_id="nc", base_url="https://nc.example.com/dav")
    assert src.base_url == "https://nc.example.com/dav/"
    assert src.supports("https://nc.example.com/dav/x.md")


@pytest.mark.asyncio
async def test_fetch_returns_bytes_md5_revision_and_content_type():
    body = b"hello dav"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(
            200,
            content=body,
            headers={
                "content-type": "text/markdown; charset=utf-8",
                "etag": '"rev-1"',
                "last-modified": "Wed, 21 Oct 2025 07:28:00 GMT",
            },
        )

    src = WebDAVSource(
        source_id="nc",
        base_url="https://nc.example.com/dav/",
        transport=_transport(handler),
    )
    result = await src.fetch("https://nc.example.com/dav/a.md")
    assert result.body == body
    assert result.content_hash == hashlib.md5(body, usedforsecurity=False).hexdigest()
    assert result.content_type == "text/markdown"
    assert result.revision == "rev-1"
    assert result.extra_metadata == {"last_modified": "Wed, 21 Oct 2025 07:28:00 GMT"}


@pytest.mark.asyncio
async def test_fetch_falls_back_to_last_modified_when_no_etag():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"x",
            headers={
                "content-type": "text/plain",
                "last-modified": "Wed, 21 Oct 2025 07:28:00 GMT",
            },
        )

    src = WebDAVSource(
        source_id="nc",
        base_url="https://nc.example.com/dav/",
        transport=_transport(handler),
    )
    result = await src.fetch("https://nc.example.com/dav/a.txt")
    assert result.revision == "Wed, 21 Oct 2025 07:28:00 GMT"


@pytest.mark.asyncio
async def test_head_returns_etag_from_propfind_depth_zero():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PROPFIND"
        assert request.headers["Depth"] == "0"
        return httpx.Response(
            207,
            content=_multistatus(
                {"href": "/dav/a.md", "etag": '"rev-9"'},
            ),
        )

    src = WebDAVSource(
        source_id="nc",
        base_url="https://nc.example.com/dav/",
        transport=_transport(handler),
    )
    assert await src.head("https://nc.example.com/dav/a.md") == "rev-9"


@pytest.mark.asyncio
async def test_head_returns_none_on_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    src = WebDAVSource(
        source_id="nc",
        base_url="https://nc.example.com/dav/",
        transport=_transport(handler),
    )
    assert await src.head("https://nc.example.com/dav/missing.md") is None


@pytest.mark.asyncio
async def test_discover_yields_upserts_and_skips_collections_and_unsupported():
    multistatus = _multistatus(
        {"href": "/dav/", "collection": True},
        {"href": "/dav/sub/", "collection": True},
        {"href": "/dav/a.md", "etag": '"rev-a"', "content_type": "text/markdown"},
        {"href": "/dav/sub/b.txt", "etag": '"rev-b"', "content_type": "text/plain"},
        {"href": "/dav/skip.log", "etag": '"rev-log"', "content_type": "text/plain"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PROPFIND"
        assert request.headers["Depth"] == "infinity"
        return httpx.Response(207, content=multistatus)

    src = WebDAVSource(
        source_id="nc",
        base_url="https://nc.example.com/dav/",
        transport=_transport(handler),
    )

    events = [event async for event in src.discover()]
    by_uri = {e.uri: e for e in events}
    assert set(by_uri) == {
        "https://nc.example.com/dav/a.md",
        "https://nc.example.com/dav/sub/b.txt",
    }
    assert all(e.kind is SourceEventKind.UPSERT for e in events)
    assert by_uri["https://nc.example.com/dav/a.md"].revision == "rev-a"


@pytest.mark.asyncio
async def test_discover_emits_unchanged_when_snapshot_matches():
    multistatus = _multistatus(
        {"href": "/dav/", "collection": True},
        {"href": "/dav/a.md", "etag": '"rev-a"', "content_type": "text/markdown"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(207, content=multistatus)

    src = WebDAVSource(
        source_id="nc",
        base_url="https://nc.example.com/dav/",
        transport=_transport(handler),
    )
    snapshot = {"https://nc.example.com/dav/a.md": "rev-a"}
    events = [event async for event in src.discover(since=snapshot)]
    assert [e.kind for e in events] == [SourceEventKind.UNCHANGED]


@pytest.mark.asyncio
async def test_discover_emits_delete_for_files_no_longer_listed():
    multistatus = _multistatus(
        {"href": "/dav/", "collection": True},
        {"href": "/dav/a.md", "etag": '"rev-a"', "content_type": "text/markdown"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(207, content=multistatus)

    src = WebDAVSource(
        source_id="nc",
        base_url="https://nc.example.com/dav/",
        transport=_transport(handler),
    )
    revisions = {"https://nc.example.com/dav/a.md": "rev-a"}
    known = {
        "https://nc.example.com/dav/a.md",
        "https://nc.example.com/dav/gone.md",
    }
    events = [event async for event in src.discover(since=revisions, known_uris=known)]
    kinds = {e.uri: e.kind for e in events}
    assert kinds == {
        "https://nc.example.com/dav/a.md": SourceEventKind.UNCHANGED,
        "https://nc.example.com/dav/gone.md": SourceEventKind.DELETE,
    }


@pytest.mark.asyncio
async def test_discover_uses_last_modified_when_etag_absent():
    multistatus = _multistatus(
        {
            "href": "/dav/a.md",
            "last_modified": "Wed, 21 Oct 2025 07:28:00 GMT",
            "content_type": "text/markdown",
        },
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(207, content=multistatus)

    src = WebDAVSource(
        source_id="nc",
        base_url="https://nc.example.com/dav/",
        transport=_transport(handler),
    )
    events = [event async for event in src.discover()]
    assert events[0].revision == "Wed, 21 Oct 2025 07:28:00 GMT"


@pytest.mark.asyncio
async def test_discover_raises_on_malformed_xml():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(207, content=b"<not></valid xml")

    src = WebDAVSource(
        source_id="nc",
        base_url="https://nc.example.com/dav/",
        transport=_transport(handler),
    )
    with pytest.raises(ValueError, match="Invalid PROPFIND"):
        [event async for event in src.discover()]


@pytest.mark.asyncio
async def test_discover_propagates_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    src = WebDAVSource(
        source_id="nc",
        base_url="https://nc.example.com/dav/",
        transport=_transport(handler),
    )
    with pytest.raises(httpx.HTTPStatusError):
        [event async for event in src.discover()]


@pytest.mark.asyncio
async def test_basic_auth_sent_when_credentials_configured():
    seen_auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization"))
        return httpx.Response(
            207,
            content=_multistatus({"href": "/dav/", "collection": True}),
        )

    src = WebDAVSource(
        source_id="nc",
        base_url="https://nc.example.com/dav/",
        username="alice",
        password="hunter2",
        transport=_transport(handler),
    )
    [event async for event in src.discover()]
    assert seen_auth and seen_auth[0] is not None
    assert seen_auth[0].startswith("Basic ")


@pytest.mark.asyncio
async def test_custom_headers_forwarded():
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(
            207,
            content=_multistatus({"href": "/dav/", "collection": True}),
        )

    src = WebDAVSource(
        source_id="nc",
        base_url="https://nc.example.com/dav/",
        headers={"Authorization": "Bearer tok-123"},
        transport=_transport(handler),
    )
    [event async for event in src.discover()]
    assert seen == ["Bearer tok-123"]


@pytest.mark.asyncio
async def test_discover_resolves_absolute_href():
    """Some servers return absolute URLs in href, others return server paths.
    Both must produce the same stored URI."""
    multistatus = _multistatus(
        {
            "href": "https://nc.example.com/dav/a.md",
            "etag": '"rev"',
            "content_type": "text/markdown",
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(207, content=multistatus)

    src = WebDAVSource(
        source_id="nc",
        base_url="https://nc.example.com/dav/",
        transport=_transport(handler),
    )
    events = [event async for event in src.discover()]
    assert [e.uri for e in events] == ["https://nc.example.com/dav/a.md"]


@pytest.mark.asyncio
async def test_discover_url_decodes_href_path():
    """PROPFIND hrefs are percent-encoded per RFC 3986. We unquote them so
    the stored URI matches what a user types in `add-src`."""
    multistatus = _multistatus(
        {
            "href": "/dav/my%20docs/Hello%20World.md",
            "etag": '"rev"',
            "content_type": "text/markdown",
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(207, content=multistatus)

    src = WebDAVSource(
        source_id="nc",
        base_url="https://nc.example.com/dav/",
        transport=_transport(handler),
    )
    events = [event async for event in src.discover()]
    assert [e.uri for e in events] == [
        "https://nc.example.com/dav/my docs/Hello World.md"
    ]


@pytest.mark.asyncio
async def test_discover_emits_unchanged_for_known_uri_without_revision():
    """A WebDAV entry with no ETag or Last-Modified should not cause
    re-ingestion every sweep once the URI has been ingested."""
    multistatus = _multistatus(
        {"href": "/dav/", "collection": True},
        {"href": "/dav/norev.md", "content_type": "text/markdown"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(207, content=multistatus)

    src = WebDAVSource(
        source_id="nc",
        base_url="https://nc.example.com/dav/",
        transport=_transport(handler),
    )
    events = [
        e
        async for e in src.discover(known_uris={"https://nc.example.com/dav/norev.md"})
    ]
    non_delete = [e for e in events if e.kind is not SourceEventKind.DELETE]
    assert len(non_delete) == 1
    assert non_delete[0].kind is SourceEventKind.UNCHANGED


@pytest.mark.asyncio
async def test_discover_emits_upsert_for_unknown_uri_without_revision():
    """A brand-new WebDAV entry with no revision should UPSERT on first sight."""
    multistatus = _multistatus(
        {"href": "/dav/", "collection": True},
        {"href": "/dav/new.md", "content_type": "text/markdown"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(207, content=multistatus)

    src = WebDAVSource(
        source_id="nc",
        base_url="https://nc.example.com/dav/",
        transport=_transport(handler),
    )
    events = [e async for e in src.discover()]
    non_delete = [e for e in events if e.kind is not SourceEventKind.DELETE]
    assert len(non_delete) == 1
    assert non_delete[0].kind is SourceEventKind.UPSERT


@pytest.mark.asyncio
async def test_fetch_rejects_file_exceeding_max_size():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": "5000"})
        return httpx.Response(200, content=b"big")

    src = WebDAVSource(
        source_id="nc",
        base_url="https://nc.example.com/dav/",
        transport=_transport(handler),
        max_file_size=1000,
    )
    with pytest.raises(FileTooLargeError):
        await src.fetch("https://nc.example.com/dav/big.bin")


@pytest.mark.asyncio
async def test_fetch_allows_file_within_max_size():
    body = b"small"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(body))})
        return httpx.Response(200, content=body, headers={"content-type": "text/plain"})

    src = WebDAVSource(
        source_id="nc",
        base_url="https://nc.example.com/dav/",
        transport=_transport(handler),
        max_file_size=1000,
    )
    result = await src.fetch("https://nc.example.com/dav/a.txt")
    assert result.body == body


@pytest.mark.asyncio
async def test_fetch_skips_head_when_no_max_size():
    """When max_file_size is None, no HEAD request is made."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        return httpx.Response(200, content=b"ok")

    src = WebDAVSource(
        source_id="nc",
        base_url="https://nc.example.com/dav/",
        transport=_transport(handler),
        max_file_size=None,
    )
    await src.fetch("https://nc.example.com/dav/a.txt")
    assert calls == ["GET"]
