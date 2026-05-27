import hashlib
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from urllib.parse import unquote, urljoin, urlparse
from xml.etree.ElementTree import Element, fromstring

import httpx

from haiku.rag.ingester.sources.base import (
    FetchResult,
    RevisionSnapshot,
    SourceEvent,
    SourceEventKind,
)
from haiku.rag.ingester.sources.filter import (
    FileFilter,
    _default_supported_extensions,
)

# WebDAV PROPFIND uses XML with the DAV: namespace. Element tags arrive as
# Clark-notation strings like "{DAV:}response", so we precompile them once.
_DAV_NS = "DAV:"
_TAG_RESPONSE = f"{{{_DAV_NS}}}response"
_TAG_HREF = f"{{{_DAV_NS}}}href"
_TAG_PROPSTAT = f"{{{_DAV_NS}}}propstat"
_TAG_PROP = f"{{{_DAV_NS}}}prop"
_TAG_STATUS = f"{{{_DAV_NS}}}status"
_TAG_RESOURCETYPE = f"{{{_DAV_NS}}}resourcetype"
_TAG_COLLECTION = f"{{{_DAV_NS}}}collection"
_TAG_GETETAG = f"{{{_DAV_NS}}}getetag"
_TAG_GETLASTMODIFIED = f"{{{_DAV_NS}}}getlastmodified"
_TAG_GETCONTENTTYPE = f"{{{_DAV_NS}}}getcontenttype"


_PROPFIND_BODY = b"""<?xml version="1.0" encoding="utf-8"?>
<propfind xmlns="DAV:">
  <prop>
    <resourcetype/>
    <getetag/>
    <getlastmodified/>
    <getcontenttype/>
  </prop>
</propfind>
"""


class _PropfindEntry:
    """One <response> element decoded into the fields the source actually
    uses. `is_collection` separates folders from files; revision is ETag
    when present, otherwise the Last-Modified header value."""

    __slots__ = ("href", "is_collection", "revision", "content_type")

    def __init__(
        self,
        href: str,
        *,
        is_collection: bool,
        revision: str | None,
        content_type: str | None,
    ) -> None:
        self.href = href
        self.is_collection = is_collection
        self.revision = revision
        self.content_type = content_type


# Matches an ETag value with optional leading whitespace, optional weak
# marker ``W/``, optional surrounding double quotes, and optional trailing
# whitespace. The non-greedy capture pulls out just the opaque inner value.
_ETAG_RE = re.compile(r'^\s*(?:W/)?"?(.*?)"?\s*$')


def _strip_etag(value: str | None) -> str | None:
    """Return the opaque part of an ETag header value (or ``getetag`` element):
    strip surrounding whitespace, the optional ``W/`` weak marker, and
    optional surrounding double quotes. Returns ``None`` for empty input."""
    if value is None:
        return None
    match = _ETAG_RE.match(value)
    cleaned = match.group(1) if match else value.strip()
    return cleaned or None


def _entry_from_response(response: Element) -> _PropfindEntry | None:
    """Decode a single <response>. Returns None if the prop block is missing
    or the entry didn't return HTTP 200 (e.g. 404 for a known-bad path)."""
    href_el = response.find(_TAG_HREF)
    if href_el is None or not href_el.text:
        return None
    href = href_el.text

    is_collection = False
    revision: str | None = None
    last_modified: str | None = None
    content_type: str | None = None
    ok = False

    for propstat in response.findall(_TAG_PROPSTAT):
        status_el = propstat.find(_TAG_STATUS)
        if status_el is None or not status_el.text:
            continue
        if " 200 " not in status_el.text:
            continue
        ok = True
        prop = propstat.find(_TAG_PROP)
        if prop is None:
            continue
        resourcetype = prop.find(_TAG_RESOURCETYPE)
        if resourcetype is not None and resourcetype.find(_TAG_COLLECTION) is not None:
            is_collection = True
        etag_el = prop.find(_TAG_GETETAG)
        if etag_el is not None and etag_el.text:
            revision = _strip_etag(etag_el.text)
        lm_el = prop.find(_TAG_GETLASTMODIFIED)
        if lm_el is not None and lm_el.text:
            last_modified = lm_el.text.strip() or None
        ct_el = prop.find(_TAG_GETCONTENTTYPE)
        if ct_el is not None and ct_el.text:
            content_type = ct_el.text.split(";")[0].strip().lower() or None

    if not ok:
        return None
    # Prefer ETag — stronger validator. Fall back to Last-Modified so revision
    # detection still works against servers that don't return ETags on PROPFIND.
    return _PropfindEntry(
        href=href,
        is_collection=is_collection,
        revision=revision or last_modified,
        content_type=content_type,
    )


def _parse_multistatus(body: bytes) -> list[_PropfindEntry]:
    """Top-level multistatus parser. Raises ValueError on garbage XML so the
    poller's circuit breaker can record a failure."""
    try:
        root = fromstring(body)
    except Exception as exc:  # ParseError + any defensive surprise
        raise ValueError(f"Invalid PROPFIND response XML: {exc}") from exc
    return [
        entry
        for response in root.findall(_TAG_RESPONSE)
        if (entry := _entry_from_response(response)) is not None
    ]


def _resolve_href(href: str, base_url: str) -> str:
    """PROPFIND href values can be either absolute URLs or server-relative
    paths. Resolve to absolute against base_url either way, then URL-decode
    the path so the URI we store matches what a user would type."""
    absolute = urljoin(base_url, href)
    parsed = urlparse(absolute)
    decoded_path = unquote(parsed.path)
    rebuilt = parsed._replace(path=decoded_path)
    return rebuilt.geturl()


class WebDAVSource:
    def __init__(
        self,
        *,
        source_id: str,
        base_url: str,
        username: str | None = None,
        password: str | None = None,
        headers: dict[str, str] | None = None,
        ignore_patterns: list[str] | None = None,
        include_patterns: list[str] | None = None,
        supported_extensions: list[str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.source_id = source_id
        # Trailing slash matters: urljoin treats path-without-slash as a sibling
        # link, so "https://srv/dav" joined with "subdir/x" gives ".../x".
        self.base_url = base_url if base_url.endswith("/") else base_url + "/"
        self.username = username
        self.password = password
        self.headers = dict(headers or {})
        self.supported_extensions = (
            list(supported_extensions)
            if supported_extensions is not None
            else _default_supported_extensions()
        )
        self.filter = FileFilter(
            ignore_patterns=ignore_patterns,
            include_patterns=include_patterns,
            supported_extensions=self.supported_extensions,
        )
        # transport is for testing — production callers leave it None.
        self._transport = transport

    def supports(self, uri: str) -> bool:
        return uri.startswith(self.base_url)

    def _client(self) -> httpx.AsyncClient:
        auth = (
            (self.username, self.password)
            if self.username is not None and self.password is not None
            else None
        )
        return httpx.AsyncClient(
            auth=auth, headers=self.headers, transport=self._transport
        )

    async def head(self, uri: str) -> str | None:
        async with self._client() as http:
            response = await http.request(
                "PROPFIND",
                uri,
                headers={"Depth": "0", "Content-Type": "application/xml"},
                content=_PROPFIND_BODY,
            )
            if response.is_error:
                return None
            entries = _parse_multistatus(response.content)
        if not entries:
            return None
        return entries[0].revision

    async def fetch(self, uri: str) -> FetchResult:
        async with self._client() as http:
            response = await http.get(uri)
            response.raise_for_status()
            body = response.content
        content_type = (
            response.headers.get("content-type", "application/octet-stream")
            .split(";")[0]
            .strip()
            .lower()
        )
        # ETag from the GET response is the freshest revision; fall back to
        # Last-Modified, matching HTTPSource's preference order.
        revision = (
            _strip_etag(response.headers.get("etag"))
            or (response.headers.get("last-modified") or "").strip()
            or None
        )
        extra: dict[str, str] = {}
        last_modified = (response.headers.get("last-modified") or "").strip()
        if last_modified:
            extra["last_modified"] = last_modified
        return FetchResult(
            uri=uri,
            body=body,
            content_type=content_type,
            content_hash=hashlib.md5(body, usedforsecurity=False).hexdigest(),
            revision=revision,
            extra_metadata=extra,
        )

    async def discover(
        self, since: RevisionSnapshot | None = None
    ) -> AsyncIterator[SourceEvent]:
        snapshot: dict[str, str | None] = dict(since) if since else {}
        now = datetime.now(UTC)
        seen: set[str] = set()

        async with self._client() as http:
            response = await http.request(
                "PROPFIND",
                self.base_url,
                headers={"Depth": "infinity", "Content-Type": "application/xml"},
                content=_PROPFIND_BODY,
            )
            response.raise_for_status()
            entries = _parse_multistatus(response.content)

        for entry in entries:
            if entry.is_collection:
                continue
            uri = _resolve_href(entry.href, self.base_url)
            # The base URL itself sometimes appears as a non-collection on
            # broken servers; skip anything that's not strictly under it.
            if uri == self.base_url.rstrip("/") or not uri.startswith(self.base_url):
                continue
            if not self.filter.include_file(uri):
                continue
            seen.add(uri)
            revision = entry.revision
            if revision is not None and snapshot.get(uri) == revision:
                kind = SourceEventKind.UNCHANGED
            else:
                kind = SourceEventKind.UPSERT
            yield SourceEvent(
                source_id=self.source_id,
                uri=uri,
                kind=kind,
                revision=revision,
                discovered_at=now,
            )

        for uri in snapshot:
            if uri in seen:
                continue
            yield SourceEvent(
                source_id=self.source_id,
                uri=uri,
                kind=SourceEventKind.DELETE,
                revision=None,
                discovered_at=now,
            )
