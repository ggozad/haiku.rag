import hashlib
import logging
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx

from haiku.rag.sources.base import (
    FetchResult,
    RevisionSnapshot,
    SourceEvent,
    SourceEventKind,
    check_file_size,
)
from haiku.rag.utils.http import is_transient_status

logger = logging.getLogger(__name__)

_FAILURE_MARKER = re.compile(r"http-\d{3}")


def failure_marker(status: int) -> str:
    """The revision recorded for a URL that answers a permanent error status."""
    return f"http-{status}"


def is_failure_marker(revision: str | None) -> bool:
    return revision is not None and _FAILURE_MARKER.fullmatch(revision) is not None


def _extract_revision(headers: httpx.Headers) -> tuple[str | None, dict[str, str]]:
    """Return (canonical_revision, extras). ETag is the stronger validator
    and is the canonical revision when present; Last-Modified backs it up.
    Last-Modified always goes into extras as separate per-source provenance
    (some pipelines want both signals)."""
    extra: dict[str, str] = {}
    etag = (headers.get("etag") or "").strip('"').strip()
    last_modified = (headers.get("last-modified") or "").strip()
    if last_modified:
        extra["last_modified"] = last_modified
    revision = etag or last_modified or None
    if is_failure_marker(revision):
        revision = f'"{revision}"'
    return revision, extra


class HTTPSource:
    def __init__(
        self,
        *,
        source_id: str,
        urls: list[str] | None = None,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        max_file_size: int | None = None,
    ) -> None:
        self.source_id = source_id
        self.urls = list(urls or [])
        self._configured = frozenset(self.urls) if urls is not None else None
        self.headers = dict(headers or {})
        self._http = httpx.AsyncClient(headers=self.headers, transport=transport)
        self._max_file_size = max_file_size

    def supports(self, uri: str) -> bool:
        """A source given ``urls``, empty included, supports only those. One
        built without them supports any http(s) URI."""
        if self._configured is not None:
            return uri in self._configured
        return urlparse(uri).scheme in ("http", "https")

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _probe(self, uri: str) -> httpx.Response:
        """HEAD, or the headers of an unread GET when the server refuses HEAD."""
        response = await self._http.head(uri)
        if response.status_code not in (405, 501):
            return response
        async with self._http.stream("GET", uri) as response:
            return response

    async def head(self, uri: str) -> str | None:
        """HEAD probe for the cheap revision short-circuit. Returns the
        ETag (or Last-Modified) so an unchanged remote URL can skip the
        full GET. None on HTTP error so the caller falls back to fetch();
        network errors propagate and the worker's classifier handles them."""
        response = await self._probe(uri)
        if response.is_error:
            return None
        revision, _ = _extract_revision(response.headers)
        return revision

    async def fetch(self, uri: str) -> FetchResult:
        if self._max_file_size is not None:
            head = await self._http.head(uri)
            content_length = head.headers.get("content-length")
            if content_length is not None:
                check_file_size(int(content_length), self._max_file_size, uri)
        response = await self._http.get(uri)
        response.raise_for_status()
        body = response.content
        content_type = (
            response.headers.get("content-type", "application/octet-stream")
            .split(";")[0]
            .strip()
            .lower()
        )
        revision, extra = _extract_revision(response.headers)
        return FetchResult(
            uri=uri,
            body=body,
            content_type=content_type,
            content_hash=hashlib.md5(body, usedforsecurity=False).hexdigest(),
            revision=revision,
            extra_metadata=extra,
        )

    async def discover(
        self,
        since: RevisionSnapshot | None = None,
        *,
        known_uris: set[str] | None = None,
    ) -> AsyncIterator[SourceEvent]:
        # HTTP has no listing concept — discover() only reports on what is
        # currently configured in self.urls. URLs that were previously
        # known to this source (sync_state) but aren't in the current
        # config emit DELETE so the poller can clean up alongside the
        # in-source 410 signal.
        #
        # 410 Gone deletes at once. Another permanent error status is
        # recorded as a failure marker revision, so the worker's GET fails
        # once and later sweeps see the URL unchanged. A 404 against a stored
        # http-404 marker deletes. Transient statuses UPSERT with no revision.
        snapshot: dict[str, str] = dict(since) if since else {}
        known = known_uris or set()
        now = datetime.now(UTC)

        for url in self.urls:
            try:
                head = await self._probe(url)
            except httpx.TransportError as exc:
                logger.debug(
                    "HEAD %s failed (%s); emitting UPSERT with no revision",
                    url,
                    exc,
                )
                yield SourceEvent(
                    source_id=self.source_id,
                    uri=url,
                    kind=SourceEventKind.UPSERT,
                    revision=None,
                    discovered_at=now,
                )
                continue

            if head.status_code == 410:
                yield SourceEvent(
                    source_id=self.source_id,
                    uri=url,
                    kind=SourceEventKind.DELETE,
                    revision=None,
                    discovered_at=now,
                )
                continue

            if head.is_error:
                marker = (
                    None
                    if is_transient_status(head.status_code)
                    else failure_marker(head.status_code)
                )
                stored = snapshot.get(url)
                if marker is None or stored != marker:
                    kind = SourceEventKind.UPSERT
                elif head.status_code == 404:
                    kind = SourceEventKind.DELETE
                else:
                    kind = SourceEventKind.UNCHANGED
                yield SourceEvent(
                    source_id=self.source_id,
                    uri=url,
                    kind=kind,
                    revision=marker,
                    discovered_at=now,
                )
                continue

            revision, _ = _extract_revision(head.headers)
            if revision is not None and snapshot.get(url) == revision:
                kind = SourceEventKind.UNCHANGED
            elif (
                revision is None
                and url in known
                and not is_failure_marker(snapshot.get(url))
            ):
                # Server provides no revision header (no ETag, no
                # Last-Modified). We can't detect changes, but the URL was
                # already ingested — skip rather than re-fetch every sweep. A
                # real change is only picked up if the server starts returning
                # revision headers or the operator forces a re-ingest.
                kind = SourceEventKind.UNCHANGED
            else:
                kind = SourceEventKind.UPSERT

            yield SourceEvent(
                source_id=self.source_id,
                uri=url,
                kind=kind,
                revision=revision,
                discovered_at=now,
            )

        # Anything previously known to this source that's no longer in
        # config emits DELETE so delete_orphans can clean up. Without this,
        # removing a URL from config leaves the document and sync_state
        # indefinitely.
        for url in known - set(self.urls):
            yield SourceEvent(
                source_id=self.source_id,
                uri=url,
                kind=SourceEventKind.DELETE,
                revision=None,
                discovered_at=now,
            )
