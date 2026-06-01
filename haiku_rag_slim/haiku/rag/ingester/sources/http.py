import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx

from haiku.rag.ingester.sources.base import (
    FetchResult,
    RevisionSnapshot,
    SourceEvent,
    SourceEventKind,
)


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
    return revision, extra


class HTTPSource:
    def __init__(
        self,
        *,
        source_id: str,
        urls: list[str] | None = None,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.source_id = source_id
        self.urls = list(urls or [])
        self.headers = dict(headers or {})
        self._http = httpx.AsyncClient(headers=self.headers, transport=transport)

    def supports(self, uri: str) -> bool:
        return urlparse(uri).scheme in ("http", "https")

    async def aclose(self) -> None:
        await self._http.aclose()

    async def head(self, uri: str) -> str | None:
        """HEAD probe for the cheap revision short-circuit. Returns the
        ETag (or Last-Modified) so an unchanged remote URL can skip the
        full GET. None on HTTP error so the caller falls back to fetch();
        network errors propagate and the worker's classifier handles them."""
        response = await self._http.head(uri)
        if response.is_error:
            return None
        revision, _ = _extract_revision(response.headers)
        return revision

    async def fetch(self, uri: str) -> FetchResult:
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
        # 410 Gone is the one real source-side deletion signal: the origin
        # explicitly says "permanently gone". 404 and other failures are
        # ambiguous (transient outage, misconfigured URL, auth blip), so we
        # fall back to UPSERT with no revision and let the worker decide
        # via GET.
        snapshot: dict[str, str] = dict(since) if since else {}
        known = known_uris or set()
        now = datetime.now(UTC)
        configured = set(self.urls)

        for url in self.urls:
            try:
                head = await self._http.head(url)
            except Exception:
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
                yield SourceEvent(
                    source_id=self.source_id,
                    uri=url,
                    kind=SourceEventKind.UPSERT,
                    revision=None,
                    discovered_at=now,
                )
                continue

            revision, _ = _extract_revision(head.headers)
            if revision is not None and snapshot.get(url) == revision:
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
        for url in known - configured:
            yield SourceEvent(
                source_id=self.source_id,
                uri=url,
                kind=SourceEventKind.DELETE,
                revision=None,
                discovered_at=now,
            )
