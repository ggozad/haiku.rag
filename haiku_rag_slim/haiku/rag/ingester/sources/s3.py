import hashlib
import mimetypes
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from urllib.parse import urlparse

from haiku.rag.client.exceptions import UnsupportedSourceError
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


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise UnsupportedSourceError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _parse_s3_object_uri(uri: str) -> tuple[str, str]:
    """Like _parse_s3_uri but rejects bucket-only URIs (no key)."""
    bucket, key = _parse_s3_uri(uri)
    if not key:
        raise UnsupportedSourceError(f"Invalid S3 URI: {uri}")
    return bucket, key


class S3Source:
    def __init__(
        self,
        *,
        uri: str,
        storage_options: dict[str, str] | None = None,
        ignore_patterns: list[str] | None = None,
        include_patterns: list[str] | None = None,
        supported_extensions: list[str] | None = None,
        source_id: str | None = None,
    ) -> None:
        self.bucket, self.prefix = _parse_s3_uri(uri)
        # uri_prefix is the canonical "everything I own" — used by supports()
        # to scope dispatch, and by discover() to build per-key URIs.
        self.uri_prefix = f"s3://{self.bucket}/{self.prefix}"
        self.source_id = source_id or f"s3:{self.bucket}/{self.prefix}"
        self.storage_options = dict(storage_options or {})
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

    def supports(self, uri: str) -> bool:
        return uri.startswith(self.uri_prefix)

    async def head(self, uri: str) -> str | None:
        import obstore  # type: ignore[import-not-found]

        from haiku.rag.s3 import make_s3_store

        bucket, key = _parse_s3_object_uri(uri)
        store = make_s3_store(bucket, self.storage_options)
        head = await obstore.head_async(store, key)
        return (head.get("e_tag") or "").strip('"').strip() or None

    async def fetch(self, uri: str) -> FetchResult:
        import obstore  # type: ignore[import-not-found]

        from haiku.rag.s3 import make_s3_store

        bucket, key = _parse_s3_object_uri(uri)
        store = make_s3_store(bucket, self.storage_options)

        head = await obstore.head_async(store, key)
        etag = (head.get("e_tag") or "").strip('"').strip() or None

        resp = await obstore.get_async(store, key)
        body = await resp.bytes_async()
        # obstore returns a Bytes view; convert to plain bytes so the rest of
        # the pipeline (hashing, tempfile write) doesn't have to know.
        body = bytes(body)

        content_type, _ = mimetypes.guess_type(key)
        if not content_type:
            content_type = "application/octet-stream"

        return FetchResult(
            uri=uri,
            body=body,
            content_type=content_type,
            content_hash=hashlib.md5(body, usedforsecurity=False).hexdigest(),
            revision=etag,
        )

    async def discover(
        self,
        since: RevisionSnapshot | None = None,
        *,
        known_uris: set[str] | None = None,
    ) -> AsyncIterator[SourceEvent]:
        import obstore  # type: ignore[import-not-found]

        from haiku.rag.s3 import make_s3_store

        snapshot: dict[str, str] = dict(since) if since else {}
        known = known_uris or set()
        now = datetime.now(UTC)
        seen: set[str] = set()
        store = make_s3_store(self.bucket, self.storage_options)

        async for batch in obstore.list(store, prefix=self.prefix or None):
            for obj in batch:
                key = obj["path"]
                if not self.filter.include_file(key):
                    continue
                uri = f"s3://{self.bucket}/{key}"
                seen.add(uri)
                revision = (obj.get("e_tag") or "").strip('"').strip() or None

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

        # URIs previously known to this source that no longer appear under
        # the prefix have been deleted upstream — emit DELETE so the poller
        # cleans up.
        for uri in known - seen:
            yield SourceEvent(
                source_id=self.source_id,
                uri=uri,
                kind=SourceEventKind.DELETE,
                revision=None,
                discovered_at=now,
            )
