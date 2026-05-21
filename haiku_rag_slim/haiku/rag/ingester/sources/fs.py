import hashlib
import mimetypes
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

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


def _uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        path = parsed.path if parsed.scheme == "file" else uri
        return Path(unquote(path))
    raise ValueError(f"Unsupported URI scheme for FSSource: {uri}")


class FSSource:
    def __init__(
        self,
        *,
        root: Path,
        ignore_patterns: list[str] | None = None,
        include_patterns: list[str] | None = None,
        supported_extensions: list[str] | None = None,
    ) -> None:
        # Resolve so symlinks and relative paths collapse to one canonical
        # source_id. The queue uses source_id as a foreign key — two paths
        # for the same root would mean duplicate sync_state rows.
        self.root = Path(root).resolve()
        self.source_id = f"fs:{self.root}"
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
        scheme = urlparse(uri).scheme
        if scheme not in ("", "file"):
            return False
        try:
            _uri_to_path(uri)
        except ValueError:
            return False
        return True

    async def fetch(self, uri: str) -> FetchResult:
        path = _uri_to_path(uri)
        body = path.read_bytes()
        content_type, _ = mimetypes.guess_type(path.name)
        if content_type is None:
            content_type = "application/octet-stream"
        # mtime_ns rather than st_mtime: nanosecond integer avoids float
        # precision collisions on rapid edits.
        revision = str(path.stat().st_mtime_ns)
        return FetchResult(
            uri=path.as_uri(),
            body=body,
            content_type=content_type,
            content_hash=hashlib.md5(body, usedforsecurity=False).hexdigest(),
            revision=revision,
        )

    async def discover(
        self, since: RevisionSnapshot | None = None
    ) -> AsyncIterator[SourceEvent]:
        snapshot: dict[str, str] = dict(since) if since else {}
        now = datetime.now(UTC)
        seen: set[str] = set()

        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            if not self.filter.include_file(str(path)):
                continue
            uri = path.as_uri()
            revision = str(path.stat().st_mtime_ns)
            seen.add(uri)
            previous = snapshot.get(uri)
            kind = (
                SourceEventKind.UNCHANGED
                if previous == revision
                else SourceEventKind.UPSERT
            )
            yield SourceEvent(
                source_id=self.source_id,
                uri=uri,
                kind=kind,
                revision=revision,
                discovered_at=now,
            )

        # Anything in the snapshot we didn't encounter during the walk is
        # gone from the source. Emit DELETE so the poller can clean up.
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
