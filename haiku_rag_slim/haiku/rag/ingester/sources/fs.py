import hashlib
import mimetypes
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

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
        source_id: str | None = None,
    ) -> None:
        # Resolve so symlinks and relative paths collapse to one canonical
        # root. The queue uses source_id as a foreign key — two paths for
        # the same root would mean duplicate sync_state rows.
        self.root = Path(root).resolve()
        self.source_id = source_id or f"fs:{self.root}"
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

    def _resolve_within_root(self, uri: str) -> Path | None:
        """Resolve a URI to a real path guaranteed to live under ``self.root``.

        Returns ``None`` if the URI parses but resolves outside the root
        (path-traversal via ``..``, symlinks pointing elsewhere). Callers
        treat this as "not ours" — `supports()` returns False, `head()`
        returns None, `fetch()` raises ``UnsupportedSourceError``.
        """
        try:
            path = _uri_to_path(uri).resolve(strict=False)
        except (ValueError, OSError):
            return None
        if not path.is_relative_to(self.root):
            return None
        return path

    def supports(self, uri: str) -> bool:
        scheme = urlparse(uri).scheme
        if scheme not in ("", "file"):
            return False
        return self._resolve_within_root(uri) is not None

    async def aclose(self) -> None:  # pragma: no cover - no resources to release
        pass

    async def head(self, uri: str) -> str | None:
        path = self._resolve_within_root(uri)
        if path is None or not path.exists():
            return None
        return str(path.stat().st_mtime_ns)

    async def fetch(self, uri: str) -> FetchResult:
        path = self._resolve_within_root(uri)
        if path is None:
            raise UnsupportedSourceError(f"Path escapes FS root ({self.root}): {uri}")
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
            disk_path=path,
        )

    async def discover(
        self,
        since: RevisionSnapshot | None = None,
        *,
        known_uris: set[str] | None = None,
    ) -> AsyncIterator[SourceEvent]:
        snapshot: dict[str, str] = dict(since) if since else {}
        known = known_uris or set()
        now = datetime.now(UTC)
        seen: set[str] = set()

        # os.walk with followlinks=False so symlinked directories aren't
        # traversed (avoids cycles and unbounded recursion). File symlinks
        # are followed iff their target resolves inside root, matching
        # supports/head/fetch's resolve-then-check behaviour. Out-of-root
        # targets stay skipped so a stray link can't exfiltrate data the
        # operator didn't intend to expose.
        candidates: list[Path] = []
        for dirpath, _dirnames, filenames in os.walk(self.root, followlinks=False):
            for filename in filenames:
                path = Path(dirpath) / filename
                if path.is_symlink():
                    try:
                        resolved = path.resolve(strict=False)
                    except OSError:
                        continue
                    if not resolved.is_relative_to(self.root):
                        continue
                    path = resolved
                candidates.append(path)
        candidates.sort()

        for path in candidates:
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

        # Anything previously known to this source that we didn't encounter
        # during the walk is gone. Emit DELETE so the poller can clean up.
        for uri in known - seen:
            yield SourceEvent(
                source_id=self.source_id,
                uri=uri,
                kind=SourceEventKind.DELETE,
                revision=None,
                discovered_at=now,
            )
