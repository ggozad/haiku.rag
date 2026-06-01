from collections.abc import AsyncIterator, Mapping
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

# uri -> revision. Captures what revisions of which URIs we had last seen
# for a given source. Passed to discover() so the source can yield only
# UPSERT / UNCHANGED / DELETE deltas instead of a full re-scan.
RevisionSnapshot = Mapping[str, str]


class SourceEventKind(StrEnum):
    UPSERT = "upsert"
    DELETE = "delete"
    # Emitted for resources whose revision matches the snapshot. The poller
    # uses these to bump last_seen_at without enqueueing work.
    UNCHANGED = "unchanged"


class SourceEvent(BaseModel):
    source_id: str
    uri: str
    kind: SourceEventKind
    # Backend's own change indicator (mtime for FS, ETag for HTTP/S3, etc.).
    # Opaque to consumers — only compared, never parsed. None for DELETE.
    revision: str | None = None
    discovered_at: datetime


class FetchResult(BaseModel):
    uri: str
    body: bytes
    content_type: str
    # MD5 of body. Stored in document metadata as the dedup key — lets the
    # pipeline short-circuit when bytes are identical but the revision differs
    # (e.g. S3 multipart re-upload landing a new ETag on the same content).
    content_hash: str
    revision: str | None = None
    extra_metadata: dict[str, str] = Field(default_factory=dict)
    # When the body is already on disk (FSSource), points at the original
    # file so the pipeline can hand it to docling without copying through a
    # tempfile. Remote sources (HTTP/S3) leave this None — their bytes only
    # exist in memory.
    disk_path: Path | None = None


@runtime_checkable
class Source(Protocol):
    source_id: str

    def supports(self, uri: str) -> bool: ...

    async def head(self, uri: str) -> str | None:
        """Return the current revision for `uri` cheaply, if the backend
        supports it. Returning None means "I have no cheap revision lookup —
        you'll have to fetch()". The pipeline uses this to short-circuit
        re-ingest when the stored revision is unchanged.
        """
        ...

    async def aclose(self) -> None:
        """Release any resources held by the source (e.g. HTTP connection
        pools). Called once during shutdown, after all workers have stopped."""
        ...

    async def fetch(self, uri: str) -> FetchResult: ...

    def discover(
        self,
        since: RevisionSnapshot | None = None,
        *,
        known_uris: set[str] | None = None,
    ) -> AsyncIterator[SourceEvent]: ...
