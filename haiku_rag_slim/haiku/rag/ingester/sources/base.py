from collections.abc import AsyncIterator, Mapping
from datetime import datetime
from enum import StrEnum
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


@runtime_checkable
class Source(Protocol):
    source_id: str

    def supports(self, uri: str) -> bool: ...

    async def fetch(self, uri: str) -> FetchResult: ...

    def discover(
        self, since: RevisionSnapshot | None = None
    ) -> AsyncIterator[SourceEvent]: ...
