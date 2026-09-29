from dataclasses import dataclass, field
from enum import StrEnum


class SweepStatus(StrEnum):
    OK = "ok"
    UNCHANGED = "unchanged"
    MOVED = "moved"
    ERROR = "error"


@dataclass(frozen=True)
class LastSweep:
    id: int
    table_versions: dict[str, int]
    embedder: str | None


@dataclass(frozen=True)
class CurrentFingerprint:
    id: int
    document_id: str
    change_key: str


@dataclass
class NewFingerprint:
    document_id: str
    uri: str | None
    title: str | None
    change_key: str
    md5: str | None
    content_type: str | None
    source_revision: str | None
    metadata_keys: list[str]
    chunks: int
    embedded_chunks: int
    chars: int
    centroid: bytes | None
    embedder: str | None
    replacement_chars: int
    chunk_stats: dict
    created_at: str | None
    updated_at: str | None
    chunk_texts: list[tuple[str, int]]


@dataclass(frozen=True)
class Refresh:
    """Attributes of a current fingerprint that change without new content."""

    fingerprint_id: int
    uri: str | None
    title: str | None
    source_revision: str | None
    metadata_keys: list[str]
    updated_at: str | None


@dataclass
class DatabaseSweep:
    database: str
    started_at: str
    finished_at: str
    status: SweepStatus
    table_versions: dict[str, int] | None = None
    embedder: str | None = None
    rebaseline: bool = False
    error: str | None = None
    documents: int | None = None
    new: list[NewFingerprint] = field(default_factory=list)
    replaced: list[int] = field(default_factory=list)
    deleted: list[int] = field(default_factory=list)
    refreshed: list[Refresh] = field(default_factory=list)


@dataclass(frozen=True)
class Fingerprint:
    id: int
    database: str
    document_id: str
    uri: str | None
    title: str | None
    change_key: str
    md5: str | None
    source_revision: str | None
    metadata_keys: list[str]
    chunks: int
    embedded_chunks: int
    chars: int
    centroid: bytes | None
    embedder: str | None
    replacement_chars: int
    chunk_stats: dict
    became_current_sweep: int
    ended_sweep: int | None
    deleted: bool
