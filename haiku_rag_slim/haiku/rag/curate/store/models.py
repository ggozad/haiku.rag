import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum


class SweepStatus(StrEnum):
    OK = "ok"
    UNCHANGED = "unchanged"
    ERROR = "error"


class FlagKind(StrEnum):
    BAD_UPDATE = "bad_update"
    BAD_DOCUMENT = "bad_document"
    WATCHED_CHANGE = "watched_change"
    WATCHED_DELETION = "watched_deletion"
    DUPLICATE_GROUP = "duplicate_group"
    REPEATED_CHUNK = "repeated_chunk"
    MISSING_METADATA = "missing_metadata"


class FlagStatus(StrEnum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    SUPERSEDED = "superseded"
    RESOLVED = "resolved"


@dataclass(frozen=True)
class Detection:
    """A condition a detector found; its identity decides which flag it is."""

    kind: FlagKind
    database: str | None
    subject: str | None = None
    fingerprint_id: int | None = None
    previous_fingerprint_id: int | None = None
    members: list[dict] | None = None
    reasons: list[dict] = field(default_factory=list)

    @property
    def identity(self) -> str:
        signature = (
            sorted((m["database"], m["document_id"]) for m in self.members)
            if self.kind is FlagKind.DUPLICATE_GROUP and self.members
            else None
        )
        key = [self.kind, self.database, self.subject, self.fingerprint_id, signature]
        return hashlib.sha256(json.dumps(key).encode()).hexdigest()


@dataclass(frozen=True)
class Flag:
    id: int
    identity: str
    kind: FlagKind
    database: str | None
    subject: str | None
    document_id: str | None
    fingerprint_id: int | None
    previous_fingerprint_id: int | None
    members: list[dict] | None
    reasons: list[dict]
    status: FlagStatus
    raised_at: str
    status_changed_at: str
    note: str | None


@dataclass(frozen=True)
class Revision:
    """A fingerprint as the detectors read it."""

    id: int
    database: str
    document_id: str
    subject: str
    md5: str | None
    embedder: str | None
    centroid: bytes | None
    chars: int
    chunks: int
    embedded_chunks: int
    replacement_chars: int
    metadata_keys: list[str]
    became_current_at: str
    ended_at: str | None
    deleted: bool


@dataclass(frozen=True)
class RepeatedText:
    text_hash: str
    chars: int
    document_ids: list[str]


@dataclass
class DatabaseView:
    """The store's state of one database, as the detectors need it."""

    database: str
    current: list[Revision]
    previous: dict[int, Revision]
    deletions: list[Revision]
    watched: dict[str, str]
    repeated: list[RepeatedText]


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
    health: list[dict] | None = None


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
    became_current_at: str
    ended_sweep: int | None
    ended_at: str | None
    deleted: bool


@dataclass(frozen=True)
class DatabaseSummary:
    database: str
    documents: int
    open_flags: int
    last_status: SweepStatus | None
    last_sweep_at: str | None
    last_error: str | None
    embedder: str | None
    failed_checks: int
    warned_checks: int


class ChangeKind(StrEnum):
    ADDED = "added"
    UPDATED = "updated"
    DELETED = "deleted"


@dataclass(frozen=True)
class Change:
    kind: ChangeKind
    database: str
    document_id: str
    uri: str | None
    title: str | None
    fingerprint_id: int
    at: str


@dataclass(frozen=True)
class CurrentDocument:
    database: str
    document_id: str
    uri: str | None
    title: str | None
    fingerprint_id: int
    chunks: int
    chars: int
    replacement_chars: int
    chunk_stats: dict
    isolation: float | None
    open_flags: list[FlagKind]


@dataclass(frozen=True)
class Health:
    """Doctor's checks of a database, from the last sweep that ran them."""

    database: str
    sweep_id: int
    checked_at: str
    results: list[dict]


@dataclass(frozen=True)
class Watch:
    database: str
    uri: str
    note: str | None
    added_at: str
