from datetime import datetime
from enum import StrEnum
from typing import NamedTuple

from pydantic import BaseModel, computed_field


class JobStatus(StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    SUCCEEDED = "succeeded"
    DEAD = "dead"
    REJECTED = "rejected"


class RetrySelection(StrEnum):
    DEAD = "dead"
    REJECTED = "rejected"
    ALL = "all"

    @property
    def statuses(self) -> set[JobStatus]:
        if self is RetrySelection.ALL:
            return {JobStatus.DEAD, JobStatus.REJECTED}
        return {JobStatus(self.value)}


class JobOp(StrEnum):
    UPSERT = "upsert"
    DELETE = "delete"


class Job(BaseModel):
    id: str
    source_id: str
    uri: str
    op: JobOp
    content_hash: str | None = None
    revision: str | None = None
    status: JobStatus
    attempts: int
    max_attempts: int
    last_error: str | None = None
    conversion_stalled: bool = False
    # Free-form per-job payload (e.g. storage_options snapshot). Serialized
    # to a JSON TEXT column.
    extra: dict | None = None
    enqueued_at: datetime
    scheduled_at: datetime
    claimed_at: datetime | None = None
    claimed_by: str | None = None
    last_heartbeat_at: datetime | None = None
    completed_at: datetime | None = None


class SyncStateRow(BaseModel):
    source_id: str
    uri: str
    revision: str | None = None
    content_hash: str | None = None
    last_seen_at: datetime
    last_ingested_at: datetime | None = None


class SyncRow(NamedTuple):
    """A pending sync_state write collected during a sweep and flushed in one
    transaction by SyncStateRepo.batch_upsert()."""

    source_id: str
    uri: str
    revision: str | None
    content_hash: str | None
    ingested: bool


class FileCounts(BaseModel):
    """A source's discovered files, each in one bucket. A file accounted for
    at a revision without a document (a retired URI) is in none.

    A pruned failure that never ingested reads as retired, and a document the
    index lost counts as ingested until the next sweep queues it again.
    """

    pending: int = 0
    failed: int = 0
    ingested: int = 0
    waiting: int = 0

    @computed_field
    @property
    def known(self) -> int:
        return self.pending + self.failed + self.ingested + self.waiting
