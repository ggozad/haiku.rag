from datetime import datetime
from enum import StrEnum
from typing import NamedTuple

from pydantic import BaseModel


class JobStatus(StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    SUCCEEDED = "succeeded"
    DEAD = "dead"


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
    # Free-form per-job payload (e.g. storage_options snapshot). Serialized
    # to a JSON TEXT column.
    extra: dict | None = None
    enqueued_at: datetime
    scheduled_at: datetime
    claimed_at: datetime | None = None
    claimed_by: str | None = None
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
