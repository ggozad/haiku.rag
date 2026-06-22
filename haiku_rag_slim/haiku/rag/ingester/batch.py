from datetime import datetime

from pydantic import BaseModel

from haiku.rag.ingester.queue.models import JobOp


class BatchChange(BaseModel):
    op: JobOp
    source_id: str
    uri: str
    revision: str | None = None
    discovered_at: datetime


class BatchSourceSummary(BaseModel):
    source_id: str
    upsert_count: int = 0
    delete_count: int = 0
    unchanged_count: int = 0
    ignored_delete_count: int = 0


class BatchManifest(BaseModel):
    version: int = 1
    generated_at: datetime
    sources: list[BatchSourceSummary] = []
    changes: list[BatchChange] = []


class BatchDryRunReport(BaseModel):
    manifest: BatchManifest
    failed_sweeps: list[str] = []
