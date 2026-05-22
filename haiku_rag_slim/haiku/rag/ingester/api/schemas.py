from datetime import datetime

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    queue_counts: dict[str, int]
    worker_count: int
    poller_count: int


class SourceSummary(BaseModel):
    source_id: str
    type: str
    last_polled_at: datetime | None
    circuit_breaker_open: bool


class RefreshResponse(BaseModel):
    source_id: str
    refreshed: bool


class CancelResponse(BaseModel):
    job_id: str
    cancelled: bool
