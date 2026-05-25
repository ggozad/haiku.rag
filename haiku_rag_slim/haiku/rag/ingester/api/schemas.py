from datetime import datetime

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    queue_counts: dict[str, int]
    worker_count: int
    poller_count: int
    # Live counters — non-zero shortfalls vs the configured count signal a
    # crashed task. status="degraded" when either shortfall is non-zero so
    # uptime monitors can alert without needing to do the math themselves.
    workers_alive: int
    pollers_alive: int


class SourceSummary(BaseModel):
    source_id: str
    type: str
    last_polled_at: datetime | None
    circuit_breaker_open: bool
    # Reason the most recent sweep attempt was skipped (e.g. "pending_work"),
    # or None when the most recent attempt actually polled. Lets operators
    # see at a glance why a source isn't picking up new work.
    last_skip_reason: str | None = None


class RefreshResponse(BaseModel):
    source_id: str
    refreshed: bool


class CancelResponse(BaseModel):
    job_id: str
    cancelled: bool


class ThroughputStats(BaseModel):
    succeeded_5m: int
    succeeded_30m: int
    succeeded_1h: int


class WorkerStats(BaseModel):
    busy: int
    total: int


class StatsResponse(BaseModel):
    """Aggregated counters and per-source breakdowns that drive the dashboard.
    Cheap to compute (all SQL aggregations against the queue file)."""

    throughput: ThroughputStats
    workers: WorkerStats
    oldest_queued_age_s: float | None
    dlq_by_source: dict[str, int]
    queue_depth_by_source: dict[str, int]
