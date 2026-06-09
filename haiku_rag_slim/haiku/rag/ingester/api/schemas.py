from datetime import datetime
from typing import Literal

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
    # Pool-wide breaker: open when N consecutive transient job failures
    # have paused worker claims. status="degraded" while open.
    worker_breaker_open: bool = False
    worker_breaker_consecutive_failures: int = 0


class ProviderEndpoint(BaseModel):
    base_url: str
    reachable: bool
    # Status code from the probe (200 on success, the HTTP code on a non-2xx
    # response, or None when the probe failed before getting a response).
    status_code: int | None = None
    # Error string when reachable is False and we have one (DNS failure,
    # connection refused, timeout). None on success or unknown failure.
    error: str | None = None


class ProvidersResponse(BaseModel):
    """Reachability snapshot of configured external providers. Probed
    on demand when /providers is hit; not cached."""

    docling_serve: list[ProviderEndpoint]


class SourceSummary(BaseModel):
    source_id: str
    type: Literal["fs", "http", "s3", "webdav"]
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


class ConfigResponse(BaseModel):
    """The full effective config (defaults filled in) as YAML, with secrets
    redacted."""

    yaml: str


class StatsResponse(BaseModel):
    """Aggregated counters and per-source breakdowns that drive the dashboard.
    Cheap to compute (all SQL aggregations against the queue file)."""

    throughput: ThroughputStats
    workers: WorkerStats
    oldest_queued_age_s: float | None
    dlq_by_source: dict[str, int]
    queue_depth_by_source: dict[str, int]
