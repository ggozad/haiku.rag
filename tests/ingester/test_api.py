from datetime import UTC, datetime

import httpx
import pytest
from httpx import ASGITransport

from haiku.rag.config import AppConfig
from haiku.rag.ingester.api.server import APIState, build_app
from haiku.rag.ingester.queue.models import JobOp, JobStatus
from haiku.rag.ingester.sources.base import (
    FetchResult,
    SourceEvent,
    SourceEventKind,
)


@pytest.fixture
def state(jobs, sync):
    return APIState(
        config=AppConfig(),
        job_repo=jobs,
        sync_repo=sync,
    )


def _client(state, *, auth_token: str | None = None) -> httpx.AsyncClient:
    app = build_app(state, auth_token=auth_token)
    return httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    )


# --- /health ---


@pytest.mark.asyncio
async def test_health_ok_with_counts(state, jobs):
    # Enqueue j2 first so claim_next reaches it before u1; then transition
    # via claim → mark_dead matches the production path.
    j2 = await jobs.enqueue("src", "u2", JobOp.UPSERT)
    assert j2 is not None
    claimed = await jobs.claim_next("w")
    assert claimed is not None and claimed.id == j2.id
    await jobs.mark_dead(j2.id, "boom", "w")
    await jobs.enqueue("src", "u1", JobOp.UPSERT)

    async with _client(state) as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["queue_counts"] == {"queued": 1, "dead": 1}
    assert body["worker_count"] == 0  # pool not attached in the test state
    assert body["poller_count"] == 0
    assert body["workers_alive"] == 0
    assert body["pollers_alive"] == 0


@pytest.mark.asyncio
async def test_health_degraded_when_worker_died(jobs, sync):
    """If a worker task crashed (live_workers < worker_count), /health must
    flip to status='degraded' so uptime monitors notice."""
    from unittest.mock import MagicMock

    from haiku.rag.config import AppConfig

    config = AppConfig()
    config.ingester.workers.worker_count = 4

    pool = MagicMock()
    pool.live_workers = 3  # one dead

    state = APIState(config=config, job_repo=jobs, sync_repo=sync, pool=pool)
    async with _client(state) as client:
        resp = await client.get("/health")
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["worker_count"] == 4
    assert body["workers_alive"] == 3


@pytest.mark.asyncio
async def test_health_degraded_when_worker_breaker_open(jobs, sync):
    """The pool-wide breaker opens after a streak of transient job failures.
    /health must surface that and flip status='degraded' even when worker
    and poller task counts are healthy."""
    from unittest.mock import MagicMock

    from haiku.rag.config import AppConfig

    config = AppConfig()
    config.ingester.workers.worker_count = 4

    pool = MagicMock()
    pool.live_workers = 4
    pool.breaker_open = True
    pool.breaker_consecutive_failures = 7

    state = APIState(config=config, job_repo=jobs, sync_repo=sync, pool=pool)
    async with _client(state) as client:
        resp = await client.get("/health")
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["worker_breaker_open"] is True
    assert body["worker_breaker_consecutive_failures"] == 7


@pytest.mark.asyncio
async def test_health_skips_auth(state):
    async with _client(state, auth_token="secret") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200


# --- auth ---


@pytest.mark.asyncio
async def test_protected_endpoint_rejects_without_token(state):
    async with _client(state, auth_token="secret") as client:
        resp = await client.get("/jobs")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_protected_endpoint_rejects_wrong_token(state):
    async with _client(state, auth_token="secret") as client:
        resp = await client.get("/jobs", headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_protected_endpoint_accepts_correct_token(state):
    async with _client(state, auth_token="secret") as client:
        resp = await client.get("/jobs", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_no_auth_token_allows_everything(state, jobs):
    async with _client(state, auth_token=None) as client:
        assert (await client.get("/jobs")).status_code == 200
        assert (await client.get("/health")).status_code == 200


@pytest.mark.asyncio
async def test_mutation_endpoints_require_auth(state, jobs):
    """Existing tests prove auth gates GETs; this pins that the *mutation*
    endpoints (retry, cancel, DLQ requeue, source refresh) also require the
    bearer. A missing-auth regression on these would silently let anyone
    cancel jobs or reset the DLQ."""
    j = await jobs.enqueue("src", "u", JobOp.UPSERT)
    assert j is not None
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_dead(j.id, "boom", "w")

    async with _client(state, auth_token="secret") as client:
        # Cancel: blocked without token
        resp = await client.delete(f"/jobs/{j.id}")
        assert resp.status_code == 401
        # Retry: blocked without token
        resp = await client.post(f"/jobs/{j.id}/retry")
        assert resp.status_code == 401
        # DLQ retry: blocked without token
        resp = await client.post(f"/dlq/{j.id}/retry")
        assert resp.status_code == 401
        # Source refresh: blocked without token
        resp = await client.post("/sources/anything/refresh")
        assert resp.status_code == 401

        # With correct token: 200 for retry (job is dead, gets resurrected).
        ok = await client.post(
            f"/jobs/{j.id}/retry", headers={"Authorization": "Bearer secret"}
        )
        assert ok.status_code == 200


# --- /jobs ---


@pytest.mark.asyncio
async def test_list_jobs_returns_recent_first(state, jobs):
    j1 = await jobs.enqueue("a", "u1", JobOp.UPSERT)
    j2 = await jobs.enqueue("b", "u2", JobOp.UPSERT)
    assert j1 is not None and j2 is not None

    async with _client(state) as client:
        resp = await client.get("/jobs")
    assert resp.status_code == 200
    payload = resp.json()
    assert [j["id"] for j in payload] == [j2.id, j1.id]


@pytest.mark.asyncio
async def test_list_jobs_rejects_out_of_range_limit_and_offset(state):
    """Limit is capped at 500 and >=1; offset is >=0. Without bounds a
    malicious or careless ?limit=10000000 would block the event loop on
    serialization."""
    async with _client(state) as client:
        for q in ("/jobs?limit=0", "/jobs?limit=501", "/jobs?offset=-1"):
            resp = await client.get(q)
            assert resp.status_code == 422, q
        for q in ("/dlq?limit=0", "/dlq?limit=501", "/dlq?offset=-1"):
            resp = await client.get(q)
            assert resp.status_code == 422, q


@pytest.mark.asyncio
async def test_list_jobs_filters_by_source_and_status(state, jobs):
    # Enqueue b first so claim_next reaches it before the a row.
    j = await jobs.enqueue("b", "u", JobOp.UPSERT)
    assert j is not None
    claimed = await jobs.claim_next("w")
    assert claimed is not None and claimed.id == j.id
    await jobs.mark_dead(j.id, "err", "w")
    await jobs.enqueue("a", "u", JobOp.UPSERT)

    async with _client(state) as client:
        resp = await client.get("/jobs?source_id=b&status=dead")
    payload = resp.json()
    assert len(payload) == 1
    assert payload[0]["id"] == j.id


@pytest.mark.asyncio
async def test_get_job_returns_record(state, jobs):
    job = await jobs.enqueue("src", "u", JobOp.UPSERT)
    assert job is not None
    async with _client(state) as client:
        resp = await client.get(f"/jobs/{job.id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == job.id


@pytest.mark.asyncio
async def test_get_job_404(state):
    async with _client(state) as client:
        resp = await client.get("/jobs/nope")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_retry_revives_dead_job(state, jobs):
    job = await jobs.enqueue("src", "u", JobOp.UPSERT)
    assert job is not None
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_dead(job.id, "err", "w")

    async with _client(state) as client:
        resp = await client.post(f"/jobs/{job.id}/retry")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == JobStatus.QUEUED.value
    assert body["attempts"] == 0


@pytest.mark.asyncio
async def test_retry_404(state):
    async with _client(state) as client:
        resp = await client.post("/jobs/missing/retry")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cancel_queued_job(state, jobs):
    job = await jobs.enqueue("src", "u", JobOp.UPSERT)
    assert job is not None

    async with _client(state) as client:
        resp = await client.delete(f"/jobs/{job.id}")
    assert resp.status_code == 200
    assert resp.json() == {"job_id": job.id, "cancelled": True}
    assert await jobs.get_job(job.id) is None


@pytest.mark.asyncio
async def test_cancel_succeeded_returns_404(state, jobs):
    job = await jobs.enqueue("src", "u", JobOp.UPSERT)
    assert job is not None
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_succeeded(claimed.id, "w")

    async with _client(state) as client:
        resp = await client.delete(f"/jobs/{job.id}")
    assert resp.status_code == 404


# --- /dlq ---


@pytest.mark.asyncio
async def test_dlq_lists_dead_jobs_only(state, jobs):
    # Enqueue j2 first so claim_next picks it before j1.
    j2 = await jobs.enqueue("src", "u2", JobOp.UPSERT)
    assert j2 is not None
    claimed = await jobs.claim_next("w")
    assert claimed is not None and claimed.id == j2.id
    await jobs.mark_dead(j2.id, "err", "w")
    j1 = await jobs.enqueue("src", "u1", JobOp.UPSERT)
    assert j1 is not None

    async with _client(state) as client:
        resp = await client.get("/dlq")
    payload = resp.json()
    assert len(payload) == 1
    assert payload[0]["id"] == j2.id


@pytest.mark.asyncio
async def test_dlq_retry_resurrects(state, jobs):
    job = await jobs.enqueue("src", "u", JobOp.UPSERT)
    assert job is not None
    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_dead(job.id, "err", "w")

    async with _client(state) as client:
        resp = await client.post(f"/dlq/{job.id}/retry")
    assert resp.status_code == 200
    assert resp.json()["status"] == JobStatus.QUEUED.value


@pytest.mark.asyncio
async def test_dlq_retry_404_on_missing_job(state):
    async with _client(state) as client:
        resp = await client.post("/dlq/missing/retry")
    assert resp.status_code == 404


# --- /sources ---


class _StubSource:
    def __init__(self, source_id, sweeps=()):
        self.source_id = source_id
        self._sweeps = list(sweeps)

    def supports(self, uri):  # pragma: no cover
        return True

    async def head(self, uri):  # pragma: no cover
        return None

    async def fetch(self, uri) -> FetchResult:  # pragma: no cover
        raise NotImplementedError

    async def discover(self, since=None, *, known_uris=None):
        events = self._sweeps.pop(0) if self._sweeps else []
        for event in events:
            yield event


def _build_pollers_state(tmp_path, jobs, sync, source_id: str = "local"):
    """Build an APIState with a real PollerManager containing one FS poller."""
    from haiku.rag.config import FSSourceConfig
    from haiku.rag.ingester.pollers.manager import PollerManager

    cfg = FSSourceConfig(type="fs", id=source_id, root=tmp_path)
    manager = PollerManager(configs=[cfg], job_repo=jobs, sync_repo=sync)
    state = APIState(
        config=AppConfig(),
        job_repo=jobs,
        sync_repo=sync,
        pollers=manager,
    )
    return state, manager


@pytest.mark.asyncio
async def test_sources_empty_when_no_pollers(state):
    async with _client(state) as client:
        resp = await client.get("/sources")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_sources_lists_configured(tmp_path, jobs, sync):
    state, _ = _build_pollers_state(tmp_path, jobs, sync)
    async with _client(state) as client:
        resp = await client.get("/sources")
    payload = resp.json()
    assert len(payload) == 1
    assert payload[0]["source_id"] == "local"
    assert payload[0]["type"] == "fs"
    assert payload[0]["circuit_breaker_open"] is False


@pytest.mark.asyncio
async def test_source_refresh_triggers_sweep(tmp_path, jobs, sync):
    state, manager = _build_pollers_state(tmp_path, jobs, sync)
    # Replace the real source with a stub that records the sweep + emits an event.
    poller = manager.pollers[0]
    poller.source = _StubSource(
        poller.source_id,
        [
            [
                SourceEvent(
                    source_id=poller.source_id,
                    uri="file:///x.md",
                    kind=SourceEventKind.UPSERT,
                    revision="v1",
                    discovered_at=datetime.now(UTC),
                )
            ]
        ],
    )

    async with _client(state) as client:
        resp = await client.post(f"/sources/{poller.source_id}/refresh")
    assert resp.status_code == 200
    body = resp.json()
    assert body["refreshed"] is True
    assert body["source_id"] == poller.source_id

    queued = await jobs.list_jobs(source_id=poller.source_id)
    assert len(queued) == 1


@pytest.mark.asyncio
async def test_source_refresh_unknown_id_404(tmp_path, jobs, sync):
    state, _ = _build_pollers_state(tmp_path, jobs, sync)
    async with _client(state) as client:
        resp = await client.post("/sources/missing/refresh")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_source_refresh_503_when_pollers_absent(state):
    async with _client(state) as client:
        resp = await client.post("/sources/anything/refresh")
    assert resp.status_code == 503


# --- /stats ---


@pytest.mark.asyncio
async def test_stats_returns_shape_on_empty_queue(state):
    async with _client(state) as client:
        resp = await client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["throughput"] == {
        "succeeded_5m": 0,
        "succeeded_30m": 0,
        "succeeded_1h": 0,
    }
    assert body["workers"] == {"busy": 0, "total": 0}
    assert body["oldest_queued_age_s"] is None
    assert body["dlq_by_source"] == {}
    assert body["queue_depth_by_source"] == {}


@pytest.mark.asyncio
async def test_stats_aggregates_real_queue(state, jobs):
    j1 = await jobs.enqueue("s1", "u1", JobOp.UPSERT)
    j2 = await jobs.enqueue("s1", "u2", JobOp.UPSERT)
    j3 = await jobs.enqueue("s2", "u3", JobOp.UPSERT)
    assert j1 and j2 and j3

    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_succeeded(claimed.id, "w")
    dead = await jobs.claim_next("w")
    assert dead is not None
    await jobs.mark_dead(dead.id, "boom", "w")

    async with _client(state) as client:
        resp = await client.get("/stats")
    body = resp.json()
    # One succeeded in the last 5m, 30m, 1h (we just marked it).
    assert body["throughput"]["succeeded_5m"] == 1
    assert body["throughput"]["succeeded_30m"] == 1
    assert body["throughput"]["succeeded_1h"] == 1
    # Last enqueued (s2/u3) remains queued.
    assert body["queue_depth_by_source"] == {"s2": 1}
    # The dead job was claim_next-ed from s1.
    assert body["dlq_by_source"] == {"s1": 1}


@pytest.mark.asyncio
async def test_stats_requires_auth(state):
    async with _client(state, auth_token="secret") as client:
        resp = await client.get("/stats")
    assert resp.status_code == 401


# --- providers ---


@pytest.mark.asyncio
async def test_providers_probes_each_docling_serve_url(state, monkeypatch):
    """Reachable URLs come back with status_code from the probe; unreachable
    URLs come back with reachable=False and the httpx error message."""
    from haiku.rag.ingester.api.routes import providers as providers_mod
    from haiku.rag.ingester.api.schemas import ProviderEndpoint

    async def _fake_probe(client, base_url):
        if "down" in base_url:
            return ProviderEndpoint(
                base_url=base_url,
                reachable=False,
                error="Name or service not known",
            )
        return ProviderEndpoint(base_url=base_url, reachable=True, status_code=200)

    monkeypatch.setattr(providers_mod, "_probe", _fake_probe)
    state.config.processing.converter = "docling-serve"
    state.config.processing.chunker = "docling-serve"
    state.config.providers.docling_serve.base_url = [
        "http://docling-serve-up:5001",
        "http://docling-serve-down:5001",
    ]
    async with _client(state) as client:
        resp = await client.get("/providers")
    assert resp.status_code == 200
    body = resp.json()
    assert [d["base_url"] for d in body["docling_serve"]] == [
        "http://docling-serve-up:5001",
        "http://docling-serve-down:5001",
    ]
    assert body["docling_serve"][0]["reachable"] is True
    assert body["docling_serve"][0]["status_code"] == 200
    assert body["docling_serve"][1]["reachable"] is False
    assert "Name or service not known" in body["docling_serve"][1]["error"]


@pytest.mark.asyncio
async def test_providers_skips_docling_serve_when_not_in_use(state, monkeypatch):
    """With docling-local for both converter and chunker, /providers
    returns an empty docling_serve list — and crucially does not probe."""
    from haiku.rag.ingester.api.routes import providers as providers_mod

    probed: list[str] = []

    async def _spy_probe(client, base_url):  # pragma: no cover - asserted not called
        probed.append(base_url)
        raise AssertionError("probe should not run when docling-serve is not in use")

    monkeypatch.setattr(providers_mod, "_probe", _spy_probe)
    state.config.processing.converter = "docling-local"
    state.config.processing.chunker = "docling-local"
    async with _client(state) as client:
        resp = await client.get("/providers")
    assert resp.status_code == 200
    assert resp.json() == {"docling_serve": []}
    assert probed == []


@pytest.mark.asyncio
async def test_providers_probes_when_only_chunker_uses_docling_serve(
    state, monkeypatch
):
    """A mixed config (local converter + docling-serve chunker, or vice versa)
    still counts as 'in use' and triggers the probe."""
    from haiku.rag.ingester.api.routes import providers as providers_mod
    from haiku.rag.ingester.api.schemas import ProviderEndpoint

    async def _fake_probe(client, base_url):
        return ProviderEndpoint(base_url=base_url, reachable=True, status_code=200)

    monkeypatch.setattr(providers_mod, "_probe", _fake_probe)
    state.config.processing.converter = "docling-local"
    state.config.processing.chunker = "docling-serve"
    async with _client(state) as client:
        resp = await client.get("/providers")
    assert resp.status_code == 200
    assert len(resp.json()["docling_serve"]) == 1


@pytest.mark.asyncio
async def test_providers_probe_with_real_httpx_transport():
    """End-to-end through the actual _probe — MockTransport drives the
    branches: 200, non-2xx, and a transport error all map to the right
    ProviderEndpoint shape."""
    import httpx

    from haiku.rag.ingester.api.routes.providers import _probe

    def _handler(request: httpx.Request) -> httpx.Response:
        path = str(request.url)
        if "ok" in path:
            return httpx.Response(200, json={"status": "ok"})
        if "bad" in path:
            return httpx.Response(503)
        raise httpx.ConnectError("boom")

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        good = await _probe(client, "http://ok:5001")
        assert good.reachable is True
        assert good.status_code == 200
        assert good.error is None

        unhealthy = await _probe(client, "http://bad:5001")
        assert unhealthy.reachable is False
        assert unhealthy.status_code == 503

        dead = await _probe(client, "http://dead:5001")
        assert dead.reachable is False
        assert dead.status_code is None
        assert dead.error is not None and "boom" in dead.error


@pytest.mark.asyncio
async def test_providers_requires_auth(state):
    async with _client(state, auth_token="secret") as client:
        resp = await client.get("/providers")
    assert resp.status_code == 401


# --- dashboard ---


@pytest.mark.asyncio
async def test_dashboard_served_unauthenticated(state):
    """The dashboard is markup-only. The JS it serves attaches the bearer
    token to its own JSON fetches, so the page itself must load without one
    even when auth is enabled."""
    async with _client(state, auth_token="secret") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    assert "haiku-ingester · status" in body
    # The JS calls the JSON endpoints; sanity-check it's wired up.
    assert "/stats" in body
    assert "/sources" in body
    assert "/jobs?status=claimed" in body
    # Op badge helper is present so DELETE rows render distinctly.
    assert "opBadge" in body


def test_api_access_log_gated_on_debug():
    """uvicorn per-request access logging is off at the ingester's normal INFO
    level (the dashboard polls every few seconds) and on at DEBUG."""
    import logging

    from haiku.rag.ingester.app import _api_access_log_enabled

    haiku_logger = logging.getLogger("haiku.rag")
    original = haiku_logger.level
    try:
        haiku_logger.setLevel(logging.INFO)
        assert _api_access_log_enabled() is False
        haiku_logger.setLevel(logging.DEBUG)
        assert _api_access_log_enabled() is True
    finally:
        haiku_logger.setLevel(original)


@pytest.mark.asyncio
async def test_dashboard_wires_database_and_config_panels(state):
    """The on-demand Database and Configuration panels are present and call
    their endpoints lazily (not in the POLL_MS loop)."""
    async with _client(state) as client:
        resp = await client.get("/")
    body = resp.text
    assert 'id="db-panel"' in body
    assert 'id="config-panel"' in body
    assert "/database" in body
    assert "/config" in body
    assert "loadDatabase" in body
    assert "loadConfig" in body


# --- config ---


@pytest.mark.asyncio
async def test_config_returns_full_yaml_with_redacted_secrets(jobs, sync):
    from haiku.rag.config import APIConfig, IngesterConfig

    config = AppConfig(ingester=IngesterConfig(api=APIConfig(auth_token="supersecret")))
    state = APIState(config=config, job_repo=jobs, sync_repo=sync)
    async with _client(state) as client:
        resp = await client.get("/config")
    assert resp.status_code == 200
    text = resp.json()["yaml"]
    # Full effective config: a default section the user never wrote is present.
    assert "embeddings:" in text
    assert "processing:" in text
    # Secret is masked, not echoed.
    assert "supersecret" not in text
    assert "auth_token: '***'" in text


@pytest.mark.asyncio
async def test_config_requires_auth(state):
    async with _client(state, auth_token="secret") as client:
        resp = await client.get("/config")
    assert resp.status_code == 401


# --- database ---


async def _seed_lancedb(path):
    """Create a minimal LanceDB with settings/documents/chunks tables so
    gather_database_info has something real to report."""
    import json

    import lancedb
    from lancedb.pydantic import LanceModel, Vector
    from pydantic import Field

    class SettingsRecord(LanceModel):
        id: str = Field(default="settings")
        settings: str = Field(default="{}")

    class DocumentRecord(LanceModel):
        id: str
        content: str

    class ChunkRecord(LanceModel):
        id: str
        document_id: str
        content: str
        vector: Vector(3)  # type: ignore

    db = await lancedb.connect_async(str(path))
    settings_tbl = await db.create_table("settings", schema=SettingsRecord)
    docs_tbl = await db.create_table("documents", schema=DocumentRecord)
    chunks_tbl = await db.create_table("chunks", schema=ChunkRecord)
    await settings_tbl.add(
        [
            SettingsRecord(
                settings=json.dumps(
                    {
                        "version": "1.2.3",
                        "embeddings": {
                            "model": {
                                "provider": "openai",
                                "name": "text-embedding-3-small",
                                "vector_dim": 3,
                            }
                        },
                    }
                )
            )
        ]
    )
    await docs_tbl.add([DocumentRecord(id="doc-1", content="hello")])
    await chunks_tbl.add(
        [ChunkRecord(id="c1", document_id="doc-1", content="c", vector=[0.1, 0.2, 0.3])]
    )


@pytest.mark.asyncio
async def test_database_reports_info(tmp_path, jobs, sync):
    db_path = tmp_path / "docs.lancedb"
    await _seed_lancedb(db_path)
    state = APIState(config=AppConfig(), job_repo=jobs, sync_repo=sync, db_path=db_path)
    async with _client(state) as client:
        resp = await client.get("/database")
    assert resp.status_code == 200
    body = resp.json()
    assert body["exists"] is True
    assert body["stored_version"] == "1.2.3"
    assert body["embeddings"]["provider"] == "openai"
    assert body["embeddings"]["vector_dim"] == 3
    tables = {t["name"]: t for t in body["tables"]}
    assert tables["documents"]["num_rows"] == 1
    assert tables["chunks"]["num_rows"] == 1
    assert tables["document_items"]["exists"] is False
    assert body["vector_index"]["exists"] is False


@pytest.mark.asyncio
async def test_database_503_when_db_path_unset(state):
    async with _client(state) as client:
        resp = await client.get("/database")
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_database_requires_auth(state):
    async with _client(state, auth_token="secret") as client:
        resp = await client.get("/database")
    assert resp.status_code == 401
