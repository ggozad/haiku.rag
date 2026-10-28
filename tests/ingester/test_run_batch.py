"""IngesterApp lifecycle: run_batch (one sweep -> drain -> exit) and serve
(pollers + workers + API until shutdown). The document store (HaikuRAG) is
patched out — the behavior under test is the orchestration and orphan
pruning, not embedding."""

import asyncio
import logging
import re
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.client.exceptions import UnsupportedSourceError
from haiku.rag.config import (
    APIConfig,
    AppConfig,
    FSSourceConfig,
    IngestedDatabase,
    IngesterConfig,
    LanceDBConfig,
    QueueConfig,
    RetryPolicyConfig,
    WorkerConfig,
)
from haiku.rag.ingester.app import IngesterApp
from haiku.rag.ingester.batch import BatchChange, BatchManifest
from haiku.rag.ingester.exceptions import EmptyListingError, QueueError
from haiku.rag.ingester.pollers.manager import PollerManager
from haiku.rag.ingester.queue.migrations import open_queue
from haiku.rag.ingester.queue.models import JobOp, JobStatus
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo
from haiku.rag.ingester.workers.pool import WorkerPool
from haiku.rag.store.models.document import Document
from haiku.rag.uri import uri_to_path
from tests.conftest import capture_logs

_POLLER_LOGGER = logging.getLogger("haiku.rag.ingester.pollers.base")


def _config(tmp_path, root=None, **worker_kwargs) -> AppConfig:
    workers = WorkerConfig(
        worker_count=2,
        poll_idle_interval_s=0.05,
        retry=RetryPolicyConfig(max_attempts=1),
        **worker_kwargs,
    )
    return AppConfig(
        lancedb=LanceDBConfig(databases={"haiku.rag": str(tmp_path / "db.lancedb")}),
        ingester=IngesterConfig(
            queue=QueueConfig(path=tmp_path / "queue.db"),
            databases={
                "haiku.rag": IngestedDatabase(
                    sources=[
                        FSSourceConfig(
                            type="fs",
                            id="local",
                            root=root or tmp_path,
                            poll_interval_s=3600.0,
                        )
                    ]
                )
            },
            workers=workers,
            api=APIConfig(enabled=False),
        ),
    )


def _mock_client() -> AsyncMock:
    """A HaikuRAG mock whose upserts return a fresh Document (with an md5 so
    the worker records sync_state) and whose lookups return a Document with a
    deterministic id so the DELETE path resolves.

    It keeps the documents it was told to write, so `list_documents` answers
    what the store holds: startup reconciliation reads it, and a double that
    always answered "nothing" would invalidate every revision.
    """
    client = AsyncMock(spec=HaikuRAG)
    counter = {"n": 0}
    stored: dict[str, Document] = {}

    async def _create(uri, *_, source_id=None, **__):
        counter["n"] += 1
        # Mirror the real client: persist the FS revision (mtime_ns) so the
        # poller's change-detection skips unchanged files on the next sweep.
        path = uri_to_path(uri)
        metadata = {
            "content_type": "text/markdown",
            "md5": f"md5-{counter['n']}",
            "source_revision": str(path.stat().st_mtime_ns),
        }
        if source_id is not None:
            metadata["source_id"] = source_id
        document = Document(
            id=f"doc-{counter['n']}", content="x", uri=uri, metadata=metadata
        )
        stored[uri] = document
        return document

    async def _get_by_uri(uri):
        return stored.get(uri) or Document(
            id=f"doc-for-{uri}", content="x", uri=uri, metadata={}
        )

    async def _delete(document_id, *, source_id=None):
        for uri, document in list(stored.items()):
            if document.id == document_id:
                del stored[uri]
        return True

    async def _list_documents(*_, **__):
        return list(stored.values())

    client._ingest_observed.side_effect = _create
    client.get_document_by_uri.side_effect = _get_by_uri
    client.delete_document.side_effect = _delete
    client.list_documents.side_effect = _list_documents
    return client


def _manifest(*changes: BatchChange) -> BatchManifest:
    return BatchManifest(generated_at=datetime.now(UTC), changes=list(changes))


async def test_run_batch_drains_upserts(tmp_path, use_client):
    (tmp_path / "a.md").write_text("hello")
    (tmp_path / "b.md").write_text("world")

    client = _mock_client()
    use_client(client)

    report = await IngesterApp(config=_config(tmp_path)).run_batch()

    assert report.succeeded == 2
    assert report.dead == 0
    assert client._ingest_observed.await_count == 2
    ingested = {call.args[0] for call in client._ingest_observed.await_args_list}
    assert ingested == {
        (tmp_path / "a.md").as_uri(),
        (tmp_path / "b.md").as_uri(),
    }


async def test_run_batch_reports_progress(tmp_path, use_client):
    (tmp_path / "a.md").write_text("hello")
    (tmp_path / "b.md").write_text("world")

    client = _mock_client()
    use_client(client)
    progress = []

    report = await IngesterApp(config=_config(tmp_path)).run_batch(
        progress_callback=progress.append
    )

    assert report.succeeded == 2
    assert report.dead == 0
    assert progress
    assert progress[-1].total == 2
    assert progress[-1].completed == 2
    assert progress[-1].succeeded == 2
    assert progress[-1].dead == 0
    assert progress[-1].queued == 0
    assert progress[-1].claimed == 0


async def test_run_batch_prunes_orphans(tmp_path, use_client):
    (tmp_path / "a.md").write_text("hello")
    (tmp_path / "b.md").write_text("world")

    client = _mock_client()
    use_client(client)
    config = _config(tmp_path)
    tmp_path / "db.lancedb"

    # First batch ingests both files and records sync_state for each.
    first = await IngesterApp(config=config).run_batch()
    assert first.succeeded == 2
    client.delete_document.assert_not_awaited()

    # b.md disappears from the source tree. The next sweep sees it in
    # sync_state but not on disk -> enqueues a DELETE for it.
    (tmp_path / "b.md").unlink()

    second = await IngesterApp(config=config).run_batch()

    # a.md is unchanged (same mtime) so it's not re-ingested; only the orphan
    # delete runs.
    assert client._ingest_observed.await_count == 2
    client.delete_document.assert_awaited_once()
    assert second.succeeded == 1
    assert second.dead == 0


async def test_run_batch_keeps_documents_when_the_root_is_gone(tmp_path, use_client):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("hello")
    (docs / "b.md").write_text("world")
    client = _mock_client()
    use_client(client)
    config = _config(tmp_path, root=docs)
    tmp_path / "db.lancedb"

    first = await IngesterApp(config=config).run_batch()
    assert first.succeeded == 2

    docs.rename(tmp_path / "unmounted")
    with capture_logs(_POLLER_LOGGER, logging.ERROR) as records:
        second = await IngesterApp(config=config).run_batch()

    assert second.failed_sweeps == ["local"]
    client.delete_document.assert_not_awaited()
    assert any(
        isinstance(r.exc_info[1], FileNotFoundError) for r in records if r.exc_info
    )


async def test_run_batch_keeps_recovered_documents_when_the_source_empties(
    tmp_path, use_client
):
    """Documents recovered into a lost queue by reconcile still block an empty
    listing from deleting them."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("hello")
    (docs / "b.md").write_text("world")
    client = _mock_client()
    use_client(client)
    config = _config(tmp_path, root=docs)
    tmp_path / "db.lancedb"

    first = await IngesterApp(config=config).run_batch()
    assert first.succeeded == 2

    for queue_file in tmp_path.glob("queue.db*"):
        queue_file.unlink()
    for document in docs.iterdir():
        document.unlink()
    with capture_logs(_POLLER_LOGGER, logging.ERROR) as records:
        second = await IngesterApp(config=config).run_batch()

    assert second.failed_sweeps == ["local"]
    client.delete_document.assert_not_awaited()
    assert any(
        isinstance(r.exc_info[1], EmptyListingError) for r in records if r.exc_info
    )


async def test_run_batch_retries_with_the_sources_backoff(tmp_path, use_client):
    from haiku.rag.ingester.exceptions import TransientError

    (tmp_path / "a.md").write_text("hello")
    client = _mock_client()
    ingest = client._ingest_observed.side_effect
    calls = {"n": 0}

    async def _fails_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TransientError("blip")
        return await ingest(*args, **kwargs)

    client._ingest_observed.side_effect = _fails_once
    use_client(client)
    config = _config(tmp_path)
    config.ingester.workers.retry = RetryPolicyConfig(
        max_attempts=2, base_delay_s=3600.0, jitter=0.0
    )
    config.ingester.databases["haiku.rag"].sources[0].retry = RetryPolicyConfig(
        max_attempts=2, base_delay_s=0.0, jitter=0.0
    )

    report = await asyncio.wait_for(
        IngesterApp(config=config).run_batch(),
        timeout=10.0,
    )

    assert report.succeeded == 1
    assert calls["n"] == 2


async def test_run_batch_reports_dead_on_permanent_failure(tmp_path, use_client):
    (tmp_path / "a.md").write_text("hello")

    client = _mock_client()
    client._ingest_observed.side_effect = UnsupportedSourceError("nope")
    use_client(client)

    report = await IngesterApp(config=_config(tmp_path)).run_batch()

    assert report.succeeded == 0
    assert report.dead == 1


async def test_run_batch_reports_rejections_apart_from_dead(tmp_path, use_client):
    from haiku.rag.hooks import DocumentRejected

    (tmp_path / "a.md").write_text("hello")
    (tmp_path / "b.md").write_text("world")

    client = _mock_client()
    client._ingest_observed.side_effect = DocumentRejected("draft")
    use_client(client)
    progress = []

    report = await IngesterApp(config=_config(tmp_path)).run_batch(
        progress_callback=progress.append
    )

    assert (report.succeeded, report.dead, report.rejected) == (0, 0, 2)
    assert progress[-1].total == 2
    assert progress[-1].completed == 2
    assert progress[-1].rejected == 2


async def test_run_batch_recovered_doc_is_not_counted_as_dead(tmp_path, use_client):
    """A doc that dead-lettered in an earlier run and succeeds in this one
    shows as a success. The report counts only this run's terminal jobs, so
    the earlier dead row adds no stale dead count."""
    (tmp_path / "a.md").write_text("hello")
    config = _config(tmp_path)
    tmp_path / "db.lancedb"

    failing = _mock_client()
    failing._ingest_observed.side_effect = UnsupportedSourceError("nope")
    use_client(failing)
    first = await IngesterApp(config=config).run_batch()
    assert first.dead == 1

    healthy = _mock_client()
    use_client(healthy)
    # Bump the file's revision so discovery re-attempts it. A permanent failure
    # records the revision in sync_state, so a plain re-run no longer retries an
    # unchanged file — recovery needs the content (mtime) to change.
    (tmp_path / "a.md").write_text("hello again")
    second = await IngesterApp(config=config).run_batch()
    assert second.dead == 0
    assert second.succeeded == 1


async def _jobs_with(config: AppConfig, status: JobStatus) -> set[str]:
    engine = await open_queue(config.ingester.queue)
    try:
        jobs = await JobRepo(engine).list_jobs(status=status)
    finally:
        await engine.dispose()
    return {uri_to_path(job.uri).name for job in jobs}


async def test_run_batch_retries_failures_beside_the_sweep(tmp_path, use_client):
    """One run retries earlier failures, a stalled conversion included,
    ingests a new file and deletes a removed one, with no sweep skipped."""
    from haiku.rag.converters.exceptions import ConversionTimeoutError

    for name in ("failed.md", "stalled.md", "removed.md"):
        (tmp_path / name).write_text(name)
    config = _config(tmp_path)
    tmp_path / "db.lancedb"

    client = _mock_client()
    ingest = client._ingest_observed.side_effect

    async def _fails(uri, *args, **kwargs):
        if uri.endswith("failed.md"):
            raise UnsupportedSourceError("nope")
        if uri.endswith("stalled.md"):
            raise ConversionTimeoutError("slow", converter_wedged=False)
        return await ingest(uri, *args, **kwargs)

    client._ingest_observed.side_effect = _fails
    use_client(client)
    first = await IngesterApp(config=config).run_batch()
    assert (first.succeeded, first.dead) == (1, 2)

    client._ingest_observed.side_effect = ingest
    (tmp_path / "removed.md").unlink()
    (tmp_path / "new.md").write_text("new")
    second = await IngesterApp(config=config).run_batch(retry={JobStatus.DEAD})

    assert second.failed_sweeps == []
    assert second.retried == 2
    assert (second.succeeded, second.dead, second.rejected) == (4, 0, 0)
    client.delete_document.assert_awaited_once()
    assert await _jobs_with(config, JobStatus.DEAD) == set()


async def test_run_batch_leaves_a_recovered_failure_dead(tmp_path, use_client):
    (tmp_path / "a.md").write_text("hello")
    config = _config(tmp_path)
    tmp_path / "db.lancedb"

    failing = _mock_client()
    failing._ingest_observed.side_effect = UnsupportedSourceError("nope")
    use_client(failing)
    await IngesterApp(config=config).run_batch()
    healthy = _mock_client()
    use_client(healthy)
    (tmp_path / "a.md").write_text("hello again")
    await IngesterApp(config=config).run_batch()

    third = await IngesterApp(config=config).run_batch(retry={JobStatus.DEAD})

    assert third.retried == 0
    assert (third.succeeded, third.dead) == (0, 0)
    assert await _jobs_with(config, JobStatus.DEAD) == {"a.md"}


async def test_run_batch_without_retry_leaves_failures_dead(tmp_path, use_client):
    (tmp_path / "a.md").write_text("hello")
    config = _config(tmp_path)
    tmp_path / "db.lancedb"
    client = _mock_client()
    ingest = client._ingest_observed.side_effect
    client._ingest_observed.side_effect = UnsupportedSourceError("nope")
    use_client(client)
    await IngesterApp(config=config).run_batch()

    client._ingest_observed.side_effect = ingest
    second = await IngesterApp(config=config).run_batch()

    assert second.retried == 0
    assert second.succeeded == 0
    assert await _jobs_with(config, JobStatus.DEAD) == {"a.md"}


async def test_run_batch_reports_failed_sweep(
    tmp_path, use_client, monkeypatch, caplog
):
    """A source whose discover() raises is reported in failed_sweeps so the
    run can be treated as failed rather than a silent empty success."""
    (tmp_path / "a.md").write_text("hello")
    use_client(_mock_client())

    async def _failing_discover(self, **kwargs):
        raise RuntimeError("discover blew up")
        yield  # unreachable; makes this an async generator

    monkeypatch.setattr("haiku.rag.sources.fs.FSSource.discover", _failing_discover)

    with caplog.at_level("ERROR", logger="haiku.rag.ingester.pollers.base"):
        report = await IngesterApp(config=_config(tmp_path)).run_batch()

    assert report.failed_sweeps == ["local"]
    assert report.succeeded == 0
    assert report.dead == 0
    assert "discover() failed" in caplog.text


async def test_run_batch_empty_source_returns_immediately(tmp_path, use_client):
    client = _mock_client()
    use_client(client)

    report = await asyncio.wait_for(
        IngesterApp(config=_config(tmp_path)).run_batch(),
        timeout=5.0,
    )

    assert report.succeeded == 0
    assert report.dead == 0
    client._ingest_observed.assert_not_awaited()


async def test_run_batch_leaves_foreign_namespace_rows_alone(tmp_path, use_client):
    """Another ingester's queued and dead jobs for the same source and URIs
    neither block the batch, nor count in its report, nor change."""
    (tmp_path / "a.md").write_text("hello")
    (tmp_path / "b.md").write_text("world")
    config = _config(tmp_path)
    config.ingester.queue.namespace = "t1"
    engine = await open_queue(config.ingester.queue)
    try:
        foreign = JobRepo(engine, namespace="other")
        dead = await foreign.enqueue("local", (tmp_path / "b.md").as_uri())
        assert dead is not None
        claimed = await foreign.claim_next("w")
        assert claimed is not None and claimed.id == dead.id
        assert await foreign.mark_dead(dead.id, "boom", "w")
        queued = await foreign.enqueue("local", (tmp_path / "a.md").as_uri())
        assert queued is not None
    finally:
        await engine.dispose()
    use_client(_mock_client())

    report = await IngesterApp(config=config).run_batch()

    assert (report.succeeded, report.dead) == (2, 0)
    engine = await open_queue(config.ingester.queue)
    try:
        foreign = JobRepo(engine, namespace="other")
        still_queued = await foreign.get_job(queued.id)
        still_dead = await foreign.get_job(dead.id)
        assert still_queued is not None and still_queued.status is JobStatus.QUEUED
        assert still_dead is not None and still_dead.status is JobStatus.DEAD
        own = JobRepo(engine, namespace="t1")
        assert {job.status for job in await own.list_jobs()} == {JobStatus.SUCCEEDED}
        assert await SyncStateRepo(engine, namespace="t1").list_ingested_uris(
            "local"
        ) == {(tmp_path / "a.md").as_uri(), (tmp_path / "b.md").as_uri()}
    finally:
        await engine.dispose()


async def test_run_batch_dry_run_reads_its_own_revisions(tmp_path):
    """The dry run compares against the revisions its own namespace recorded."""
    (tmp_path / "a.md").write_text("hello")
    config = _config(tmp_path)
    config.ingester.queue.namespace = "t1"
    uri = (tmp_path / "a.md").as_uri()
    revision = str((tmp_path / "a.md").stat().st_mtime_ns)
    engine = await open_queue(config.ingester.queue)
    try:
        await SyncStateRepo(engine, namespace="t1").upsert(
            "local", uri, revision=revision, ingested=True
        )
    finally:
        await engine.dispose()

    report = await IngesterApp(config=config).run_batch_dry_run()

    assert report.failed_sweeps == []
    assert report.manifest.changes == []


async def test_run_batch_from_manifest_ignores_foreign_pending_work(
    tmp_path, use_client
):
    (tmp_path / "a.md").write_text("hello")
    config = _config(tmp_path)
    config.ingester.queue.namespace = "t1"
    use_client(_mock_client())
    engine = await open_queue(config.ingester.queue)
    try:
        foreign = JobRepo(engine, namespace="other")
        await foreign.enqueue("local", (tmp_path / "a.md").as_uri(), op=JobOp.UPSERT)
    finally:
        await engine.dispose()

    report = await IngesterApp(config=config).run_batch_from_manifest(
        _manifest(
            BatchChange(
                op=JobOp.UPSERT,
                source_id="local",
                uri=(tmp_path / "a.md").as_uri(),
                discovered_at=datetime.now(UTC),
            )
        )
    )

    assert (report.succeeded, report.dead) == (1, 0)


@pytest.mark.parametrize("namespace", ["", "   "])
def test_ingester_app_requires_a_namespace_on_a_server_queue(tmp_path, namespace):
    config = _config(tmp_path)
    config.ingester.queue = QueueConfig(
        dburi="postgresql+asyncpg://u:p@host/db", namespace=namespace
    )
    with pytest.raises(QueueError, match="namespace"):
        IngesterApp(config=config)


def test_ingester_app_accepts_a_sqlite_dburi_without_a_namespace(tmp_path):
    config = _config(tmp_path)
    config.ingester.queue = QueueConfig(
        dburi=f"sqlite+aiosqlite:///{tmp_path / 'queue.db'}"
    )
    IngesterApp(config=config)


async def test_run_batch_dry_run_reports_manifest_without_mutating_queue(tmp_path):
    (tmp_path / "a.md").write_text("hello")
    config = _config(tmp_path)
    tmp_path / "db.lancedb"

    engine = await open_queue(config.ingester.queue)
    try:
        sync = SyncStateRepo(engine)
        await sync.upsert("local", (tmp_path / "gone.md").as_uri(), revision="old")
    finally:
        await engine.dispose()

    report = await IngesterApp(config=config).run_batch_dry_run()

    assert report.failed_sweeps == []
    assert report.manifest.version == 1
    assert [(change.op, change.uri) for change in report.manifest.changes] == [
        (JobOp.UPSERT, (tmp_path / "a.md").as_uri()),
        (JobOp.DELETE, (tmp_path / "gone.md").as_uri()),
    ]
    source_summary = report.manifest.sources[0]
    assert source_summary.source_id == "local"
    assert source_summary.upsert_count == 1
    assert source_summary.delete_count == 1

    engine = await open_queue(config.ingester.queue)
    try:
        jobs = JobRepo(engine)
        sync = SyncStateRepo(engine)
        assert await jobs.list_jobs(source_id="local") == []
        assert await sync.list_known_uris("local") == {(tmp_path / "gone.md").as_uri()}
    finally:
        await engine.dispose()


async def test_run_batch_without_docling_fails_at_startup(tmp_path, monkeypatch):
    """`run_batch_dry_run` with `converter: docling-local` and no docling raises
    before any sweep, naming the extra."""
    (tmp_path / "a.md").write_text("hello")
    monkeypatch.setitem(sys.modules, "docling", None)

    with pytest.raises(ImportError, match=r"haiku\.rag-slim\[docling\]"):
        await IngesterApp(config=_config(tmp_path)).run_batch_dry_run()


async def test_run_batch_from_manifest_drains_changes_without_sweeping(
    tmp_path, use_client, monkeypatch
):
    (tmp_path / "a.md").write_text("hello")
    revision = str((tmp_path / "a.md").stat().st_mtime_ns)
    client = _mock_client()
    use_client(client)
    sweep_all = AsyncMock(side_effect=AssertionError("manifest replay must not sweep"))
    monkeypatch.setattr(PollerManager, "sweep_all", sweep_all)

    report = await IngesterApp(config=_config(tmp_path)).run_batch_from_manifest(
        _manifest(
            BatchChange(
                op=JobOp.UPSERT,
                source_id="local",
                uri=(tmp_path / "a.md").as_uri(),
                revision=revision,
                discovered_at=datetime.now(UTC),
            )
        )
    )

    assert report.succeeded == 1
    assert report.dead == 0
    client._ingest_observed.assert_awaited_once()
    sweep_all.assert_not_awaited()


async def test_run_batch_from_manifest_rejects_stale_upsert_revision(
    tmp_path, use_client
):
    (tmp_path / "a.md").write_text("hello")
    client = _mock_client()
    use_client(client)

    report = await IngesterApp(config=_config(tmp_path)).run_batch_from_manifest(
        _manifest(
            BatchChange(
                op=JobOp.UPSERT,
                source_id="local",
                uri=(tmp_path / "a.md").as_uri(),
                revision="stale",
                discovered_at=datetime.now(UTC),
            )
        )
    )

    assert report.succeeded == 0
    assert report.dead == 1
    client._ingest_observed.assert_not_awaited()


async def test_run_batch_from_manifest_delete_uses_manifest_even_if_file_reappears(
    tmp_path, use_client
):
    path = tmp_path / "gone.md"
    path.write_text("back")
    client = _mock_client()
    client.get_document_by_uri.side_effect = None
    client.get_document_by_uri.return_value = Document(
        id="doc-gone", content="x", uri=path.as_uri(), metadata={"source_id": "local"}
    )
    use_client(client)

    report = await IngesterApp(config=_config(tmp_path)).run_batch_from_manifest(
        _manifest(
            BatchChange(
                op=JobOp.DELETE,
                source_id="local",
                uri=path.as_uri(),
                discovered_at=datetime.now(UTC),
            )
        )
    )

    assert report.succeeded == 1
    assert report.dead == 0
    client.delete_document.assert_awaited_once()


async def test_run_batch_from_manifest_resumes_same_manifest_work(tmp_path, use_client):
    (tmp_path / "a.md").write_text("hello")
    revision = str((tmp_path / "a.md").stat().st_mtime_ns)
    config = _config(tmp_path)
    client = _mock_client()
    use_client(client)
    manifest = _manifest(
        BatchChange(
            op=JobOp.UPSERT,
            source_id="local",
            uri=(tmp_path / "a.md").as_uri(),
            revision=revision,
            discovered_at=datetime.now(UTC),
        )
    )
    engine = await open_queue(config.ingester.queue)
    try:
        jobs = JobRepo(engine)
        await jobs.enqueue(
            "local",
            (tmp_path / "a.md").as_uri(),
            op=JobOp.UPSERT,
            revision=revision,
            extra={
                "_manifest": {
                    "version": manifest.version,
                    "generated_at": manifest.generated_at.isoformat(),
                    "discovered_at": manifest.changes[0].discovered_at.isoformat(),
                }
            },
        )
    finally:
        await engine.dispose()

    report = await IngesterApp(config=config).run_batch_from_manifest(manifest)

    assert report.succeeded == 1
    assert report.dead == 0
    client._ingest_observed.assert_awaited_once()


async def test_run_batch_from_manifest_rejects_non_manifest_pending_work(
    tmp_path, use_client
):
    (tmp_path / "a.md").write_text("hello")
    config = _config(tmp_path)
    client = _mock_client()
    use_client(client)
    engine = await open_queue(config.ingester.queue)
    try:
        jobs = JobRepo(engine)
        await jobs.enqueue("local", (tmp_path / "a.md").as_uri(), op=JobOp.UPSERT)
    finally:
        await engine.dispose()

    with pytest.raises(ValueError, match="non-manifest pending work"):
        await IngesterApp(config=config).run_batch_from_manifest(
            _manifest(
                BatchChange(
                    op=JobOp.UPSERT,
                    source_id="local",
                    uri=(tmp_path / "a.md").as_uri(),
                    discovered_at=datetime.now(UTC),
                )
            )
        )


async def test_run_batch_from_manifest_rejects_different_manifest_pending_work(
    tmp_path, use_client
):
    (tmp_path / "a.md").write_text("hello")
    config = _config(tmp_path)
    client = _mock_client()
    use_client(client)
    manifest = _manifest(
        BatchChange(
            op=JobOp.UPSERT,
            source_id="local",
            uri=(tmp_path / "a.md").as_uri(),
            discovered_at=datetime.now(UTC),
        )
    )
    engine = await open_queue(config.ingester.queue)
    try:
        jobs = JobRepo(engine)
        await jobs.enqueue(
            "local",
            (tmp_path / "a.md").as_uri(),
            op=JobOp.UPSERT,
            extra={
                "_manifest": {
                    "version": 1,
                    "generated_at": "2026-01-01T00:00:00+00:00",
                    "discovered_at": manifest.changes[0].discovered_at.isoformat(),
                }
            },
        )
    finally:
        await engine.dispose()

    with pytest.raises(ValueError, match="non-manifest pending work"):
        await IngesterApp(config=config).run_batch_from_manifest(manifest)


async def test_run_batch_from_manifest_rejects_unrelated_pending_work(
    tmp_path, use_client
):
    (tmp_path / "a.md").write_text("hello")
    config = _config(tmp_path)
    client = _mock_client()
    use_client(client)
    engine = await open_queue(config.ingester.queue)
    try:
        jobs = JobRepo(engine)
        await jobs.enqueue("local", "file:///outside.md", op=JobOp.UPSERT)
    finally:
        await engine.dispose()

    with pytest.raises(ValueError, match="non-manifest pending work"):
        await IngesterApp(config=config).run_batch_from_manifest(
            _manifest(
                BatchChange(
                    op=JobOp.UPSERT,
                    source_id="local",
                    uri=(tmp_path / "a.md").as_uri(),
                    discovered_at=datetime.now(UTC),
                )
            )
        )


async def test_run_batch_from_manifest_rejects_duplicate_changes(tmp_path, use_client):
    path = tmp_path / "a.md"
    path.write_text("hello")
    client = _mock_client()
    use_client(client)
    change = BatchChange(
        op=JobOp.UPSERT,
        source_id="local",
        uri=path.as_uri(),
        discovered_at=datetime.now(UTC),
    )

    with pytest.raises(ValueError, match="duplicate change"):
        await IngesterApp(config=_config(tmp_path)).run_batch_from_manifest(
            _manifest(change, change)
        )


async def test_run_batch_aborts_when_all_workers_die(
    tmp_path, use_client, monkeypatch, caplog
):
    """If all workers crash with outstanding jobs, run_batch should break
    out of the drain loop instead of spinning forever."""
    (tmp_path / "a.md").write_text("hello")

    client = _mock_client()
    use_client(client)
    config = _config(tmp_path)

    config.ingester.workers.worker_count = 1

    # Patch _process to raise an unhandled exception, simulating a hard
    # worker crash.  _process only catches CancelledError, PermanentError,
    # and TransientError — anything else propagates and kills the task.
    monkeypatch.setattr(
        WorkerPool,
        "_process",
        AsyncMock(side_effect=Exception("worker crash")),
    )

    with caplog.at_level("ERROR", logger="haiku.rag.ingester.app"):
        report = await asyncio.wait_for(
            IngesterApp(config=config).run_batch(),
            timeout=10.0,
        )

    assert report is not None
    assert "All workers have died" in caplog.text


async def _wait_until(
    predicate, *, timeout: float = 5.0, serving: asyncio.Task | None = None
):
    """Wait for `predicate`, raising the error of a `serving` task that ended."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if serving is not None and serving.done():
            serving.result()
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached within timeout")
        await asyncio.sleep(0.02)


async def test_waiting_on_a_serve_that_failed_raises_its_error():
    async def failing_serve():
        raise RuntimeError("queue unavailable")

    serving = asyncio.create_task(failing_serve())

    with pytest.raises(RuntimeError, match="queue unavailable"):
        await _wait_until(lambda: False, serving=serving)


@pytest.mark.parametrize("api", [True, False])
async def test_serve_starts_workers_pollers_and_shuts_down(tmp_path, use_client, api):
    """serve brings up pollers, workers and (when enabled) the HTTP API, then
    tears them all down. Shutdown is driven here by cancelling the serve task,
    which runs the same drain-and-close path as a SIGINT/SIGTERM."""
    use_client(_mock_client())
    config = _config(tmp_path)
    config.ingester.api = APIConfig(enabled=api, host="127.0.0.1", port=0)
    app = IngesterApp(config=config)

    task = asyncio.create_task(app.serve(api=api))
    try:
        await _wait_until(
            lambda: (
                app._pool is not None
                and app._pool.live_workers > 0
                and app._pollers is not None
                and app._pollers.live_pollers > 0
            ),
            serving=task,
        )
    finally:
        task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5.0)

    # Workers and pollers are stopped after the shutdown path runs.
    assert app._pool is not None and app._pollers is not None
    assert app._pool.live_workers == 0
    assert app._pollers.live_pollers == 0


class _SlowPool(WorkerPool):
    """A WorkerPool whose stop never finishes within the grace, to exercise
    _stop_pool's timeout path. The real wiring is bypassed since _stop_pool
    only calls stop() and drain_pending_releases()."""

    def __init__(self) -> None:
        self.released = 0

    async def stop(self) -> None:
        await asyncio.sleep(1.0)

    async def drain_pending_releases(self, timeout: float = 2.0) -> int:
        self.released += 1
        return 2


async def test_stop_pool_warns_when_shutdown_grace_elapses(tmp_path, caplog):
    """When a worker doesn't stop within the shutdown grace, _stop_pool logs a
    warning and still drains any pending cancel-cleanup releases."""
    config = _config(tmp_path, shutdown_grace_s=0.01)
    app = IngesterApp(config=config)
    pool = _SlowPool()
    app._pool = pool

    with caplog.at_level("WARNING", logger="haiku.rag.ingester.app"):
        await app._stop_pool()

    assert pool.released == 1
    assert "Shutdown grace" in caplog.text


def _record_close_order(monkeypatch) -> list[str]:
    """Record the order of _stop_pool and PollerManager.close_sources.

    Workers share the pollers' Source instances for fetch(), so the source
    clients must be closed only after the pool has stopped — otherwise an
    in-flight fetch during the shutdown grace hits a closed client.
    """
    order: list[str] = []
    orig_stop_pool = IngesterApp._stop_pool
    orig_close = PollerManager.close_sources

    async def rec_stop(self):
        order.append("stop_pool")
        await orig_stop_pool(self)

    async def rec_close(self):
        order.append("close_sources")
        await orig_close(self)

    monkeypatch.setattr(IngesterApp, "_stop_pool", rec_stop)
    monkeypatch.setattr(PollerManager, "close_sources", rec_close)
    return order


async def test_run_batch_closes_sources_after_pool_stops(
    tmp_path, use_client, monkeypatch
):
    (tmp_path / "a.md").write_text("hello")
    use_client(_mock_client())
    app = IngesterApp(config=_config(tmp_path))
    order = _record_close_order(monkeypatch)

    await app.run_batch()

    assert order == ["stop_pool", "close_sources"]


async def test_serve_closes_sources_after_pool_stops(tmp_path, use_client, monkeypatch):
    use_client(_mock_client())
    config = _config(tmp_path)
    config.ingester.api = APIConfig(enabled=False)
    app = IngesterApp(config=config)
    order = _record_close_order(monkeypatch)

    task = asyncio.create_task(app.serve(api=False))
    try:
        await _wait_until(
            lambda: app._pool is not None and app._pool.live_workers > 0,
            serving=task,
        )
    finally:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5.0)

    assert order == ["stop_pool", "close_sources"]


# --- several databases ---


def _two_databases(tmp_path) -> AppConfig:
    for name in ("papers", "wiki"):
        (tmp_path / name).mkdir()
        (tmp_path / name / f"{name}.md").write_text(name)
    return AppConfig.model_validate(
        {
            "lancedb": {
                "databases": {
                    name: str(tmp_path / f"{name}.lancedb")
                    for name in ("papers", "wiki")
                }
            },
            "ingester": {
                "queue": {"path": str(tmp_path / "queue.db")},
                "workers": {
                    "worker_count": 2,
                    "poll_idle_interval_s": 0.05,
                    "retry": {"max_attempts": 1},
                },
                "api": {"enabled": False},
                "databases": {
                    name: {
                        "sources": [
                            {
                                "type": "fs",
                                "id": name,
                                "root": str(tmp_path / name),
                                "poll_interval_s": 3600.0,
                            }
                        ]
                    }
                    for name in ("papers", "wiki")
                },
            },
        }
    )


def _ingested(client) -> list[str]:
    return [
        uri_to_path(call.args[0]).name
        for call in client._ingest_observed.await_args_list
    ]


async def test_run_batch_ingests_each_database_through_its_own_client(
    tmp_path, use_client
):
    papers, wiki = _mock_client(), _mock_client()
    use_client({"papers": papers, "wiki": wiki})

    report = await IngesterApp(config=_two_databases(tmp_path)).run_batch()

    assert (report.succeeded, report.dead) == (2, 0)
    assert _ingested(papers) == ["papers.md"]
    assert _ingested(wiki) == ["wiki.md"]


async def test_serve_ingests_each_database_through_its_own_client(tmp_path, use_client):
    papers, wiki = _mock_client(), _mock_client()
    use_client({"papers": papers, "wiki": wiki})
    app = IngesterApp(config=_two_databases(tmp_path))

    task = asyncio.create_task(app.serve(api=False))
    try:
        await _wait_until(lambda: _ingested(papers) and _ingested(wiki))
    finally:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5.0)

    assert _ingested(papers) == ["papers.md"]
    assert _ingested(wiki) == ["wiki.md"]


async def test_each_database_is_reconciled_against_its_own_sources(
    tmp_path, use_client, monkeypatch
):
    papers, wiki = _mock_client(), _mock_client()
    use_client({"papers": papers, "wiki": wiki})
    seen = []

    async def _reconcile(rag, sync, source_ids):
        seen.append((rag, source_ids))
        return []

    monkeypatch.setattr("haiku.rag.ingester.app.reconcile", _reconcile)

    await IngesterApp(config=_two_databases(tmp_path)).run_batch()

    assert seen == [(papers, ["papers"]), (wiki, ["wiki"])]


@pytest.mark.parametrize("explicit", [True, False])
async def test_duplicate_source_ids_are_refused(tmp_path, use_client, explicit):
    from haiku.rag.ingester.exceptions import DuplicateSourceError

    use_client(_mock_client())
    config = _two_databases(tmp_path)
    for name, entry in config.ingester.databases.items():
        source = entry.sources[0]
        assert isinstance(source, FSSourceConfig)
        source.root = tmp_path / "papers"
        source.id = "shared" if explicit else None
    shared = "shared" if explicit else f"fs:{tmp_path / 'papers'}"

    with pytest.raises(DuplicateSourceError, match=re.escape(shared)):
        await IngesterApp(config=config).run_batch()


async def test_a_lone_database_failing_to_open_is_named(tmp_path, monkeypatch):
    from haiku.rag.ingester.exceptions import NoDatabaseStartedError
    from haiku.rag.store.exceptions import MigrationRequiredError

    @asynccontextmanager
    async def _open(scope):
        if scope.names == ("wiki",):
            raise MigrationRequiredError("Database requires migration")
        yield _mock_client()

    stub = MagicMock()
    stub._covering = lambda scope, *a, **k: _open(scope)
    monkeypatch.setattr("haiku.rag.client.HaikuRAG", stub)

    config = _two_databases(tmp_path)
    del config.ingester.databases["papers"]

    with capture_logs(
        logging.getLogger("haiku.rag.ingester.app"), logging.ERROR
    ) as records:
        with pytest.raises(
            NoDatabaseStartedError,
            match="every configured database failed to start: 'wiki': "
            "database 'wiki': Database requires migration",
        ):
            await IngesterApp(config=config).run_batch()
    assert [record.getMessage() for record in records] == [
        "Database 'wiki' failed to start and is skipped: database 'wiki': Database requires migration"
    ]


def _case_insensitive(directory) -> bool:
    (directory / "Probe").mkdir()
    return (directory / "probe").exists()


async def test_two_names_for_one_new_directory_are_refused_once_opened(tmp_path):
    """Neither database exists at startup, so only the opened stores show that
    the two names reach one directory."""
    if not _case_insensitive(tmp_path):
        pytest.skip("case-sensitive filesystem")
    config = _two_databases(tmp_path)
    config.lancedb.databases["papers"] = str(tmp_path / "Shared.lancedb")
    config.lancedb.databases["wiki"] = str(tmp_path / "shared.lancedb")
    from haiku.rag.store.exceptions import AmbiguousDatabaseError

    with pytest.raises(AmbiguousDatabaseError, match="'papers' and 'wiki'"):
        await IngesterApp(config=config).run_batch()
    assert await _jobs_of(config, "papers") == {}


def test_two_names_for_one_existing_directory_are_refused(tmp_path):
    if not _case_insensitive(tmp_path):
        pytest.skip("case-sensitive filesystem")
    (tmp_path / "Shared.lancedb").mkdir()
    config = _two_databases(tmp_path)
    config.lancedb.databases["papers"] = str(tmp_path / "Shared.lancedb")
    config.lancedb.databases["wiki"] = str(tmp_path / "shared.lancedb")
    from haiku.rag.store.exceptions import AmbiguousDatabaseError

    with pytest.raises(AmbiguousDatabaseError, match="'papers' and 'wiki'"):
        IngesterApp(config=config)


def test_two_spellings_of_an_existing_directory_are_refused(tmp_path):
    (tmp_path / "shared.lancedb").mkdir()
    config = _two_databases(tmp_path)
    config.lancedb.databases["papers"] = str(tmp_path / "shared.lancedb")
    config.lancedb.databases["wiki"] = str(
        tmp_path / "papers" / ".." / "shared.lancedb"
    )
    from haiku.rag.store.exceptions import AmbiguousDatabaseError

    with pytest.raises(AmbiguousDatabaseError, match="'papers' and 'wiki'"):
        IngesterApp(config=config)


async def _seed_unconfigured_jobs(config) -> JobRepo:
    engine = await open_queue(config.ingester.queue)
    jobs = JobRepo(engine)
    claimed = await jobs.enqueue("removed", "claimed", JobOp.UPSERT)
    assert claimed is not None
    assert await jobs.claim_next("w") is not None
    dead = await jobs.enqueue("removed", "dead", JobOp.UPSERT)
    assert dead is not None and await jobs.claim_next("w") is not None
    assert await jobs.mark_dead(dead.id, "boom", "w")
    assert await jobs.enqueue("removed", "queued", JobOp.UPSERT) is not None
    await engine.dispose()
    return jobs


async def _jobs_of(config, source_id: str) -> dict[str, JobStatus]:
    engine = await open_queue(config.ingester.queue)
    try:
        return {
            job.uri: job.status
            for job in await JobRepo(engine).list_jobs(source_id=source_id)
        }
    finally:
        await engine.dispose()


async def test_startup_deletes_live_jobs_of_unconfigured_sources(tmp_path, use_client):
    use_client({"papers": _mock_client(), "wiki": _mock_client()})
    config = _two_databases(tmp_path)
    await _seed_unconfigured_jobs(config)

    await IngesterApp(config=config).run_batch()

    assert await _jobs_of(config, "removed") == {"dead": JobStatus.DEAD}


async def test_a_dry_run_leaves_jobs_of_unconfigured_sources(tmp_path):
    config = _two_databases(tmp_path)
    await _seed_unconfigured_jobs(config)

    await IngesterApp(config=config).run_batch_dry_run()

    assert await _jobs_of(config, "removed") == {
        "claimed": JobStatus.CLAIMED,
        "dead": JobStatus.DEAD,
        "queued": JobStatus.QUEUED,
    }


async def test_retry_revives_only_the_failures_of_configured_sources(
    tmp_path, use_client
):
    use_client({"papers": _mock_client(), "wiki": _mock_client()})
    config = _two_databases(tmp_path)
    await _seed_unconfigured_jobs(config)

    report = await asyncio.wait_for(
        IngesterApp(config=config).run_batch(retry={JobStatus.DEAD}), timeout=10
    )

    assert report.retried == 0
    assert await _jobs_of(config, "removed") == {"dead": JobStatus.DEAD}


# --- one database of several ---


async def _seed(config, *jobs_to_add: tuple[str, str, JobStatus]) -> dict[str, str]:
    """Seed jobs as (source_id, uri, status) and return their ids by uri."""
    engine = await open_queue(config.ingester.queue)
    jobs = JobRepo(engine)
    ids = {}
    try:
        for source_id, uri, status in jobs_to_add:
            job = await jobs.enqueue(source_id, uri, JobOp.UPSERT)
            assert job is not None
            ids[uri] = job.id
            if status is JobStatus.DEAD:
                claimed = await jobs.claim_next("w", source_ids={source_id})
                assert claimed is not None and claimed.id == job.id
                assert await jobs.mark_dead(job.id, "boom", "w")
    finally:
        await engine.dispose()
    return ids


async def test_a_selected_database_is_ingested_alone(tmp_path, use_client):
    papers, wiki = _mock_client(), _mock_client()
    use_client({"papers": papers, "wiki": wiki})
    config = _two_databases(tmp_path)
    await _seed(config, ("wiki", "file:///queued.md", JobStatus.QUEUED))

    report = await IngesterApp(config=config, database_name="papers").run_batch()

    assert (report.succeeded, report.dead) == (1, 0)
    assert _ingested(papers) == ["papers.md"]
    assert _ingested(wiki) == []
    assert await _jobs_of(config, "wiki") == {"file:///queued.md": JobStatus.QUEUED}


async def test_a_selection_retries_its_own_failures_only(tmp_path, use_client):
    use_client({"papers": _mock_client(), "wiki": _mock_client()})
    config = _two_databases(tmp_path)
    await _seed(config, ("wiki", "file:///failed.md", JobStatus.DEAD))

    report = await asyncio.wait_for(
        IngesterApp(config=config, database_name="papers").run_batch(
            retry={JobStatus.DEAD}
        ),
        timeout=10,
    )

    assert report.retried == 0
    assert await _jobs_of(config, "wiki") == {"file:///failed.md": JobStatus.DEAD}


async def test_a_selected_dry_run_sweeps_its_own_sources(tmp_path):
    config = _two_databases(tmp_path)

    report = await IngesterApp(config=config, database_name="wiki").run_batch_dry_run()

    assert [summary.source_id for summary in report.manifest.sources] == ["wiki"]


async def test_a_selected_replay_ignores_other_databases_pending_work(
    tmp_path, use_client
):
    use_client({"papers": _mock_client(), "wiki": _mock_client()})
    config = _two_databases(tmp_path)
    await _seed(config, ("wiki", "file:///queued.md", JobStatus.QUEUED))
    app = IngesterApp(config=config, database_name="papers")
    manifest = (await app.run_batch_dry_run()).manifest

    report = await app.run_batch_from_manifest(manifest)

    assert report.succeeded == 1
    assert await _jobs_of(config, "wiki") == {"file:///queued.md": JobStatus.QUEUED}


async def test_a_selected_replay_refuses_another_databases_changes(
    tmp_path, use_client
):
    """A manifest from a sweep over every database names sources a selected
    replay does not ingest."""
    use_client({"papers": _mock_client(), "wiki": _mock_client()})
    config = _two_databases(tmp_path)
    manifest = (await IngesterApp(config=config).run_batch_dry_run()).manifest

    with pytest.raises(ValueError, match="does not ingest: wiki"):
        await IngesterApp(
            config=config, database_name="papers"
        ).run_batch_from_manifest(manifest)
    assert await _jobs_of(config, "papers") == {}


def test_an_unknown_selection_is_refused(tmp_path):
    from haiku.rag.store.exceptions import UnknownDatabaseError

    with pytest.raises(UnknownDatabaseError, match="'nope'.*papers, wiki"):
        IngesterApp(config=_two_databases(tmp_path), database_name="nope")


async def test_an_unselected_source_that_fails_to_build_closes_the_selected(
    tmp_path, use_client, monkeypatch
):
    from haiku.rag.config import PluginSourceConfig

    use_client({"papers": _mock_client(), "wiki": _mock_client()})
    config = _two_databases(tmp_path)
    config.ingester.databases["wiki"].sources = [
        PluginSourceConfig(type="plugin", id="wiki", plugin="missing")
    ]
    closed = []
    monkeypatch.setattr(PollerManager, "close_sources", lambda self: _closing(closed))

    with pytest.raises(ValueError, match="unknown source plugin 'missing'"):
        await IngesterApp(config=config, database_name="papers").run_batch()

    assert closed == [True]


async def _closing(closed: list) -> None:
    closed.append(True)
