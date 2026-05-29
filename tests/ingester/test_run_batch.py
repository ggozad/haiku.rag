"""IngesterApp.run_batch: one discover sweep across configured sources,
drain the queue, then exit. The document store (HaikuRAG) is patched out —
the behavior under test is the sweep -> queue -> worker -> drain
orchestration and orphan pruning, not embedding."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import unquote, urlparse

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.client.exceptions import UnsupportedSourceError
from haiku.rag.config import (
    APIConfig,
    AppConfig,
    FSSourceConfig,
    IngesterConfig,
    QueueConfig,
    RetryPolicyConfig,
    WorkerConfig,
)
from haiku.rag.ingester.app import IngesterApp
from haiku.rag.store.models.document import Document


def _config(tmp_path, **worker_kwargs) -> AppConfig:
    workers = WorkerConfig(
        worker_count=2,
        poll_idle_interval_s=0.05,
        retry=RetryPolicyConfig(max_attempts=1),
        **worker_kwargs,
    )
    return AppConfig(
        ingester=IngesterConfig(
            queue=QueueConfig(path=tmp_path / "queue.db"),
            sources=[
                FSSourceConfig(
                    type="fs",
                    id="local",
                    root=tmp_path,
                    poll_interval_s=3600.0,
                )
            ],
            workers=workers,
            api=APIConfig(enabled=False),
        ),
    )


def _mock_client() -> AsyncMock:
    """A HaikuRAG mock whose upserts return a fresh Document (with an md5 so
    the worker records sync_state) and whose lookups return a Document with a
    deterministic id so the DELETE path resolves."""
    client = AsyncMock(spec=HaikuRAG)
    counter = {"n": 0}

    async def _create(uri, *_, **__):
        counter["n"] += 1
        # Mirror the real client: persist the FS revision (mtime_ns) so the
        # poller's change-detection skips unchanged files on the next sweep.
        path = Path(unquote(urlparse(uri).path))
        return Document(
            id=f"doc-{counter['n']}",
            content="x",
            uri=uri,
            metadata={
                "content_type": "text/markdown",
                "md5": f"md5-{counter['n']}",
                "source_revision": str(path.stat().st_mtime_ns),
            },
        )

    async def _get_by_uri(uri):
        return Document(id=f"doc-for-{uri}", content="x", uri=uri, metadata={})

    client.create_document_from_source.side_effect = _create
    client.get_document_by_uri.side_effect = _get_by_uri
    return client


@pytest.fixture
def use_client(monkeypatch):
    """Make IngesterApp's internally-created HaikuRAG resolve to the given
    mock client. Call again to swap the client for a later run in the same
    test."""

    def _install(client):
        @asynccontextmanager
        async def _cm(*_, **__):
            yield client

        monkeypatch.setattr("haiku.rag.client.HaikuRAG", lambda *a, **k: _cm())

    return _install


@pytest.mark.asyncio
async def test_run_batch_drains_upserts(tmp_path, use_client):
    (tmp_path / "a.md").write_text("hello")
    (tmp_path / "b.md").write_text("world")

    client = _mock_client()
    use_client(client)

    report = await IngesterApp(
        config=_config(tmp_path), db_path=tmp_path / "db.lancedb"
    ).run_batch()

    assert report.succeeded == 2
    assert report.dead == 0
    assert client.create_document_from_source.await_count == 2
    ingested = {
        call.args[0] for call in client.create_document_from_source.await_args_list
    }
    assert ingested == {
        (tmp_path / "a.md").as_uri(),
        (tmp_path / "b.md").as_uri(),
    }


@pytest.mark.asyncio
async def test_run_batch_prunes_orphans(tmp_path, use_client):
    (tmp_path / "a.md").write_text("hello")
    (tmp_path / "b.md").write_text("world")

    client = _mock_client()
    use_client(client)
    config = _config(tmp_path)
    db_path = tmp_path / "db.lancedb"

    # First batch ingests both files and records sync_state for each.
    first = await IngesterApp(config=config, db_path=db_path).run_batch()
    assert first.succeeded == 2
    client.delete_document.assert_not_awaited()

    # b.md disappears from the source tree. The next sweep sees it in
    # sync_state but not on disk -> enqueues a DELETE for it.
    (tmp_path / "b.md").unlink()

    second = await IngesterApp(config=config, db_path=db_path).run_batch()

    # a.md is unchanged (same mtime) so it's not re-ingested; only the orphan
    # delete runs.
    assert client.create_document_from_source.await_count == 2
    client.delete_document.assert_awaited_once()
    assert second.succeeded == 1
    assert second.dead == 0


@pytest.mark.asyncio
async def test_run_batch_reports_dead_on_permanent_failure(tmp_path, use_client):
    (tmp_path / "a.md").write_text("hello")

    client = _mock_client()
    client.create_document_from_source.side_effect = UnsupportedSourceError("nope")
    use_client(client)

    report = await IngesterApp(
        config=_config(tmp_path), db_path=tmp_path / "db.lancedb"
    ).run_batch()

    assert report.succeeded == 0
    assert report.dead == 1


@pytest.mark.asyncio
async def test_run_batch_recovered_doc_is_not_counted_as_dead(tmp_path, use_client):
    """A doc that dead-lettered in an earlier run and succeeds in this one
    prunes its dead row. The report counts only this run's terminal jobs, so
    the recovery shows as a success, never a negative or stale dead count."""
    (tmp_path / "a.md").write_text("hello")
    config = _config(tmp_path)
    db_path = tmp_path / "db.lancedb"

    failing = _mock_client()
    failing.create_document_from_source.side_effect = UnsupportedSourceError("nope")
    use_client(failing)
    first = await IngesterApp(config=config, db_path=db_path).run_batch()
    assert first.dead == 1

    healthy = _mock_client()
    use_client(healthy)
    second = await IngesterApp(config=config, db_path=db_path).run_batch()
    assert second.dead == 0
    assert second.succeeded == 1


@pytest.mark.asyncio
async def test_run_batch_empty_source_returns_immediately(tmp_path, use_client):
    client = _mock_client()
    use_client(client)

    report = await asyncio.wait_for(
        IngesterApp(
            config=_config(tmp_path), db_path=tmp_path / "db.lancedb"
        ).run_batch(),
        timeout=5.0,
    )

    assert report.succeeded == 0
    assert report.dead == 0
    client.create_document_from_source.assert_not_awaited()
