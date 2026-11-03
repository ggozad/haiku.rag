"""Ingestion through supervised conversion processes, end to end, with a
document whose conversion hangs. The conversion process is the stub in
tests/converters, so no docling model runs; chunking, embedding and storage
are real, with the embedder answering locally."""

import asyncio
import contextlib
import os
import signal
import sys
from pathlib import Path

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.client.exceptions import DirectoryIngestError
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
from haiku.rag.converters import supervisor as supervisor_module
from haiku.rag.converters.supervisor import close_supervisors, running_supervisors
from haiku.rag.embeddings import EmbedderWrapper
from haiku.rag.ingester.app import IngesterApp
from haiku.rag.ingester.queue.migrations import open_queue
from haiku.rag.ingester.queue.models import JobStatus
from haiku.rag.ingester.queue.repository import JobRepo
from tests.converters.test_supervisor import STUB
from tests.ingester.conftest import _wait_until


@pytest.fixture(autouse=True)
def local_conversion_and_embedding(monkeypatch):
    monkeypatch.setattr(supervisor_module, "WORKER_COMMAND", STUB)

    async def embed_documents(self, texts):
        return [[0.1] * self.vector_dim for _ in texts]

    async def embed_query(self, text):  # noqa: ARG001
        return [0.1] * self.vector_dim

    monkeypatch.setattr(EmbedderWrapper, "_embed_documents", embed_documents)
    monkeypatch.setattr(EmbedderWrapper, "_embed_query", embed_query)


def _config(tmp_path: Path, folder: Path) -> AppConfig:
    config = AppConfig(
        lancedb=LanceDBConfig(databases={"files": str(tmp_path / "files.lancedb")}),
        ingester=IngesterConfig(
            queue=QueueConfig(path=tmp_path / "queue.db"),
            databases={
                "files": IngestedDatabase(
                    sources=[
                        FSSourceConfig(
                            type="fs", id="docs", root=folder, poll_interval_s=3600.0
                        )
                    ]
                )
            },
            workers=WorkerConfig(
                worker_count=3,
                poll_idle_interval_s=0.05,
                retry=RetryPolicyConfig(max_attempts=1),
            ),
            api=APIConfig(enabled=False),
        ),
    )
    config.processing.conversion_options.ocr_engine = "ocrmac"
    supervisor = config.processing.supervisor
    supervisor.enabled = True
    supervisor.workers = 2
    supervisor.unit_timeout = 2.0
    supervisor.background = False
    return config


def _folder(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    for name in ("a.md", "b.md", "hang-always.md", "hang-once.md"):
        (folder / name).write_text(f"# {name}\n", encoding="utf-8")
    return folder


@pytest.mark.slow
async def test_the_ingester_records_a_hanging_document_and_ingests_the_rest(
    tmp_path,
):
    folder = _folder(tmp_path)
    config = _config(tmp_path, folder)

    report = await IngesterApp(config=config).run_batch()

    assert report.succeeded == 3
    assert report.dead == 1
    engine = await open_queue(config.ingester.queue)
    try:
        [dead] = await JobRepo(engine).list_jobs(status=JobStatus.DEAD)
    finally:
        await engine.dispose()
    assert dead.uri == (folder / "hang-always.md").as_uri()
    assert dead.last_error is not None
    assert "timed out after 2s (ocrmac); retried with rapidocr" in dead.last_error
    assert dead.conversion_stalled

    async with HaikuRAG(config=config, read_only=True) as rag:
        uris = {doc.uri for doc in await rag.list_documents()}
    assert uris == {
        (folder / name).as_uri() for name in ("a.md", "b.md", "hang-once.md")
    }


@pytest.mark.slow
async def test_a_directory_ingest_reports_a_hanging_document_and_keeps_the_rest(
    tmp_path,
):
    folder = _folder(tmp_path)
    config = _config(tmp_path, folder)

    try:
        async with HaikuRAG(config=config, create=True) as rag:
            with pytest.raises(DirectoryIngestError) as failed:
                await rag.create_document_from_source(folder)
    finally:
        await close_supervisors()

    assert set(failed.value.failures) == {str(folder / "hang-always.md")}
    assert "timed out" in failed.value.failures[str(folder / "hang-always.md")]
    assert sorted(doc.uri for doc in failed.value.documents) == sorted(
        (folder / name).as_uri() for name in ("a.md", "b.md", "hang-once.md")
    )


async def test_a_clean_directory_ingest_returns_every_document(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    for name in ("a.md", "b.md"):
        (folder / name).write_text(f"# {name}\n", encoding="utf-8")
    (folder / "notes.txt").write_text("plain text", encoding="utf-8")
    config = _config(tmp_path, folder)

    try:
        async with HaikuRAG(config=config, create=True) as rag:
            documents = await rag.create_document_from_source(folder)
    finally:
        await close_supervisors()

    assert isinstance(documents, list)
    assert len(documents) == 3


async def test_a_directory_ingest_keeps_going_past_a_failing_file(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    for name in ("a.md", "bad.md"):
        (folder / name).write_text(f"# {name}\n", encoding="utf-8")
    config = _config(tmp_path, folder)

    try:
        async with HaikuRAG(config=config, create=True) as rag:
            with pytest.raises(DirectoryIngestError) as failed:
                await rag.create_document_from_source(folder)
    finally:
        await close_supervisors()

    [reason] = failed.value.failures.values()
    assert "ValueError: cannot parse" in reason
    assert [doc.uri for doc in failed.value.documents] == [(folder / "a.md").as_uri()]


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows has no signals to send a process"
)
async def test_a_document_converting_when_serve_stops_is_queued_again(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "hang-always.md").write_text("# hang\n", encoding="utf-8")
    config = _config(tmp_path, folder)
    config.processing.supervisor.unit_timeout = 60.0
    # Long enough for a retry to start, and to be signalled in turn.
    config.ingester.workers.shutdown_grace_s = 4.0
    serving = asyncio.create_task(IngesterApp(config=config).serve(api=False))

    def conversions() -> list[asyncio.subprocess.Process]:
        return [
            worker.process
            for supervisor in running_supervisors()
            for worker in (*supervisor._pool.busy, *supervisor._pool.idle)
            if worker.process.returncode is None
        ]

    async def stop_every_process_of_the_service() -> None:
        while True:
            for process in conversions():
                os.kill(process.pid, signal.SIGTERM)
            await asyncio.sleep(0.02)

    await _wait_until(
        lambda: any(s.busy for s in running_supervisors()),
        timeout=30.0,
        serving=serving,
    )
    signalling = asyncio.create_task(stop_every_process_of_the_service())
    try:
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(serving, timeout=10.0)
    finally:
        signalling.cancel()

    engine = await open_queue(config.ingester.queue)
    try:
        [job] = await JobRepo(engine).list_jobs()
    finally:
        await engine.dispose()
    assert job.status == JobStatus.QUEUED
    assert job.attempts == 0
