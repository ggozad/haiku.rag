import logging

import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.client import HaikuRAG
from haiku.rag.client.scope import DatabaseScope
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
from haiku.rag.ingester.queue.repository import SyncStateRepo
from haiku.rag.ingester.reconcile import reconcile
from haiku.rag.store.models.chunk import Chunk

from ..conftest import capture_logs

RECONCILE_LOGGER = logging.getLogger("haiku.rag.ingester.reconcile")


def _docling_document(text: str) -> DoclingDocument:
    doc = DoclingDocument(name="reconcile")
    doc.add_text(label=DocItemLabel.TEXT, text=text)
    return doc


async def _document(client: HaikuRAG, uri: str, source_id: str | None = None):
    """A stored document, attributed when `source_id` is given, without an
    embedder call."""
    vector_dim = client.store.embedder.vector_dim
    doc = await client.import_document(
        _docling_document(f"Content of {uri}."),
        [Chunk(content=f"Content of {uri}.", embedding=[0.1] * vector_dim)],
        uri=uri,
    )
    assert doc.id is not None
    if source_id is not None:
        (doc,) = await client.set_document_source([doc.id], source_id)
    return doc


@pytest.fixture
async def client(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        yield rag


async def test_recovers_a_lost_sync_state_row(client, sync):
    """#643: the queue lost its rows, so nothing knew the document was ours."""
    await _document(client, "file:///corpus/a.md", "fs:corpus")

    reports = await reconcile(client, sync, ["fs:corpus"])

    assert await sync.list_known_uris("fs:corpus") == {"file:///corpus/a.md"}
    assert await sync.get_revision_snapshot("fs:corpus") == {}
    assert reports[0].recovered == 1


async def test_invalidates_a_revision_the_store_lost(client, sync):
    """A document deleted by hand leaves a revision that suppresses re-ingest."""
    await sync.upsert(
        "fs:corpus", "file:///corpus/gone.md", revision="v1", ingested=True
    )

    reports = await reconcile(client, sync, ["fs:corpus"])

    assert await sync.get_revision_snapshot("fs:corpus") == {}
    assert await sync.list_known_uris("fs:corpus") == {"file:///corpus/gone.md"}
    assert reports[0].invalidated == 1


async def test_attributes_a_document_the_queue_claims(client, sync):
    doc = await _document(client, "file:///corpus/legacy.md")
    await sync.upsert(
        "fs:corpus", "file:///corpus/legacy.md", revision="v1", ingested=True
    )

    reports = await reconcile(client, sync, ["fs:corpus"])

    assert doc.id is not None
    refreshed = await client.get_document_by_id(doc.id)
    assert refreshed is not None
    assert refreshed.metadata["source_id"] == "fs:corpus"
    assert reports[0].attributed == 1


async def test_does_not_attribute_from_a_row_that_never_ingested(client, sync):
    """A row without last_ingested_at was enqueued or gave up, so the source
    may not have written the document."""
    doc = await _document(client, "file:///corpus/pending.md")
    await sync.upsert("fs:corpus", "file:///corpus/pending.md")

    reports = await reconcile(client, sync, ["fs:corpus"])

    assert doc.id is not None
    refreshed = await client.get_document_by_id(doc.id)
    assert refreshed is not None
    assert "source_id" not in refreshed.metadata
    assert reports[0].attributed == 0


async def test_does_not_attribute_from_a_permanent_failure_marker(client, sync):
    """A dead job writes a revision to suppress re-enqueue; the document at
    that URI came from somewhere else."""
    doc = await _document(client, "file:///corpus/poison.md")
    await sync.upsert(
        "fs:corpus", "file:///corpus/poison.md", revision="v1", ingested=False
    )

    reports = await reconcile(client, sync, ["fs:corpus"])

    assert doc.id is not None
    refreshed = await client.get_document_by_id(doc.id)
    assert refreshed is not None
    assert "source_id" not in refreshed.metadata
    assert reports[0].attributed == 0


async def test_keeps_a_permanent_failure_marker(client, sync):
    """A dead job's revision is not evidence its document is missing."""
    await sync.upsert(
        "fs:corpus", "file:///corpus/poison.md", revision="v1", ingested=False
    )

    reports = await reconcile(client, sync, ["fs:corpus"])

    assert await sync.get_revision_snapshot("fs:corpus") == {
        "file:///corpus/poison.md": "v1"
    }
    assert reports[0].invalidated == 0


async def test_attributes_a_revisionless_ingestion(client, sync):
    """An HTTP source without an ETag stores no revision and still ingested."""
    doc = await _document(client, "https://example.com/a.html")
    await sync.upsert("http:site", "https://example.com/a.html", ingested=True)

    reports = await reconcile(client, sync, ["http:site"])

    assert doc.id is not None
    refreshed = await client.get_document_by_id(doc.id)
    assert refreshed is not None
    assert refreshed.metadata["source_id"] == "http:site"
    assert reports[0].attributed == 1


async def test_leaves_a_document_two_sources_ingested_unattributed(client, sync):
    doc = await _document(client, "file:///corpus/shared.md")
    await sync.upsert("fs:outer", "file:///corpus/shared.md", ingested=True)
    await sync.upsert("fs:inner", "file:///corpus/shared.md", ingested=True)

    with capture_logs(RECONCILE_LOGGER, logging.WARNING) as records:
        reports = await reconcile(client, sync, ["fs:outer", "fs:inner"])

    assert doc.id is not None
    refreshed = await client.get_document_by_id(doc.id)
    assert refreshed is not None
    assert "source_id" not in refreshed.metadata
    assert all(report.attributed == 0 for report in reports)
    messages = [record.getMessage() for record in records]
    assert [m for m in messages if "fs:outer" in m and "fs:inner" in m]


async def test_keeps_a_revision_for_a_document_another_source_owns(client, sync):
    """The document is still there, so nothing needs re-ingesting."""
    await _document(client, "file:///corpus/shared.md", "fs:other")
    await sync.upsert(
        "fs:corpus", "file:///corpus/shared.md", revision="v1", ingested=True
    )

    reports = await reconcile(client, sync, ["fs:corpus"])

    assert await sync.get_revision_snapshot("fs:corpus") == {
        "file:///corpus/shared.md": "v1"
    }
    assert reports[0] == reports[0].__class__(source_id="fs:corpus")


async def test_leaves_another_sources_documents_alone(client, sync):
    await _document(client, "file:///other/a.md", "fs:other")

    reports = await reconcile(client, sync, ["fs:corpus"])

    assert await sync.list_known_uris("fs:corpus") == set()
    assert not reports[0].drifted


async def test_a_reconciled_database_reconciles_to_a_noop(client, sync):
    await _document(client, "file:///corpus/a.md", "fs:corpus")
    await reconcile(client, sync, ["fs:corpus"])

    reports = await reconcile(client, sync, ["fs:corpus"])

    assert not reports[0].drifted


async def test_warns_about_documents_of_an_unconfigured_source(client, sync):
    await _document(client, "file:///retired/a.md", "fs:retired")
    await _document(client, "file:///retired/b.md", "fs:retired")

    with capture_logs(RECONCILE_LOGGER, logging.WARNING) as records:
        await reconcile(client, sync, ["fs:corpus"])

    messages = [record.getMessage() for record in records]
    assert [m for m in messages if "fs:retired" in m and "2 document" in m]


async def test_unattributed_documents_are_left_alone(client, sync):
    """Nothing records that the source wrote them, so nothing may claim them."""
    await _document(client, "file:///corpus/handmade.md")

    reports = await reconcile(client, sync, ["fs:corpus"])

    assert not reports[0].drifted
    assert await sync.list_known_uris("fs:corpus") == set()


async def test_ignores_documents_without_a_uri(client, sync):
    """A document with no URI matches no source item."""
    vector_dim = client.store.embedder.vector_dim
    await client.import_document(
        _docling_document("Content with no source."),
        [Chunk(content="Content with no source.", embedding=[0.1] * vector_dim)],
    )

    reports = await reconcile(client, sync, ["fs:corpus"])

    assert not reports[0].drifted


async def test_attribution_writes_one_table_version(client, sync, monkeypatch):
    """A legacy corpus migrates in one document_meta version, not one each."""
    # auto_vacuum writes document_meta versions of its own.
    monkeypatch.setattr(client._config.storage, "auto_vacuum", False)
    uris = [f"file:///corpus/legacy-{n}.md" for n in range(5)]
    for uri in uris:
        await _document(client, uri)
        await sync.upsert("fs:corpus", uri, ingested=True)
    before = (await client.store.current_table_versions())["document_meta"]

    reports = await reconcile(client, sync, ["fs:corpus"])

    after = (await client.store.current_table_versions())["document_meta"]
    assert reports[0].attributed == 5
    assert after - before == 1


async def test_queue_repairs_write_one_transaction_each(client, sync, monkeypatch):
    """Recovery and invalidation are one transaction each, not one per URI."""
    calls: dict[str, int] = {}

    def count(name):
        original = getattr(SyncStateRepo, name)

        async def wrapper(self, *args, **kwargs):
            calls[name] = calls.get(name, 0) + 1
            return await original(self, *args, **kwargs)

        monkeypatch.setattr(SyncStateRepo, name, wrapper)

    for n in range(3):
        await _document(client, f"file:///corpus/owned-{n}.md", "fs:corpus")
        await sync.upsert(
            "fs:corpus", f"file:///corpus/lost-{n}.md", revision="v1", ingested=True
        )
    count("upsert")
    count("batch_upsert")
    count("invalidate")

    reports = await reconcile(client, sync, ["fs:corpus"])

    assert reports[0].recovered == 3
    assert reports[0].invalidated == 3
    assert calls.get("upsert", 0) == 0
    assert calls.get("batch_upsert") == 1
    assert calls.get("invalidate") == 1


def _ingester_config(tmp_path, queue_path):
    return AppConfig(
        ingester=IngesterConfig(
            queue=QueueConfig(path=queue_path),
            sources=[
                FSSourceConfig(
                    type="fs",
                    id="corpus",
                    root=tmp_path / "corpus",
                    poll_interval_s=3600,
                )
            ],
            workers=WorkerConfig(
                worker_count=1,
                poll_idle_interval_s=0.05,
                retry=RetryPolicyConfig(max_attempts=1),
            ),
            api=APIConfig(enabled=False),
        )
    )


@pytest.mark.vcr()
async def test_queue_loss_does_not_strand_documents(tmp_path):
    """#643: the queue database is lost between runs, so the swapped-out
    document has nothing left saying it was ever ours."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    queue_path = tmp_path / "queue.db"
    db_path = tmp_path / "db.lancedb"
    config = _ingester_config(tmp_path, queue_path)

    async def run_batch():
        return await IngesterApp(
            config=config, scope=DatabaseScope.at(db_path)
        ).run_batch()

    async def stored_uris():
        async with HaikuRAG(db_path, config) as rag:
            return {doc.uri for doc in await rag.list_documents()}

    first = corpus / "first.md"
    second = corpus / "second.md"
    first.write_text("The first document of the corpus.")
    assert (await run_batch()).dead == 0
    assert await stored_uris() == {first.as_uri()}

    first.unlink()
    second.write_text("The second document of the corpus.")
    assert (await run_batch()).dead == 0
    assert await stored_uris() == {second.as_uri()}

    for leftover in tmp_path.glob("queue.db*"):
        leftover.unlink()
    second.unlink()
    first.write_text("The first document of the corpus.")
    assert (await run_batch()).dead == 0

    assert await stored_uris() == {first.as_uri()}
