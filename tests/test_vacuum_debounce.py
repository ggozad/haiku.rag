import asyncio

import pytest

import haiku.rag.client as client_mod
from haiku.rag.client import HaikuRAG
from haiku.rag.client.documents import _refresh_doc_metadata
from haiku.rag.config import Config
from haiku.rag.store.models.chunk import Chunk


def _docling_doc(name: str, text: str):
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel

    doc = DoclingDocument(name=name)
    doc.add_text(label=DocItemLabel.TEXT, text=text)
    return doc


@pytest.mark.asyncio
async def test_schedule_vacuum_is_debounced(temp_db_path, monkeypatch):
    """Rapid writes within the throttle window schedule only one background
    vacuum; once the interval elapses, a new one is scheduled."""
    t = {"now": 1000.0}
    monkeypatch.setattr(client_mod, "monotonic", lambda: t["now"])

    async with HaikuRAG(temp_db_path, create=True) as client:
        calls: list[int] = []

        async def fake_vacuum(*_a, **_k):
            calls.append(1)

        monkeypatch.setattr(client.store, "vacuum", fake_vacuum)

        for _ in range(3):
            client._schedule_vacuum()
        await asyncio.gather(*client._vacuum_tasks)
        assert len(calls) == 1  # debounced within the interval

        t["now"] += client_mod._VACUUM_MIN_INTERVAL_S + 1
        client._schedule_vacuum()
        await asyncio.gather(*client._vacuum_tasks)
        assert len(calls) == 2  # interval elapsed -> a new vacuum scheduled


@pytest.mark.asyncio
async def test_debounced_writes_still_collapse_on_close(temp_db_path, monkeypatch):
    """Even when scheduled vacuums after the first are debounced, the writes are
    marked dirty so the close-time drain runs a final collapse."""
    t = {"now": 1000.0}
    monkeypatch.setattr(client_mod, "monotonic", lambda: t["now"])
    calls: list[int] = []

    async with HaikuRAG(temp_db_path, create=True) as client:

        async def fake_vacuum(*_a, **_k):
            calls.append(1)

        monkeypatch.setattr(client.store, "vacuum", fake_vacuum)

        client._schedule_vacuum()  # schedules the first background pass
        client._schedule_vacuum()  # debounced (no task)

        await client._await_vacuum_tasks()
        # one scheduled background pass + one final collapse on drain
        assert len(calls) == 2
        assert client._vacuum_dirty is False


@pytest.mark.asyncio
async def test_metadata_refresh_sweep_schedules_vacuum(temp_db_path):
    """A source re-sweep that only rolls source_revision (MD5/revision
    short-circuit) writes document_meta and must still schedule the (debounced)
    vacuum, so that tiny churn gets reclaimed instead of accumulating."""
    dim = Config.embeddings.model.vector_dim
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.import_document(
            _docling_doc("d", "body"),
            [Chunk(content="body", embedding=[0.1] * dim, order=0)],
            uri="mem://sweep",
            metadata={"source_revision": "r1"},
        )
        # Isolate the refresh: the import already scheduled a vacuum.
        client._vacuum_dirty = False

        await _refresh_doc_metadata(
            client,
            doc,
            title=None,
            user_metadata={},
            source_metadata={"source_revision": "r2", "md5": "same"},
        )
        assert client._vacuum_dirty is True
