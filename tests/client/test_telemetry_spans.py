import asyncio

import logfire
import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from logfire.testing import SimpleSpanProcessor, TestExporter

from haiku.rag.client import HaikuRAG
from haiku.rag.client.documents import DocumentImport
from haiku.rag.config import get_config
from haiku.rag.embeddings import EmbedderWrapper
from haiku.rag.store.models.chunk import Chunk
from tests.locks import ObservedLock, assert_waiting_for_lock


class _StubEmbedder(EmbedderWrapper):
    """A constant vector, so no HTTP call and no cassette."""

    def __init__(self, vector_dim: int):
        super().__init__(None, vector_dim)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.vector_dim for _ in texts]


def _docling_doc(name: str, text: str) -> DoclingDocument:
    doc = DoclingDocument(name=name)
    doc.add_text(label=DocItemLabel.TEXT, text=text)
    return doc


@pytest.fixture
def exporter():
    """Collect spans in memory, restoring logfire's inert default on teardown."""
    test_exporter = TestExporter()
    logfire.configure(
        send_to_logfire=False,
        console=False,
        additional_span_processors=[SimpleSpanProcessor(test_exporter)],
    )
    yield test_exporter
    logfire.configure(send_to_logfire=False, console=False)


class Tree:
    """The exported spans, indexed for parent/child questions."""

    def __init__(self, exporter: TestExporter):
        # logfire exports a pending span when one opens and the real span when
        # it closes; only the latter carries final attributes.
        self.spans = [
            span
            for span in exporter.exported_spans
            if (span.attributes or {}).get("logfire.span_type") != "pending_span"
        ]
        self._names = {
            span.context.span_id: span.name for span in self.spans if span.context
        }

    def names(self) -> list[str]:
        return [span.name for span in self.spans]

    def one(self, name: str):
        matches = [span for span in self.spans if span.name == name]
        assert len(matches) == 1, f"expected exactly one {name!r}, got {len(matches)}"
        return matches[0]

    def parent_of(self, name: str) -> str | None:
        span = self.one(name)
        if span.parent is None:
            return None
        return self._names.get(span.parent.span_id)

    def attrs(self, name: str) -> dict:
        return dict(self.one(name).attributes or {})


async def test_embed_items_and_store_are_siblings(temp_db_path, exporter):
    """No phase's duration contains another's."""
    dim = get_config().embeddings.model.vector_dim

    async with HaikuRAG(temp_db_path, create=True) as client:
        client.store.embedder = _StubEmbedder(dim)
        with logfire.span("test.parent"):
            await client.import_document(
                _docling_doc("a", "Alpha document body"),
                [Chunk(content="Alpha document body", order=0)],
                uri="mem://a",
            )

    tree = Tree(exporter)
    assert tree.parent_of("document.embed") == "test.parent"
    assert tree.parent_of("document.items") == "test.parent"
    assert tree.parent_of("document.store") == "test.parent"


async def test_embed_span_counts_what_it_embedded(temp_db_path, exporter):
    dim = get_config().embeddings.model.vector_dim

    async with HaikuRAG(temp_db_path, create=True) as client:
        client.store.embedder = _StubEmbedder(dim)
        await client.import_document(
            _docling_doc("a", "Alpha document body"),
            [
                Chunk(content="Alpha document body", order=0),
                Chunk(content="Beta document body", order=1),
            ],
            uri="mem://a",
        )

    attrs = Tree(exporter).attrs("document.embed")
    assert attrs["chunks"] == 2
    assert attrs["chunks_embedded"] == 2
    assert attrs["images"] == 0
    assert attrs["batch_size"] == get_config().embeddings.batch_size


async def test_embed_span_is_emitted_when_nothing_needs_embedding(
    temp_db_path, exporter
):
    """Pre-embedded chunks still get the phase, with `chunks_embedded=0`."""
    dim = get_config().embeddings.model.vector_dim

    async with HaikuRAG(temp_db_path, create=True) as client:
        client.store.embedder = _StubEmbedder(dim)
        await client.import_document(
            _docling_doc("a", "Alpha document body"),
            [Chunk(content="Alpha document body", embedding=[0.1] * dim, order=0)],
            uri="mem://a",
        )

    attrs = Tree(exporter).attrs("document.embed")
    assert attrs["chunks"] == 1
    assert attrs["chunks_embedded"] == 0


async def test_store_span_reports_create(temp_db_path, exporter):
    dim = get_config().embeddings.model.vector_dim

    async with HaikuRAG(temp_db_path, create=True) as client:
        client.store.embedder = _StubEmbedder(dim)
        doc = await client.import_document(
            _docling_doc("a", "Alpha document body"),
            [Chunk(content="Alpha document body", order=0)],
            uri="mem://a",
        )

    attrs = Tree(exporter).attrs("document.store")
    assert attrs["op"] == "create"
    assert attrs["document_id"] == doc.id
    assert attrs["uri"] == "mem://a"
    assert attrs["chunks"] == 1


async def test_store_span_reports_update_when_the_uri_is_already_taken(
    temp_db_path, exporter
):
    """The create path resolves create-vs-update under the write lock."""
    dim = get_config().embeddings.model.vector_dim

    async with HaikuRAG(temp_db_path, create=True) as client:
        client.store.embedder = _StubEmbedder(dim)
        first = await client.import_document(
            _docling_doc("a", "Alpha document body"),
            [Chunk(content="Alpha document body", order=0)],
            uri="mem://a",
        )
        exporter.clear()
        await client.import_document(
            _docling_doc("a", "Alpha document body, revised"),
            [Chunk(content="Alpha document body, revised", order=0)],
            uri="mem://a",
        )

    attrs = Tree(exporter).attrs("document.store")
    assert attrs["op"] == "update"
    assert attrs["document_id"] == first.id


async def test_update_path_store_span(temp_db_path, exporter):
    dim = get_config().embeddings.model.vector_dim

    async with HaikuRAG(temp_db_path, create=True) as client:
        client.store.embedder = _StubEmbedder(dim)
        doc = await client.import_document(
            _docling_doc("a", "Alpha document body"),
            [Chunk(content="Alpha document body", order=0)],
            uri="mem://a",
        )
        exporter.clear()
        await client.update_document(
            doc.id,
            chunks=[Chunk(content="Revised body", order=0)],
            docling_document=_docling_doc("a", "Revised body"),
        )

    tree = Tree(exporter)
    attrs = tree.attrs("document.store")
    assert attrs["op"] == "update"
    assert attrs["document_id"] == doc.id
    assert tree.parent_of("document.embed") != "document.store"


async def test_update_without_a_docling_document_skips_the_items_phase(
    temp_db_path, exporter
):
    """Existing items are preserved, so nothing is extracted."""
    dim = get_config().embeddings.model.vector_dim

    async with HaikuRAG(temp_db_path, create=True) as client:
        client.store.embedder = _StubEmbedder(dim)
        doc = await client.import_document(
            _docling_doc("a", "Alpha document body"),
            [Chunk(content="Alpha document body", order=0)],
            uri="mem://a",
        )
        exporter.clear()
        await client.update_document(
            doc.id, chunks=[Chunk(content="Revised body", order=0)]
        )

    tree = Tree(exporter)
    assert "document.items" not in tree.names()
    assert "items" not in tree.attrs("document.store")


async def test_batch_store_span(temp_db_path, exporter):
    """`import_documents` writes each table once, so the batch gets one span."""
    dim = get_config().embeddings.model.vector_dim

    async with HaikuRAG(temp_db_path, create=True) as client:
        client.store.embedder = _StubEmbedder(dim)
        await client.import_documents(
            [
                DocumentImport(
                    docling_document=_docling_doc(name, text),
                    chunks=[Chunk(content=text, order=0)],
                    uri=f"mem://{name}",
                )
                for name, text in (("a", "Alpha body"), ("b", "Beta body"))
            ]
        )

    tree = Tree(exporter)
    attrs = tree.attrs("document.store")
    assert attrs["op"] == "create_batch"
    assert attrs["documents"] == 2
    assert attrs["chunks"] == 2
    assert tree.attrs("document.items")["documents"] == 2


async def test_items_span_counts_what_it_extracted(temp_db_path, exporter):
    dim = get_config().embeddings.model.vector_dim

    async with HaikuRAG(temp_db_path, create=True) as client:
        client.store.embedder = _StubEmbedder(dim)
        await client.import_document(
            _docling_doc("a", "Alpha document body"),
            [Chunk(content="Alpha document body", order=0)],
            uri="mem://a",
        )

    attrs = Tree(exporter).attrs("document.items")
    assert attrs["uri"] == "mem://a"
    assert attrs["pictures"] == 0
    assert attrs["items"] == 1


async def test_spans_carry_the_haiku_rag_scope(temp_db_path, exporter):
    dim = get_config().embeddings.model.vector_dim

    async with HaikuRAG(temp_db_path, create=True) as client:
        client.store.embedder = _StubEmbedder(dim)
        await client.import_document(
            _docling_doc("a", "Alpha document body"),
            [Chunk(content="Alpha document body", order=0)],
            uri="mem://a",
        )

    tree = Tree(exporter)
    for name in ("document.embed", "document.items", "document.store"):
        assert tree.one(name).instrumentation_scope.name == "haiku.rag"


async def test_ingest_phases_report_the_fetched_uri(temp_db_path, tmp_path, exporter):
    """Every phase span of an ingest, on the create and the update path, reports
    the fetched uri, not the uri the document is stored under."""
    dim = get_config().embeddings.model.vector_dim
    source = tmp_path / "alpha.md"
    source.write_text("# Alpha\n\nAlpha document body.\n", encoding="utf-8")

    async with HaikuRAG(temp_db_path, create=True) as client:
        client.store.embedder = _StubEmbedder(dim)
        doc = await client.create_document_from_source(source, uri="mem://override")

        tree = Tree(exporter)
        convert_uri = tree.attrs("document.convert")["uri"]
        assert doc.uri == "mem://override"
        assert convert_uri != doc.uri
        assert tree.attrs("document.store")["op"] == "create"
        assert tree.attrs("document.store")["uri"] == convert_uri
        assert tree.attrs("document.items")["uri"] == convert_uri

        source.write_text(
            "# Alpha\n\nAlpha document body, revised.\n", encoding="utf-8"
        )
        exporter.clear()
        updated = await client.create_document_from_source(source, uri="mem://override")

    tree = Tree(exporter)
    assert updated.id == doc.id
    assert tree.attrs("document.store")["op"] == "update"
    assert tree.attrs("document.store")["uri"] == convert_uri
    assert tree.attrs("document.items")["uri"] == convert_uri


async def test_store_span_reports_time_waiting_for_the_write_lock(
    temp_db_path, exporter, monkeypatch
):
    dim = get_config().embeddings.model.vector_dim

    async with HaikuRAG(temp_db_path, create=True) as client:
        client.store.embedder = _StubEmbedder(dim)
        lock = ObservedLock()
        monkeypatch.setattr(client.store, "_write_lock", lock)

        async with lock:
            write = asyncio.create_task(
                client.import_document(
                    _docling_doc("a", "Alpha document body"),
                    [Chunk(content="Alpha document body", order=0)],
                    uri="mem://a",
                )
            )
            await assert_waiting_for_lock(write, lock)
        await write

    assert Tree(exporter).attrs("document.store")["lock_wait_ms"] > 0
