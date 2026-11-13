from haiku.rag.client import HaikuRAG
from haiku.rag.client.processing import chunk
from haiku.rag.config import AppConfig
from haiku.rag.embeddings import EmbedderWrapper
from haiku.rag.store.models.chunk import Chunk, ChunkMetadata
from haiku.rag.store.models.document_item import extract_items, time_range
from haiku.rag.store.repositories.document_item import DocumentItemRepository
from tests.documents import docling_doc, timed_docling_doc


class _TextEmbedder(EmbedderWrapper):
    def __init__(self):
        super().__init__(embedder=None, vector_dim=4)


class _MultimodalEmbedder(_TextEmbedder):
    supports_images = True


def test_time_range_spans_the_items():
    doc = timed_docling_doc()

    assert time_range(doc.texts) == (0.0, 12.12)
    assert time_range([doc.pictures[1]]) == (10.0, 10.001)


def test_time_range_is_none_without_track_sources():
    assert time_range(docling_doc("plain", "No timing here.").texts) is None
    assert time_range([]) is None


def test_extract_items_records_item_times():
    items = extract_items("doc-1", timed_docling_doc())

    assert [(i.self_ref, i.start_time, i.end_time) for i in items] == [
        ("#/pictures/0", 0.0, 0.001),
        ("#/texts/0", 0.0, 4.16),
        ("#/texts/1", 4.48, 7.6),
        ("#/texts/2", 7.92, 12.12),
        ("#/pictures/1", 10.0, 10.001),
    ]


def test_extract_items_leaves_untimed_items_without_times():
    [item] = extract_items("doc-1", docling_doc("plain", "No timing here."))

    assert (item.start_time, item.end_time) == (None, None)


async def test_item_times_survive_a_database_round_trip(temp_db_path):
    items = extract_items("doc-1", timed_docling_doc())
    async with HaikuRAG(temp_db_path, create=True) as rag:
        await DocumentItemRepository(rag.store).create_items("doc-1", items)

    async with HaikuRAG(temp_db_path) as rag:
        stored = (
            await DocumentItemRepository(rag.store).get_items_in_ranges(
                {"doc-1": (0, len(items) - 1)}
            )
        )["doc-1"]

    assert [(i.start_time, i.end_time) for i in stored] == [
        (i.start_time, i.end_time) for i in items
    ]


async def test_chunks_carry_the_time_range_of_their_items(monkeypatch):
    class Chunker:
        async def chunk(self, document):
            return [
                Chunk(content="a", metadata={"doc_item_refs": ["#/texts/0"]}),
                Chunk(
                    content="b",
                    metadata={"doc_item_refs": ["#/texts/1", "#/texts/2"]},
                ),
                Chunk(
                    content="c",
                    metadata={"doc_item_refs": ["#/texts/2", "#/pictures/1"]},
                ),
            ]

    monkeypatch.setattr("haiku.rag.chunkers.get_chunker", lambda *a, **kw: Chunker())

    chunks = await chunk(AppConfig(), timed_docling_doc(), embedder=_TextEmbedder())

    assert [ChunkMetadata.model_validate(c.metadata).time_range for c in chunks] == [
        (0.0, 4.16),
        (4.48, 12.12),
        (7.92, 12.12),
    ]


async def test_picture_chunks_carry_their_frame_time():
    chunks = await chunk(
        AppConfig(), timed_docling_doc(), embedder=_MultimodalEmbedder()
    )

    frames = [
        ChunkMetadata.model_validate(c.metadata)
        for c in chunks
        if c.metadata.get("labels") == ["picture"]
    ]
    assert [m.time_range for m in frames] == [(0.0, 0.001), (10.0, 10.001)]


async def test_untimed_chunks_have_no_time_range():
    chunks = await chunk(
        AppConfig(), docling_doc("plain", "No timing here."), embedder=_TextEmbedder()
    )

    assert [ChunkMetadata.model_validate(c.metadata).time_range for c in chunks] == [
        None
    ]
