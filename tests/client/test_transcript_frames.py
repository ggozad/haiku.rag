import base64

import pytest
from docling_core.types.doc import ContentLayer, ImageRef, TrackSource
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from PIL import Image

from haiku.rag.client import HaikuRAG
from haiku.rag.client.search import populate_image_data
from haiku.rag.store.models.chunk import Chunk, SearchResult
from haiku.rag.store.models.citation import resolve_citations
from haiku.rag.utils.images import build_image_content_from_results


def _frame(doc: DoclingDocument, at: float, shade: int) -> None:
    doc.add_picture(
        image=ImageRef.from_pil(Image.new("RGB", (16, 16), (shade, 0, 0)), dpi=72),
        content_layer=ContentLayer.BODY,
        source=TrackSource(start_time=at, end_time=at + 0.001),
    )


def _recording() -> DoclingDocument:
    """Two transcript items, 0-100 s and 100-130 s, with a frame every 10 s.
    The 120 s frame is placed first, out of time order."""
    doc = DoclingDocument(name="interview")
    _frame(doc, 120.0, 250)
    for n, at in enumerate(range(0, 100, 10)):
        _frame(doc, float(at), 10 + n)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="This is a lithium hydroxide can that is a square cube.",
        content_layer=ContentLayer.BODY,
        source=TrackSource(start_time=0.0, end_time=100.0),
    )
    _frame(doc, 100.0, 200)
    _frame(doc, 110.0, 210)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="We taped a bag over it with the cover of a flight plan.",
        content_layer=ContentLayer.BODY,
        source=TrackSource(start_time=100.0, end_time=130.0),
    )
    return doc


# self_ref of the frame at each time, in the order they were added
FRAME_AT = {120.0: "#/pictures/0"} | {
    float(at): f"#/pictures/{n + 1}" for n, at in enumerate(range(0, 100, 10))
}
FRAME_AT |= {100.0: "#/pictures/11", 110.0: "#/pictures/12"}

CHUNKS = {
    "first": (["#/texts/0"], [0.0, 100.0], ["text"]),
    "second": (["#/texts/1"], [100.0, 130.0], ["text"]),
    "described": (
        ["#/texts/0", *(FRAME_AT[float(at)] for at in range(0, 100, 10))],
        [0.0, 100.0],
        ["picture", "text"],
    ),
    "frame": ([FRAME_AT[30.0]], [30.0, 30.001], ["picture"]),
}


@pytest.fixture
async def recording(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        dim = rag.store.embedder.vector_dim
        doc = await rag.import_document(
            _recording(),
            [
                Chunk(
                    content=name,
                    embedding=[0.1] * dim,
                    order=order,
                    metadata={
                        "doc_item_refs": refs,
                        "time_range": times,
                        "labels": labels,
                    },
                )
                for order, (name, (refs, times, labels)) in enumerate(CHUNKS.items())
            ],
            uri="test://interview",
        )
        assert doc.id is not None
        chunks = await rag.chunk_repository.get_by_document_id(doc.id)
        yield rag, {c.content: c for c in chunks}


async def _results(recording, *names: str) -> list[SearchResult]:
    rag, chunks = recording
    results = [SearchResult.from_chunk(chunks[name], score=1.0) for name in names]
    await populate_image_data(rag, results)
    return results


def _times(result: SearchResult) -> list[float]:
    return sorted(result.picture_times[ref] for ref in result.image_data or {})


async def test_a_transcript_result_carries_frames_spread_over_its_time(recording):
    [result] = await _results(recording, "first")

    assert _times(result) == [0.0, 30.0, 60.0, 90.0]
    assert set(result.picture_times) == set(result.image_data or {})


async def test_frames_are_chosen_by_time_not_position(recording):
    [result] = await _results(recording, "second")

    assert _times(result) == [100.0, 110.0, 120.0]
    assert FRAME_AT[120.0] in (result.image_data or {})


async def test_described_frames_in_a_transcript_chunk_are_capped(recording):
    [result] = await _results(recording, "described")

    assert _times(result) == [0.0, 30.0, 60.0, 90.0]


async def test_a_frame_chunk_carries_only_its_frame(recording):
    [result] = await _results(recording, "frame")

    assert list(result.image_data or {}) == [FRAME_AT[30.0]]


async def test_a_merged_result_keeps_the_cap(recording):
    results = await _results(recording, "first", "second")

    [merged] = await recording[0].expand_context(results)

    assert len(merged.image_data or {}) == 4
    assert _times(merged) == [0.0, 60.0, 100.0, 120.0]


async def test_a_citation_records_the_frames_attached(recording):
    [result] = await _results(recording, "described")

    assert result.chunk_id is not None
    [citation] = resolve_citations([result.chunk_id], [result])

    assert sorted(citation.picture_refs) == sorted(result.image_data or {})


async def test_a_frame_is_labelled_by_its_time(recording):
    [result] = await _results(recording, "first")

    content, _ = build_image_content_from_results([result])

    labels = [part for part in content if isinstance(part, str)]
    assert labels[1].startswith("Video frame at 00:00:30, 2 of 4,")
    assert all("Page image" not in label for label in labels)


async def test_an_untimed_result_is_unchanged(temp_db_path):
    from tests.documents import docling_doc_with_picture

    async with HaikuRAG(temp_db_path, create=True) as rag:
        dim = rag.store.embedder.vector_dim
        doc = await rag.import_document(
            docling_doc_with_picture(),
            [
                Chunk(
                    content="Hello world",
                    embedding=[0.1] * dim,
                    metadata={"doc_item_refs": ["#/texts/0", "#/pictures/0"]},
                )
            ],
            uri="test://page",
        )
        assert doc.id is not None
        [chunk] = await rag.chunk_repository.get_by_document_id(doc.id)
        result = SearchResult.from_chunk(chunk, score=1.0)
        await populate_image_data(rag, [result])

    assert list(result.image_data or {}) == ["#/pictures/0"]
    assert result.picture_times == {}
    content, _ = build_image_content_from_results([result])
    assert isinstance(content[0], str)
    assert content[0].startswith("Page image 1 of 1")
    assert base64.b64decode((result.image_data or {})["#/pictures/0"])


async def test_a_frame_chunk_keeps_its_frame_beside_a_frame_at_the_same_time(
    temp_db_path,
):
    doc = DoclingDocument(name="burst")
    _frame(doc, 30.0, 10)
    _frame(doc, 30.0005, 20)
    async with HaikuRAG(temp_db_path, create=True) as rag:
        dim = rag.store.embedder.vector_dim
        stored = await rag.import_document(
            doc,
            [
                Chunk(
                    content="first",
                    embedding=[0.1] * dim,
                    metadata={
                        "doc_item_refs": ["#/pictures/0"],
                        "time_range": [30.0, 30.001],
                        "labels": ["picture"],
                    },
                )
            ],
            uri="test://burst",
        )
        assert stored.id is not None
        [chunk] = await rag.chunk_repository.get_by_document_id(stored.id)
        result = SearchResult.from_chunk(chunk, score=1.0)
        await populate_image_data(rag, [result])

    assert list(result.image_data or {}) == ["#/pictures/0"]


def test_a_split_result_carries_only_its_own_frames():
    from haiku.rag.context import expand_with_items
    from haiku.rag.store.models.document_item import DocumentItem

    texts = ["LOW " + "a" * 400, "b" * 400, "c" * 400 + " HIGH"]
    items = [
        DocumentItem(
            document_id="doc-1",
            position=n,
            self_ref=f"#/texts/{n}",
            label="text",
            text=text,
            start_time=60.0 * n,
            end_time=60.0 * (n + 1),
        )
        for n, text in enumerate(texts)
    ]

    def timed(n: int, score: float) -> SearchResult:
        frame = f"#/pictures/{n}"
        return SearchResult(
            content=texts[n],
            score=score,
            chunk_id=f"c-{n}",
            document_id="doc-1",
            doc_item_refs=[f"#/texts/{n}"],
            time_range=(60.0 * n, 60.0 * (n + 1)),
            image_data={frame: f"bytes {n}"},
            picture_captions={frame: f"caption {n}"},
            picture_times={frame: 60.0 * n + 10.0},
        )

    expanded = expand_with_items(
        [timed(0, 0.4), timed(2, 0.9)], 500, {"#/texts/0": 0, "#/texts/2": 2}, items
    )

    assert {e.chunk_id for e in expanded} == {"c-0", "c-2"}
    for e in expanded:
        assert e.chunk_id is not None
        n = int(e.chunk_id[-1])
        frame = f"#/pictures/{n}"
        assert e.image_data == {frame: f"bytes {n}"}
        assert e.picture_captions == {frame: f"caption {n}"}
        assert e.picture_times == {frame: 60.0 * n + 10.0}
