from docling_core.types.doc import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    ImageRef,
    TrackSource,
)
from PIL import Image

from haiku.rag.chunkers import get_chunker
from haiku.rag.chunkers.base import DocumentChunker
from haiku.rag.chunkers.transcript import TranscriptChunker
from haiku.rag.client.processing import chunk
from haiku.rag.config import AppConfig
from haiku.rag.embeddings import get_embedder
from haiku.rag.providers.picture_description import set_picture_descriptions
from haiku.rag.store.models.chunk import Chunk
from tests.documents import timed_docling_doc


def _recording(*segments: tuple[float, float, str]) -> DoclingDocument:
    doc = DoclingDocument(
        name="talk",
        origin=DocumentOrigin(
            filename="talk.mp3", mimetype="audio/mpeg", binary_hash="ab" * 32
        ),
    )
    for start, end, text in segments:
        _segment(doc, start, end, text)
    return doc


def _segment(doc: DoclingDocument, start: float, end: float, text: str) -> None:
    doc.add_text(
        label=DocItemLabel.TEXT,
        text=text,
        content_layer=ContentLayer.BODY,
        source=TrackSource(start_time=start, end_time=end),
    )


def _described_frame(doc: DoclingDocument, at: float, description: str) -> None:
    doc.add_picture(
        image=ImageRef.from_pil(Image.new("RGB", (8, 8)), dpi=72),
        content_layer=ContentLayer.BODY,
        source=TrackSource(start_time=at, end_time=at + 0.001),
    )
    set_picture_descriptions(doc, {doc.pictures[-1].self_ref: description})


def _config(**processing) -> AppConfig:
    return AppConfig.model_validate({"processing": processing})


async def _chunks(doc: DoclingDocument, **processing) -> list[tuple[str, list[str]]]:
    config = _config(**processing)
    chunks = await TranscriptChunker(get_chunker(config)).chunk(doc)
    return [(c.content, c.metadata["doc_item_refs"]) for c in chunks]


class _Returning(DocumentChunker):
    """A chunker answering with the given texts, whatever the document."""

    def __init__(self, *contents: str):
        self.contents = contents

    async def chunk(self, document):
        return [Chunk(content=c) for c in self.contents]


async def test_chunks_split_the_joined_transcript_between_sentences():
    doc = _recording(
        (0.0, 3.0, "The oxygen tank failed"),
        (3.0, 6.0, "fifty five hours in. The crew"),
        (6.0, 9.0, "moved into the lunar module. Engineers"),
        (9.0, 12.0, "improvised a carbon dioxide filter."),
    )

    chunks = await _chunks(doc, chunk_size=12)

    assert [content for content, _ in chunks] == [
        "The oxygen tank failed fifty five hours in.",
        "The crew moved into the lunar module.",
        "Engineers improvised a carbon dioxide filter.",
    ]
    assert [refs for _, refs in chunks] == [
        ["#/texts/0", "#/texts/1"],
        ["#/texts/1", "#/texts/2"],
        ["#/texts/2", "#/texts/3"],
    ]


async def test_a_chunk_with_rewritten_whitespace_maps_to_its_segments():
    doc = _recording((0.0, 2.0, "One two"), (2.0, 4.0, "three four"))

    chunks = await TranscriptChunker(_Returning("One\ntwo three", "four")).chunk(doc)

    assert [c.metadata["doc_item_refs"] for c in chunks] == [
        ["#/texts/0", "#/texts/1"],
        ["#/texts/1"],
    ]
    assert [c.metadata["labels"] for c in chunks] == [["text", "text"], ["text"]]


async def test_a_chunk_missing_from_the_transcript_fails():
    doc = _recording((0.0, 2.0, "One two"))

    try:
        await TranscriptChunker(_Returning("unrelated")).chunk(doc)
    except ValueError as error:
        assert "unrelated" in str(error)
    else:
        raise AssertionError("expected ValueError")


async def test_a_chunk_longer_than_the_cap_splits_between_segments():
    doc = _recording(
        (0.0, 10.0, "One."),
        (10.0, 20.0, "Two."),
        (20.0, 30.0, "Three."),
        (30.0, 40.0, "Four."),
    )

    assert await _chunks(doc) == [
        ("One. Two. Three.", ["#/texts/0", "#/texts/1", "#/texts/2"]),
        ("Four.", ["#/texts/3"]),
    ]


async def test_a_segment_longer_than_the_cap_is_a_chunk_alone():
    doc = _recording((0.0, 40.0, "Long."), (40.0, 45.0, "Short."))

    assert await _chunks(doc) == [
        ("Long.", ["#/texts/0"]),
        ("Short.", ["#/texts/1"]),
    ]


async def test_the_cap_measures_every_segment_of_a_chunk():
    doc = _recording((10.0, 20.0, "One."), (0.0, 5.0, "Two."), (20.0, 35.0, "Three."))

    assert await _chunks(doc) == [
        ("One. Two.", ["#/texts/0", "#/texts/1"]),
        ("Three.", ["#/texts/2"]),
    ]


async def test_untimed_segments_stay_in_one_chunk():
    doc = _recording()
    for text in ("One.", "Two.", "Three."):
        doc.add_text(
            label=DocItemLabel.TEXT, text=text, content_layer=ContentLayer.BODY
        )

    assert await _chunks(doc) == [
        ("One. Two. Three.", ["#/texts/0", "#/texts/1", "#/texts/2"])
    ]


async def test_a_split_chunk_keeps_the_text_it_starts_with():
    doc = _recording(
        (0.0, 20.0, "One two"), (20.0, 40.0, "three four"), (40.0, 50.0, "five")
    )

    chunks = await TranscriptChunker(_Returning("One", "two three four five")).chunk(
        doc
    )

    assert [(c.content, c.metadata["doc_item_refs"]) for c in chunks] == [
        ("One", ["#/texts/0"]),
        ("two", ["#/texts/0"]),
        ("three four five", ["#/texts/1", "#/texts/2"]),
    ]


async def test_frames_are_in_no_chunk_even_when_described():
    doc = _recording((0.0, 2.0, "The oxygen tank failed."))
    _described_frame(doc, 2.0, "A tank on a test stand.")
    _segment(doc, 2.0, 4.0, "The crew moved.")

    assert await _chunks(doc) == [
        ("The oxygen tank failed. The crew moved.", ["#/texts/0", "#/texts/1"])
    ]


async def test_a_blank_segment_is_in_no_chunk():
    doc = timed_docling_doc()
    _segment(doc, 12.0, 13.0, "   ")

    chunks = await _chunks(doc)

    refs = [r for _, rs in chunks for r in rs]
    assert refs == ["#/texts/0", "#/texts/1", "#/texts/2"]


async def test_a_recording_without_speech_has_no_chunks():
    doc = _recording()
    _described_frame(doc, 1.0, "A slide.")

    assert await _chunks(doc) == []
    assert await TranscriptChunker(get_chunker(_config())).chunk(None) == []


async def test_recordings_take_the_transcript_chunker_with_their_times():
    config = _config(chunk_size=5)
    doc = _recording(
        (0.0, 6.0, "First point here."),
        (6.0, 12.0, "Second point"),
        (12.0, 15.0, "here."),
    )

    chunks = await chunk(config, doc, embedder=get_embedder(config))

    assert [(c.content, c.metadata["time_range"]) for c in chunks] == [
        ("First point here.", [0.0, 6.0]),
        ("Second point here.", [6.0, 15.0]),
    ]


async def test_recordings_split_by_tokens_under_the_hierarchical_chunker():
    config = _config(chunk_size=5, chunker_type="hierarchical")
    doc = _recording((0.0, 6.0, "First point here."), (6.0, 12.0, "Second point here."))

    chunks = await chunk(config, doc, embedder=get_embedder(config))

    assert [c.content for c in chunks] == ["First point here.", "Second point here."]


async def test_documents_that_are_not_recordings_keep_the_configured_chunker():
    config = _config()
    doc = timed_docling_doc()

    chunks = await chunk(config, doc, embedder=get_embedder(config))
    hybrid = await get_chunker(config).chunk(doc)

    assert [c.content for c in chunks] == [c.content for c in hybrid]
