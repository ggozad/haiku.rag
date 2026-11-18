import pytest
from docling_core.types.doc import DocumentOrigin

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, LanceDBConfig, get_config
from haiku.rag.embeddings import EmbedderWrapper
from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk, TimeWindow
from tests.documents import docling_doc, timed_docling_doc

SEGMENTS = [
    ("#/texts/0", 0.0, 4.16),
    ("#/texts/1", 4.48, 7.6),
    ("#/texts/2", 7.92, 12.12),
]
FRAMES = [("#/pictures/0", 0.0, 0.001), ("#/pictures/1", 10.0, 10.001)]


@pytest.fixture(autouse=True)
def query_embedding(monkeypatch):
    async def embed_query(self, text):
        return [0.1] * get_config().embeddings.model.vector_dim

    monkeypatch.setattr(EmbedderWrapper, "embed_query", embed_query)


class _TextReranker(RerankerBase):
    """A text reranker: it drops chunks with no text, as every shipped one does."""

    async def _rerank(self, query, chunks, top_n=10):
        return [(c, 1.0) for c in chunks][:top_n]


def _with_reranker(rag: HaikuRAG) -> None:
    rag.__dict__["_own_reranker"] = _TextReranker()


async def _seed(rag: HaikuRAG) -> str:
    """A recording, one chunk per segment and per frame, and a PDF."""
    dim = get_config().embeddings.model.vector_dim
    video = timed_docling_doc()
    video.origin = DocumentOrigin(
        mimetype="video/mp4", filename="talk.mp4", binary_hash=1
    )
    texts = {t.self_ref: t.text for t in video.texts}
    chunks = [
        Chunk(
            content=texts[ref],
            embedding=[0.1] * dim,
            order=i,
            metadata={
                "doc_item_refs": [ref],
                "labels": ["text"],
                "time_range": [start, end],
            },
        )
        for i, (ref, start, end) in enumerate(SEGMENTS)
    ] + [
        Chunk(
            content="",
            embedding=[0.1] * dim,
            order=len(SEGMENTS) + i,
            metadata={
                "doc_item_refs": [ref],
                "labels": ["picture"],
                "time_range": [start, end],
            },
        )
        for i, (ref, start, end) in enumerate(FRAMES)
    ]
    stored = await rag.import_document(video, chunks, uri="file:///talk.mp4")
    pdf = docling_doc("paper", "The oxygen tank failed.")
    await rag.import_document(
        pdf,
        [
            Chunk(
                content="The oxygen tank failed.",
                embedding=[0.1] * dim,
                order=0,
                metadata={"doc_item_refs": ["#/texts/0"], "labels": ["text"]},
            )
        ],
        uri="file:///paper.pdf",
        title="Paper",
    )
    assert stored.id is not None
    return stored.id


def _kinds(results) -> set[tuple[str, ...]]:
    return {tuple(r.doc_item_refs) for r in results}


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("documents", {("pdf", "#/texts/0")}),
        ("speech", {("video", ref) for ref, _, _ in SEGMENTS}),
        ("frames", {("video", ref) for ref, _, _ in FRAMES}),
    ],
)
async def test_a_search_returns_only_the_content_it_asks_for(
    temp_db_path, content, expected
):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        video_id = await _seed(rag)

        results = await rag.search(
            "oxygen", limit=10, search_type="vector", content=content
        )

        assert {
            ("video" if r.document_id == video_id else "pdf", *r.doc_item_refs)
            for r in results
        } == expected


async def test_a_search_without_content_returns_everything(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        await _seed(rag)

        results = await rag.search("oxygen", limit=10, search_type="vector")

        assert len(results) == len(SEGMENTS) + len(FRAMES) + 1


async def test_a_window_returns_only_chunks_overlapping_it(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        video_id = await _seed(rag)
        window = TimeWindow(video_id, start=5.0, end=8.0)

        speech = await rag.search(
            "crew", limit=10, search_type="vector", content="speech", window=window
        )
        frames = await rag.search(
            "crew", limit=10, search_type="vector", content="frames", window=window
        )

        assert _kinds(speech) == {("#/texts/1",), ("#/texts/2",)}
        assert frames == []


async def test_a_window_without_times_covers_the_whole_recording(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        video_id = await _seed(rag)

        results = await rag.search(
            "crew",
            limit=10,
            search_type="vector",
            content="frames",
            window=TimeWindow(video_id),
        )

        assert _kinds(results) == {(ref,) for ref, _, _ in FRAMES}


@pytest.mark.parametrize(("start", "end"), [(8.0, 8.0), (8.0, 5.0)])
def test_a_window_must_end_after_it_starts(start, end):
    with pytest.raises(ValueError, match="end after it starts"):
        TimeWindow("doc", start=start, end=end)


async def test_expanding_a_windowed_result_keeps_to_the_window(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        video_id = await _seed(rag)
        window = TimeWindow(video_id, start=5.0, end=7.0)
        [result] = await rag.search(
            "crew", limit=10, search_type="vector", content="speech", window=window
        )

        [unclipped] = await rag.expand_context([result])
        [clipped] = await rag.expand_context([result], window=window)

        assert "oxygen tank" in unclipped.content
        assert clipped.content == "[00:00:04] The crew moved into the lunar module."
        assert clipped.time_range == (4.48, 7.6)


async def test_expansion_drops_a_result_with_nothing_in_the_window(
    temp_db_path,
):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        video_id = await _seed(rag)
        [result] = await rag.search(
            "crew",
            limit=10,
            search_type="vector",
            content="speech",
            window=TimeWindow(video_id, start=5.0, end=7.0),
        )

        expanded = await rag.expand_context(
            [result], window=TimeWindow(video_id, start=100.0, end=200.0)
        )

        assert expanded == []


async def test_a_frame_search_under_a_text_reranker_ranks_frames_by_vector(
    temp_db_path,
):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        await _seed(rag)
        _with_reranker(rag)

        frames = await rag.search("crew", limit=10, content="frames")
        speech = await rag.search("crew", limit=10, content="speech")

        assert _kinds(frames) == {(ref,) for ref, _, _ in FRAMES}
        assert [r.score for r in speech] == [1.0] * len(SEGMENTS)


def _two_databases(tmp_path) -> AppConfig:
    return AppConfig(
        lancedb=LanceDBConfig(
            databases={n: str(tmp_path / f"{n}.lancedb") for n in ("a", "b")}
        )
    )


async def test_a_federated_search_returns_only_the_content_it_asks_for(tmp_path):
    config = _two_databases(tmp_path)
    for name in ("a", "b"):
        async with HaikuRAG(config=config, create=True, sources=[name]) as rag:
            await _seed(rag)

    async with HaikuRAG(config=config) as rag:
        _with_reranker(rag)
        frames = await rag.search("crew", limit=10, content="frames")
        documents = await rag.search("crew", limit=10, content="documents")

    assert sorted((r.source, r.doc_item_refs[0]) for r in frames) == [
        (name, ref) for name in ("a", "b") for ref, _, _ in FRAMES
    ]
    assert sorted(r.source for r in documents) == ["a", "b"]


async def test_list_recordings_lists_each_recording_once(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        video_id = await _seed(rag)

        [recording] = await rag.list_recordings()

    assert recording.document_id == video_id
    assert recording.uri == "file:///talk.mp4"
    assert recording.last_time == 12.12
    assert recording.has_frames
    assert recording.source == rag.source


async def test_list_recordings_marks_audio_without_frames(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        dim = get_config().embeddings.model.vector_dim
        await rag.chunk_repository.create(
            Chunk(
                document_id="audio-1",
                content="Hello.",
                metadata={"doc_item_refs": ["#/texts/0"], "time_range": [0.0, 2.0]},
                embedding=[0.1] * dim,
            )
        )

        [recording] = await rag.list_recordings()

    assert (recording.document_id, recording.has_frames) == ("audio-1", False)
    assert (recording.title, recording.uri) == (None, None)


async def test_list_recordings_honours_the_document_filter(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        await _seed(rag)

        assert await rag.list_recordings(filter="uri = 'file:///paper.pdf'") == []
        assert await rag.list_recordings(filter="uri = 'file:///missing'") == []
        assert await rag.list_recordings(sources=[]) == []
        assert len(await rag.list_recordings(filter="uri LIKE '%talk%'")) == 1


async def test_list_recordings_covers_every_database(tmp_path):
    config = _two_databases(tmp_path)
    for name in ("a", "b"):
        async with HaikuRAG(config=config, create=True, sources=[name]) as rag:
            await _seed(rag)

    async with HaikuRAG(config=config) as rag:
        recordings = await rag.list_recordings()
        only_b = await rag.list_recordings(sources=["b"])

    assert sorted(r.source for r in recordings) == ["a", "b"]
    assert [r.source for r in only_b] == ["b"]


async def _seed_spanning_chunk(rag: HaikuRAG) -> str:
    """A recording whose one speech chunk spans all three segments."""
    dim = get_config().embeddings.model.vector_dim
    video = timed_docling_doc()
    video.origin = DocumentOrigin(
        mimetype="video/mp4", filename="talk.mp4", binary_hash=1
    )
    stored = await rag.import_document(
        video,
        [
            Chunk(
                content=" ".join(t.text for t in video.texts),
                embedding=[0.1] * dim,
                order=0,
                metadata={
                    "doc_item_refs": [ref for ref, _, _ in SEGMENTS],
                    "labels": ["text"] * 3,
                    "time_range": [0.0, 12.12],
                },
            )
        ],
        uri="file:///talk.mp4",
    )
    assert stored.id is not None
    return stored.id


async def _windowed(rag: HaikuRAG, window: TimeWindow):
    results = await rag.search(
        "crew", limit=10, search_type="vector", content="speech", window=window
    )
    return results, await rag.expand_context(results, window=window)


async def test_a_windowed_chunk_spanning_several_segments_expands_to_those_inside(
    temp_db_path,
):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        video_id = await _seed_spanning_chunk(rag)

        [result], [expanded] = await _windowed(
            rag, TimeWindow(video_id, start=5.0, end=7.0)
        )

        assert set(result.picture_times) == {"#/pictures/0", "#/pictures/1"}
        assert expanded.content == "[00:00:04] The crew moved into the lunar module."
        assert expanded.time_range == (4.48, 7.6)
        assert (expanded.image_data, expanded.picture_times) == (None, {})


async def test_a_windowed_chunk_keeps_only_the_frames_inside_the_window(
    temp_db_path,
):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        video_id = await _seed_spanning_chunk(rag)

        _, [expanded] = await _windowed(rag, TimeWindow(video_id, start=9.0, end=11.0))

        assert expanded.content.splitlines() == [
            "[00:00:07] Engineers improvised a carbon dioxide filter.",
            "[00:00:10] (frame)",
        ]
        assert set(expanded.image_data or {}) == {"#/pictures/1"}
        assert set(expanded.picture_times) == {"#/pictures/1"}


async def test_a_window_in_a_gap_between_segments_drops_the_chunk(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        video_id = await _seed_spanning_chunk(rag)

        [_], expanded = await _windowed(rag, TimeWindow(video_id, start=4.2, end=4.4))

        assert expanded == []


async def test_after_search_sees_the_vector_search_a_frame_search_ran(temp_db_path):
    from haiku.rag.hooks import Hook

    seen = []

    class _AskForFullText(Hook):
        async def before_search(self, client, request):
            request.search_type = "fts"
            return request

        async def after_search(self, client, request, results):
            seen.append(request.search_type)
            return results

    async with HaikuRAG(temp_db_path, create=True) as rag:
        await _seed(rag)
        rag._hooks = [_AskForFullText()]

        results = await rag.search("crew", limit=10, content="frames")

    assert seen == ["vector"]
    assert _kinds(results) == {(ref,) for ref, _, _ in FRAMES}


async def test_a_windowed_expansion_drops_items_outside_the_window_in_reading_order(
    temp_db_path,
):
    from docling_core.types.doc import ContentLayer, DocItemLabel, TrackSource
    from docling_core.types.doc.document import DoclingDocument

    dim = get_config().embeddings.model.vector_dim
    doc = DoclingDocument(name="talk")
    for start, end, text in (
        (20.0, 25.0, "Inside the window."),
        (0.0, 5.0, "Long before it."),
        (21.0, 24.0, "Also inside."),
    ):
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=text,
            content_layer=ContentLayer.BODY,
            source=TrackSource(start_time=start, end_time=end),
        )
    doc.origin = DocumentOrigin(mimetype="audio/mpeg", filename="t.mp3", binary_hash=1)
    async with HaikuRAG(temp_db_path, create=True) as rag:
        stored = await rag.import_document(
            doc,
            [
                Chunk(
                    content="Inside the window.",
                    embedding=[0.1] * dim,
                    order=0,
                    metadata={
                        "doc_item_refs": ["#/texts/0"],
                        "labels": ["text"],
                        "time_range": [20.0, 25.0],
                    },
                )
            ],
            uri="file:///t.mp3",
        )
        assert stored.id is not None
        window = TimeWindow(stored.id, start=19.0, end=26.0)
        [result] = await rag.search(
            "inside", limit=10, search_type="vector", content="speech", window=window
        )

        [expanded] = await rag.expand_context([result], window=window)

        assert "Long before it." not in expanded.content
        assert "Also inside." in expanded.content


@pytest.mark.parametrize("search_type", [None, "hybrid", "fts"])
async def test_a_frame_search_is_vector_only(temp_db_path, search_type):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        await _seed(rag)

        results = await rag.search(
            "crew", limit=10, search_type=search_type, content="frames"
        )

        assert _kinds(results) == {(ref,) for ref, _, _ in FRAMES}


@pytest.mark.parametrize(
    ("span", "overlaps"),
    [
        ((None, None), False),
        ((0.0, 5.0), False),
        ((4.0, 6.0), True),
        ((9.0, 12.0), True),
        ((10.0, 11.0), False),
    ],
)
def test_a_window_overlaps_spans_that_share_time_with_it(span, overlaps):
    assert TimeWindow("doc", start=5.0, end=10.0).overlaps(*span) is overlaps


async def test_a_clipped_transcript_keeps_its_late_match_within_the_budget(tmp_path):
    from docling_core.types.doc import ContentLayer, DocItemLabel, TrackSource
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.config import AppConfig

    config = AppConfig()
    config.search.max_context_chars = 300
    dim = config.embeddings.model.vector_dim
    doc = DoclingDocument(name="long")
    for i in range(60):
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=f"Line {i} spoken here.",
            content_layer=ContentLayer.BODY,
            source=TrackSource(start_time=i * 5.0, end_time=i * 5.0 + 4.5),
        )
    doc.origin = DocumentOrigin(mimetype="audio/mpeg", filename="l.mp3", binary_hash=1)
    async with HaikuRAG(tmp_path / "long.lancedb", config=config, create=True) as rag:
        await rag.import_document(
            doc,
            [
                Chunk(
                    content="Line 55 spoken here.",
                    embedding=[0.1] * dim,
                    order=0,
                    metadata={
                        "doc_item_refs": ["#/texts/55"],
                        "labels": ["text"],
                        "time_range": [275.0, 279.5],
                    },
                )
            ],
            uri="file:///long.mp3",
        )
        [result] = await rag.expand_context(await rag.search("Line 55", limit=1))

    lines = result.content.splitlines()
    assert "[00:04:35] Line 55 spoken here." in lines
    assert all(line.startswith("[00:") for line in lines)
    assert len(result.content) <= 300
