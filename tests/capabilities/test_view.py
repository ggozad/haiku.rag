import hashlib
import re
import shutil
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from haiku.rag.capabilities.rag import RAGState
from haiku.rag.capabilities.rag import create_capability as create_rag
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from tests.capabilities.helpers import Deps
from tests.capabilities.test_media_tools import UUID, _config, images, returns, scripted


async def _film(
    path,
    frames: bool = True,
    original: bytes | None = None,
    filename: str = "f.mp4",
    mimetype: str = "video/mp4",
):
    """Ten ten-second sentences, each its own chunk, and a frame five seconds
    into each, each its own chunk. `original` is stored as the file the film
    was converted from, named `filename`."""
    from docling_core.types.doc import (
        ContentLayer,
        DocItemLabel,
        DocumentOrigin,
        ImageRef,
        TrackSource,
    )
    from docling_core.types.doc.document import DoclingDocument
    from PIL import Image

    from haiku.rag.config import get_config
    from haiku.rag.store.models.chunk import Chunk

    dim = get_config().embeddings.model.vector_dim
    doc = DoclingDocument(name="film")
    chunks = []
    for n, second in enumerate(range(0, 100, 10)):
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=f"Said at {second} seconds.",
            content_layer=ContentLayer.BODY,
            source=TrackSource(start_time=second, end_time=second + 9.5),
        )
        chunks.append(
            Chunk(
                content=f"Said at {second} seconds.",
                embedding=[0.1] * dim,
                order=len(chunks),
                metadata={
                    "doc_item_refs": [f"#/texts/{n}"],
                    "labels": ["text"],
                    "time_range": [second, second + 9.5],
                },
            )
        )
        if frames:
            doc.add_picture(
                image=ImageRef.from_pil(Image.new("RGB", (16, 16), "red"), dpi=72),
                content_layer=ContentLayer.BODY,
                source=TrackSource(start_time=second + 5, end_time=second + 5.001),
            )
            chunks.append(
                Chunk(
                    content="",
                    embedding=[0.1] * dim,
                    order=len(chunks),
                    metadata={
                        "doc_item_refs": [f"#/pictures/{n}"],
                        "labels": ["picture"],
                        "time_range": [second + 5, second + 5.001],
                    },
                )
            )
    digest = hashlib.sha256(original).hexdigest() if original else "1"
    doc.origin = DocumentOrigin(
        mimetype=mimetype,
        filename=filename,
        binary_hash=int(digest, 16) & (2**64 - 1),
    )
    async with HaikuRAG(path, create=True) as rag:
        stored = await rag.import_document(
            doc, chunks, uri="file:///film.mp4", original=original
        )
        await rag.create_document("A manual.", uri="file:///manual.md")
    assert stored.id is not None
    return stored.id


@pytest.fixture
async def film(tmp_path):
    path = tmp_path / "film.lancedb"
    return path, await _film(path)


async def view(
    path,
    calls: list[dict[str, Any]],
    config: AppConfig | None = None,
    deps: Deps | None = None,
):
    seen: list[AgentInfo] = []
    capability = create_rag(db_path=path, config=config or _config())
    agent = Agent(
        scripted([[("view", args)] for args in calls], seen),
        deps_type=Deps,
        capabilities=[capability],
    )
    deps = deps or Deps()
    result = await agent.run("question", deps=deps)
    return result, RAGState.model_validate(deps.state["rag"])


def _times(state: RAGState) -> list[tuple[float, str]]:
    """The registered results' start times and kinds, in time order."""
    return sorted(
        (r.time_range[0], "frame" if not r.content.count("Said") else "speech")
        for results in state.searches.values()
        for r in results
        if r.time_range
    )


async def test_a_window_returns_its_sentences_in_order_and_at_most_eight_frames(film):
    path, film_id = film

    result, state = await view(path, [{"recording": film_id, "start": 0, "end": 100}])

    [part] = returns(result, "view")
    said = re.findall(r"\[(\d\d:\d\d:\d\d)\] Said at", str(part.content))
    assert said == [f"00:0{s // 60}:{s % 60:02d}" for s in range(0, 100, 10)]
    frames = [at for at, kind in _times(state) if kind == "frame"]
    assert frames == [5.0, 15.0, 35.0, 45.0, 55.0, 65.0, 85.0, 95.0]
    assert len(images(result)) == 8


async def test_a_window_with_eight_frames_returns_every_one(film):
    path, film_id = film

    result, state = await view(path, [{"recording": film_id, "start": 0, "end": 80}])

    frames = [at for at, kind in _times(state) if kind == "frame"]
    assert frames == [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]
    assert len(images(result)) == 8


async def test_a_narrow_window_returns_only_what_overlaps_it(film):
    path, film_id = film

    _, state = await view(path, [{"recording": film_id, "start": 20, "end": 40}])

    assert _times(state) == [
        (20.0, "speech"),
        (25.0, "frame"),
        (30.0, "speech"),
        (35.0, "frame"),
    ]


async def test_a_long_window_stops_at_a_sentence_and_continues_from_there(film):
    path, film_id = film
    config = _config()
    config.search.max_context_chars = 70

    result, _ = await view(
        path,
        [
            {"recording": film_id, "start": 0, "end": 100},
            {"recording": film_id, "start": 19.5, "end": 100},
        ],
        config,
    )

    first, second = (str(p.content) for p in returns(result, "view"))
    assert re.findall(r"Said at (\d+)", first) == ["0", "10"]
    assert "continues at [00:00:19] (19.5 s)" in first
    assert re.findall(r"Said at (\d+)", second) == ["20", "30"]


async def test_a_sentence_longer_than_the_budget_is_returned_whole(film):
    path, film_id = film
    config = _config()
    config.search.max_context_chars = 5

    result, _ = await view(
        path, [{"recording": film_id, "start": 0, "end": 30}], config
    )

    [part] = returns(result, "view")
    assert re.findall(r"Said at (\d+)", str(part.content)) == ["0"]


async def test_cited_frames_carry_only_their_own_frame(film):
    path, film_id = film

    def model_function(messages, info: AgentInfo) -> ModelResponse:
        parts = [p for m in messages for p in m.parts]
        viewed = [
            p for p in parts if isinstance(p, ToolReturnPart) and p.tool_name == "view"
        ]
        if not viewed:
            args = {"recording": film_id, "start": 0, "end": 100}
            return ModelResponse(parts=[ToolCallPart("view", args)])
        if not any(
            isinstance(p, ToolReturnPart) and p.tool_name == "cite" for p in parts
        ):
            ids = re.findall(rf"\[({UUID})\]", str(viewed[0].content))
            return ModelResponse(parts=[ToolCallPart("cite", {"chunk_ids": ids})])
        return ModelResponse(parts=[TextPart("done")])

    deps = Deps()
    await Agent(
        FunctionModel(model_function),
        deps_type=Deps,
        capabilities=[create_rag(db_path=path, config=_config())],
    ).run("question", deps=deps)

    state = RAGState.model_validate(deps.state["rag"])
    framed = [c for c in state.citation_index.values() if c.picture_refs]
    assert len(framed) == 8
    assert all(len(c.picture_refs) == 1 for c in framed)


async def test_no_images_without_vision(film):
    path, film_id = film
    config = _config()
    config.qa.model.vision = False

    result, state = await view(
        path, [{"recording": film_id, "start": 0, "end": 100}], config
    )

    assert images(result) == []
    assert [kind for _, kind in _times(state)].count("frame") == 8


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"start": 50, "end": 50}, "must end after it starts"),
        ({"recording": "no-such-id"}, "No recording no-such-id"),
    ],
)
async def test_a_bad_view_fails_the_call(film, args, message):
    path, film_id = film

    result, _ = await view(
        path, [{"recording": film_id, "start": 0, "end": 10, **args}]
    )

    [part] = returns(result, "view")
    assert message in str(part.content)


async def test_a_recording_outside_the_filter_is_unknown(film):
    path, film_id = film
    deps = Deps(
        state={
            "rag": RAGState(document_filter="uri = 'file:///manual.md'").model_dump(
                mode="json"
            )
        }
    )

    result, _ = await view(
        path, [{"recording": film_id, "start": 0, "end": 10}], deps=deps
    )

    [part] = returns(result, "view")
    assert f"No recording {film_id}" in str(part.content)


async def test_a_view_pays_the_search_budget(film):
    path, film_id = film
    window = {"recording": film_id, "start": 0, "end": 10}

    result, _ = await view(path, [window, window], _config(max_searches=1))

    first, second = returns(result, "view")
    assert "Said at 0 seconds." in str(first.content)
    assert "Search limit reached" in str(second.content)


@pytest.fixture
async def two_copies(tmp_path):
    """One database holding the film, configured twice under two names."""
    film_id = await _film(tmp_path / "a.lancedb")
    shutil.copytree(tmp_path / "a.lancedb", tmp_path / "b.lancedb")
    config = _config()
    config.lancedb.databases = {
        "a": str(tmp_path / "a.lancedb"),
        "b": str(tmp_path / "b.lancedb"),
    }
    return config, film_id


async def test_a_recording_in_two_collections_needs_its_source(two_copies):
    config, film_id = two_copies
    window = {"recording": film_id, "start": 0, "end": 10}

    seen: list[AgentInfo] = []
    capability = create_rag(config=config)
    agent = Agent(
        scripted([[("view", window)], [("view", {**window, "source": "b"})]], seen),
        deps_type=Deps,
        capabilities=[capability],
    )
    result = await agent.run("question", deps=Deps())

    ambiguous, named = returns(result, "view")
    assert "in collections a, b" in str(ambiguous.content)
    assert "Collection: b" in str(named.content)


async def test_view_offers_source_only_over_several_collections(film, two_copies):
    path, _ = film
    config, _ = two_copies

    one: list[AgentInfo] = []
    several: list[AgentInfo] = []
    for db, cfg, seen in ((path, _config(), one), (None, config, several)):
        agent = Agent(
            scripted([], seen),
            deps_type=Deps,
            capabilities=[create_rag(db_path=db, config=cfg)],
        )
        await agent.run("question", deps=Deps())

    def view_params(seen):
        [tool] = [t for t in seen[0].function_tools if t.name == "view"]
        return set(tool.parameters_json_schema["properties"])

    assert view_params(one) == {"recording", "start", "end"}
    assert view_params(several) == {"recording", "start", "end", "source"}


async def _segments(path, spans: list[tuple[float, float]], chunks: list[list[int]]):
    """A recording of the given segments, chunked as `chunks` lists their indexes."""
    from docling_core.types.doc import (
        ContentLayer,
        DocItemLabel,
        DocumentOrigin,
        TrackSource,
    )
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.config import get_config
    from haiku.rag.store.models.chunk import Chunk

    dim = get_config().embeddings.model.vector_dim
    doc = DoclingDocument(name="talk")
    for n, (start, end) in enumerate(spans):
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=f"Segment {n} is said.",
            content_layer=ContentLayer.BODY,
            source=TrackSource(start_time=start, end_time=end),
        )
    doc.origin = DocumentOrigin(mimetype="audio/mpeg", filename="t.mp3", binary_hash=1)
    async with HaikuRAG(path, create=True) as rag:
        stored = await rag.import_document(
            doc,
            [
                Chunk(
                    content=" ".join(f"Segment {n} is said." for n in members),
                    embedding=[0.1] * dim,
                    order=order,
                    metadata={
                        "doc_item_refs": [f"#/texts/{n}" for n in members],
                        "labels": ["text"] * len(members),
                        "time_range": [spans[members[0]][0], spans[members[-1]][1]],
                    },
                )
                for order, members in enumerate(chunks)
            ],
            uri="file:///talk.mp3",
        )
    assert stored.id is not None
    return stored.id


async def test_a_sentence_in_two_chunks_is_shown_once(tmp_path):
    path = tmp_path / "shared.lancedb"
    talk = await _segments(path, [(0, 5), (5, 10), (10, 15)], [[0, 1], [1, 2]])

    result, _ = await view(path, [{"recording": talk, "start": 0, "end": 15}])

    [part] = returns(result, "view")
    assert str(part.content).count("Segment 1 is said.") == 1


async def test_reading_on_never_repeats_an_overlapping_segment(tmp_path):
    path = tmp_path / "overlap.lancedb"
    talk = await _segments(path, [(0, 2), (1, 3), (4, 6)], [[0], [1], [2]])
    config = _config()
    config.search.max_context_chars = 5

    result, _ = await view(
        path,
        [
            {"recording": talk, "start": 0, "end": 6},
            {"recording": talk, "start": 2, "end": 6},
        ],
        config,
    )

    first, second = (str(p.content) for p in returns(result, "view"))
    assert re.findall(r"Segment (\d)", first) == ["0"]
    assert "(2 s)" in first
    assert re.findall(r"Segment (\d)", second) == ["1"]


async def test_a_full_page_keeps_the_segments_nested_in_it(tmp_path):
    path = tmp_path / "nested.lancedb"
    talk = await _segments(path, [(0, 10), (1, 2), (11, 12)], [[0], [1], [2]])
    config = _config()
    config.search.max_context_chars = 5

    result, _ = await view(
        path,
        [
            {"recording": talk, "start": 0, "end": 12},
            {"recording": talk, "start": 10, "end": 12},
        ],
        config,
    )

    first, second = (str(p.content) for p in returns(result, "view"))
    assert re.findall(r"Segment (\d)", first) == ["0", "1"]
    assert "(10 s)" in first
    assert re.findall(r"Segment (\d)", second) == ["2"]


async def test_reading_on_names_the_time_to_the_millisecond(tmp_path):
    path = tmp_path / "touching.lancedb"
    talk = await _segments(path, [(1230.0, 1234.564), (1234.564, 1238.0)], [[0], [1]])
    config = _config()
    config.search.max_context_chars = 5

    result, _ = await view(
        path,
        [
            {"recording": talk, "start": 1230, "end": 1240},
            {"recording": talk, "start": 1234.564, "end": 1240},
        ],
        config,
    )

    first, second = (str(p.content) for p in returns(result, "view"))
    assert "(1234.564 s)" in first
    assert re.findall(r"Segment (\d)", second) == ["1"]
