import re
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.run import AgentRunResult

from haiku.rag.capabilities.rag import RAGState
from haiku.rag.capabilities.rag import create_capability as create_rag
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from tests.capabilities.conftest import _seeded_vector
from tests.capabilities.helpers import Deps
from tests.client.test_media_search import FRAMES, SEGMENTS, _seed

MEDIA_TOOLS = {"search_audio", "search_video", "list_recordings"}
UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


@pytest.fixture
async def media_db(tmp_path):
    """A recording, one chunk per segment and per frame, and a PDF."""
    path = tmp_path / "media.lancedb"
    async with HaikuRAG(path, create=True) as rag:
        video_id = await _seed(rag)
    return path, video_id


def _config(media_tools: bool = True, max_searches: int = 5) -> AppConfig:
    config = AppConfig()
    config.qa.media_tools = media_tools
    config.qa.max_searches = max_searches
    config.qa.model.vision = True
    return config


def scripted(
    responses: list[list[tuple[str, dict[str, Any]]]], seen: list[AgentInfo]
) -> FunctionModel:
    """Emit each response's tool calls in turn, then answer, recording what
    the model was offered on every request."""
    sent = 0

    def model_function(_messages, info: AgentInfo) -> ModelResponse:
        nonlocal sent
        seen.append(info)
        sent += 1
        if sent <= len(responses):
            return ModelResponse(
                parts=[ToolCallPart(name, args) for name, args in responses[sent - 1]]
            )
        return ModelResponse(parts=[TextPart("done")])

    return FunctionModel(model_function)


async def run(
    db_path,
    responses: list[list[tuple[str, dict[str, Any]]]],
    config: AppConfig | None = None,
) -> tuple[AgentRunResult[Any], list[AgentInfo], RAGState]:
    seen: list[AgentInfo] = []
    capability = create_rag(db_path=db_path, config=config or _config())
    agent = Agent(scripted(responses, seen), deps_type=Deps, capabilities=[capability])
    deps = Deps()
    result = await agent.run("question", deps=deps)
    return result, seen, RAGState.model_validate(deps.state["rag"])


def returns(result: AgentRunResult[Any], name: str) -> list[ToolReturnPart]:
    return [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == name
    ]


def images(result: AgentRunResult[Any]) -> list[BinaryContent]:
    """Every image the run sent the model, wherever the request carried it."""
    return [
        item
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, list)
        for item in part.content
        if isinstance(item, BinaryContent)
    ]


def _anchors(state: RAGState, query: str) -> set[str]:
    """The refs of the chunks each result was matched on."""
    return {ref for r in state.searches[query] for ref in r.chunk_meta["doc_item_refs"]}


async def test_without_media_tools_search_covers_recordings(media_db):
    db_path, video_id = media_db

    result, seen, state = await run(
        db_path, [[("search", {"query": "crew"})]], _config(media_tools=False)
    )

    assert not MEDIA_TOOLS & {tool.name for tool in seen[0].function_tools}
    assert "search_audio" not in (seen[0].instructions or "")
    assert video_id in {r.document_id for r in state.searches["crew"]}


async def test_media_tools_and_their_instructions_appear_under_qa_media_tools(
    media_db,
):
    db_path, _ = media_db

    _, seen, _ = await run(db_path, [])

    assert MEDIA_TOOLS <= {tool.name for tool in seen[0].function_tools}
    assert "search_audio" in (seen[0].instructions or "")


async def test_search_leaves_recordings_out_under_qa_media_tools(media_db):
    db_path, video_id = media_db

    _, _, state = await run(db_path, [[("search", {"query": "crew"})]])

    assert state.searches["crew"]
    assert video_id not in {r.document_id for r in state.searches["crew"]}


async def test_search_audio_returns_speech_without_frames(media_db):
    db_path, video_id = media_db

    result, _, state = await run(db_path, [[("search_audio", {"query": "crew"})]])

    [part] = returns(result, "search_audio")
    assert images(result) == []
    assert "Time:" in str(part.content)
    assert all(r.document_id == video_id for r in state.searches["crew"])
    assert _anchors(state, "crew") <= {ref for ref, _, _ in SEGMENTS}
    assert all(not r.image_data for r in state.searches["crew"])


async def test_search_video_returns_frames_as_images(media_db):
    db_path, _ = media_db

    result, _, state = await run(db_path, [[("search_video", {"query": "a door"})]])

    assert returns(result, "search_video")
    assert images(result)
    assert _anchors(state, "a door") <= {ref for ref, _, _ in FRAMES}


async def test_a_window_narrows_a_media_search(media_db):
    db_path, video_id = media_db
    window = {"recording": video_id, "start": 5.0, "end": 7.0}

    _, _, state = await run(db_path, [[("search_audio", {"query": "crew", **window})]])

    assert {r.time_range for r in state.searches["crew"]} == {(4.48, 7.6)}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"start": 5.0}, "needs a recording"),
        ({"recording": "x", "start": 7.0, "end": 5.0}, "end after it starts"),
    ],
)
async def test_a_bad_window_fails_the_call(media_db, args, message):
    db_path, _ = media_db

    result, _, _ = await run(db_path, [[("search_audio", {"query": "crew", **args})]])

    [part] = returns(result, "search_audio")
    assert part.outcome == "failed"
    assert message in str(part.content)


async def test_one_response_mixing_the_searches_pays_one_unit(media_db):
    db_path, _ = media_db
    burst = [
        ("search", {"query": "paper"}),
        ("search_audio", {"query": "crew"}),
        ("search_video", {"query": "a door"}),
    ]

    result, seen, _ = await run(
        db_path,
        [burst, [("search_audio", {"query": "again"})]],
        _config(max_searches=1),
    )

    outcomes = [
        part.outcome
        for name in ("search", "search_audio", "search_video")
        for part in returns(result, name)
    ]
    assert outcomes.count("failed") == 1
    assert returns(result, "search_audio")[-1].outcome == "failed"
    assert "search_audio" in (seen[-1].instructions or "")


async def test_list_recordings_names_each_recording_and_costs_nothing(media_db):
    db_path, video_id = media_db

    result, _, _ = await run(
        db_path, [[("list_recordings", {})]], _config(max_searches=0)
    )

    [part] = returns(result, "list_recordings")
    assert part.outcome != "failed"
    assert video_id in str(part.content)
    assert "file:///talk.mp4" in str(part.content)


async def test_a_speech_citation_carries_its_time_and_no_frames(media_db):
    db_path, _ = media_db

    def model_function(messages, info: AgentInfo) -> ModelResponse:
        searched = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "search_audio"
        ]
        cited = any(
            isinstance(part, ToolReturnPart) and part.tool_name == "cite"
            for message in messages
            for part in message.parts
        )
        if not searched:
            return ModelResponse(
                parts=[ToolCallPart("search_audio", {"query": "crew"})]
            )
        if not cited:
            chunk_id = re.search(UUID, str(searched[0].content))
            assert chunk_id is not None
            return ModelResponse(
                parts=[ToolCallPart("cite", {"chunk_ids": [chunk_id.group(0)]})]
            )
        return ModelResponse(parts=[TextPart("done")])

    capability = create_rag(db_path=db_path, config=_config())
    agent = Agent(
        FunctionModel(model_function), deps_type=Deps, capabilities=[capability]
    )
    deps = Deps()
    await agent.run("question", deps=deps)

    state = RAGState.model_validate(deps.state["rag"])
    [citation] = [state.citation_index[c] for c in state.citations]
    assert citation.time_range is not None
    assert not citation.picture_refs


async def test_list_recordings_names_the_collection_over_several(tmp_path):
    from haiku.rag.config import LanceDBConfig

    config = _config()
    config.lancedb = LanceDBConfig(
        databases={n: str(tmp_path / f"{n}.lancedb") for n in ("a", "b")}
    )
    for name in ("a", "b"):
        async with HaikuRAG(config=config, create=True, sources=[name]) as rag:
            await _seed(rag)

    seen: list[AgentInfo] = []
    capability = create_rag(config=config)
    agent = Agent(
        scripted([[("list_recordings", {"sources": ["b"]})]], seen),
        deps_type=Deps,
        capabilities=[capability],
    )
    result = await agent.run("question", deps=Deps())

    [part] = returns(result, "list_recordings")
    assert "Collection: b" in str(part.content)
    assert "Collection: a" not in str(part.content)


async def _lecture(tmp_path):
    """Ten ten-second segments of speech, one chunk each, and one frame at 50
    seconds."""
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
    doc = DoclingDocument(name="lecture")
    for second in range(0, 100, 10):
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=f"Said at {second} seconds.",
            content_layer=ContentLayer.BODY,
            source=TrackSource(start_time=second, end_time=second + 9.5),
        )
        if second == 50:
            doc.add_picture(
                image=ImageRef.from_pil(Image.new("RGB", (96, 96), "red"), dpi=72),
                content_layer=ContentLayer.BODY,
                source=TrackSource(start_time=50.0, end_time=50.001),
            )
    doc.origin = DocumentOrigin(mimetype="video/mp4", filename="l.mp4", binary_hash=1)
    path = tmp_path / "lecture.lancedb"
    # The query "50" embeds closest to the 50 s chunk, so hybrid search ranks it
    # first on both lists instead of tying every chunk on the vector side.
    speech = [
        Chunk(
            content=item.text,
            embedding=_seeded_vector("50")
            if item.text == "Said at 50 seconds."
            else [0.1] * dim,
            order=i + 1,
            metadata={
                "doc_item_refs": [item.self_ref],
                "labels": ["text"],
                "time_range": [item.source[0].start_time, item.source[0].end_time],
            },
        )
        for i, item in enumerate(doc.texts)
    ]
    async with HaikuRAG(path, create=True) as rag:
        stored = await rag.import_document(
            doc,
            [
                Chunk(
                    content="",
                    embedding=[0.1] * dim,
                    order=0,
                    metadata={
                        "doc_item_refs": ["#/pictures/0"],
                        "labels": ["picture"],
                        "time_range": [50.0, 50.001],
                    },
                ),
                *speech,
            ],
            uri="file:///lecture.mp4",
        )
    assert stored.id is not None
    return path, stored.id


async def test_a_frame_hit_carries_only_the_speech_near_the_frame(tmp_path):
    path, _ = await _lecture(tmp_path)

    _, _, state = await run(path, [[("search_video", {"query": "red"})]])

    [result] = state.searches["red"]
    assert "Said at 50 seconds." in result.content
    assert "Said at 30 seconds." in result.content
    assert "Said at 60 seconds." in result.content
    assert "Said at 20 seconds." not in result.content
    assert "Said at 70 seconds." not in result.content


async def test_a_frame_hit_keeps_to_the_window_it_was_searched_in(tmp_path):
    path, lecture_id = await _lecture(tmp_path)
    window = {"recording": lecture_id, "start": 45.0, "end": 52.0}

    _, _, state = await run(path, [[("search_video", {"query": "red", **window})]])

    [result] = state.searches["red"]
    assert "Said at 40 seconds." in result.content
    assert "Said at 50 seconds." in result.content
    assert "Said at 30 seconds." not in result.content
    assert "Said at 60 seconds." not in result.content


async def test_a_frame_hit_shows_each_line_with_its_time_and_the_frame_among_them(
    tmp_path,
):
    path, _ = await _lecture(tmp_path)

    result, _, state = await run(path, [[("search_video", {"query": "red"})]])

    [hit] = state.searches["red"]
    assert hit.content.splitlines() == [
        "[00:00:30] Said at 30 seconds.",
        "[00:00:40] Said at 40 seconds.",
        "[00:00:50] (frame)",
        "[00:00:50] Said at 50 seconds.",
        "[00:01:00] Said at 60 seconds.",
    ]
    [part] = returns(result, "search_video")
    assert "Type: frame" in str(part.content)


async def test_a_speech_hit_shows_the_lines_within_15_seconds_of_its_chunk(tmp_path):
    path, _ = await _lecture(tmp_path)

    result, _, state = await run(
        path, [[("search_audio", {"query": "50", "limit": 1})]]
    )

    [hit] = state.searches["50"]
    assert hit.content.splitlines() == [
        "[00:00:30] Said at 30 seconds.",
        "[00:00:40] Said at 40 seconds.",
        "[00:00:50] Said at 50 seconds.",
        "[00:01:00] Said at 60 seconds.",
        "[00:01:10] Said at 70 seconds.",
    ]
    [part] = returns(result, "search_audio")
    assert "Type: speech" in str(part.content)


async def test_a_window_narrower_than_15_seconds_clips_a_speech_hit(tmp_path):
    path, lecture_id = await _lecture(tmp_path)
    window = {"recording": lecture_id, "start": 48.0, "end": 62.0}

    _, _, state = await run(
        path, [[("search_audio", {"query": "50", "limit": 1, **window})]]
    )

    [hit] = state.searches["50"]
    assert hit.content.splitlines() == [
        "[00:00:40] Said at 40 seconds.",
        "[00:00:50] Said at 50 seconds.",
        "[00:01:00] Said at 60 seconds.",
    ]


async def test_citing_a_chunk_again_holds_the_slice_cited_last(tmp_path):
    from tests.client.test_media_search import _seed_spanning_chunk

    path = tmp_path / "spanning.lancedb"
    async with HaikuRAG(path, create=True) as rag:
        video_id = await _seed_spanning_chunk(rag)
    windows = [(5.0, 7.0), (9.0, 11.0)]

    def model_function(messages, info: AgentInfo) -> ModelResponse:
        parts = [part for message in messages for part in message.parts]
        searched = [
            p
            for p in parts
            if isinstance(p, ToolReturnPart) and p.tool_name == "search_audio"
        ]
        cited = [
            p for p in parts if isinstance(p, ToolReturnPart) and p.tool_name == "cite"
        ]
        if len(searched) == len(cited) < len(windows):
            start, end = windows[len(searched)]
            args = {"query": "crew", "recording": video_id, "start": start, "end": end}
            return ModelResponse(parts=[ToolCallPart("search_audio", args)])
        if len(cited) < len(searched):
            chunk_id = re.search(UUID, str(searched[-1].content))
            assert chunk_id is not None
            return ModelResponse(
                parts=[ToolCallPart("cite", {"chunk_ids": [chunk_id.group(0)]})]
            )
        return ModelResponse(parts=[TextPart("done")])

    capability = create_rag(db_path=path, config=_config())
    agent = Agent(
        FunctionModel(model_function), deps_type=Deps, capabilities=[capability]
    )
    deps = Deps()
    await agent.run("question", deps=deps)

    state = RAGState.model_validate(deps.state["rag"])
    # The compaction capsule renders the held citation's content.
    [citation] = state.citation_index.values()
    assert (
        citation.content == "[00:00:07] Engineers improvised a carbon dioxide filter."
    )
    assert citation.time_range == (7.92, 12.12)
    assert citation.index == 1


async def test_citing_a_held_chunk_without_searching_keeps_the_held_slice(media_db):
    db_path, video_id = media_db
    chunk_ids: list[str] = []

    def first_question(messages, info: AgentInfo) -> ModelResponse:
        parts = [part for message in messages for part in message.parts]
        searched = [
            p
            for p in parts
            if isinstance(p, ToolReturnPart) and p.tool_name == "search_audio"
        ]
        if not searched:
            args = {"query": "crew", "recording": video_id, "start": 5.0, "end": 7.0}
            return ModelResponse(parts=[ToolCallPart("search_audio", args)])
        if not any(
            isinstance(p, ToolReturnPart) and p.tool_name == "cite" for p in parts
        ):
            found = re.search(UUID, str(searched[0].content))
            assert found is not None
            chunk_ids.append(found.group(0))
            return ModelResponse(parts=[ToolCallPart("cite", {"chunk_ids": chunk_ids})])
        return ModelResponse(parts=[TextPart("done")])

    def second_question(messages, info: AgentInfo) -> ModelResponse:
        last = messages[-1].parts
        if any(isinstance(p, ToolReturnPart) and p.tool_name == "cite" for p in last):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("cite", {"chunk_ids": chunk_ids})])

    capability = create_rag(db_path=db_path, config=_config())
    deps = Deps()
    first = await Agent(
        FunctionModel(first_question), deps_type=Deps, capabilities=[capability]
    ).run("first", deps=deps)
    held = RAGState.model_validate(deps.state["rag"]).citation_index[chunk_ids[0]]
    await Agent(
        FunctionModel(second_question), deps_type=Deps, capabilities=[capability]
    ).run("second", deps=deps, message_history=first.all_messages())

    state = RAGState.model_validate(deps.state["rag"])
    assert state.citation_index[chunk_ids[0]] == held
    assert held.content.startswith("[00:00:04] ")


@pytest.mark.parametrize(
    "queries",
    [
        [("crew", 5.0, 7.0), ("crew", 9.0, 11.0)],
        [("crew", 5.0, 7.0), ("module", 0.0, 4.0), ("crew", 9.0, 11.0)],
        [("crew", 0.0, 4.0), ("module", 9.0, 11.0), ("crew", 100.0, 200.0)],
    ],
    ids=["same-query", "another-query-between", "empty-repeat"],
)
async def test_a_repeated_query_in_another_window_cites_the_latest_slice(
    tmp_path, queries
):
    from tests.client.test_media_search import _seed_spanning_chunk

    path = tmp_path / "spanning.lancedb"
    async with HaikuRAG(path, create=True) as rag:
        video_id = await _seed_spanning_chunk(rag)

    def model_function(messages, info: AgentInfo) -> ModelResponse:
        searched = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "search_audio"
        ]
        cited = any(
            isinstance(part, ToolReturnPart) and part.tool_name == "cite"
            for message in messages
            for part in message.parts
        )
        if len(searched) < len(queries):
            query, start, end = queries[len(searched)]
            args = {"query": query, "recording": video_id, "start": start, "end": end}
            return ModelResponse(parts=[ToolCallPart("search_audio", args)])
        if not cited:
            chunk_id = re.search(UUID, " ".join(str(p.content) for p in searched))
            assert chunk_id is not None
            return ModelResponse(
                parts=[ToolCallPart("cite", {"chunk_ids": [chunk_id.group(0)]})]
            )
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(
        FunctionModel(model_function),
        deps_type=Deps,
        capabilities=[create_rag(db_path=path, config=_config())],
    )
    deps = Deps()
    await agent.run("question", deps=deps)

    state = RAGState.model_validate(deps.state["rag"])
    [citation] = [state.citation_index[c] for c in state.citations]
    assert (
        citation.content == "[00:00:07] Engineers improvised a carbon dioxide filter."
    )
    assert citation.time_range == (7.92, 12.12)


def test_refreshing_replaces_every_held_version_of_a_returned_chunk():
    from haiku.rag.capabilities._tools import refresh_results
    from haiku.rag.store.models.chunk import SearchResult

    first = [
        SearchResult(content="early slice", score=1.0, chunk_id="c1"),
        SearchResult(content="unnamed", score=1.0),
    ]
    second = [SearchResult(content="early slice", score=1.0, chunk_id="c1")]

    refresh_results(
        [first, second],
        [
            SearchResult(content="late slice", score=1.0, chunk_id="c1"),
            SearchResult(content="another unnamed", score=1.0),
        ],
    )

    assert [r.content for r in first] == ["late slice", "unnamed"]
    assert [r.content for r in second] == ["late slice"]


async def test_every_media_tool_parameter_is_described_and_search_says_what_it_covers(
    media_db,
):
    db_path, _ = media_db

    _, without, _ = await run(db_path, [], _config(media_tools=False))
    _, under, _ = await run(db_path, [])

    offered = {tool.name: tool for tool in under[0].function_tools}
    for name in MEDIA_TOOLS:
        properties = offered[name].parameters_json_schema.get("properties", {})
        assert all("description" in p for p in properties.values()), name
    [plain] = [t for t in without[0].function_tools if t.name == "search"]
    assert offered["search"].description != plain.description


@pytest.mark.parametrize("media_tools", [False, True])
async def test_no_capability_tool_is_strict(media_db, media_tools):
    db_path, _ = media_db

    _, seen, _ = await run(db_path, [], _config(media_tools=media_tools))

    assert {tool.name: tool.strict for tool in seen[0].function_tools} == {
        tool.name: False for tool in seen[0].function_tools
    }
