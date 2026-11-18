import json
import re

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from haiku.rag.capabilities.rag import RAGState
from haiku.rag.capabilities.rag import create_capability as create_rag
from haiku.rag.client import HaikuRAG
from tests.capabilities.helpers import Deps
from tests.capabilities.test_media_tools import (
    UUID,
    _config,
    media_db,  # noqa: F401
    returns,
    run,
)
from tests.client.test_media_search import FRAMES, SEGMENTS


async def _printed(db_path, code: str, media_tools: bool = True) -> dict:
    result, _, _ = await run(
        db_path, [[("execute_code", {"code": code})]], _config(media_tools)
    )
    [part] = returns(result, "execute_code")
    return json.loads(str(part.content).strip().splitlines()[-1])


LISTING = """
import json
from pathlib import Path
found = await search("crew")
docs = await list_documents()
items = Path("/documents/{video_id}/items.jsonl").read_text().splitlines()
print(json.dumps({{
    "searched": sorted(set(r["document_id"] for r in found)),
    "listed": [d["id"] for d in docs],
    "items": len(items),
}}))
"""


async def test_in_code_search_and_listing_leave_recordings_out(media_db):  # noqa: F811
    db_path, video_id = media_db

    printed = await _printed(db_path, LISTING.format(video_id=video_id))

    assert printed["searched"]
    assert video_id not in printed["searched"]
    assert video_id not in printed["listed"]
    assert printed["items"] > 0


async def test_without_media_tools_in_code_search_and_listing_cover_recordings(
    media_db,  # noqa: F811
):
    db_path, video_id = media_db

    printed = await _printed(db_path, LISTING.format(video_id=video_id), False)

    assert video_id in printed["searched"]
    assert video_id in printed["listed"]


async def test_the_media_functions_run_from_a_program(media_db):  # noqa: F811
    db_path, video_id = media_db
    code = f"""
import json
speech = await search_audio("crew", recording="{video_id}", start=5.0, end=7.0)
frames = await search_video("a door")
[recording] = await list_recordings()
print(json.dumps({{
    "speech": [r["time_range"] for r in speech],
    "frames": [r["chunk_meta"]["doc_item_refs"] for r in frames],
    "frame_times": [r["time_range"] for r in frames],
    "recording": recording,
}}))
"""

    printed = await _printed(db_path, code)

    assert printed["speech"] == [[4.48, 7.6]]
    assert printed["frames"]
    assert {ref for refs in printed["frames"] for ref in refs} <= {
        ref for ref, _, _ in FRAMES
    }
    assert printed["frame_times"] == [[0.0, 12.12], [0.0, 12.12]]
    assert printed["recording"]["document_id"] == video_id
    assert printed["recording"]["uri"] == "file:///talk.mp4"
    assert printed["recording"]["has_frames"] is True
    assert printed["recording"]["last_time"] == SEGMENTS[-1][2]


async def test_in_code_media_searches_run_past_a_spent_search_budget(
    media_db,  # noqa: F811
):
    db_path, _ = media_db
    code = """
import json
print(json.dumps({"speech": len(await search_audio("crew")),
                  "frames": len(await search_video("a door"))}))
"""

    result, _, _ = await run(
        db_path,
        [[("execute_code", {"code": code})]],
        _config(max_searches=0),
    )

    [part] = returns(result, "execute_code")
    printed = json.loads(str(part.content).strip().splitlines()[-1])
    assert printed["speech"] > 0
    assert printed["frames"] > 0


async def test_a_window_without_a_recording_fails_the_program(media_db):  # noqa: F811
    db_path, _ = media_db

    result, _, _ = await run(
        db_path,
        [[("execute_code", {"code": 'await search_audio("crew", start=5.0)'})]],
    )

    [part] = returns(result, "execute_code")
    assert "needs a recording" in str(part.content)


async def test_a_citation_holds_the_slice_an_in_code_search_returned_last(tmp_path):
    from tests.client.test_media_search import _seed_spanning_chunk

    path = tmp_path / "spanning.lancedb"
    async with HaikuRAG(path, create=True) as rag:
        video_id = await _seed_spanning_chunk(rag)

    def in_code(start: float, end: float) -> ToolCallPart:
        code = (
            f'r = await search_audio("crew", recording="{video_id}", '
            f"start={start}, end={end})\nprint(r[0]['chunk_id'])"
        )
        return ToolCallPart("execute_code", {"code": code})

    window = {"recording": video_id, "start": 5.0, "end": 7.0}
    calls = [
        in_code(0.0, 4.0),
        ToolCallPart("search_audio", {"query": "module", **window}),
        in_code(9.0, 11.0),
    ]

    def model_function(messages, info: AgentInfo) -> ModelResponse:
        done = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if len(done) < len(calls):
            return ModelResponse(parts=[calls[len(done)]])
        if len(done) == len(calls):
            chunk_id = re.search(UUID, str(done[-1].content))
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
