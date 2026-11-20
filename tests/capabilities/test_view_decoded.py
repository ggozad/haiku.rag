import re
import subprocess

import pytest

from haiku.rag.capabilities.rag import RAGState
from haiku.rag.client import HaikuRAG
from haiku.rag.converters import media
from haiku.rag.converters.exceptions import ConversionFailedError
from haiku.rag.store.models.recording import RecordingFile
from tests.capabilities.test_media_tools import images, returns
from tests.capabilities.test_view import _film, view
from tests.platform import needs_ffmpeg

pytestmark = needs_ffmpeg


@pytest.fixture(scope="module")
def film_file(tmp_path_factory) -> bytes:
    """A 95 s test pattern at 5 frames a second, shorter than the film's
    transcript, which runs to 99.5 s."""
    path = tmp_path_factory.mktemp("film") / "film.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=95:size=64x48:rate=5",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path.read_bytes()


@pytest.fixture
async def stored_film(tmp_path, film_file):
    path = tmp_path / "film.lancedb"
    return path, await _film(path, recording=RecordingFile(film_file, ".mp4"))


def _frames(state: RAGState) -> dict[str, float]:
    return {
        ref: at
        for results in state.searches.values()
        for r in results
        for ref, at in (r.picture_times or {}).items()
    }


async def test_a_window_shows_eight_frames_decoded_at_even_times(stored_film):
    path, film_id = stored_film

    result, state = await view(path, [{"recording": film_id, "start": 20, "end": 36}])

    times = [20 + (i + 0.5) * 2 for i in range(8)]
    assert sorted(_frames(state).values()) == times
    assert sorted(_frames(state)) == sorted(f"#/frames/{t:.3f}" for t in times)
    assert len(images(result)) == 8
    assert all(image.data[:2] == b"\xff\xd8" for image in images(result))


async def test_a_decoded_frame_belongs_to_the_chunk_whose_time_holds_it(stored_film):
    path, film_id = stored_film

    _, state = await view(path, [{"recording": film_id, "start": 20, "end": 36}])

    async with HaikuRAG(path) as rag:
        spans = await rag.chunk_repository.get_time_spans(film_id)
    owned = [
        (r.chunk_id, at)
        for results in state.searches.values()
        for r in results
        for at in (r.picture_times or {}).values()
    ]
    assert len(owned) == 8
    for chunk_id, at in owned:
        assert chunk_id is not None
        first, last = spans[chunk_id]
        assert first <= at <= last


async def test_a_window_past_the_end_of_the_file_shows_no_frames(stored_film):
    path, film_id = stored_film

    result, state = await view(path, [{"recording": film_id, "start": 96, "end": 99}])

    assert "Said at 90" in str(returns(result, "view")[0].content)
    assert _frames(state) == {}
    assert images(result) == []
    assert "failed" not in str(returns(result, "view")[0].content)


async def test_a_cited_decoded_frame_is_decoded_again_on_fetch(stored_film):
    path, film_id = stored_film

    async with HaikuRAG(path) as rag:
        data = await rag.get_picture_bytes(film_id, "#/frames/12.500")
        missing = await rag.get_picture_bytes("no-such-document", "#/frames/1.000")

    assert data is not None and data[:2] == b"\xff\xd8"
    assert missing is None


async def test_a_failed_decode_shows_the_stored_frames_and_says_why(
    stored_film, monkeypatch
):
    path, film_id = stored_film

    async def broken(*args, **kwargs):
        raise ConversionFailedError("ffmpeg gave no frame")

    monkeypatch.setattr(media, "_frame_at", broken)

    result, state = await view(path, [{"recording": film_id, "start": 0, "end": 100}])

    assert all(ref.startswith("#/pictures/") for ref in _frames(state))
    assert len(_frames(state)) == 8
    content = str(returns(result, "view")[0].content)
    assert re.search(r"Frames could not be decoded.*ffmpeg gave no frame", content)


async def test_a_frame_between_chunks_belongs_to_the_nearest(stored_film):
    path, film_id = stored_film

    _, state = await view(path, [{"recording": film_id, "start": 29, "end": 31}])

    owner = {
        ref: r.content
        for results in state.searches.values()
        for r in results
        for ref in (r.picture_times or {})
    }
    assert "Said at 20" in owner["#/frames/29.625"]
    assert "Said at 30" in owner["#/frames/29.875"]


async def test_a_decode_that_runs_out_of_time_shows_the_stored_frames(
    stored_film, monkeypatch
):
    path, film_id = stored_film
    monkeypatch.setattr(media, "DECODE_TIMEOUT_SECONDS", 0.000001)

    result, state = await view(path, [{"recording": film_id, "start": 0, "end": 100}])

    assert all(ref.startswith("#/pictures/") for ref in _frames(state))
    assert "(it timed out)" in str(returns(result, "view")[0].content)


async def test_a_malformed_decoded_frame_ref_fetches_nothing(stored_film):
    path, film_id = stored_film

    async with HaikuRAG(path) as rag:
        assert await rag.get_picture_bytes(film_id, "#/frames/later") is None


class _Stored:
    suffix = ".mp4"

    def __init__(self, content: bytes):
        self.content = content
        self.size = len(content)

    def read_range(self, offset: int, length: int) -> bytes:
        return self.content[offset : offset + length]


async def test_the_recording_server_answers_byte_ranges_at_its_secret_path_only(
    film_file,
):
    import httpx

    size = len(film_file)
    async with media._served(_Stored(film_file)) as url:  # ty: ignore[invalid-argument-type]
        async with httpx.AsyncClient() as client:
            whole = await client.get(url)
            ranged = await client.get(url, headers={"Range": "bytes=10-19"})
            open_ended = await client.get(url, headers={"Range": "bytes=100-"})
            suffix = await client.get(url, headers={"Range": "bytes=-10"})
            past = await client.get(url, headers={"Range": f"bytes={size}-"})
            malformed = await client.get(url, headers={"Range": "bytes=x-y"})
            empty = await client.get(url, headers={"Range": "bytes=-"})
            guessed = await client.get(url.rsplit("/", 1)[0] + "/recording.mp4")

    assert (whole.status_code, whole.content) == (200, film_file)
    assert (ranged.status_code, ranged.content) == (206, film_file[10:20])
    assert ranged.headers["Content-Range"] == f"bytes 10-19/{size}"
    assert open_ended.content == film_file[100:]
    assert suffix.content == film_file[-10:]
    assert past.status_code == malformed.status_code == empty.status_code == 416
    assert guessed.status_code == 404


async def test_an_empty_window_between_chunks_shows_frames_of_the_nearest(
    stored_film,
):
    path, film_id = stored_film

    result, state = await view(path, [{"recording": film_id, "start": 9.6, "end": 9.9}])

    assert len(_frames(state)) == 8
    owners = [
        r.content
        for results in state.searches.values()
        for r in results
        if r.picture_times
    ]
    assert owners and all("Said" not in content for content in owners)
    assert len(images(result)) == 8


async def test_a_cited_decoded_frame_is_in_the_citation(stored_film):
    from pydantic_ai import Agent
    from pydantic_ai.messages import (
        ModelResponse,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from haiku.rag.capabilities.rag import create_capability as create_rag
    from tests.capabilities.helpers import Deps
    from tests.capabilities.test_media_tools import UUID, _config

    path, film_id = stored_film

    def model_function(messages, info: AgentInfo) -> ModelResponse:
        parts = [p for m in messages for p in m.parts]
        viewed = [
            p for p in parts if isinstance(p, ToolReturnPart) and p.tool_name == "view"
        ]
        if not viewed:
            args = {"recording": film_id, "start": 20, "end": 36}
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
    refs = [ref for c in state.citation_index.values() for ref in c.picture_refs]
    assert len(refs) == 8 and all(ref.startswith("#/frames/") for ref in refs)
    async with HaikuRAG(path) as rag:
        fetched = await rag.get_picture_bytes(film_id, refs[0])
    assert fetched is not None and fetched[:2] == b"\xff\xd8"


async def test_without_ffmpeg_the_stored_frames_are_shown(stored_film, monkeypatch):
    path, film_id = stored_film
    monkeypatch.setattr(media.shutil, "which", lambda name: None)

    result, state = await view(path, [{"recording": film_id, "start": 0, "end": 100}])

    assert all(ref.startswith("#/pictures/") for ref in _frames(state))
    assert "needs ffmpeg" in str(returns(result, "view")[0].content)


@pytest.mark.parametrize("ref", ["#/frames/nan", "#/frames/-1", "#/frames/inf"])
async def test_a_decoded_frame_ref_without_a_time_fetches_nothing(stored_film, ref):
    path, film_id = stored_film

    async with HaikuRAG(path) as rag:
        assert await rag.get_picture_bytes(film_id, ref) is None


def test_frame_times_stay_inside_a_window_narrower_than_a_millisecond():
    assert media.frame_times(1.0001, 1.0002, 10.0, 8) == []
    assert media.frame_times(1.0, 1.004, 10.0, 8) == [1.0, 1.001, 1.002, 1.003, 1.004]


async def test_a_stored_audio_file_is_not_decoded_for_frames(tmp_path, film_file):
    path = tmp_path / "film.lancedb"
    film_id = await _film(path, recording=RecordingFile(film_file, ".mp3"))

    result, state = await view(path, [{"recording": film_id, "start": 0, "end": 100}])

    assert all(ref.startswith("#/pictures/") for ref in _frames(state))
    assert "could not be decoded" not in str(returns(result, "view")[0].content)


async def test_a_request_still_open_ends_with_the_server(film_file):
    import asyncio
    from urllib.parse import urlsplit

    async with media._served(_Stored(film_file)) as url:  # ty: ignore[invalid-argument-type]
        address = urlsplit(url)
        reader, writer = await asyncio.open_connection(address.hostname, address.port)
        writer.write(f"GET {address.path} HTTP/1.1\r\n".encode())
        await writer.drain()
        await asyncio.sleep(0.2)

    assert await asyncio.wait_for(reader.read(), timeout=5) == b""
    writer.close()


async def test_an_uppercase_video_suffix_is_decoded(tmp_path, film_file):
    path = tmp_path / "film.lancedb"
    film_id = await _film(path, recording=RecordingFile(film_file, ".MP4"))

    _, state = await view(path, [{"recording": film_id, "start": 20, "end": 36}])

    assert len(_frames(state)) == 8
    assert all(ref.startswith("#/frames/") for ref in _frames(state))


async def test_a_result_lists_its_lines_and_decoded_frames_in_time_order(stored_film):
    path, film_id = stored_film

    # The frame at 9.938 s falls before the sentence starting at 10 s, whose
    # chunk is the nearest to it.
    result, _ = await view(path, [{"recording": film_id, "start": 9.8, "end": 12}])

    content = str(returns(result, "view")[0].content)
    assert "[00:00:09] (frame)" in content
    for block in content.split("\n---\n"):
        clocks = re.findall(r"^\[(\d\d:\d\d:\d\d)\]", block, re.M)
        assert clocks == sorted(clocks)
