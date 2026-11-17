import asyncio
import sys
from pathlib import Path

import httpx
import pytest
from docling_core.types.doc import TrackSource

from haiku.rag.client.processing import convert
from haiku.rag.config import AppConfig
from haiku.rag.converters import media
from haiku.rag.converters.exceptions import (
    ConversionFailedError,
    ConversionTimeoutError,
)
from haiku.rag.converters.media import MediaConverter, cut_points
from tests.platform import needs_ffmpeg

DATA = Path(__file__).parents[1] / "data"
WHISPER = {
    "base_url": "http://localhost:11470/v1",
    "model": "Systran/faster-whisper-small",
}


def _config(converter: str = "docling-local", **processing) -> AppConfig:
    return AppConfig.model_validate(
        {
            "processing": {
                "converter": converter,
                "media": {"asr": WHISPER},
                **processing,
            }
        }
    )


def _serve(reply) -> tuple[list[httpx.Request], httpx.MockTransport]:
    """Route transcription requests to `reply` and record them."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return reply(request)

    return seen, httpx.MockTransport(handler)


@pytest.fixture
def transcription_server(monkeypatch):
    def install(reply):
        seen, transport = _serve(reply)
        monkeypatch.setattr(
            media,
            "pooled_client",
            lambda timeout: httpx.AsyncClient(transport=transport, timeout=timeout),
        )
        return seen

    return install


def _segments(*segments: tuple[float, float, str]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "segments": [
                {"start": start, "end": end, "text": text}
                for start, end, text in segments
            ]
        },
    )


def test_a_cut_moves_back_to_the_longest_silence_before_its_target():
    silences = [(500.0, 510.0), (540.0, 544.0), (580.0, 581.0), (595.0, 620.0)]

    assert cut_points(1000.0, silences) == [542.0]


def test_without_a_silence_before_it_the_cut_is_at_the_target():
    assert cut_points(1300.0, [(100.0, 120.0)]) == [590.0, 1180.0]


def test_no_piece_is_longer_than_a_default_vllm_server_accepts():
    import random

    rng = random.Random(7)
    for _ in range(200):
        duration = rng.uniform(600.0, 5000.0)
        starts = sorted(rng.uniform(0.0, duration) for _ in range(rng.randint(0, 80)))
        silences = [(s, min(duration, s + rng.uniform(0.1, 20.0))) for s in starts]
        bounds = [0.0, *cut_points(duration, silences), duration]

        assert max(b - a for a, b in zip(bounds, bounds[1:])) <= 590.0 + 1e-9


def test_a_short_recording_is_not_cut():
    assert cut_points(590.0, []) == []


@needs_ffmpeg
@pytest.mark.parametrize("converter", ["docling-local", "docling-serve"])
@pytest.mark.vcr()
async def test_audio_converts_to_a_timed_transcript_with_either_converter(
    converter,
):
    doc = await convert(_config(converter), DATA / "speech.mp3")

    texts = [t.text for t in doc.texts]
    assert "carbon dioxide filter" in " ".join(texts)
    sources = [t.source[0] for t in doc.texts]
    assert all(isinstance(s, TrackSource) for s in sources)
    assert sources[0].start_time == 0.0
    assert doc.origin is not None and doc.origin.mimetype == "audio/mpeg"


@needs_ffmpeg
async def test_segments_are_transcribed_separately_and_offset(
    monkeypatch, transcription_server
):
    monkeypatch.setattr(media, "SEGMENT_SECONDS", 5.0)
    monkeypatch.setattr(media, "CUT_WINDOW_SECONDS", 0.0)
    seen = transcription_server(lambda request: _segments((0.5, 1.5, "words")))

    doc = await MediaConverter(_config()).convert_file(DATA / "speech.mp3")

    assert len(seen) == 3
    assert [(t.source[0].start_time, t.source[0].end_time) for t in doc.texts] == [
        (0.5, 1.5),
        (5.5, 6.5),
        (10.5, 11.5),
    ]


@needs_ffmpeg
async def test_a_zero_length_segment_is_kept(transcription_server):
    transcription_server(lambda request: _segments((2.0, 2.0, "Yes.")))

    doc = await MediaConverter(_config()).convert_file(DATA / "speech.mp3")

    source = doc.texts[0].source[0]
    assert doc.texts[0].text == "Yes."
    assert source.end_time > source.start_time


@needs_ffmpeg
async def test_a_server_without_timed_segments_fails_naming_verbose_json(
    transcription_server,
):
    transcription_server(lambda request: httpx.Response(200, json={"text": "hi"}))

    with pytest.raises(ConversionFailedError, match="verbose_json"):
        await MediaConverter(_config()).convert_file(DATA / "speech.mp3")


@needs_ffmpeg
async def test_a_server_error_is_an_http_error(transcription_server):
    transcription_server(lambda request: httpx.Response(503))

    with pytest.raises(httpx.HTTPStatusError):
        await MediaConverter(_config()).convert_file(DATA / "speech.mp3")


async def test_without_ffmpeg_conversion_names_it(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(ConversionFailedError, match="ffmpeg"):
        await MediaConverter(_config()).convert_file(DATA / "speech.mp3")


async def test_the_deadline_kills_running_commands(monkeypatch):
    started: list[asyncio.subprocess.Process] = []
    spawn = asyncio.create_subprocess_exec

    async def record(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        started.append(process)
        return process

    async def stalled_probe(path):
        await media._run(sys.executable, "-c", "import time; time.sleep(30)")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", record)
    monkeypatch.setattr(media, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(media, "probe", stalled_probe)
    config = _config()
    config.processing.conversion_timeout = 0.5

    with pytest.raises(ConversionTimeoutError) as excinfo:
        await MediaConverter(config).convert_file(DATA / "speech.mp3")

    assert not excinfo.value.converter_wedged
    assert [p.returncode is not None for p in started] == [True]


VIDEO = DATA / "speech.mp4"


@needs_ffmpeg
async def test_a_video_without_audio_has_frames_and_no_transcription(
    tmp_path, transcription_server
):
    silent = tmp_path / "silent.mp4"
    await media._run(
        "ffmpeg", "-v", "error", "-i", str(VIDEO), "-an", "-c:v", "copy", str(silent)
    )
    seen = transcription_server(lambda request: _segments((0.0, 1.0, "never")))

    doc = await MediaConverter(_config()).convert_file(silent)

    assert seen == []
    assert not doc.texts
    assert [p.source[0].start_time for p in doc.pictures] == [0.0, 10.0]


@needs_ffmpeg
async def test_pictures_none_samples_no_frames(transcription_server):
    transcription_server(lambda request: _segments((0.0, 1.0, "words")))

    doc = await MediaConverter(_config(pictures="none")).convert_file(VIDEO)

    assert not doc.pictures
    assert [t.text for t in doc.texts] == ["words"]


@needs_ffmpeg
async def test_scene_change_takes_one_frame_per_scene_at_its_middle(
    transcription_server,
):
    transcription_server(lambda request: _segments())
    config = _config()
    assert config.processing.media is not None
    config.processing.media.frames.mode = "scene_change"

    doc = await MediaConverter(config).convert_file(VIDEO)

    sources = [p.source[0] for p in doc.pictures]
    assert sources
    assert [s.identifier for s in sources] == [
        f"scene:{i}" for i in range(len(sources))
    ]
    assert all(0 < s.start_time < 12.5 for s in sources)


async def test_a_scene_probe_short_of_the_end_fails(monkeypatch):
    async def short(what, target, *args):
        return [(0, 0.0, 4.0)]

    monkeypatch.setattr(media, "run_in_child", short)

    with pytest.raises(ConversionFailedError, match="stopped at 4.0s of 12.0s"):
        await media.scene_frames(VIDEO, 12.0, None)


def _start_and_wait(pid_file: str) -> None:
    import subprocess
    import time

    ignore_term = (
        "import os, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)"
    )
    subprocess.Popen([sys.executable, "-c", ignore_term, pid_file])
    time.sleep(60)


def _fail() -> None:
    raise RuntimeError("no scenes here")


def _exit() -> None:
    import os

    os._exit(3)


def _alive(pid: int) -> bool:
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        waited, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return True
    return waited == 0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
async def test_cancelling_a_child_kills_the_processes_it_started(tmp_path):
    pid_file = tmp_path / "pid"
    task = asyncio.create_task(
        media.run_in_child("Waiting", _start_and_wait, str(pid_file))
    )
    for _ in range(1200):
        if pid_file.exists() and pid_file.read_text():
            break
        await asyncio.sleep(0.05)
    grandchild = int(pid_file.read_text())

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(100):
        if not _alive(grandchild):
            break
        await asyncio.sleep(0.05)
    assert not _alive(grandchild)


async def test_a_failing_child_names_its_error():
    with pytest.raises(ConversionFailedError, match="Probe failed: RuntimeError"):
        await media.run_in_child("Probe", _fail)


async def test_a_child_exiting_without_a_result_fails():
    with pytest.raises(ConversionFailedError, match="exited with code 3"):
        await media.run_in_child("Probe", _exit)


@needs_ffmpeg
async def test_an_interval_frame_is_the_frame_shown_at_its_time(tmp_path):
    counter = tmp_path / "counter.mp4"
    await media._run(
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc=rate=1:duration=25:size=160x120",
        "-pix_fmt",
        "yuv420p",
        str(counter),
    )

    frames = await media.interval_frames(counter, 10.0, None)

    assert [f.time for f in frames] == [0.0, 10.0, 20.0]
    for frame in frames:
        assert frame.jpeg == await media._frame_at(counter, frame.time, None)


async def test_cancelling_a_child_before_it_starts_work_returns_promptly():
    import time

    task = asyncio.create_task(media.run_in_child("Waiting", _sleep))
    await asyncio.sleep(0)
    started = time.monotonic()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - started < 10


def _sleep() -> None:
    import time

    time.sleep(60)


@needs_ffmpeg
async def test_a_file_ffprobe_cannot_read_fails_naming_ffprobe(tmp_path):
    fake = tmp_path / "talk.mp3"
    fake.write_text("not audio")

    with pytest.raises(ConversionFailedError, match="ffprobe failed"):
        await media.probe(fake)


@needs_ffmpeg
async def test_a_file_without_a_duration_fails(tmp_path):
    from PIL import Image

    still = tmp_path / "still.png"
    Image.new("RGB", (8, 8)).save(still)

    with pytest.raises(ConversionFailedError, match="no duration"):
        await media.probe(still)


@needs_ffmpeg
async def test_silences_are_found_including_one_running_to_the_end(tmp_path):
    tone = tmp_path / "tone.wav"
    await media._run(
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=duration=2,apad=pad_dur=2,aloop=loop=1:size=192000,apad=pad_dur=1",
        str(tone),
    )
    duration, _ = await media.probe(tone)

    silences = await media._silences(tone, duration)

    assert len(silences) == 2
    assert silences[0][0] == pytest.approx(2.0, abs=0.1)
    assert silences[-1][1] == duration


def test_silences_pair_starts_with_ends_and_close_at_the_duration():
    log = (
        "[silencedetect] silence_end: 0.5 | silence_duration: 0.5\n"
        "[silencedetect] silence_start: 2.0\n"
        "[silencedetect] silence_end: 3.0 | silence_duration: 1.0\n"
        "[silencedetect] silence_start: 8.5\n"
    )

    assert media.parse_silences(log, 10.0) == [(2.0, 3.0), (8.5, 10.0)]


@needs_ffmpeg
async def test_the_language_reaches_the_server_and_blank_segments_are_dropped(
    transcription_server,
):
    seen = transcription_server(
        lambda request: _segments((0.0, 1.0, "  "), (1.0, 2.0, "Words."))
    )
    config = _config()
    assert config.processing.media is not None and config.processing.media.asr
    config.processing.media.asr.language = "en"

    doc = await MediaConverter(config).convert_file(DATA / "speech.mp3")

    assert b'name="language"\r\n\r\nen' in seen[0].content
    assert [t.text for t in doc.texts] == ["Words."]


@needs_ffmpeg
async def test_a_time_past_the_end_fails_in_ffmpeg():
    with pytest.raises(ConversionFailedError, match="ffmpeg"):
        await media._frame_at(VIDEO, 100.0, None)


async def test_ffmpeg_without_output_fails_naming_the_time(monkeypatch):
    async def nothing(*args):
        return b"", b""

    monkeypatch.setattr(media, "_run", nothing)

    with pytest.raises(ConversionFailedError, match="no frame at 3.000s"):
        await media._frame_at(VIDEO, 3.0, None)


async def test_frames_without_logged_times_fail(monkeypatch):
    async def unlogged(*args):
        Path(args[-1]).parent.joinpath("00000001.jpg").write_bytes(b"jpeg")
        return b"", b""

    monkeypatch.setattr(media, "_run", unlogged)

    with pytest.raises(ConversionFailedError, match="0 frame times for 1 frames"):
        await media.interval_frames(VIDEO, 10.0, None)


@needs_ffmpeg
async def test_a_frame_after_a_timestamp_gap_keeps_its_own_time(tmp_path):
    gapped = tmp_path / "gap.mkv"
    await media._run(
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc=rate=1:duration=5:size=160x120,setpts='if(lt(N,2),N,N+18)/TB'",
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "yuv420p",
        str(gapped),
    )

    frames = await media.interval_frames(gapped, 10.0, None)

    assert [f.time for f in frames] == [0.0, 20.0, 21.0]


@needs_ffmpeg
async def test_audio_under_picture_description_asks_for_no_description(
    monkeypatch, transcription_server
):
    def never(*args, **kwargs):
        raise AssertionError("no picture to describe")

    monkeypatch.setattr(
        "haiku.rag.providers.picture_description.describe_pictures", never
    )
    transcription_server(lambda request: _segments((0.0, 1.0, "Words.")))

    doc = await MediaConverter(_config(pictures="description")).convert_file(
        DATA / "speech.mp3"
    )

    assert [t.text for t in doc.texts] == ["Words."]


def test_only_text_shaped_like_an_invention_is_dropped_where_no_one_speaks():
    from haiku.rag.converters.media import Utterance, spoken

    utterances = [
        Utterance(0.0, 29.0, "Thank you."),  # sparse, silent
        Utterance(30.0, 31.0, "Shhh."),  # repeated, silent
        Utterance(40.0, 41.0, "Shhh."),
        Utterance(50.0, 51.0, "Shhh."),
        Utterance(60.0, 62.0, "How you feeling, though?"),  # silent, neither
        Utterance(70.0, 71.0, "Yes."),  # repeated, spoken
        Utterance(72.0, 73.0, "Yes."),
        Utterance(74.0, 75.0, "Yes."),
    ]
    shares = [0.0, 0.0, 0.01, 0.0, 0.0, 0.9, 0.8, 0.7]

    assert [u.text for u in spoken(utterances, shares)] == [
        "How you feeling, though?",
        "Yes.",
        "Yes.",
        "Yes.",
    ]


def test_a_speech_share_counts_a_second_either_side():
    import numpy as np

    from haiku.rag.converters.media import Utterance, speech_shares

    window = media.VAD_WINDOW / 16000
    speech = np.zeros(int(20 / window), dtype=bool)
    speech[int(4.5 / window) : int(5 / window)] = True

    [share] = speech_shares([Utterance(15.5, 16.0, "Late.")], speech, offset=10.0)

    assert share > 0


def test_a_stopped_speech_scan_returns_what_it_had():
    import threading

    stop = threading.Event()
    stop.set()

    assert media._speech_windows(b"\0\0" * 16000, stop).size == 0


@needs_ffmpeg
async def test_text_over_noise_is_dropped_and_speech_is_kept(
    tmp_path, transcription_server
):
    noise = tmp_path / "noise.mp3"
    await media._run(
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "anoisesrc=duration=40:color=pink:amplitude=0.3",
        str(noise),
    )
    transcription_server(lambda request: _segments((0.0, 29.0, "Thank you.")))

    silent = await MediaConverter(_config()).convert_file(noise)
    spoken = await MediaConverter(_config()).convert_file(DATA / "speech.mp3")

    assert not silent.texts
    assert [t.text for t in spoken.texts] == ["Thank you."]


@pytest.mark.parametrize("module", ["onnxruntime", "huggingface_hub"])
def test_the_speech_detector_names_the_media_extra_when_it_is_missing(
    module, monkeypatch
):
    monkeypatch.setitem(sys.modules, module, None)
    media.speech_detector.cache_clear()

    with pytest.raises(ImportError, match=r"haiku\.rag-slim\[media\]"):
        media.speech_detector()
