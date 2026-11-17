import asyncio
import base64
import hashlib
import io
import json
import multiprocessing
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from haiku.rag.config import AppConfig
from haiku.rag.config.models import FrameSamplingConfig, TranscriptionServerConfig
from haiku.rag.converters.base import VIDEO_EXTENSIONS
from haiku.rag.converters.exceptions import (
    ConversionFailedError,
    ConversionTimeoutError,
)
from haiku.rag.utils.dependencies import raise_missing_extra
from haiku.rag.utils.http import pooled_client, post_retrying_dropped_connection

if TYPE_CHECKING:
    import httpx
    import numpy as np
    from docling_core.types.doc.document import DoclingDocument
    from onnxruntime import InferenceSession

MEDIA_MIMETYPES = {
    ".wav": "audio/x-wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}

# vLLM refuses audio over 600 s or 25 MB by default. A piece of at most 590 s
# leaves room for MP3 padding, and 16 kHz mono MP3 at 32 kbit/s keeps it near 2.4 MB.
SEGMENT_SECONDS = 590.0
CUT_WINDOW_SECONDS = 60.0
MAX_CONCURRENT_TRANSCRIPTIONS = 4
MIN_SEGMENT_SECONDS = 0.01
FRAME_SECONDS = 0.001
# Silero VAD scores 512-sample windows of 16 kHz audio, each with the 64 samples
# before it as context.
VAD_REPO = "onnx-community/silero-vad"
VAD_FILE = "onnx/model.onnx"
VAD_REVISION = "e71cae966052b992a7eca6b17738916ce0eca4ec"
VAD_WINDOW = 512
VAD_CONTEXT = 64
VAD_THRESHOLD = 0.5
# The model's state drifts over long audio until it misses speech.
VAD_RESET_SECONDS = 30
VAD_MARGIN_SECONDS = 1.0
MIN_SPEECH_SHARE = 0.05
REPEATS = 3
SPARSE_SECONDS = 10.0
SPARSE_WORDS_PER_SECOND = 0.4
JPEG_QUALITY = "2"
SCENE_PROBE_FPS = 1.0
CHILD_GRACE_SECONDS = 2.0

_FFMPEG = ("ffmpeg", "-nostdin", "-hide_banner", "-nostats")
_SILENCE = re.compile(r"silence_(start|end): (-?[\d.]+)")
_PTS_TIME = re.compile(r"pts_time:(-?[\d.]+)")


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Frame:
    time: float
    jpeg: bytes
    scene: int | None = None


async def _run(*args: str) -> tuple[bytes, bytes]:
    """Run a command and return its stdout and stderr. A cancelled or failed
    caller kills and reaps it."""
    process = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await process.communicate()
    except BaseException:
        if process.returncode is None:  # pragma: no branch - exited as cancelled
            process.kill()
        await process.wait()
        raise
    if process.returncode != 0:
        detail = err.decode(errors="replace").strip()[-500:]
        raise ConversionFailedError(f"{args[0]} failed: {detail}")
    return out, err


async def probe(path: Path) -> tuple[float, bool]:
    """The duration of `path` in seconds, and whether it has an audio stream."""
    out, _ = await _run(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type",
        "-of",
        "json",
        str(path),
    )
    answer = json.loads(out)
    try:
        duration = float(answer["format"]["duration"])
    except (KeyError, ValueError):
        raise ConversionFailedError(f"ffprobe reports no duration for {path}") from None
    audio = any(s.get("codec_type") == "audio" for s in answer.get("streams", []))
    return duration, audio


async def _silences(path: Path, duration: float) -> list[tuple[float, float]]:
    _, err = await _run(
        *_FFMPEG,
        "-i",
        str(path),
        "-vn",
        "-af",
        "silencedetect=noise=-30dB:d=0.3",
        "-f",
        "null",
        "-",
    )
    return parse_silences(err.decode(errors="replace"), duration)


def parse_silences(log: str, duration: float) -> list[tuple[float, float]]:
    """The silences `silencedetect` logged. One still open at the end runs to
    `duration`."""
    silences: list[tuple[float, float]] = []
    start: float | None = None
    for kind, value in _SILENCE.findall(log):
        if kind == "start":
            start = max(0.0, float(value))
        elif start is not None:
            silences.append((start, float(value)))
            start = None
    if start is not None:
        silences.append((start, duration))
    return silences


def cut_points(duration: float, silences: list[tuple[float, float]]) -> list[float]:
    """Where to cut the audio into pieces of at most `SEGMENT_SECONDS`: each cut
    moves back to the middle of the longest silence within `CUT_WINDOW_SECONDS`
    before its target."""
    cuts: list[float] = []
    target = SEGMENT_SECONDS
    while target < duration:
        nearby = [
            (end - start, (start + end) / 2)
            for start, end in silences
            if 0 <= target - (start + end) / 2 <= CUT_WINDOW_SECONDS
        ]
        cut = max(nearby)[1] if nearby else target
        cuts.append(cut)
        target = cut + SEGMENT_SECONDS
    return cuts


_MP3 = ("-c:a", "libmp3lame", "-b:a", "32k", "-f", "mp3")
_PCM = ("-f", "s16le")


async def _extract_audio(
    path: Path, start: float, end: float | None, encoding: tuple[str, ...] = _MP3
) -> bytes:
    """16 kHz mono audio of `path` from `start` to `end`."""
    length = [] if end is None else ["-t", f"{end - start:.3f}"]
    out, _ = await _run(
        *_FFMPEG,
        "-v",
        "error",
        "-ss",
        f"{start:.3f}",
        *length,
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        *encoding,
        "pipe:1",
    )
    return out


@cache
def speech_detector() -> "InferenceSession":
    """Silero VAD, downloaded on first use."""
    try:
        from huggingface_hub import hf_hub_download
        from onnxruntime import InferenceSession
    except ModuleNotFoundError as e:
        raise_missing_extra(e.name or "", "media", e)

    model = hf_hub_download(VAD_REPO, VAD_FILE, revision=VAD_REVISION)
    return InferenceSession(model, providers=["CPUExecutionProvider"])


def _speech_windows(pcm: bytes, stop: threading.Event) -> "np.ndarray":
    """Whether each `VAD_WINDOW` of 16 kHz PCM holds speech, by Silero VAD."""
    import numpy as np

    session = speech_detector()
    audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
    state = np.zeros((2, 1, 128), np.float32)
    rate = np.array(16000, np.int64)
    context = np.zeros(VAD_CONTEXT, np.float32)
    speech = []
    reset = VAD_RESET_SECONDS * 16000 // VAD_WINDOW * VAD_WINDOW
    for i in range(0, len(audio) - VAD_WINDOW + 1, VAD_WINDOW):
        if stop.is_set():
            break
        if i % reset == 0:
            state = np.zeros((2, 1, 128), np.float32)
        window = audio[i : i + VAD_WINDOW]
        probability, state = session.run(
            None,
            {
                "input": np.concatenate([context, window])[None, :],
                "state": state,
                "sr": rate,
            },
        )
        context = window[-VAD_CONTEXT:]
        speech.append(np.asarray(probability).item() > VAD_THRESHOLD)
    return np.array(speech, dtype=bool)


async def speech_windows(path: Path, start: float, end: float | None) -> "np.ndarray":
    pcm = await _extract_audio(path, start, end, _PCM)
    stop = threading.Event()
    try:
        return await asyncio.to_thread(_speech_windows, pcm, stop)
    finally:
        stop.set()


def speech_shares(
    utterances: list[Utterance], speech: "np.ndarray", offset: float
) -> list[float]:
    """The share of speech windows within `VAD_MARGIN_SECONDS` of each utterance."""
    window = VAD_WINDOW / 16000
    shares = []
    for u in utterances:
        first = max(0, int((u.start - VAD_MARGIN_SECONDS - offset) / window))
        last = max(first + 1, int((u.end + VAD_MARGIN_SECONDS - offset) / window))
        windows = speech[first:last]
        shares.append(float(windows.mean()) if windows.size else 0.0)
    return shares


def spoken(utterances: list[Utterance], shares: list[float]) -> list[Utterance]:
    """The utterances that are not Whisper's text over music, noise or silence:
    almost no speech around it, and text the recording repeats or a few words
    over a long stretch. Silero misses speech over a music bed, so a low share
    alone drops real speech."""
    counts = Counter(u.text for u in utterances)

    def invented(u: Utterance, share: float) -> bool:
        seconds = u.end - u.start
        sparse = (
            seconds >= SPARSE_SECONDS
            and len(u.text.split()) / seconds < SPARSE_WORDS_PER_SECOND
        )
        return share < MIN_SPEECH_SHARE and (counts[u.text] >= REPEATS or sparse)

    return [
        u for u, share in zip(utterances, shares, strict=True) if not invented(u, share)
    ]


async def _transcribe(
    client: "httpx.AsyncClient",
    server: TranscriptionServerConfig,
    audio: bytes,
    offset: float,
) -> list[Utterance]:
    data = {
        "model": server.model,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "segment",
    }
    if server.language:
        data["language"] = server.language
    headers = {"Authorization": f"Bearer {server.api_key}"} if server.api_key else {}
    response = await post_retrying_dropped_connection(
        client,
        f"{server.base_url.rstrip('/')}/audio/transcriptions",
        data=data,
        files={"file": ("audio.mp3", audio, "audio/mpeg")},
        headers=headers,
    )
    response.raise_for_status()
    segments = response.json().get("segments")
    if not isinstance(segments, list):
        raise ConversionFailedError(
            "The transcription server gave no timed segments. It must support "
            "response_format=verbose_json"
        )
    utterances = []
    for segment in segments:
        text = segment["text"].strip()
        if text:
            start = offset + segment["start"]
            end = max(offset + segment["end"], start + MIN_SEGMENT_SECONDS)
            utterances.append(Utterance(start, end, text))
    return utterances


async def transcribe(
    path: Path, duration: float, server: TranscriptionServerConfig
) -> list[Utterance]:
    """Transcribe the audio of `path` in segments cut at silences, at most
    `MAX_CONCURRENT_TRANSCRIPTIONS` at a time."""
    silences = await _silences(path, duration) if duration > SEGMENT_SECONDS else []
    bounds = [0.0, *cut_points(duration, silences)]
    ends: list[float | None] = [*bounds[1:], None]
    gate = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIPTIONS)

    async with pooled_client(server.timeout) as client:

        async def segment(
            start: float, end: float | None
        ) -> list[tuple[Utterance, float]]:
            async with gate:
                audio = await _extract_audio(path, start, end)
                utterances, speech = await _gather(
                    _transcribe(client, server, audio, start),
                    speech_windows(path, start, end),
                )
                return list(
                    zip(
                        utterances,
                        speech_shares(utterances, speech, start),
                        strict=True,
                    )
                )

        pieces = await _gather(
            *(segment(start, end) for start, end in zip(bounds, ends, strict=True))
        )
    scored = [pair for piece in pieces for pair in piece]
    return spoken([u for u, _ in scored], [share for _, share in scored])


def _scale(max_width: int | None) -> list[str]:
    return [] if max_width is None else [f"scale='min(iw,{max_width})':-1"]


async def interval_frames(
    path: Path, interval: float, max_width: int | None
) -> list[Frame]:
    """The first frame at or after every multiple of `interval` seconds, by one
    ffmpeg pass, at the time `showinfo` logs for it. Each lands in a temporary
    file: the document holds every frame anyway."""
    with tempfile.TemporaryDirectory() as directory:
        _, log = await _run(
            *_FFMPEG,
            "-v",
            "info",
            "-i",
            str(path),
            "-an",
            "-vf",
            ",".join(
                [
                    f"select='gte(t,selected_n*{interval})'",
                    "showinfo",
                    *_scale(max_width),
                ]
            ),
            "-fps_mode",
            "passthrough",
            "-q:v",
            JPEG_QUALITY,
            str(Path(directory) / "%08d.jpg"),
        )
        files = sorted(Path(directory).glob("*.jpg"))
        jpegs = await asyncio.to_thread(lambda: [f.read_bytes() for f in files])
    times = [float(t) for t in _PTS_TIME.findall(log.decode(errors="replace"))]
    if len(times) != len(jpegs):
        raise ConversionFailedError(
            f"ffmpeg logged {len(times)} frame times for {len(jpegs)} frames of {path}"
        )
    return [Frame(t, jpeg) for t, jpeg in zip(times, jpegs, strict=True)]


async def _frame_at(path: Path, time: float, max_width: int | None) -> bytes:
    scale = _scale(max_width)
    out, _ = await _run(
        *_FFMPEG,
        "-v",
        "error",
        "-ss",
        f"{time:.3f}",
        "-i",
        str(path),
        "-an",
        "-frames:v",
        "1",
        *(["-vf", scale[0]] if scale else []),
        "-q:v",
        JPEG_QUALITY,
        "-f",
        "image2pipe",
        "-c:v",
        "mjpeg",
        "pipe:1",
    )
    if not out:
        raise ConversionFailedError(f"ffmpeg gave no frame at {time:.3f}s of {path}")
    return out


def _detect_scenes(  # pragma: no cover - runs in a spawned child
    path: str, duration: float
) -> list[tuple[int, float, float]]:
    from docling.utils.video_frame_sampling import (
        FfmpegRunner,
        SimpleSceneChangeFrameSampler,
    )

    runner = FfmpegRunner()
    runner.scale_to_duration(duration)
    sampler = SimpleSceneChangeFrameSampler(probe_fps=SCENE_PROBE_FPS, runner=runner)
    return [
        (s.scene_id, s.start_time, s.end_time)
        for s in sampler.detect_scenes(Path(path))
    ]


async def scene_frames(
    path: Path, duration: float, max_width: int | None
) -> list[Frame]:
    """One JPEG per scene, at the scene's midpoint. Scenes come from docling's
    scene-change sampler, run in a child process."""
    scenes = await run_in_child("Scene detection", _detect_scenes, str(path), duration)
    covered = scenes[-1][2] if scenes else 0.0
    if covered < duration - 2 / SCENE_PROBE_FPS:
        raise ConversionFailedError(
            f"The scene probe of {path} stopped at {covered:.1f}s of {duration:.1f}s"
        )
    frames = []
    for scene, start, end in scenes:
        middle = (start + end) / 2
        frames.append(Frame(middle, await _frame_at(path, middle, max_width), scene))
    return frames


def _child_main(  # pragma: no cover - runs in a spawned child
    sender: Any, target: Callable[..., Any], args: tuple
) -> None:
    if sys.platform != "win32":
        os.setpgrp()
    try:
        sender.send((True, target(*args)))
    except BaseException as exc:  # noqa: BLE001
        sender.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        sender.close()


def _stop_child(process: Any) -> None:
    """Stop the child and every process it started."""
    if sys.platform == "win32":  # pragma: no cover - POSIX CI
        if process.is_alive():
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
            )
        process.join()
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            # No group: the child has exited, or has not yet called setpgrp and
            # so has started nothing. macOS answers EPERM when only the
            # unreaped child is left.
            if process.is_alive():
                process.kill()
            break
        process.join(CHILD_GRACE_SECONDS)
    process.join()


async def run_in_child(what: str, target: Callable[..., Any], *args: Any) -> Any:
    """Run `target(*args)` in a child process and return its result. Cancelling
    the caller stops the child and every process it started."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_child_main, args=(sender, target, args), daemon=True
    )
    process.start()
    sender.close()
    try:
        while not await asyncio.to_thread(receiver.poll, 0.5):
            pass
        try:
            ok, value = receiver.recv()
        except EOFError:
            process.join(CHILD_GRACE_SECONDS)
            raise ConversionFailedError(
                f"{what} exited with code {process.exitcode} and no result"
            ) from None
    finally:
        _stop_child(process)
        receiver.close()
    if not ok:
        raise ConversionFailedError(f"{what} failed: {value}")
    return value


async def _gather(*awaitables: Any) -> list[Any]:
    """Await concurrently, cancelling the rest when one fails with its error."""
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(a) for a in awaitables]
    except ExceptionGroup as failed:
        raise failed.exceptions[0] from None
    return [task.result() for task in tasks]


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise ConversionFailedError(
            "Converting audio and video needs ffmpeg and ffprobe on PATH"
        )


class MediaConverter:
    """Converts audio and video into a `DoclingDocument` of timed transcript
    items and video frames, with ffmpeg and the transcription server of
    `processing.media.asr`."""

    def __init__(self, config: AppConfig):
        self.config = config

    async def convert_file(self, path: Path) -> "DoclingDocument":
        timeout = self.config.processing.conversion_timeout
        try:
            async with asyncio.timeout(timeout):
                return await self._document(path)
        except TimeoutError:
            raise ConversionTimeoutError(
                f"Converting {path} exceeded processing.conversion_timeout "
                f"({timeout}s)",
                converter_wedged=False,
            ) from None

    async def _document(self, path: Path) -> "DoclingDocument":
        media = self.config.processing.media
        assert media is not None and media.asr is not None
        require_ffmpeg()
        duration, audio = await probe(path)
        video = path.suffix.lower() in VIDEO_EXTENSIONS
        utterances, frames = await _gather(
            transcribe(path, duration, media.asr) if audio else _nothing(),
            self._frames(path, duration, media.frames) if video else _nothing(),
        )
        return await asyncio.to_thread(_build_document, path, utterances, frames)

    async def _frames(
        self, path: Path, duration: float, frames: FrameSamplingConfig
    ) -> list[Frame]:
        if self.config.processing.pictures == "none":
            return []
        if frames.mode == "scene_change":
            return await scene_frames(path, duration, frames.max_width)
        return await interval_frames(path, frames.interval_seconds, frames.max_width)


async def _nothing() -> list:
    return []


def _file_hash(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _build_document(
    path: Path, utterances: list[Utterance], frames: list[Frame]
) -> "DoclingDocument":
    """Transcript items and frames in time order, a frame before an item
    starting at the same time."""
    from docling_core.types.doc import (
        ContentLayer,
        DocItemLabel,
        DoclingDocument,
        DocumentOrigin,
        ImageRef,
        Size,
        TrackSource,
    )
    from PIL import Image

    doc = DoclingDocument(
        name=path.stem,
        origin=DocumentOrigin(
            filename=path.name,
            mimetype=MEDIA_MIMETYPES[path.suffix.lower()],
            binary_hash=_file_hash(path),
        ),
    )
    events = sorted(
        [(f.time, 0, f) for f in frames] + [(u.start, 1, u) for u in utterances],
        key=lambda e: (e[0], e[1]),
    )
    for _, _, event in events:
        if isinstance(event, Frame):
            with Image.open(io.BytesIO(event.jpeg)) as image:
                width, height = image.size
            encoded = base64.b64encode(event.jpeg).decode("ascii")
            doc.add_picture(
                image=ImageRef(
                    mimetype="image/jpeg",
                    dpi=72,
                    size=Size(width=width, height=height),
                    uri=f"data:image/jpeg;base64,{encoded}",
                ),
                content_layer=ContentLayer.BODY,
                source=TrackSource(
                    start_time=event.time,
                    end_time=event.time + FRAME_SECONDS,
                    identifier=None if event.scene is None else f"scene:{event.scene}",
                ),
            )
        else:
            doc.add_text(
                label=DocItemLabel.TEXT,
                text=event.text,
                content_layer=ContentLayer.BODY,
                source=TrackSource(start_time=event.start, end_time=event.end),
            )
    return doc
