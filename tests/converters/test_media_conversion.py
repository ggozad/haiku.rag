import asyncio
import threading
from pathlib import Path

import pytest
from docling_core.types.doc import PictureItem, TrackSource
from pydantic import ValidationError

from haiku.rag.client.exceptions import UnsupportedSourceError
from haiku.rag.config import AppConfig
from haiku.rag.converters.base import MEDIA_EXTENSIONS
from haiku.rag.converters.docling_local import DoclingLocalConverter
from haiku.rag.converters.exceptions import ConversionFailedError
from tests.platform import needs_ffmpeg

DATA = Path(__file__).parents[1] / "data"
WHISPER = {
    "base_url": "http://localhost:11470/v1",
    "model": "Systran/faster-whisper-small",
}


def _config(**media) -> AppConfig:
    return AppConfig.model_validate(
        {"processing": {"media": {"asr": WHISPER, **media}}}
    )


def test_media_extensions_are_docling_s():
    from docling.datamodel.base_models import FormatToExtensions, InputFormat

    docling = {
        f".{ext}"
        for fmt in (InputFormat.AUDIO, InputFormat.VIDEO)
        for ext in FormatToExtensions[fmt]
    }
    assert set(MEDIA_EXTENSIONS) == docling


def test_media_is_off_by_default():
    converter = DoclingLocalConverter(AppConfig())

    assert not set(MEDIA_EXTENSIONS) & set(converter.supported_extensions)
    with pytest.raises(UnsupportedSourceError):
        converter.require_supported(Path("talk.mp3"))


def test_configuring_media_supports_its_extensions():
    converter = DoclingLocalConverter(_config())

    assert set(MEDIA_EXTENSIONS) <= set(converter.supported_extensions)


def test_docling_local_media_needs_a_transcription_server():
    with pytest.raises(ValidationError, match="processing.media.asr"):
        AppConfig.model_validate({"processing": {"media": {}}})


@pytest.mark.parametrize(
    "media", [{"asr": WHISPER}, {"frames": {"interval_seconds": 5}}]
)
def test_docling_serve_refuses_media_settings_it_cannot_honour(media):
    with pytest.raises(ValidationError, match="docling-serve"):
        AppConfig.model_validate(
            {"processing": {"converter": "docling-serve", "media": media}}
        )


@pytest.mark.parametrize("converter", ["docling-local", "docling-serve"])
def test_media_config_survives_a_dump_and_reload(converter):
    media = {"asr": WHISPER} if converter == "docling-local" else {}
    config = AppConfig.model_validate(
        {"processing": {"converter": converter, "media": media}}
    )

    assert AppConfig.model_validate(config.model_dump()) == config


def test_docling_serve_accepts_media_without_settings():
    config = AppConfig.model_validate(
        {"processing": {"converter": "docling-serve", "media": {}}}
    )
    assert config.processing.media is not None


def _pipelines(config: AppConfig):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AsrPipelineOptions,
        VideoPipelineOptions,
    )

    options = DoclingLocalConverter(config)._media_format_options()
    audio = options[InputFormat.AUDIO].pipeline_options
    video = options[InputFormat.VIDEO].pipeline_options
    assert isinstance(audio, AsrPipelineOptions)
    assert isinstance(video, VideoPipelineOptions)
    return audio, video


def test_media_format_options():
    from docling.datamodel.pipeline_options_asr_model import ApiAsrOptions

    audio, video = _pipelines(
        AppConfig.model_validate(
            {
                "processing": {
                    "media": {
                        "asr": {**WHISPER, "api_key": "secret"},
                        "frames": {"mode": "scene_change", "max_frames": 20},
                    }
                }
            }
        )
    )

    for pipeline in (audio, video):
        assert pipeline.enable_remote_services
        asr = pipeline.asr_options
        assert isinstance(asr, ApiAsrOptions)
        assert str(asr.url) == "http://localhost:11470/v1/audio/transcriptions"
        assert asr.model == "Systran/faster-whisper-small"
        assert asr.headers == {"Authorization": "Bearer secret"}
    assert video.frame_sampling_mode.value == "scene_change"
    assert video.max_sampled_frames == 20
    assert video.generate_frame_images


def test_pictures_none_turns_frame_sampling_off():
    config = _config()
    config.processing.pictures = "none"

    _, video = _pipelines(config)

    assert not video.generate_frame_images


@pytest.mark.vcr()
async def test_converts_audio_into_a_timed_transcript():
    doc = await DoclingLocalConverter(_config()).convert_file(DATA / "speech.mp3")

    assert "carbon dioxide filter" in " ".join(t.text for t in doc.texts)
    sources = [t.source[0] for t in doc.texts]
    assert all(isinstance(s, TrackSource) for s in sources)
    assert sources[0].start_time == 0.0
    assert sources[-1].end_time == pytest.approx(12.2, abs=0.5)


@needs_ffmpeg
@pytest.mark.vcr()
async def test_converts_video_into_a_transcript_and_frames():
    doc = await DoclingLocalConverter(_config()).convert_file(DATA / "speech.mp4")

    assert "lunar module" in " ".join(t.text for t in doc.texts)
    frames = [i for i, _ in doc.iterate_items() if isinstance(i, PictureItem)]
    assert [f.source[0].start_time for f in frames] == [0.0, 10.0]


@pytest.mark.vcr()
async def test_a_transcription_does_not_wait_for_the_document_converter(monkeypatch):
    from haiku.rag.converters import docling_local

    held = threading.Lock()
    held.acquire()
    monkeypatch.setattr(docling_local, "_CONVERTER_LOCK", held)

    doc = await asyncio.wait_for(
        DoclingLocalConverter(_config()).convert_file(DATA / "speech.mp3"), 30
    )

    assert doc.texts


async def test_video_without_ffmpeg_names_it(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(ConversionFailedError, match="ffmpeg"):
        await DoclingLocalConverter(_config()).convert_file(DATA / "speech.mp4")


class TestMediaTimeout:
    @pytest.fixture(autouse=True)
    def _fresh_media_state(self, monkeypatch):
        from haiku.rag.converters import docling_local

        docling_local._MEDIA_WEDGED.clear()
        monkeypatch.setattr(docling_local, "_MEDIA_LOCK", threading.Lock())
        yield
        docling_local._MEDIA_WEDGED.clear()

    async def test_a_stalled_transcription_wedges_media_only(self, monkeypatch):
        import asyncio

        from haiku.rag.converters import docling_local
        from haiku.rag.converters.exceptions import (
            ConversionTimeoutError,
            ConverterWedgedError,
        )

        release = threading.Event()
        config = _config()
        config.processing.conversion_timeout = 0.05
        converter = DoclingLocalConverter(config)
        monkeypatch.setattr(
            converter, "_sync_convert_timed", lambda *a, **k: release.wait(1)
        )
        try:
            with pytest.raises(ConversionTimeoutError) as excinfo:
                await asyncio.wait_for(converter.convert_file(DATA / "speech.mp3"), 3)
            assert excinfo.value.converter_wedged
            assert docling_local._MEDIA_WEDGED.is_set()
            assert not docling_local._WEDGED.is_set()
            with pytest.raises(ConverterWedgedError):
                await converter.convert_file(DATA / "speech.mp3")
        finally:
            release.set()


@pytest.mark.parametrize(
    ("content_type", "extension"),
    [
        ("audio/mpeg", ".mp3"),
        ("audio/wav", ".wav"),
        ("audio/x-wav", ".wav"),
        ("audio/mp4", ".m4a"),
        ("audio/aac", ".aac"),
        ("audio/ogg", ".ogg"),
        ("audio/flac", ".flac"),
        ("video/mp4", ".mp4"),
        ("video/quicktime", ".mov"),
        ("video/x-matroska", ".mkv"),
        ("video/webm", ".webm"),
        ("video/x-msvideo", ".avi"),
    ],
)
def test_an_extensionless_media_url_takes_its_content_type(content_type, extension):
    from haiku.rag.client.processing import get_extension_from_content_type_or_url

    assert (
        get_extension_from_content_type_or_url(
            "https://example.com/stream", content_type
        )
        == extension
    )


class _ExtensionlessSource:
    """Serves the spoken fixture at a URL without an extension."""

    def __init__(self, uri: str):
        self.source_id = "stream"
        self.supported_extensions = None
        self.max_file_size = None
        self._uri = uri

    def supports(self, uri: str) -> bool:
        return uri == self._uri

    async def head(self, uri: str) -> str | None:
        return "v1"

    async def aclose(self) -> None:
        pass

    async def fetch(self, uri: str):
        import hashlib

        from haiku.rag.sources.base import FetchResult

        body = (DATA / "speech.mp3").read_bytes()
        return FetchResult(
            uri=uri,
            body=body,
            content_type="audio/mpeg",
            content_hash=hashlib.md5(body).hexdigest(),
            revision="v1",
        )

    def discover(self, since=None, *, known_uris=None):
        raise NotImplementedError


@pytest.mark.vcr()
async def test_an_extensionless_audio_url_converts_as_audio(temp_db_path, monkeypatch):
    from haiku.rag import sources as sources_module
    from haiku.rag.client import HaikuRAG

    uri = "https://example.com/stream"
    monkeypatch.setattr(
        sources_module,
        "resolve_adhoc_fetcher",
        lambda *a, **kw: _ExtensionlessSource(uri),
    )

    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        doc = await rag.create_document_from_source(uri)
        assert not isinstance(doc, list) and doc.id is not None
        chunks = await rag.chunk_repository.get_by_document_id(doc.id)

    assert "carbon dioxide filter" in " ".join(c.content for c in chunks)
    assert all(c.metadata.get("time_range") for c in chunks)
