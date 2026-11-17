import asyncio
import threading
from pathlib import Path

import pytest
from docling_core.types.doc import PictureItem, TextItem
from pydantic import ValidationError

from haiku.rag.client.exceptions import UnsupportedSourceError
from haiku.rag.config import AppConfig
from haiku.rag.converters.base import (
    AUDIO_EXTENSIONS,
    MEDIA_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from haiku.rag.converters.docling_local import DoclingLocalConverter
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


def test_video_needs_a_multimodal_embedder_and_audio_does_not():
    text = DoclingLocalConverter(_config()).supported_extensions
    multimodal = DoclingLocalConverter(
        AppConfig.model_validate(
            {
                "embeddings": {"model": {"multimodal": True}},
                "processing": {"media": {"asr": WHISPER}},
            }
        )
    ).supported_extensions

    assert set(AUDIO_EXTENSIONS) <= set(text)
    assert not set(VIDEO_EXTENSIONS) & set(text)
    assert set(MEDIA_EXTENSIONS) <= set(multimodal)


@pytest.mark.parametrize("converter", ["docling-local", "docling-serve"])
def test_media_needs_a_transcription_server(converter):
    with pytest.raises(ValidationError, match="processing.media.asr"):
        AppConfig.model_validate({"processing": {"converter": converter, "media": {}}})


def test_docling_serve_accepts_frame_settings():
    frames = {"mode": "scene_change", "interval_seconds": 5, "max_width": 640}
    config = AppConfig.model_validate(
        {
            "processing": {
                "converter": "docling-serve",
                "media": {"asr": WHISPER, "frames": frames},
            }
        }
    )

    assert config.processing.media is not None
    assert config.processing.media.frames.model_dump() == frames


@pytest.mark.parametrize("converter", ["docling-local", "docling-serve"])
def test_media_config_survives_a_dump_and_reload(converter):
    config = AppConfig.model_validate(
        {"processing": {"converter": converter, "media": {"asr": WHISPER}}}
    )

    assert AppConfig.model_validate(config.model_dump()) == config


@needs_ffmpeg
@pytest.mark.parametrize("converter", ["docling-local", "docling-serve"])
@pytest.mark.vcr()
async def test_converts_video_into_a_transcript_and_frames(converter):
    from haiku.rag.client.processing import convert

    config = _config()
    config.processing.converter = converter
    config.embeddings.model.multimodal = True
    doc = await convert(config, DATA / "speech.mp4")

    assert "lunar module" in " ".join(t.text for t in doc.texts)
    items = [i for i, _ in doc.iterate_items() if isinstance(i, PictureItem | TextItem)]
    frames = [i for i in items if isinstance(i, PictureItem)]
    assert [f.source[0].start_time for f in frames] == [0.0, 10.0]
    times = [i.source[0].start_time for i in items]
    assert times == sorted(times)
    assert items[0] is frames[0]
    assert doc.origin is not None and doc.origin.mimetype == "video/mp4"


@needs_ffmpeg
@pytest.mark.vcr()
async def test_a_transcription_does_not_wait_for_the_document_converter(monkeypatch):
    from haiku.rag.client.processing import convert
    from haiku.rag.converters import docling_local

    held = threading.Lock()
    held.acquire()
    monkeypatch.setattr(docling_local, "_CONVERTER_LOCK", held)

    doc = await asyncio.wait_for(convert(_config(), DATA / "speech.mp3"), 120)

    assert doc.texts


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


@needs_ffmpeg
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
