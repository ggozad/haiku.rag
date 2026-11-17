from pathlib import Path

import pytest

from haiku.rag.client.exceptions import UnsupportedSourceError
from haiku.rag.config import AppConfig
from haiku.rag.converters.base import AUDIO_EXTENSIONS, MEDIA_EXTENSIONS
from haiku.rag.converters.docling_serve import DoclingServeConverter

DATA = Path(__file__).parents[1] / "data"
WHISPER = {
    "base_url": "http://localhost:11470/v1",
    "model": "Systran/faster-whisper-small",
}


def _config(base_url: str = "http://localhost:5001", **processing) -> AppConfig:
    return AppConfig.model_validate(
        {
            "processing": {
                "converter": "docling-serve",
                "chunker": "docling-serve",
                **processing,
            },
            "providers": {"docling_serve": {"base_url": base_url}},
        }
    )


def test_media_is_off_by_default():
    converter = DoclingServeConverter(_config())

    assert not set(MEDIA_EXTENSIONS) & set(converter.supported_extensions)
    with pytest.raises(UnsupportedSourceError):
        converter.require_supported(Path("talk.mp3"))


def test_configuring_media_supports_audio_under_a_text_embedder():
    converter = DoclingServeConverter(_config(media={"asr": WHISPER}))

    assert set(converter.supported_extensions) & set(MEDIA_EXTENSIONS) == set(
        AUDIO_EXTENSIONS
    )
