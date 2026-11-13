import sys
from pathlib import Path

import pytest

from haiku.rag.config import AppConfig
from haiku.rag.converters.docling_local import DoclingLocalConverter
from haiku.rag.converters.exceptions import ConversionFailedError

DATA = Path(__file__).parents[1] / "data"


def _config(preset: str) -> AppConfig:
    return AppConfig.model_validate({"processing": {"media": {"asr": preset}}})


def test_a_preset_name_selects_docling_s_in_process_model():
    from docling.datamodel import asr_model_specs
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AsrPipelineOptions,
        VideoPipelineOptions,
    )

    options = DoclingLocalConverter(_config("whisper_tiny"))._media_format_options()

    for fmt in (InputFormat.AUDIO, InputFormat.VIDEO):
        pipeline = options[fmt].pipeline_options
        assert isinstance(pipeline, AsrPipelineOptions | VideoPipelineOptions)
        assert pipeline.asr_options == asr_model_specs.WHISPER_TINY
        assert not pipeline.enable_remote_services


def test_an_unknown_preset_names_the_valid_ones():
    with pytest.raises(ValueError, match="whisper_turbo"):
        DoclingLocalConverter(_config("whisper_enormous"))


def test_docling_serve_refuses_a_preset():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="docling-serve"):
        AppConfig.model_validate(
            {
                "processing": {
                    "converter": "docling-serve",
                    "media": {"asr": "whisper_tiny"},
                }
            }
        )


async def test_in_process_audio_without_ffmpeg_names_it(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(ConversionFailedError, match="ffmpeg"):
        await DoclingLocalConverter(_config("whisper_tiny")).convert_file(
            DATA / "speech.mp3"
        )


async def test_in_process_transcription_without_the_extra_names_it(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    for module in ("whisper", "mlx_whisper", "whisper_s2t"):
        monkeypatch.setitem(sys.modules, module, None)

    with pytest.raises(ImportError, match=r"haiku\.rag-slim\[asr\]"):
        await DoclingLocalConverter(_config("whisper_tiny_native")).convert_file(
            DATA / "speech.mp3"
        )


@pytest.mark.slow
async def test_whisper_tiny_transcribes_in_process():
    doc = await DoclingLocalConverter(_config("whisper_tiny")).convert_file(
        DATA / "speech.mp3"
    )

    text = " ".join(t.text for t in doc.texts).lower()
    assert "lunar module" in text
    assert doc.texts[0].source[0].start_time == 0.0
