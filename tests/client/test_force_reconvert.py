from pathlib import Path

import pytest

from haiku.rag.config import AppConfig
from haiku.rag.converters.docling_local import DoclingLocalConverter

DATA = Path(__file__).parents[1] / "data"


def _config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "processing": {
                "media": {
                    "asr": {
                        "base_url": "http://localhost:11470/v1",
                        "model": "Systran/faster-whisper-small",
                    }
                }
            }
        }
    )


@pytest.mark.vcr()
async def test_force_converts_an_unchanged_file_again(
    temp_db_path, tmp_path, monkeypatch
):
    from haiku.rag.client import HaikuRAG

    converted: list[Path] = []
    convert_file = DoclingLocalConverter.convert_file

    async def counting(self, path, *args, **kwargs):
        converted.append(path)
        return await convert_file(self, path, *args, **kwargs)

    monkeypatch.setattr(DoclingLocalConverter, "convert_file", counting)
    talk = tmp_path / "talk.mp3"
    talk.write_bytes((DATA / "speech.mp3").read_bytes())

    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        first = await rag.create_document_from_source(talk)
        await rag.create_document_from_source(talk)
        assert len(converted) == 1
        again = await rag.create_document_from_source(talk, force=True)

    assert len(converted) == 2
    assert not isinstance(first, list) and not isinstance(again, list)
    assert again.id == first.id
