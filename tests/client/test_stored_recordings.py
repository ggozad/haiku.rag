import hashlib
import shutil
from pathlib import Path

import pytest

from haiku.rag.client import HaikuRAG, documents
from haiku.rag.config import AppConfig
from haiku.rag.converters.media import probe
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.recording import RecordingRepository
from tests.platform import needs_ffmpeg

DATA = Path(__file__).parents[1] / "data"
SUBTITLES = "WEBVTT\n\n00:00.000 --> 00:02.000\nHello there.\n"


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


def _talk(tmp_path: Path) -> Path:
    talk = tmp_path / "talk.mp3"
    shutil.copy(DATA / "speech.mp3", talk)
    talk.with_suffix(".vtt").write_text(SUBTITLES, encoding="utf-8")
    return talk


async def _ingest(rag: HaikuRAG, path: Path, **kwargs) -> Document:
    doc = await rag.create_document_from_source(path, **kwargs)
    assert isinstance(doc, Document) and doc.id is not None
    return doc


async def _stored(rag: HaikuRAG, document_id: str):
    return await RecordingRepository(rag.store).open(document_id)


@needs_ffmpeg
@pytest.mark.vcr()
async def test_an_ingested_recording_is_stored_with_what_conversion_read(
    temp_db_path, tmp_path
):
    talk = _talk(tmp_path)
    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        doc = await _ingest(rag, talk)
        assert doc.id is not None

        stored = await _stored(rag, doc.id)

    assert stored is not None
    content = talk.read_bytes()
    assert stored.sha256 == hashlib.sha256(content).hexdigest()
    assert (stored.suffix, stored.subtitles) == (".mp3", SUBTITLES)
    assert stored.duration == pytest.approx((await probe(talk))[0])
    assert stored.read_range(0, 64) == content[:64]


@needs_ffmpeg
async def test_ingest_refuses_a_recording_replaced_while_it_was_converted(
    temp_db_path, tmp_path, monkeypatch
):
    talk = _talk(tmp_path)
    convert = documents.convert

    async def replacing_convert(config, path, **kwargs):
        talk.write_bytes(talk.read_bytes() + b"\x00" * 1024)
        return await convert(config, path, **kwargs)

    monkeypatch.setattr(documents, "convert", replacing_convert)
    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        with pytest.raises(ValueError, match="changed while it was converted"):
            await rag.create_document_from_source(talk)

        assert await rag.list_documents() == []


@needs_ffmpeg
async def test_ingest_refuses_subtitles_edited_while_the_recording_was_converted(
    temp_db_path, tmp_path, monkeypatch
):
    talk = _talk(tmp_path)
    convert = documents.convert

    async def editing_convert(config, path, **kwargs):
        converted = await convert(config, path, **kwargs)
        talk.with_suffix(".vtt").write_text(
            SUBTITLES.replace("Hello", "Goodbye"), encoding="utf-8"
        )
        return converted

    monkeypatch.setattr(documents, "convert", editing_convert)
    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        with pytest.raises(ValueError, match="changed while it was converted"):
            await rag.create_document_from_source(talk)

        assert await rag.list_documents() == []


@needs_ffmpeg
@pytest.mark.vcr()
async def test_updates_keep_the_recording_unless_they_replace_what_it_transcribes(
    temp_db_path, tmp_path
):
    talk = _talk(tmp_path)
    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        doc = await _ingest(rag, talk)
        assert doc.id is not None
        docling = await rag.convert(talk)

        await rag.update_document(doc.id, title="Renamed", metadata={"k": "v"})
        assert await _stored(rag, doc.id) is not None

        await rag.update_document(doc.id, docling_document=docling)
        assert await _stored(rag, doc.id) is not None

        chunks = await rag.chunk(docling)
        await rag.update_document(doc.id, chunks=chunks)
        assert await _stored(rag, doc.id) is not None

        await rag.update_document(doc.id, content="Plain text now.", chunks=chunks)
        assert await _stored(rag, doc.id) is not None

        other = docling.model_copy(deep=True)
        assert other.origin is not None
        other.origin.binary_hash += 1
        await rag.update_document(doc.id, docling_document=other)
        assert await _stored(rag, doc.id) is None


@needs_ffmpeg
@pytest.mark.vcr()
async def test_a_content_update_drops_the_recording(temp_db_path, tmp_path):
    talk = _talk(tmp_path)
    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        doc = await _ingest(rag, talk)
        assert doc.id is not None

        await rag.update_document(doc.id, content="Plain text now.")

        assert await _stored(rag, doc.id) is None


@needs_ffmpeg
@pytest.mark.vcr()
async def test_ingesting_another_file_at_the_uri_replaces_the_recording(
    temp_db_path, tmp_path
):
    talk = _talk(tmp_path)
    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        first = await _ingest(rag, talk)
        talk.write_bytes(talk.read_bytes() + b"\x00")
        again = await _ingest(rag, talk, force=True)
        assert again.id == first.id and again.id is not None

        stored = await _stored(rag, again.id)

    assert stored is not None
    assert stored.sha256 == hashlib.sha256(talk.read_bytes()).hexdigest()
