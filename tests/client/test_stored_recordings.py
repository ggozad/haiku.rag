import hashlib
import logging
import shutil
from pathlib import Path

import httpx
import pytest

from haiku.rag.client import DocumentImport, HaikuRAG, RebuildMode, documents, rebuild
from haiku.rag.config import AppConfig
from haiku.rag.converters import media
from haiku.rag.converters.media import MediaConverter, decoded_frame_ref
from haiku.rag.hooks import Hook
from haiku.rag.store.models.chunk import TimeWindow
from haiku.rag.store.models.document import Document
from tests.conftest import capture_logs
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


@pytest.fixture
def embedded(monkeypatch):
    """Chunks embed to a constant vector, without the embedder."""

    import haiku.rag.embeddings

    async def constant(chunks, embedder, config=None):
        for chunk in chunks:
            chunk.embedding = [0.1] * embedder.vector_dim
        return chunks

    async def ensured(config, chunks, embedder):
        return await constant(chunks, embedder)

    monkeypatch.setattr(documents, "ensure_chunks_embedded", ensured)
    monkeypatch.setattr(haiku.rag.embeddings, "embed_chunks", constant)


async def _stored(rag: HaikuRAG, document_id: str):
    return await rag.document_repository.open_recording(document_id)


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
        assert stored.original.sha256 == hashlib.sha256(content).hexdigest()
        assert (stored.suffix, stored.original.subtitles) == (".mp3", SUBTITLES)
        assert await stored.original.read(0, 64) == content[:64]


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
    assert stored.original.sha256 == hashlib.sha256(talk.read_bytes()).hexdigest()


async def _drop_original(rag: HaikuRAG, document_id: str) -> None:
    doc = await rag.document_repository.get_by_id(document_id, include_blobs=True)
    assert doc is not None
    doc.original = doc.original_sha256 = doc.original_subtitles = None
    await rag.document_repository.update(doc, with_original=True)


async def _converted(rag: HaikuRAG, talk: Path):
    docling = await rag.convert(talk)
    return docling, await rag.chunk(docling)


@needs_ffmpeg
@pytest.mark.vcr()
async def test_an_import_stores_the_recording_it_carries(temp_db_path, tmp_path):
    talk = _talk(tmp_path)
    content = talk.read_bytes()
    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        docling, chunks = await _converted(rag, talk)
        one = await rag.import_document(
            docling, chunks, uri="one", original=content, original_subtitles=SUBTITLES
        )
        docling, chunks = await _converted(rag, talk)
        [two] = await rag.import_documents(
            [
                DocumentImport(
                    docling,
                    chunks,
                    uri="two",
                    original=content,
                    original_subtitles=SUBTITLES,
                )
            ]
        )
        assert one.id is not None and two.id is not None

        stored = [await _stored(rag, one.id), await _stored(rag, two.id)]

    for recording in stored:
        assert recording is not None
        assert (recording.suffix, recording.original.subtitles) == (".mp3", SUBTITLES)


@needs_ffmpeg
@pytest.mark.vcr()
async def test_an_import_refuses_a_recording_that_does_not_match_its_document(
    temp_db_path, tmp_path
):
    talk = _talk(tmp_path)
    content = talk.read_bytes()
    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        docling, chunks = await _converted(rag, talk)
        assert docling.origin is not None
        text = await rag.convert("Just text.")

        with pytest.raises(ValueError, match="was not converted from"):
            await rag.import_document(docling, chunks, original=content + b"x")
        with pytest.raises(ValueError, match="neither a PDF nor an image"):
            await rag.import_document(text, await rag.chunk(text), original=content)
        for name in ("talk.txt", "talk"):
            docling.origin.filename = name
            with pytest.raises(ValueError, match="not an audio or video file"):
                await rag.import_document(docling, chunks, original=content)
        assert await rag.count_documents() == 0


@needs_ffmpeg
@pytest.mark.vcr()
async def test_a_full_rebuild_converts_a_recording_from_its_stored_file(
    temp_db_path, tmp_path, monkeypatch
):
    talk = _talk(tmp_path)
    converted: list[Path] = []
    convert_file = MediaConverter.convert_file

    async def counting(self, path, *args, **kwargs):
        converted.append(path)
        return await convert_file(self, path, *args, **kwargs)

    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        doc = await _ingest(rag, talk)
        assert doc.id is not None
        talk.with_suffix(".vtt").write_text(
            "WEBVTT\n\n00:00.000 --> 00:02.000\nEdited since.\n", encoding="utf-8"
        )
        monkeypatch.setattr(MediaConverter, "convert_file", counting)
        # The file is copied out in several reads.
        monkeypatch.setattr(rebuild, "_STORED_FILE_READ_BYTES", 4096)

        rebuilt = [doc_id async for doc_id in rag.rebuild_database()]
        content = (await rag.get_document_by_id(doc.id)).content  # type: ignore[union-attr]
        stored = await _stored(rag, doc.id)

    assert rebuilt == [doc.id]
    assert len(converted) == 1 and converted[0] != talk
    assert "Hello there." in content and "Edited" not in content
    assert stored is not None


@needs_ffmpeg
@pytest.mark.vcr()
async def test_a_full_rebuild_rechunks_a_recording_without_a_stored_file(
    temp_db_path, tmp_path, monkeypatch
):
    talk = _talk(tmp_path)
    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        doc = await _ingest(rag, talk)
        assert doc.id is not None
        await _drop_original(rag, doc.id)

        async def refuse(self, path, *args, **kwargs):
            raise AssertionError("converted")

        monkeypatch.setattr(MediaConverter, "convert_file", refuse)

        rebuilt = [doc_id async for doc_id in rag.rebuild_database()]

    assert rebuilt == [doc.id]


@needs_ffmpeg
@pytest.mark.vcr()
async def test_a_full_rebuild_transcribes_a_stored_file_without_subtitles(
    temp_db_path, tmp_path
):
    talk = tmp_path / "talk.mp3"
    shutil.copy(DATA / "speech.mp3", talk)
    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        doc = await _ingest(rag, talk)
        assert doc.id is not None
        before = (await rag.get_document_by_id(doc.id)).content  # type: ignore[union-attr]
        talk.unlink()

        rebuilt = [doc_id async for doc_id in rag.rebuild_database()]
        after = (await rag.get_document_by_id(doc.id)).content  # type: ignore[union-attr]

    assert rebuilt == [doc.id]
    assert after == before and after


@needs_ffmpeg
@pytest.mark.vcr()
async def test_a_stored_file_rebuild_keeps_an_edit_made_during_conversion(
    temp_db_path, tmp_path, monkeypatch
):
    talk = _talk(tmp_path)
    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        doc = await _ingest(rag, talk)
        assert doc.id is not None
        doc_id = doc.id
        convert = rebuild.convert

        async def editing_convert(*args, **kwargs):
            await rag.update_document(doc_id, title="Edited", metadata={"k": "v"})
            return await convert(*args, **kwargs)

        monkeypatch.setattr(rebuild, "convert", editing_convert)
        rebuilt = [i async for i in rag.rebuild_database()]
        stored = await rag.get_document_by_id(doc_id)

    assert rebuilt == [doc_id]
    assert stored is not None
    assert stored.title == "Edited"
    assert stored.metadata["k"] == "v"


@needs_ffmpeg
@pytest.mark.vcr()
async def test_imports_at_a_uri_keep_its_recording_while_they_name_its_file(
    temp_db_path, tmp_path
):
    talk = _talk(tmp_path)
    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        docling, chunks = await _converted(rag, talk)
        doc = await rag.import_document(
            docling,
            chunks,
            uri="talk",
            original=talk.read_bytes(),
            original_subtitles=SUBTITLES,
        )
        assert doc.id is not None

        await rag.update_document(doc.id, metadata={"k": "v"})
        assert await _stored(rag, doc.id) is not None

        docling, chunks = await _converted(rag, talk)
        again = await rag.import_document(docling, chunks, uri="talk")
        assert again.id == doc.id
        assert await _stored(rag, doc.id) is not None

        docling, chunks = await _converted(rag, talk)
        assert docling.origin is not None
        docling.origin.binary_hash += 1
        await rag.import_document(docling, chunks, uri="talk")
        assert await _stored(rag, doc.id) is None


VIDEO = DATA / "speech.mp4"


def _video_config(interval: float) -> AppConfig:
    return AppConfig.model_validate(
        {
            "embeddings": {
                "model": {
                    "provider": "vllm",
                    "name": "embedder",
                    "vector_dim": 8,
                    "multimodal": True,
                    "base_url": "http://localhost:1/v1",
                }
            },
            "processing": {
                "media": {
                    "asr": {"base_url": "http://localhost:11470/v1", "model": "w"},
                    "frames": {"interval_seconds": interval},
                }
            },
        }
    )


@needs_ffmpeg
async def test_a_full_rebuild_samples_a_stored_video_s_frames_again(
    temp_db_path, monkeypatch
):
    silent = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"segments": []})
    )
    monkeypatch.setattr(
        media,
        "pooled_client",
        lambda timeout: httpx.AsyncClient(transport=silent, timeout=timeout),
    )

    async def embedded(config, chunks, embedder):
        for chunk in chunks:
            chunk.embedding = [0.1] * config.embeddings.model.vector_dim
        return chunks

    monkeypatch.setattr(documents, "ensure_chunks_embedded", embedded)

    async def frames(rag: HaikuRAG, document_id: str) -> list[tuple[float, int]]:
        items = await rag.document_item_repository.get_items_in_window(
            TimeWindow(document_id=document_id, start=0.0, end=10_000.0)
        )
        found = []
        for item in items:
            if item.label == "picture" and item.start_time is not None:
                data = await rag.document_item_repository.get_picture_bytes(
                    document_id, item.self_ref
                )
                found.append((item.start_time, len(data or b"")))
        return found

    async with HaikuRAG(temp_db_path, config=_video_config(10.0), create=True) as rag:
        docling = await rag.convert(VIDEO)
        doc = await rag.import_document(
            docling,
            await rag.chunk(docling),
            uri="file:///talk.mp4",
            original=VIDEO.read_bytes(),
        )
        assert doc.id is not None
        before = await frames(rag, doc.id)

    async with HaikuRAG(temp_db_path, config=_video_config(2.0)) as rag:
        rebuilt = [i async for i in rag.rebuild_database()]
        after = await frames(rag, doc.id)
        again = [i async for i in rag.rebuild_database()]
        await rag.vacuum(retention_seconds=0)
        decoded = await rag.get_picture_bytes(doc.id, decoded_frame_ref(1.0))

    assert rebuilt == again == [doc.id]
    assert len(after) > len(before)
    assert [t for t, _ in after] == sorted(t for t, _ in after)
    assert all(size > 0 for _, size in after)
    assert decoded


@needs_ffmpeg
@pytest.mark.vcr()
async def test_a_stored_file_rebuild_does_not_bring_back_a_document_deleted_meanwhile(
    temp_db_path, tmp_path, monkeypatch
):
    talk = _talk(tmp_path)
    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        doc = await _ingest(rag, talk)
        assert doc.id is not None
        doc_id = doc.id
        convert = rebuild.convert

        async def deleting_convert(*args, **kwargs):
            await rag.delete_document(doc_id)
            return await convert(*args, **kwargs)

        monkeypatch.setattr(rebuild, "convert", deleting_convert)

        with pytest.raises(ValueError, match="not found"):
            [i async for i in rag.rebuild_database()]
        assert await rag.get_document_by_id(doc_id) is None


@needs_ffmpeg
@pytest.mark.parametrize("replaced", ["file", "subtitles"])
async def test_a_stored_file_rebuild_rechunks_a_recording_replaced_meanwhile(
    temp_db_path, tmp_path, monkeypatch, replaced, embedded
):
    talk = _talk(tmp_path)
    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        doc = await _ingest(rag, talk)
        assert doc.id is not None
        doc_id = doc.id
        before = (await rag.get_document_by_id(doc_id)).content  # type: ignore[union-attr]
        other = talk.read_bytes() + (b"\x00" if replaced == "file" else b"")
        convert = rebuild.convert

        async def replacing_convert(*args, **kwargs):
            # Stores the bytes without subtitles.
            await rag.document_repository.store_originals({doc_id: other})
            return await convert(*args, **kwargs)

        monkeypatch.setattr(rebuild, "convert", replacing_convert)
        with capture_logs(rebuild.logger, logging.WARNING) as records:
            rebuilt = [i async for i in rag.rebuild_database()]
        after = (await rag.get_document_by_id(doc_id)).content  # type: ignore[union-attr]
        stored = await rag.document_repository.open_original(doc_id)

    assert rebuilt == [doc_id]
    assert after == before
    assert stored is not None
    assert (stored.sha256, stored.subtitles) == (
        hashlib.sha256(other).hexdigest(),
        None,
    )
    [record] = records
    assert "changed while it was rebuilt" in record.getMessage()


@needs_ffmpeg
async def test_a_full_rebuild_drops_the_file_when_a_hook_names_another(
    temp_db_path, tmp_path, embedded
):
    from haiku.rag.hooks import ConvertedDocument, Hook

    class Elsewhere(Hook):
        async def after_convert(self, client, document):
            converted = document.docling_document
            assert converted.origin is not None
            converted.origin.binary_hash += 1
            return ConvertedDocument(
                uri=document.uri,
                source_id=document.source_id,
                content_type=document.content_type,
                docling_document=converted,
            )

    talk = _talk(tmp_path)
    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        doc = await _ingest(rag, talk)
        assert doc.id is not None
        rag._hooks = [Elsewhere()]

        with capture_logs(documents.logger, logging.WARNING) as records:
            rebuilt = [i async for i in rag.rebuild_database()]

        assert rebuilt == [doc.id]
        assert await rag.document_repository.open_original(doc.id) is None
    assert any(doc.id in r.getMessage() for r in records)


@needs_ffmpeg
async def test_a_descriptions_rebuild_rechunks_a_recording_under_a_new_dimension(
    temp_db_path, tmp_path, monkeypatch, embedded
):
    """The chunks table is recreated, so a recording is re-chunked and re-embedded
    as RECHUNK does, and its `documents` row is left alone."""
    talk = _talk(tmp_path)
    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        doc = await _ingest(rag, talk)
        assert doc.id is not None
        before = len(await rag.chunk_repository.get_by_document_id(doc.id))

    config = _config()
    config.processing.pictures = "description"
    config.embeddings.model.vector_dim = 8
    written: list[tuple[list[str | None], tuple]] = []
    flush = rebuild._flush_rebuild_batch

    async def recording_flush(session, documents, chunks, *, document_columns):
        written.append(([d.id for d in documents], document_columns))
        return await flush(
            session, documents, chunks, document_columns=document_columns
        )

    monkeypatch.setattr(rebuild, "_flush_rebuild_batch", recording_flush)
    async with HaikuRAG(temp_db_path, config=config, skip_validation=True) as rag:
        rebuilt = [i async for i in rag.rebuild_database(mode=RebuildMode.DESCRIPTIONS)]
        chunks = await rag.chunk_repository.get_by_document_id(doc.id)
        width = (await rag.store.chunks_table.schema()).field("vector").type.list_size
        stored = await _stored(rag, doc.id)

    assert rebuilt == [doc.id]
    assert written == [([doc.id], ())]
    assert (len(chunks), width) == (before, 8)
    assert stored is not None


@needs_ffmpeg
async def test_a_recording_is_read_by_range_only(temp_db_path, tmp_path, embedded):
    talk = _talk(tmp_path)
    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        doc = await _ingest(rag, talk)
        assert doc.id is not None
        [chunk, *_] = await rag.chunk_repository.get_by_document_id(doc.id)
        repository = rag.document_repository

        assert await repository.get_original(doc.id) is None
        assert await repository.get_original_and_sha256(doc.id) is None
        assert await rag.chunk_grounding(chunk) is None
        assert await repository.open_original(doc.id) is not None


class _Reorigin(Hook):
    """Changes one field of the converted structure's origin."""

    def __init__(self, field: str):
        self.field = field

    async def after_convert(self, client, document):
        origin = document.docling_document.origin
        assert origin is not None
        if self.field == "binary_hash":
            origin.binary_hash += 1
        elif self.field == "filename":
            origin.filename = "talk.txt"
        else:
            origin.mimetype = "text/plain"
        return document


@needs_ffmpeg
@pytest.mark.parametrize("field", ["binary_hash", "filename", "mimetype"])
async def test_an_ingest_stores_no_file_that_its_final_structure_does_not_name(
    temp_db_path, tmp_path, embedded, field
):
    talk = _talk(tmp_path)
    async with HaikuRAG(temp_db_path, config=_config(), create=True) as rag:
        stored = await _ingest(rag, talk)
        assert stored.id is not None
        rag._hooks = [_Reorigin(field)]

        with capture_logs(documents.logger, logging.WARNING) as records:
            replaced = await _ingest(rag, talk, force=True)
            other = tmp_path / "other.mp3"
            shutil.copy(talk, other)
            shutil.copy(talk.with_suffix(".vtt"), other.with_suffix(".vtt"))
            fresh = await _ingest(rag, other)
        assert replaced.id == stored.id and fresh.id is not None

        assert await rag.document_repository.open_original(stored.id) is None
        assert await rag.document_repository.open_original(fresh.id) is None
    assert sum("stored without its file" in r.getMessage() for r in records) == 2


async def _recording_state(rag: HaikuRAG, document_id: str):
    stored = await rag.document_repository.open_recording(document_id)
    assert stored is not None
    frame = await rag.get_picture_bytes(document_id, decoded_frame_ref(1.0))
    return (
        await stored.original.read(0, stored.original.size),
        stored.original.sha256,
        stored.original.subtitles,
        bool(frame),
    )


@needs_ffmpeg
async def test_a_stored_video_survives_restore_and_rollback(
    temp_db_path, tmp_path, monkeypatch, embedded
):
    video = tmp_path / "talk.mp4"
    shutil.copy(VIDEO, video)
    video.with_suffix(".vtt").write_text(SUBTITLES, encoding="utf-8")
    body = video.read_bytes()
    expected = (body, hashlib.sha256(body).hexdigest(), SUBTITLES, True)

    async with HaikuRAG(temp_db_path, config=_video_config(10.0), create=True) as rag:
        doc = await _ingest(rag, video)
        assert doc.id is not None
        assert await _recording_state(rag, doc.id) == expected

        await rag.store.create_tag("kept")
        await rag.delete_document(doc.id)
        await rag.vacuum(retention_seconds=0)
        await rag.store.restore_tag("kept")
        assert await _recording_state(rag, doc.id) == expected

        async def failing(*args, **kwargs):
            raise RuntimeError("chunk write failed")

        video.write_bytes(body + b"\x00")
        with monkeypatch.context() as patch:
            patch.setattr(rag.chunk_repository, "replace_for_document", failing)
            with pytest.raises(RuntimeError, match="chunk write failed"):
                await _ingest(rag, video, force=True)

        assert await _recording_state(rag, doc.id) == expected
