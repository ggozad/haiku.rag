import pytest
from docling_core.types.doc import DocumentOrigin, TrackSource

from haiku.rag.client import HaikuRAG, RebuildMode
from tests.documents import docling_doc, timed_docling_doc


@pytest.fixture
def fixed_embeddings(monkeypatch):
    from haiku.rag import embeddings

    async def embed(chunks, embedder, config=None):
        for chunk in chunks:
            if chunk.embedding is None:
                chunk.embedding = [0.1] * embedder.vector_dim
        return chunks

    monkeypatch.setattr(embeddings, "embed_chunks", embed)


@pytest.fixture
def refreshes(monkeypatch):
    calls: list[str] = []

    async def refresh(session, client, *, source, **kwargs):
        calls.append(source)
        raise RuntimeError("conversion failed")

    monkeypatch.setattr("haiku.rag.client.rebuild.create_document_from_source", refresh)
    return calls


async def _import_recording(
    rag: HaikuRAG, uri: str, mimetype: str = "video/mp4"
) -> str:
    doc = timed_docling_doc()
    doc.origin = DocumentOrigin(mimetype=mimetype, filename="talk", binary_hash=1)
    stored = await rag.import_document(doc, await rag.chunk(doc), uri=uri)
    assert stored.id is not None
    return stored.id


async def _snapshot(rag: HaikuRAG, doc_id: str):
    stored = await rag.document_repository.get_by_id(doc_id, include_blobs=True)
    assert stored is not None
    docling = stored.get_docling_document()
    assert docling is not None
    items = (
        await rag.document_item_repository.get_items_in_ranges({doc_id: (0, 100)})
    )[doc_id]
    return (
        [(i.self_ref, i.source) for i in [*docling.texts, *docling.pictures]],
        [(i.self_ref, i.start_time, i.end_time) for i in items],
        await rag.document_item_repository.get_all_picture_data(doc_id),
    )


@pytest.mark.parametrize("mimetype", ["video/mp4", "audio/mpeg"])
@pytest.mark.parametrize("source_exists", [True, False], ids=["present", "missing"])
async def test_full_rebuild_keeps_a_recording_transcript(
    temp_db_path, tmp_path, fixed_embeddings, refreshes, source_exists, mimetype
):
    media = tmp_path / "talk"
    if source_exists:
        media.write_bytes(b"not really a recording")

    async with HaikuRAG(temp_db_path, create=True) as rag:
        doc_id = await _import_recording(rag, media.as_uri(), mimetype)
        before = await _snapshot(rag, doc_id)

        processed = [d async for d in rag.rebuild_database(mode=RebuildMode.FULL)]

        assert processed == [doc_id]
        assert refreshes == []
        sources, item_times, pictures = await _snapshot(rag, doc_id)
        assert (sources, item_times, pictures) == before
        assert all(isinstance(s[0], TrackSource) for _, s in sources)
        assert len(pictures) == 2 and all(pictures.values())
        chunks = await rag.chunk_repository.get_by_document_id(doc_id)
        assert chunks and all(c.metadata.get("time_range") for c in chunks)


async def test_full_rebuild_still_refreshes_other_documents(
    temp_db_path, tmp_path, fixed_embeddings, refreshes
):
    page = tmp_path / "notes.md"
    page.write_text("# Notes\n\nPlain text.")

    async with HaikuRAG(temp_db_path, create=True) as rag:
        await _import_recording(rag, (tmp_path / "talk.mp4").as_uri())
        notes = docling_doc("notes", "Plain text.")
        await rag.import_document(notes, await rag.chunk(notes), uri=page.as_uri())

        [d async for d in rag.rebuild_database(mode=RebuildMode.FULL)]

    assert refreshes == [page.as_uri()]
