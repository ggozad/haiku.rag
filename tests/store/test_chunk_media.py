import pytest
from docling_core.types.doc import DocumentOrigin

from haiku.rag.client import HaikuRAG, RebuildMode
from haiku.rag.config import AppConfig
from haiku.rag.store.models.chunk import Chunk, chunk_media
from tests.documents import docling_doc, timed_docling_doc

SPEECH = {
    "doc_item_refs": ["#/texts/0", "#/texts/1"],
    "labels": ["text", "text"],
    "time_range": [0.0, 7.6],
}
FRAME = {
    "doc_item_refs": ["#/pictures/1"],
    "labels": ["picture"],
    "time_range": [10.0, 10.001],
}
PDF_TEXT = {"doc_item_refs": ["#/texts/0"], "labels": ["text"]}
PDF_PICTURE = {"doc_item_refs": ["#/pictures/0"], "labels": ["picture"]}

MULTIMODAL = AppConfig.model_validate(
    {
        "embeddings": {
            "model": {
                "provider": "vllm",
                "name": "embed-vl",
                "vector_dim": 4,
                "multimodal": True,
            }
        }
    }
)


@pytest.fixture
def fixed_embeddings(monkeypatch):
    from haiku.rag import embeddings

    async def embed(chunks, embedder, config=None):
        for chunk in chunks:
            chunk.embedding = [0.1] * embedder.vector_dim
        return chunks

    monkeypatch.setattr(embeddings, "embed_chunks", embed)


@pytest.mark.parametrize(
    ("metadata", "columns"),
    [
        (SPEECH, ("speech", 0.0, 7.6)),
        (FRAME, ("frame", 10.0, 10.001)),
        (PDF_TEXT, (None, None, None)),
        (PDF_PICTURE, (None, None, None)),
    ],
    ids=["speech", "frame", "pdf-text", "pdf-picture"],
)
def test_chunk_media_reads_the_kind_and_times_from_the_metadata(metadata, columns):
    assert chunk_media(metadata) == columns


async def _rows(rag: HaikuRAG) -> dict[str, tuple]:
    table = await rag.store.chunks_table.query().to_arrow()
    return {
        row["content"]: (row["media"], row["start_time"], row["end_time"])
        for row in table.to_pylist()
    }


def _chunks(dim: int) -> list[Chunk]:
    return [
        Chunk(
            document_id="doc-1",
            content=name,
            metadata=metadata,
            order=order,
            embedding=[0.1] * dim,
        )
        for order, (name, metadata) in enumerate(
            [
                ("speech", SPEECH),
                ("frame", FRAME),
                ("pdf-text", PDF_TEXT),
                ("pdf-picture", PDF_PICTURE),
            ]
        )
    ]


EXPECTED = {
    "speech": ("speech", 0.0, 7.6),
    "frame": ("frame", 10.0, 10.001),
    "pdf-text": (None, None, None),
    "pdf-picture": (None, None, None),
}


async def test_created_chunks_store_their_media_columns(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        await rag.chunk_repository.create(
            _chunks(rag.store._config.embeddings.model.vector_dim)
        )

        assert await _rows(rag) == EXPECTED


async def test_replaced_chunks_store_their_media_columns(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        await rag.chunk_repository.replace_for_document(
            "doc-1", _chunks(rag.store._config.embeddings.model.vector_dim)
        )

        assert await _rows(rag) == EXPECTED


async def _import_recording_and_pdf(rag: HaikuRAG) -> None:
    video = timed_docling_doc()
    video.origin = DocumentOrigin(
        mimetype="video/mp4", filename="talk.mp4", binary_hash=1
    )
    await rag.import_document(video, await rag.chunk(video), uri="file:///talk.mp4")
    pdf = docling_doc("paper", "The tank failed.")
    await rag.import_document(pdf, await rag.chunk(pdf), uri="file:///paper.pdf")


async def _media_of(rag: HaikuRAG) -> list[tuple]:
    table = await rag.store.chunks_table.query().to_arrow()
    return sorted(
        (row["media"] or "", row["start_time"] or 0.0, row["end_time"] or 0.0)
        for row in table.to_pylist()
    )


async def test_imported_documents_store_their_media_columns(
    temp_db_path, fixed_embeddings
):
    async with HaikuRAG(temp_db_path, config=MULTIMODAL, create=True) as rag:
        await _import_recording_and_pdf(rag)

        assert await _media_of(rag) == [
            ("", 0.0, 0.0),
            ("frame", 0.0, 0.001),
            ("frame", 10.0, 10.001),
            ("speech", 0.0, 12.12),
        ]


async def test_an_embed_only_rebuild_keeps_the_media_columns(
    temp_db_path, fixed_embeddings
):
    async with HaikuRAG(temp_db_path, config=MULTIMODAL, create=True) as rag:
        await _import_recording_and_pdf(rag)
        before = await _media_of(rag)

        async for _ in rag.rebuild_database(mode=RebuildMode.EMBED_ONLY):
            pass

        assert await _media_of(rag) == before


async def test_an_embed_only_rebuild_resumed_from_staging_keeps_the_media_columns(
    temp_db_path, fixed_embeddings
):
    from haiku.rag.client.rebuild import _StagingChunkRecord, _StagingMarkerRecord

    config = MULTIMODAL.model_copy(deep=True)
    config.storage.auto_vacuum = False
    async with HaikuRAG(temp_db_path, config=config, create=True) as rag:
        await _import_recording_and_pdf(rag)
        before = await _media_of(rag)
        rows = (await rag.store.chunks_table.query().to_arrow()).to_pylist()
        staging = await rag.store.db.create_table(
            "chunks_rebuild_staging", schema=_StagingChunkRecord
        )
        await staging.add(
            [
                _StagingChunkRecord(
                    id=r["id"],
                    document_id=r["document_id"],
                    content=r["content"],
                    metadata=r["metadata"],
                    order=r["order"],
                )
                for r in rows
            ]
        )
        marker = await rag.store.db.create_table(
            "chunks_rebuild_marker", schema=_StagingMarkerRecord
        )
        await marker.add([_StagingMarkerRecord(id="phase1_complete")])
        await rag.store.db.drop_table("chunks")

        async for _ in rag.rebuild_database(mode=RebuildMode.EMBED_ONLY):
            pass

        assert await _media_of(rag) == before
