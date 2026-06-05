import json
import tempfile
from pathlib import Path
from typing import TypedDict

import pytest

from haiku.rag.client import HaikuRAG, RebuildMode


class ChunkData(TypedDict):
    id: str
    document_id: str
    content: str
    content_fts: str
    metadata: str
    order: int


@pytest.mark.vcr()
async def test_rebuild_full(qa_corpus: list[dict[str, str]], temp_db_path):
    """Test full rebuild: converts, chunks, and embeds all documents."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content=qa_corpus[0]["document_extracted"])
        assert doc.id is not None
        assert doc.docling_document is not None

        chunks_before = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(chunks_before) > 0
        chunk_ids_before = {c.id for c in chunks_before}

        processed_ids = [doc_id async for doc_id in client.rebuild_database()]

        assert doc.id in processed_ids

        # Verify DoclingDocument JSON is preserved after rebuild
        doc_after = await client.document_repository.get_by_id(doc.id)
        assert doc_after is not None
        assert doc_after.docling_document is not None
        assert doc_after.docling_version is not None

        chunks_after = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(chunks_after) > 0
        chunk_ids_after = {c.id for c in chunks_after}

        # Chunk IDs should change (chunks are recreated)
        assert chunk_ids_before.isdisjoint(chunk_ids_after)


@pytest.mark.vcr()
async def test_rebuild_embed_only(qa_corpus: list[dict[str, str]], temp_db_path):
    """Test embed-only rebuild: keeps chunks, only regenerates embeddings."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content=qa_corpus[0]["document_extracted"])
        assert doc.id is not None
        original_docling_json = doc.docling_document

        chunks_before = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(chunks_before) > 0
        chunk_ids_before = {c.id for c in chunks_before}
        chunk_contents_before = {c.id: c.content for c in chunks_before}

        processed_ids = [
            doc_id
            async for doc_id in client.rebuild_database(mode=RebuildMode.EMBED_ONLY)
        ]

        assert doc.id in processed_ids

        # DoclingDocument JSON should be unchanged (embed-only doesn't touch documents)
        doc_after = await client.document_repository.get_by_id(doc.id)
        assert doc_after is not None
        assert doc_after.docling_document == original_docling_json

        chunks_after = await client.chunk_repository.get_by_document_id(doc.id)
        chunk_ids_after = {c.id for c in chunks_after}

        # Chunk IDs should be preserved (same chunks, just re-embedded)
        assert chunk_ids_before == chunk_ids_after

        # Content should be identical
        for chunk in chunks_after:
            assert chunk.content == chunk_contents_before[chunk.id]


@pytest.mark.vcr()
async def test_rebuild_embed_only_multi_doc_streams_via_staging(
    qa_corpus: list[dict[str, str]], temp_db_path
):
    """Embed-only rebuild with multiple documents preserves chunks via staging.

    Regression guard for the OOM bug: the previous implementation buffered
    all chunks across all documents in memory before flushing. The current
    streaming implementation copies chunks to a staging table, recreates
    the chunks table, then streams doc-by-doc. This test verifies:

    - chunks survive across multiple documents (correctness),
    - the staging table is dropped at the end (no leak), and
    - the rebuild yields every document with chunks.
    """
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc1 = await client.create_document(content=qa_corpus[0]["document_extracted"])
        doc2 = await client.create_document(content=qa_corpus[1]["document_extracted"])
        assert doc1.id is not None and doc2.id is not None

        chunks_before_1 = await client.chunk_repository.get_by_document_id(doc1.id)
        chunks_before_2 = await client.chunk_repository.get_by_document_id(doc2.id)
        assert chunks_before_1 and chunks_before_2
        ids_before = {c.id for c in chunks_before_1} | {c.id for c in chunks_before_2}

        processed_ids = [
            doc_id
            async for doc_id in client.rebuild_database(mode=RebuildMode.EMBED_ONLY)
        ]

        assert doc1.id in processed_ids
        assert doc2.id in processed_ids

        chunks_after_1 = await client.chunk_repository.get_by_document_id(doc1.id)
        chunks_after_2 = await client.chunk_repository.get_by_document_id(doc2.id)
        ids_after = {c.id for c in chunks_after_1} | {c.id for c in chunks_after_2}

        # Same chunk IDs survive; content unchanged.
        assert ids_before == ids_after
        contents_before = {c.id: c.content for c in chunks_before_1 + chunks_before_2}
        for chunk in chunks_after_1 + chunks_after_2:
            assert chunk.content == contents_before[chunk.id]

        # Staging table was cleaned up.
        tables = (await client.store.db.list_tables()).tables
        assert "chunks_rebuild_staging" not in tables


@pytest.mark.vcr()
async def test_rebuild_drops_leftover_staging_table(
    qa_corpus: list[dict[str, str]], temp_db_path
):
    """Staging table without marker is treated as partial phase 1 and dropped.

    Simulates a phase-1 interruption by creating only the staging table (no
    marker). On the next rebuild ``_resolve_rebuild_recovery`` should drop
    the partial staging — the live chunks table is still authoritative.
    """
    from haiku.rag.client.rebuild import _StagingChunkRecord

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content=qa_corpus[0]["document_extracted"])
        assert doc.id is not None

        # Simulate a partial phase 1 (staging exists, marker absent).
        await client.store.db.create_table(
            "chunks_rebuild_staging", schema=_StagingChunkRecord
        )
        tables = (await client.store.db.list_tables()).tables
        assert "chunks_rebuild_staging" in tables
        assert "chunks_rebuild_marker" not in tables

        processed_ids = [
            doc_id
            async for doc_id in client.rebuild_database(mode=RebuildMode.EMBED_ONLY)
        ]
        assert doc.id in processed_ids

        tables = (await client.store.db.list_tables()).tables
        assert "chunks_rebuild_staging" not in tables
        assert "chunks_rebuild_marker" not in tables


@pytest.mark.vcr()
async def test_rebuild_resumes_phase2_from_staging_after_crash(
    qa_corpus: list[dict[str, str]], temp_db_path
):
    """Marker + staging present → phase 2 resumes from staging instead of
    redoing phase 1.

    Simulates a phase-2 crash: pre-populate staging with the original chunks,
    create the marker, then drop the live chunks table entirely (the worst
    case — crash right after ``recreate_embeddings_table`` succeeded but
    before any phase-2 batch flushed). The rebuild must reconstruct the
    chunks table from staging without losing data.
    """
    from haiku.rag.client.rebuild import (
        _StagingChunkRecord,
        _StagingMarkerRecord,
    )

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content=qa_corpus[0]["document_extracted"])
        assert doc.id is not None
        original_chunks = await client.chunk_repository.get_by_document_id(doc.id)
        assert original_chunks
        original_ids = {c.id for c in original_chunks}
        original_contents = {c.id: c.content for c in original_chunks}

        # Snapshot chunk data into staging (simulating phase 1's output).
        staging = await client.store.db.create_table(
            "chunks_rebuild_staging", schema=_StagingChunkRecord
        )
        await staging.add(
            [
                _StagingChunkRecord(
                    id=c.id or "",
                    document_id=c.document_id or "",
                    content=c.content,
                    metadata=json.dumps(c.metadata),
                    order=c.order,
                )
                for c in original_chunks
            ]
        )

        # Mark phase 1 complete (simulating the marker write that happens
        # just before phase 2 starts).
        marker = await client.store.db.create_table(
            "chunks_rebuild_marker", schema=_StagingMarkerRecord
        )
        await marker.add([_StagingMarkerRecord(id="phase1_complete")])

        # Wipe the live chunks table to simulate a worst-case phase-2 crash
        # after recreate_embeddings_table but before any chunks were
        # written.
        await client.store.db.drop_table("chunks")

        # Recovery: rebuild_database should detect marker+staging and have
        # _rebuild_embed_only skip phase 1.
        processed_ids = [
            doc_id
            async for doc_id in client.rebuild_database(mode=RebuildMode.EMBED_ONLY)
        ]
        assert doc.id in processed_ids

        recovered = await client.chunk_repository.get_by_document_id(doc.id)
        assert {c.id for c in recovered} == original_ids
        for chunk in recovered:
            assert chunk.content == original_contents[chunk.id]

        tables = (await client.store.db.list_tables()).tables
        assert "chunks_rebuild_staging" not in tables
        assert "chunks_rebuild_marker" not in tables


def test_staging_chunk_record_mirrors_chunk_record_schema():
    """``_StagingChunkRecord`` must hold every ``ChunkRecordBase`` field except
    those that are re-derived (``content_fts``) or replaced (``vector``).

    If someone adds a column to ``ChunkRecordBase`` without updating
    ``_StagingChunkRecord``, embed-only rebuilds will silently drop that
    column on every crash-recovery cycle. This test fails loudly instead.
    """
    from haiku.rag.client.rebuild import _StagingChunkRecord
    from haiku.rag.store.engine import ChunkRecordBase

    expected = set(ChunkRecordBase.model_fields) - {"content_fts", "vector"}
    assert set(_StagingChunkRecord.model_fields) == expected


async def test_rebuild_drops_orphan_marker(temp_db_path):
    """Marker without staging is treated as corrupted and dropped.

    No embeddings are needed: ``_resolve_rebuild_recovery`` decides on
    tables before any embed call, and the empty database has no documents
    to embed.
    """
    from haiku.rag.client.rebuild import _StagingMarkerRecord

    async with HaikuRAG(temp_db_path, create=True) as client:
        marker = await client.store.db.create_table(
            "chunks_rebuild_marker", schema=_StagingMarkerRecord
        )
        await marker.add([_StagingMarkerRecord(id="phase1_complete")])

        _ = [
            doc_id
            async for doc_id in client.rebuild_database(mode=RebuildMode.EMBED_ONLY)
        ]

        tables = (await client.store.db.list_tables()).tables
        assert "chunks_rebuild_marker" not in tables
        assert "chunks_rebuild_staging" not in tables


@pytest.mark.vcr()
async def test_rebuild_non_embed_mode_drops_staging_recovery_state(
    qa_corpus: list[dict[str, str]], temp_db_path
):
    """Staging + marker from a prior embed-only crash → dropped on RECHUNK.

    If a user runs a different rebuild mode after a crashed embed-only, the
    staging tables are stale: the new mode recreates chunks from a
    different source (e.g. the stored docling blob), so the staging copy is
    not useful.
    """
    from haiku.rag.client.rebuild import (
        _StagingChunkRecord,
        _StagingMarkerRecord,
    )

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content=qa_corpus[0]["document_extracted"])
        assert doc.id is not None

        await client.store.db.create_table(
            "chunks_rebuild_staging", schema=_StagingChunkRecord
        )
        marker = await client.store.db.create_table(
            "chunks_rebuild_marker", schema=_StagingMarkerRecord
        )
        await marker.add([_StagingMarkerRecord(id="phase1_complete")])

        processed_ids = [
            doc_id async for doc_id in client.rebuild_database(mode=RebuildMode.RECHUNK)
        ]
        assert doc.id in processed_ids

        tables = (await client.store.db.list_tables()).tables
        assert "chunks_rebuild_staging" not in tables
        assert "chunks_rebuild_marker" not in tables


@pytest.mark.vcr()
async def test_rebuild_embed_only_skips_unchanged(
    qa_corpus: list[dict[str, str]], temp_db_path
):
    """Test embed-only rebuild skips chunks with unchanged embeddings."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content=qa_corpus[0]["document_extracted"])
        assert doc.id is not None

        # Get embeddings before rebuild
        records_before = await (
            client.store.chunks_table.query()
            .where(f"document_id = '{doc.id}'")
            .to_pydantic(client.store.ChunkRecord)
        )
        embeddings_before = {rec.id: rec.vector for rec in records_before}

        # Run embed-only rebuild with same embedder - embeddings should be identical
        processed_ids = [
            doc_id
            async for doc_id in client.rebuild_database(mode=RebuildMode.EMBED_ONLY)
        ]
        assert doc.id in processed_ids

        # Get embeddings after rebuild
        records_after = await (
            client.store.chunks_table.query()
            .where(f"document_id = '{doc.id}'")
            .to_pydantic(client.store.ChunkRecord)
        )
        embeddings_after = {rec.id: rec.vector for rec in records_after}

        # Embeddings should be identical (same content, same embedder)
        assert embeddings_before.keys() == embeddings_after.keys()
        for chunk_id in embeddings_before:
            assert embeddings_before[chunk_id] == embeddings_after[chunk_id]


@pytest.mark.vcr()
async def test_rebuild_embed_only_with_changed_vector_dim(
    qa_corpus: list[dict[str, str]], temp_db_path
):
    """Test embed-only rebuild when vector dimension changes.

    This tests the scenario where a database was created with one embedding model
    (e.g., qwen3-embedding:8b with 4096 dims) and rebuild is run with a different
    model (e.g., qwen3-embedding:4b with 2560 dims).

    The Store should use the stored vector_dim for reading existing chunks,
    then rebuild should handle changing to the new dimension.
    """
    import json

    import lancedb
    from lancedb.pydantic import LanceModel, Vector
    from pydantic import Field

    # Step 1: Create a database with normal 2560-dim embeddings
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content=qa_corpus[0]["document_extracted"])
        assert doc.id is not None

        chunks_before = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(chunks_before) > 0
        chunk_data: list[ChunkData] = [
            ChunkData(
                id=c.id or "",
                document_id=c.document_id or "",
                content=c.content,
                content_fts=c.content,
                metadata=json.dumps(c.metadata),
                order=c.order,
            )
            for c in chunks_before
        ]

    # Step 2: Manually recreate chunks table with 4096-dim vectors (simulating old DB)
    db = await lancedb.connect_async(temp_db_path)

    class ChunkRecord4096(LanceModel):
        id: str
        document_id: str
        content: str
        content_fts: str = Field(default="")
        metadata: str = Field(default="{}")
        order: int = Field(default=0)
        vector: Vector(4096) = Field(default_factory=lambda: [0.0] * 4096)  # type: ignore

    await db.drop_table("chunks")
    chunks_table = await db.create_table("chunks", schema=ChunkRecord4096)

    # Insert chunks with 4096-dim fake vectors
    records_4096 = [
        ChunkRecord4096(
            id=c["id"],
            document_id=c["document_id"],
            content=c["content"],
            content_fts=c["content_fts"],
            metadata=c["metadata"],
            order=c["order"],
            vector=[0.1] * 4096,
        )
        for c in chunk_data
    ]
    await chunks_table.add(records_4096)

    # Update settings to reflect the 4096-dim model used
    settings_table = await db.open_table("settings")
    rows = (
        await settings_table.query().where("id = 'settings'").limit(1).to_arrow()
    ).to_pylist()
    settings = json.loads(rows[0]["settings"])
    settings["embeddings"]["model"]["vector_dim"] = 4096
    settings["embeddings"]["model"]["name"] = "qwen3-embedding:8b"
    await settings_table.update(
        {"settings": json.dumps(settings)}, where="id = 'settings'"
    )
    db.close()

    # Step 3: Open with skip_validation (different config) and run embed-only rebuild
    # This should work: Store should use stored vector_dim for reading,
    # then rebuild should migrate to new dimension
    async with HaikuRAG(temp_db_path, skip_validation=True) as client:
        processed_ids = [
            doc_id
            async for doc_id in client.rebuild_database(mode=RebuildMode.EMBED_ONLY)
        ]
        assert doc.id in processed_ids

        # Verify chunks now have 2560-dim embeddings (from current config's model)
        chunks_after = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(chunks_after) == len(chunks_before)

        # Check that embeddings in DB are now 2560-dim
        raw_chunks = (
            await client.store.chunks_table.query()
            .where(f"document_id = '{doc.id}'")
            .to_arrow()
        ).to_pylist()
        for raw_chunk in raw_chunks:
            assert len(raw_chunk["vector"]) == 2560

        # Chunk IDs should be preserved
        chunk_ids_before = {c.id for c in chunks_before}
        chunk_ids_after = {c.id for c in chunks_after}
        assert chunk_ids_before == chunk_ids_after


@pytest.mark.vcr()
async def test_rebuild_rechunk(qa_corpus: list[dict[str, str]], temp_db_path):
    """Test rechunk rebuild: re-chunks from content without accessing source files."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content=qa_corpus[0]["document_extracted"])
        assert doc.id is not None
        assert doc.docling_document is not None

        # Set a fake URI to simulate a document that came from a file
        doc.uri = "file:///nonexistent/path.txt"
        await client.document_repository.update(doc)

        chunks_before = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(chunks_before) > 0
        chunk_ids_before = {c.id for c in chunks_before}
        content_before = doc.content

        processed_ids = [
            doc_id async for doc_id in client.rebuild_database(mode=RebuildMode.RECHUNK)
        ]

        assert doc.id in processed_ids

        # Document content should be unchanged, but docling JSON should be updated
        doc_after = await client.document_repository.get_by_id(doc.id)
        assert doc_after is not None
        assert doc_after.content == content_before
        assert doc_after.docling_document is not None
        assert doc_after.docling_version is not None

        chunks_after = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(chunks_after) > 0
        chunk_ids_after = {c.id for c in chunks_after}

        # Chunk IDs should change (chunks are recreated)
        assert chunk_ids_before.isdisjoint(chunk_ids_after)


@pytest.mark.vcr()
async def test_rebuild_full_with_accessible_source(temp_db_path):
    """FULL rebuild re-ingests from source when the URI is accessible.

    Covers the main path in _rebuild_full (source-accessible branch): the
    document is deleted and re-created from its URI, producing a new ID.
    """
    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.txt"
            source_path.write_text("Fresh content from an accessible file source.")

            original = await client.create_document_from_source(source=source_path)
            assert not isinstance(original, list)
            assert original.id is not None
            original_id = original.id

            processed_ids = [
                doc_id
                async for doc_id in client.rebuild_database(mode=RebuildMode.FULL)
            ]

            # Original doc was deleted and a new one created; the old ID
            # must not appear, and exactly one new ID must have been yielded.
            assert original_id not in processed_ids
            assert len(processed_ids) == 1

            new_doc = await client.get_document_by_id(processed_ids[0])
            assert new_doc is not None
            assert new_doc.uri == source_path.as_uri()
            assert "Fresh content" in new_doc.content


async def test_rebuild_title_only_handles_llm_failure(temp_db_path, monkeypatch):
    """TITLE_ONLY: a failure on one document does not abort the generator.

    The first document raises during title generation (simulated LLM error);
    the second succeeds. Rebuild must log-and-skip the failure, yield only
    the successful document, and persist its new title.
    """
    from haiku.rag.store.models.document import Document

    async with HaikuRAG(temp_db_path, create=True) as client:
        # Skip embedding — TITLE_ONLY only touches documents.
        doc1 = await client.document_repository.create(
            Document(content="doc one body", metadata={})
        )
        doc2 = await client.document_repository.create(
            Document(content="doc two body", metadata={})
        )
        assert doc1.id is not None and doc2.id is not None

        async def fake_generate_title(doc):
            if doc.id == doc1.id:
                raise RuntimeError("simulated LLM failure")
            return "Second Title"

        monkeypatch.setattr(client, "generate_title", fake_generate_title)

        processed_ids = [
            doc_id
            async for doc_id in client.rebuild_database(mode=RebuildMode.TITLE_ONLY)
        ]

        assert processed_ids == [doc2.id]

        refreshed = await client.get_document_by_id(doc2.id)
        assert refreshed is not None
        assert refreshed.title == "Second Title"

        untouched = await client.get_document_by_id(doc1.id)
        assert untouched is not None
        assert untouched.title is None


@pytest.mark.vcr()
async def test_rebuild_full_source_failure_is_logged_and_skipped(
    temp_db_path, monkeypatch
):
    """FULL rebuild logs-and-continues when re-ingesting from source fails.

    Covers _rebuild_full's `except Exception` branch: when
    create_document_from_source raises, the doc is skipped (no yield) and
    the error is logged. Regression guard against silent failures.
    """
    import logging

    from haiku.rag.client import rebuild as rebuild_module

    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.txt"
            source_path.write_text("Content that will vanish by rebuild time.")

            original = await client.create_document_from_source(source=source_path)
            assert not isinstance(original, list)
            assert original.id is not None

            # Force the source rebuild branch to raise.
            async def failing_create(*args, **kwargs):
                raise RuntimeError("simulated ingestion failure")

            monkeypatch.setattr(client, "create_document_from_source", failing_create)

            # Attach directly to the rebuild module's logger rather than
            # relying on caplog — `haiku.rag.logging.get_logger()` (invoked
            # by other tests) sets `propagate=False` on the `haiku.rag`
            # logger, which breaks caplog under xdist ordering.
            records: list[logging.LogRecord] = []

            class _ListHandler(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    records.append(record)

            handler = _ListHandler(level=logging.ERROR)
            rebuild_module.logger.addHandler(handler)
            try:
                processed_ids = [
                    doc_id
                    async for doc_id in client.rebuild_database(mode=RebuildMode.FULL)
                ]
            finally:
                rebuild_module.logger.removeHandler(handler)

            assert processed_ids == []
            assert any(
                "Error recreating document from source" in rec.getMessage()
                for rec in records
            )


@pytest.mark.vcr()
async def test_rebuild_batch_size_flush(temp_db_path, monkeypatch):
    """RECHUNK flushes in batches and yields every document.

    Forces a tiny batch size so three docs trigger at least one mid-loop
    flush plus the final flush. Regression guard for the batched-write path
    in _rebuild_rechunk.
    """
    from haiku.rag.client import rebuild as rebuild_module

    monkeypatch.setattr(rebuild_module, "_REBUILD_BATCH_SIZE", 2)

    async with HaikuRAG(temp_db_path, create=True) as client:
        ids: list[str] = []
        for i in range(3):
            doc = await client.create_document(content=f"batch flush doc {i}")
            assert doc.id is not None
            ids.append(doc.id)

        processed = [
            doc_id async for doc_id in client.rebuild_database(mode=RebuildMode.RECHUNK)
        ]

        assert sorted(processed) == sorted(ids)
        for doc_id in ids:
            chunks = await client.chunk_repository.get_by_document_id(doc_id)
            assert len(chunks) > 0


async def test_rebuild_descriptions_requires_description_mode(temp_db_path):
    """Calling rebuild --descriptions without `processing.pictures='description'`
    is a config error: the user has nothing to gain and the resulting state is
    indistinguishable from a plain --rechunk."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        with pytest.raises(ValueError, match="processing.pictures"):
            async for _ in client.rebuild_database(mode=RebuildMode.DESCRIPTIONS):
                pass


@pytest.mark.vcr()
async def test_rebuild_descriptions_patches_blob_and_chunks(temp_db_path, monkeypatch):
    """End-to-end: ingest a doc with a picture (no VLM at ingest), then run
    rebuild --descriptions with the VLM mocked. After the rebuild:

    - the docling blob has the description in meta;
    - the chunk text picks it up;
    - the docling_pages blob is preserved untouched.

    The pages assertion guards against a foot-gun in compress_docling_split:
    the docling document loaded via get_docling_document() never carries
    pages (they live in a separate column), so calling set_docling() after
    patching would write pages_bytes=None and silently destroy page rasters
    on disk — breaking visualize_chunk for the affected docs."""
    from haiku.rag.client.documents import _store_document_with_chunks
    from haiku.rag.config import AppConfig
    from haiku.rag.store.models.document import Document
    from tests.store.test_document_items import _docling_doc_with_picture

    docling_doc = _docling_doc_with_picture()

    config = AppConfig()
    config.processing.pictures = "description"

    async with HaikuRAG(temp_db_path, config=config, create=True) as rag:
        document = Document(content="x", uri="test://doc")
        document.set_docling(docling_doc)
        created = await _store_document_with_chunks(rag, document, [], docling_doc)
        assert created.id is not None

        # _docling_doc_with_picture has no PageItems, so set_docling leaves
        # docling_pages as None. Inject sentinel bytes to stand in for what
        # a real ingest with generate_page_images=True would store.
        sentinel_pages = b"\x80SENTINEL_PAGE_BYTES"
        await rag.store.documents_table.update(
            {"docling_pages": sentinel_pages}, where=f"id = '{created.id}'"
        )

        from_blob = (
            await rag.document_repository.get_by_id(created.id)
        ).get_docling_document()  # type: ignore[union-attr]
        assert from_blob is not None and from_blob.pictures
        # No description in the freshly-ingested doc
        meta = from_blob.pictures[0].meta
        existing = (
            getattr(getattr(meta, "description", None), "text", None) if meta else None
        )
        assert not existing

        async def fake_describe(image_bytes_by_ref, *, config):
            return {ref: "A red square (mocked)." for ref in image_bytes_by_ref}

        monkeypatch.setattr(
            "haiku.rag.client.rebuild.describe_pictures", fake_describe, raising=False
        )
        # The function is imported lazily inside _patch_picture_descriptions, so
        # patch the module-of-origin too.
        monkeypatch.setattr(
            "haiku.rag.providers.picture_description.describe_pictures",
            fake_describe,
        )

        processed = [
            doc_id
            async for doc_id in rag.rebuild_database(mode=RebuildMode.DESCRIPTIONS)
        ]
        assert created.id in processed

        # The stored docling blob now has the description
        after = await rag.document_repository.get_by_id(created.id)
        assert after is not None
        after_doc = after.get_docling_document()
        assert after_doc is not None and after_doc.pictures
        meta = after_doc.pictures[0].meta
        text = (
            getattr(getattr(meta, "description", None), "text", None) if meta else None
        )
        assert text == "A red square (mocked)."

        # And the description reaches chunk text
        chunks = await rag.chunk_repository.get_by_document_id(created.id)
        assert any("A red square (mocked)." in (c.content or "") for c in chunks)

        # docling_pages must survive untouched — see docstring.
        assert after.docling_pages == sentinel_pages


@pytest.mark.vcr()
async def test_rebuild_descriptions_skips_already_described(temp_db_path, monkeypatch):
    """Pictures that already carry a description must not be re-sent to the
    VLM, so the operation is safe to re-run after a partial failure."""
    from haiku.rag.client.documents import _store_document_with_chunks
    from haiku.rag.config import AppConfig
    from haiku.rag.store.models.document import Document
    from tests.store.test_document_items import _docling_doc_with_picture

    docling_doc = _docling_doc_with_picture()

    # Pre-populate the description directly on the docling document
    from docling_core.types.doc.document import DescriptionMetaField, PictureMeta

    docling_doc.pictures[0].meta = PictureMeta(
        description=DescriptionMetaField(text="Pre-existing description.")
    )

    config = AppConfig()
    config.processing.pictures = "description"

    async with HaikuRAG(temp_db_path, config=config, create=True) as rag:
        document = Document(content="x", uri="test://doc")
        document.set_docling(docling_doc)
        created = await _store_document_with_chunks(rag, document, [], docling_doc)
        assert created.id is not None

        called_with: list[dict[str, bytes]] = []

        async def fake_describe(image_bytes_by_ref, *, config):
            called_with.append(image_bytes_by_ref)
            return {ref: "Should not be used." for ref in image_bytes_by_ref}

        monkeypatch.setattr(
            "haiku.rag.providers.picture_description.describe_pictures",
            fake_describe,
        )

        async for _ in rag.rebuild_database(mode=RebuildMode.DESCRIPTIONS):
            pass

        # VLM was never called for this picture (it already had a description)
        assert called_with == [] or all(not d for d in called_with)

        after = await rag.document_repository.get_by_id(created.id)
        assert after is not None
        after_doc = after.get_docling_document()
        assert after_doc is not None
        meta = after_doc.pictures[0].meta
        text = (
            getattr(getattr(meta, "description", None), "text", None) if meta else None
        )
        assert text == "Pre-existing description."


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_patch_picture_descriptions_returns_zero_for_doc_without_pictures(
    temp_db_path,
):
    """A document with no pictures returns 0 without ever calling the VLM."""
    from haiku.rag.client.rebuild import _patch_picture_descriptions
    from haiku.rag.config import AppConfig

    config = AppConfig()
    config.processing.pictures = "description"

    async with HaikuRAG(temp_db_path, config=config, create=True) as rag:
        doc = await rag.create_document(content="Just text, no pictures.")
        assert doc.id is not None
        n = await _patch_picture_descriptions(rag, doc)
        assert n == 0


@pytest.mark.asyncio
async def test_patch_picture_descriptions_warns_on_missing_bytes(temp_db_path, caplog):
    """When the docling blob has pictures but document_items.picture_data is
    empty (e.g. legacy DB ingested before A2b), the helper logs a warning
    and returns 0 instead of trying to drive the VLM with no input."""
    import logging

    from haiku.rag.client.documents import _store_document_with_chunks
    from haiku.rag.client.rebuild import _patch_picture_descriptions
    from haiku.rag.config import AppConfig
    from haiku.rag.store.models.document import Document
    from tests.store.test_document_items import _docling_doc_with_picture

    docling_doc = _docling_doc_with_picture()
    config = AppConfig()
    config.processing.pictures = "description"

    async with HaikuRAG(temp_db_path, config=config, create=True) as rag:
        document = Document(content="x", uri="test://doc")
        document.set_docling(docling_doc)
        created = await _store_document_with_chunks(rag, document, [], docling_doc)
        assert created.id is not None

        # Wipe the stored picture bytes to simulate a doc that knows about
        # pictures but doesn't have them on disk.
        await rag.store.document_items_table.update(
            {"picture_data": None},
            where=f"document_id = '{created.id}' AND label = 'picture'",
        )

        # Capture warnings directly off the rebuild module logger — the
        # haiku.rag parent logger is configured non-propagating elsewhere
        # in the suite so caplog can miss records.
        from haiku.rag.client import rebuild as rebuild_module

        records: list[logging.LogRecord] = []

        class _ListHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _ListHandler(level=logging.WARNING)
        rebuild_module.logger.addHandler(handler)
        try:
            n = await _patch_picture_descriptions(rag, created)
        finally:
            rebuild_module.logger.removeHandler(handler)

        assert n == 0
        assert any("no stored picture bytes" in r.getMessage() for r in records)


@pytest.mark.asyncio
async def test_patch_picture_descriptions_skips_when_all_already_described(
    temp_db_path, monkeypatch
):
    """If every picture already has meta.description.text, the helper does
    not call the VLM and returns 0."""
    from docling_core.types.doc.document import DescriptionMetaField, PictureMeta

    from haiku.rag.client.documents import _store_document_with_chunks
    from haiku.rag.client.rebuild import _patch_picture_descriptions
    from haiku.rag.config import AppConfig
    from haiku.rag.store.models.document import Document
    from tests.store.test_document_items import _docling_doc_with_picture

    docling_doc = _docling_doc_with_picture()
    docling_doc.pictures[0].meta = PictureMeta(
        description=DescriptionMetaField(text="Pre-described.")
    )

    config = AppConfig()
    config.processing.pictures = "description"

    async with HaikuRAG(temp_db_path, config=config, create=True) as rag:
        document = Document(content="x", uri="test://doc")
        document.set_docling(docling_doc)
        created = await _store_document_with_chunks(rag, document, [], docling_doc)
        assert created.id is not None

        called = False

        async def fake_describe(*args, **kwargs):
            nonlocal called
            called = True
            return {}

        monkeypatch.setattr(
            "haiku.rag.providers.picture_description.describe_pictures",
            fake_describe,
        )

        n = await _patch_picture_descriptions(rag, created)
        assert n == 0
        assert called is False


@pytest.mark.asyncio
async def test_rebuild_descriptions_raises_when_blob_is_missing(
    temp_db_path, monkeypatch
):
    """Documents without a stored docling blob can't be re-described —
    surface a clear error pointing the user at full rebuild instead."""
    from haiku.rag.client.documents import _store_document_with_chunks
    from haiku.rag.config import AppConfig
    from haiku.rag.store.models.document import Document
    from tests.store.test_document_items import _docling_doc_with_picture

    docling_doc = _docling_doc_with_picture()
    config = AppConfig()
    config.processing.pictures = "description"

    async with HaikuRAG(temp_db_path, config=config, create=True) as rag:
        document = Document(content="x", uri="test://doc")
        document.set_docling(docling_doc)
        created = await _store_document_with_chunks(rag, document, [], docling_doc)
        assert created.id is not None

        # Force the stored doc to come back without a docling blob.
        await rag.store.documents_table.update(
            {"docling_document": None}, where=f"id = '{created.id}'"
        )

        with pytest.raises(ValueError, match="rebuild --descriptions requires"):
            async for _ in rag.rebuild_database(mode=RebuildMode.DESCRIPTIONS):
                pass


async def _add_chunk(client: HaikuRAG, vector: list[float]) -> str:
    """Insert a chunk row directly, bypassing the embedder."""
    record = client.store.ChunkRecord(
        id="chunk-1",
        document_id="doc-1",
        content="hello",
        content_fts="hello",
        metadata="{}",
        order=0,
        vector=vector,
    )
    await client.store.chunks_table.add([record])
    return record.id


async def _stored_embedding_name(client: HaikuRAG) -> str:
    from haiku.rag.store.repositories.settings import SettingsRepository

    settings = await SettingsRepository(client.store).get_current_settings()
    return settings["embeddings"]["model"]["name"]


async def test_rebuild_set_embedder_adopts_identity_without_reembedding(temp_db_path):
    """SET_EMBEDDER updates stored embedder identity and leaves vectors untouched."""
    from haiku.rag.config import AppConfig

    dim = AppConfig().embeddings.model.vector_dim
    sentinel = [0.5] * dim

    async with HaikuRAG(temp_db_path, create=True) as client:
        await _add_chunk(client, sentinel)

    drift = AppConfig()
    drift.embeddings.model.name = "different-model"

    async with HaikuRAG(temp_db_path, config=drift, skip_validation=True) as client:
        async for _ in client.rebuild_database(mode=RebuildMode.SET_EMBEDDER):
            pass

        assert await _stored_embedding_name(client) == "different-model"

        rows = (await client.store.chunks_table.query().to_arrow()).to_pylist()
        assert len(rows) == 1
        assert rows[0]["vector"] == pytest.approx(sentinel)


async def test_rebuild_set_embedder_works_on_empty_database(temp_db_path):
    """SET_EMBEDDER reconciles even with no documents (preflight must be bypassed)."""
    from haiku.rag.app import HaikuRAGApp
    from haiku.rag.config import AppConfig

    async with HaikuRAG(temp_db_path, create=True):
        pass

    drift = AppConfig()
    drift.embeddings.model.name = "different-model"

    app = HaikuRAGApp(db_path=temp_db_path, config=drift)
    await app.rebuild(mode=RebuildMode.SET_EMBEDDER)

    async with HaikuRAG(temp_db_path, config=drift, skip_validation=True) as client:
        assert await _stored_embedding_name(client) == "different-model"


async def test_rebuild_set_embedder_raises_on_vector_dim_mismatch(temp_db_path):
    """SET_EMBEDDER refuses when the vector dimension changed — a full rebuild is needed."""
    from haiku.rag.config import AppConfig
    from haiku.rag.store.repositories.settings import ConfigMismatchError

    async with HaikuRAG(temp_db_path, create=True):
        pass

    drift = AppConfig()
    drift.embeddings.model.vector_dim = 9999

    async with HaikuRAG(temp_db_path, config=drift, skip_validation=True) as client:
        with pytest.raises(ConfigMismatchError):
            async for _ in client.rebuild_database(mode=RebuildMode.SET_EMBEDDER):
                pass
