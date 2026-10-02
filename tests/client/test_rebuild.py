import json
import tempfile
from pathlib import Path
from typing import TypedDict

import pytest

from haiku.rag.client import HaikuRAG, RebuildMode
from haiku.rag.config import get_config
from tests.conftest import (
    capture_logs,
    for_path,
    run_with_lance_memory_pool,
    writing,
)


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
        doc_after = await client.document_repository.get_by_id(
            doc.id, include_blobs=True
        )
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
        doc_after = await client.document_repository.get_by_id(
            doc.id, include_blobs=True
        )
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
    - the staging table is dropped at the end (no leak),
    - the rebuild yields every document with chunks, and
    - the recreated chunks table carries an FTS index covering its rows.
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

        # Phase 2 recreates the chunks table, so it owns the FTS index.
        fts = [
            index
            for index in await client.store.chunks_table.list_indices()
            if index.index_type == "FTS"
        ]
        assert len(fts) == 1
        stats = await client.store.chunks_table.index_stats(fts[0].name)
        assert stats.num_indexed_rows > 0


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

    # auto_vacuum off: this test drops the chunks table by hand to simulate a
    # crash, where no background vacuum would be in flight. Leaving it on lets
    # create_document's scheduled optimize race the raw drop_table ("Directory
    # not empty").
    config = get_config().model_copy(deep=True)
    config.storage.auto_vacuum = False

    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
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
    from haiku.rag.store.schema import ChunkRecordBase

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


async def test_rebuild_embed_only_covers_the_index_from_the_first_flush(
    temp_db_path, monkeypatch
):
    """A rebuild abandoned after a flushed batch leaves the FTS index
    covering the flushed rows."""
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel

    from haiku.rag import embeddings as embeddings_module
    from haiku.rag.client import rebuild as rebuild_module
    from haiku.rag.config import get_config
    from haiku.rag.store.models.chunk import Chunk

    monkeypatch.setattr(rebuild_module, "_REBUILD_BATCH_SIZE", 1)

    async def keep_embeddings(chunks, embedder, config=None):
        for chunk in chunks:
            if chunk.embedding is None:
                chunk.embedding = [0.1] * embedder.vector_dim
        return chunks

    monkeypatch.setattr(embeddings_module, "embed_chunks", keep_embeddings)

    async with HaikuRAG(temp_db_path, create=True) as client:
        dim = get_config().embeddings.model.vector_dim
        for name in ("one", "two"):
            doc = DoclingDocument(name=name)
            doc.add_text(label=DocItemLabel.TEXT, text=f"document {name}")
            await client.import_document(
                doc,
                [Chunk(content=f"document {name}", embedding=[0.1] * dim, order=0)],
                uri=f"test://{name}",
            )

        rebuild = client.rebuild_database(mode=RebuildMode.EMBED_ONLY)
        await anext(rebuild)
        await anext(rebuild)
        await rebuild.aclose()

        fts = [
            index
            for index in await client.store.chunks_table.list_indices()
            if index.index_type == "FTS"
        ]
        assert len(fts) == 1
        stats = await client.store.chunks_table.index_stats(fts[0].name)
        assert stats.num_indexed_rows > 0


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

        # Document content and docling JSON are unchanged
        doc_after = await client.document_repository.get_by_id(
            doc.id, include_blobs=True
        )
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
    document is refreshed in place, keeping its ID. The source bytes are
    unchanged since ingestion, so this also pins that the refresh bypasses the
    revision and MD5 short-circuits instead of returning the document
    untouched.
    """
    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.txt"
            source_path.write_text("Fresh content from an accessible file source.")

            original = await client.create_document_from_source(source=source_path)
            assert not isinstance(original, list)
            assert original.id is not None
            original_id = original.id
            chunk_ids_before = await _chunk_ids(client, original_id)
            assert chunk_ids_before

            processed_ids = [
                doc_id
                async for doc_id in client.rebuild_database(mode=RebuildMode.FULL)
            ]

            assert processed_ids == [original_id]

            refreshed = await client.get_document_by_id(original_id)
            assert refreshed is not None
            assert refreshed.uri == source_path.as_uri()
            assert "Fresh content" in refreshed.content

            # The table keeps its rows through a same-dimension FULL, so the
            # refresh has to replace the document's chunks, not add to them.
            chunk_ids_after = await _chunk_ids(client, original_id)
            assert chunk_ids_after
            assert chunk_ids_after.isdisjoint(chunk_ids_before)
            assert len(chunk_ids_after) == len(chunk_ids_before)
            assert await client.store.chunks_table.count_rows() == len(chunk_ids_after)


async def test_rebuild_title_only_reads_structural_title(temp_db_path):
    """TITLE_ONLY takes the title from the stored docling structure, so it never
    reaches the LLM for a document that carries one."""
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel

    from haiku.rag.store.models.document import Document

    docling_doc = DoclingDocument(name="structured")
    docling_doc.add_text(label=DocItemLabel.TITLE, text="The Stored Title")

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = Document(content="body text", metadata={})
        doc.set_docling(docling_doc)
        created = await client.document_repository.create(doc)
        assert created.id is not None

        processed_ids = [
            doc_id
            async for doc_id in client.rebuild_database(mode=RebuildMode.TITLE_ONLY)
        ]

        assert processed_ids == [created.id]
        refreshed = await client.get_document_by_id(created.id)
        assert refreshed is not None
        assert refreshed.title == "The Stored Title"


async def test_rebuild_title_only_writes_one_meta_version_per_batch(
    temp_db_path, monkeypatch
):
    from haiku.rag.client import rebuild as rebuild_module
    from haiku.rag.config import AppConfig
    from haiku.rag.store.models.document import Document

    monkeypatch.setattr(rebuild_module, "_REBUILD_BATCH_SIZE", 2)

    async def fake_generate_title(config, doc):
        return f"Title {doc.content}"

    monkeypatch.setattr("haiku.rag.client.rebuild.generate_title", fake_generate_title)

    config = AppConfig()
    config.storage.auto_vacuum = False

    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        docs = [
            await client.document_repository.create(Document(content=f"doc {i}"))
            for i in range(3)
        ]
        before = await client.store.document_meta_table.version()

        processed = [
            doc_id
            async for doc_id in client.rebuild_database(mode=RebuildMode.TITLE_ONLY)
        ]

        assert sorted(processed) == sorted(d.id for d in docs)
        assert await client.store.document_meta_table.version() == before + 2
        for doc in docs:
            assert doc.id is not None
            refreshed = await client.get_document_by_id(doc.id)
            assert refreshed is not None
            assert refreshed.title == f"Title {doc.content}"


async def test_rebuild_title_only_yields_only_saved_titles(temp_db_path, monkeypatch):
    from haiku.rag.client import rebuild as rebuild_module
    from haiku.rag.store.models.document import Document

    monkeypatch.setattr(rebuild_module, "_REBUILD_BATCH_SIZE", 2)

    async def fake_generate_title(config, doc):
        return "Saved"

    monkeypatch.setattr("haiku.rag.client.rebuild.generate_title", fake_generate_title)

    async with HaikuRAG(temp_db_path, create=True) as client:
        for i in range(3):
            await client.document_repository.create(Document(content=f"doc {i}"))

        rebuild = client.rebuild_database(mode=RebuildMode.TITLE_ONLY)
        first = await anext(rebuild)
        await rebuild.aclose()

        saved = await client.get_document_by_id(first)
        assert saved is not None
        assert saved.title == "Saved"


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

        async def fake_generate_title(config, doc):
            if doc.id == doc1.id:
                raise RuntimeError("simulated LLM failure")
            return "Second Title"

        monkeypatch.setattr(
            "haiku.rag.client.rebuild.generate_title", fake_generate_title
        )

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
async def test_rebuild_full_source_failure_falls_back_to_stored_content(
    temp_db_path, monkeypatch
):
    """A failed source refresh must never cost the document.

    Covers _rebuild_full's `except Exception` branch: when the refresh raises,
    the document keeps its stored row and is rebuilt from stored content, so it
    is still readable and still searchable afterwards.
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

            # Force the source refresh to raise.
            async def failing_refresh(*args, **kwargs):
                raise RuntimeError("simulated ingestion failure")

            monkeypatch.setattr(
                rebuild_module, "create_document_from_source", failing_refresh
            )

            with capture_logs(rebuild_module.logger, logging.WARNING) as records:
                processed_ids = [
                    doc_id
                    async for doc_id in client.rebuild_database(mode=RebuildMode.FULL)
                ]

            assert processed_ids == [original.id]

            survivor = await client.get_document_by_id(original.id)
            assert survivor is not None
            assert "Content that will vanish" in survivor.content

            chunks = await client.chunk_repository.get_by_document_id(original.id)
            assert chunks

            assert any(
                "falling back to stored content" in rec.getMessage() for rec in records
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
        created = await _store_document_with_chunks(
            writing(rag), document, [], docling_doc
        )
        assert created.id is not None

        # _docling_doc_with_picture has no PageItems, so set_docling leaves
        # docling_pages as None. Inject sentinel bytes to stand in for what
        # a real ingest with generate_page_images=True would store.
        sentinel_pages = b"\x80SENTINEL_PAGE_BYTES"
        await rag.store.documents_table.update(
            {"docling_pages": sentinel_pages}, where=f"id = '{created.id}'"
        )

        from_blob = (
            await rag.document_repository.get_by_id(created.id, include_blobs=True)
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
        after = await rag.document_repository.get_by_id(created.id, include_blobs=True)
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
        created = await _store_document_with_chunks(
            writing(rag), document, [], docling_doc
        )
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

        after = await rag.document_repository.get_by_id(created.id, include_blobs=True)
        assert after is not None
        after_doc = after.get_docling_document()
        assert after_doc is not None
        meta = after_doc.pictures[0].meta
        text = (
            getattr(getattr(meta, "description", None), "text", None) if meta else None
        )
        assert text == "Pre-existing description."


@pytest.mark.vcr()
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
        n = await _patch_picture_descriptions(writing(rag), doc)
        assert n == 0


async def test_patch_picture_descriptions_warns_on_missing_bytes(temp_db_path):
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
        created = await _store_document_with_chunks(
            writing(rag), document, [], docling_doc
        )
        assert created.id is not None

        # Wipe the stored picture bytes to simulate a doc that knows about
        # pictures but doesn't have them on disk.
        await rag.store.document_items_table.update(
            {"picture_data": None},
            where=f"document_id = '{created.id}' AND label = 'picture'",
        )

        from haiku.rag.client import rebuild as rebuild_module

        with capture_logs(rebuild_module.logger, logging.WARNING) as records:
            n = await _patch_picture_descriptions(writing(rag), created)

        assert n == 0
        assert any("no stored picture bytes" in r.getMessage() for r in records)


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
        created = await _store_document_with_chunks(
            writing(rag), document, [], docling_doc
        )
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

        n = await _patch_picture_descriptions(writing(rag), created)
        assert n == 0
        assert called is False


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
        created = await _store_document_with_chunks(
            writing(rag), document, [], docling_doc
        )
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

    app = HaikuRAGApp(scope=for_path(temp_db_path, drift), config=drift)
    await app.rebuild(mode=RebuildMode.SET_EMBEDDER)

    async with HaikuRAG(temp_db_path, config=drift, skip_validation=True) as client:
        assert await _stored_embedding_name(client) == "different-model"


async def test_rebuild_set_embedder_raises_on_vector_dim_mismatch(temp_db_path):
    """SET_EMBEDDER refuses when the vector dimension changed — a full rebuild is needed."""
    from haiku.rag.config import AppConfig
    from haiku.rag.store.exceptions import ConfigMismatchError

    async with HaikuRAG(temp_db_path, create=True):
        pass

    drift = AppConfig()
    drift.embeddings.model.vector_dim = 9999

    async with HaikuRAG(temp_db_path, config=drift, skip_validation=True) as client:
        with pytest.raises(ConfigMismatchError):
            async for _ in client.rebuild_database(mode=RebuildMode.SET_EMBEDDER):
                pass


async def test_rebuild_blocks_tag_operations(temp_db_path, monkeypatch):
    """rebuild_database holds the rebuild lock for its whole run: tag
    operations fail mid-rebuild and work again once it completes."""
    import random

    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel

    from haiku.rag.embeddings import EmbedderWrapper
    from haiku.rag.store.models.chunk import Chunk

    async def fake_embed_documents(self, texts):
        result = []
        for t in texts:
            random.seed(hash(t) % (2**32))
            result.append([random.random() for _ in range(2560)])
        return result

    monkeypatch.setattr(EmbedderWrapper, "embed_documents", fake_embed_documents)

    dim = get_config().embeddings.model.vector_dim
    docling_doc = DoclingDocument(name="d")
    docling_doc.add_text(label=DocItemLabel.TEXT, text="body")

    async with HaikuRAG(temp_db_path, create=True) as client:
        await client.import_document(
            docling_doc,
            [Chunk(content="body", embedding=[0.1] * dim, order=0)],
            uri="mem://rebuild",
        )

        rebuild = client.rebuild_database(mode=RebuildMode.EMBED_ONLY)
        await anext(rebuild)

        assert client.store._rebuild_lock.locked()
        with pytest.raises(ValueError, match="[Rr]ebuild in progress"):
            await client.store.create_tag("mid-rebuild")

        async for _ in rebuild:
            pass

        assert not client.store._rebuild_lock.locked()
        await client.store.create_tag("post-rebuild")
        assert set(await client.store.list_tags()) == {"post-rebuild"}


def _count_flushes(monkeypatch, rebuild_module) -> list[int]:
    """Record the size of every batch handed to _flush_rebuild_batch."""
    real = rebuild_module._flush_rebuild_batch
    sizes: list[int] = []

    async def spy(client, documents, chunks, **kwargs):
        sizes.append(len(documents))
        return await real(client, documents, chunks, **kwargs)

    monkeypatch.setattr(rebuild_module, "_flush_rebuild_batch", spy)
    return sizes


# --- unit-level rebuild helpers (no embedder involved) ---


@pytest.mark.vcr()
async def test_flush_rebuild_batch_is_a_noop_without_documents(temp_db_path):
    from haiku.rag.client.rebuild import _flush_rebuild_batch

    async with HaikuRAG(temp_db_path, create=True) as client:
        # A populated table makes "unchanged" distinguishable from "wiped".
        existing = await client.create_document(content="keep me")
        before = await client.store.documents_table.count_rows()
        assert before == 1

        await _flush_rebuild_batch(
            client, [], [], document_columns=("docling_document",)
        )

        assert await client.store.documents_table.count_rows() == before
        after = await client.get_document_by_id(existing.id)
        assert after is not None
        assert after.updated_at == existing.updated_at


async def test_flush_rebuild_batch_deletes_items_in_one_version(temp_db_path):
    """One delete for the batch, one add per document, picture bytes kept."""
    from haiku.rag.client.documents import _store_document_with_chunks
    from haiku.rag.client.rebuild import _flush_rebuild_batch, _hydrate
    from haiku.rag.config import AppConfig
    from haiku.rag.store.models.document import Document
    from tests.store.test_document_items import _docling_doc_with_picture

    config = AppConfig()
    config.storage.auto_vacuum = False

    async with HaikuRAG(temp_db_path, config=config, create=True) as rag:
        created = []
        for i in range(3):
            docling_doc = _docling_doc_with_picture()
            document = Document(content=f"doc {i}", uri=f"test://doc-{i}")
            document.set_docling(docling_doc)
            created.append(
                await _store_document_with_chunks(
                    writing(rag), document, [], docling_doc
                )
            )
        items_repo = rag.document_item_repository
        before = {
            d.id: (
                await items_repo.get_item_count(d.id),
                await items_repo.get_all_picture_data(d.id),
            )
            for d in created
            if d.id is not None
        }
        assert all(pictures for _, pictures in before.values())
        version = await rag.store.document_items_table.version()

        docs = [doc async for doc in _hydrate(writing(rag), created)]
        await _flush_rebuild_batch(writing(rag), docs, [], document_columns=())

        assert await rag.store.document_items_table.version() == version + 1 + 3
        after = {
            doc_id: (
                await items_repo.get_item_count(doc_id),
                await items_repo.get_all_picture_data(doc_id),
            )
            for doc_id in before
        }
        assert after == before


async def test_mark_phase1_complete_is_idempotent(temp_db_path):
    from haiku.rag.client.rebuild import (
        _STAGING_MARKER_TABLE_NAME,
        _mark_phase1_complete,
    )

    async with HaikuRAG(temp_db_path, create=True) as client:
        await _mark_phase1_complete(client)
        await _mark_phase1_complete(client)

        tables = (await client.store.db.list_tables()).tables
        assert _STAGING_MARKER_TABLE_NAME in tables


async def test_populate_staging_returns_early_without_chunks_table(temp_db_path):
    from haiku.rag.client.rebuild import _STAGING_TABLE_NAME, _populate_staging_table

    async with HaikuRAG(temp_db_path, create=True) as client:
        await client.store.db.drop_table("chunks")

        await _populate_staging_table(client)

        staging = await client.store.db.open_table(_STAGING_TABLE_NAME)
        assert await staging.count_rows() == 0


async def test_hydrate_skips_documents_deleted_mid_rebuild(temp_db_path):
    """A document removed between listing and hydration is skipped."""
    from haiku.rag.client.rebuild import _hydrate
    from haiku.rag.store.models.document import Document
    from haiku.rag.store.repositories.document import DocumentRepository

    async with HaikuRAG(temp_db_path, create=True) as client:
        stored = await DocumentRepository(client.store).create(
            Document(content="body", uri="test://gone")
        )

        async def vanished(_document_id, include_blobs=False):
            return None

        client.document_repository.get_by_id = vanished  # type: ignore[method-assign]

        assert [doc async for doc in _hydrate(client, [stored])] == []


@pytest.mark.parametrize(
    "description,expected_text",
    [("", None), ("a red square", "a red square")],
    ids=["empty_skipped", "populated_applied"],
)
async def test_apply_descriptions_writes_only_non_empty_text(
    description, expected_text
):
    """An empty generated description leaves the picture untouched; a real one
    is written through to the picture meta."""
    from haiku.rag.client.rebuild import _apply_descriptions_sync
    from haiku.rag.store.models.document import Document
    from tests.store.test_document_items import _docling_doc_with_picture

    docling_doc = _docling_doc_with_picture()
    ref = docling_doc.pictures[0].self_ref
    document = Document(content="x", uri="test://doc")

    _apply_descriptions_sync(docling_doc, document, {ref: description})

    meta = docling_doc.pictures[0].meta
    actual = getattr(getattr(meta, "description", None), "text", None) if meta else None
    assert actual == expected_text
    # The blob is re-compressed either way; page rasters must survive it.
    assert document.docling_document is not None


async def test_patch_picture_descriptions_returns_zero_without_descriptions(
    temp_db_path, monkeypatch
):
    """When the VLM returns nothing, no blob rewrite is attempted."""
    from haiku.rag.client.documents import _store_document_with_chunks
    from haiku.rag.client.rebuild import _patch_picture_descriptions
    from haiku.rag.config import AppConfig
    from haiku.rag.store.models.document import Document
    from tests.store.test_document_items import _docling_doc_with_picture

    docling_doc = _docling_doc_with_picture()
    config = AppConfig()
    config.processing.pictures = "description"

    async def no_descriptions(_bytes_by_ref, config=None):
        return {}

    monkeypatch.setattr(
        "haiku.rag.providers.picture_description.describe_pictures",
        no_descriptions,
    )

    async with HaikuRAG(temp_db_path, config=config, create=True) as rag:
        document = Document(content="x", uri="test://doc")
        document.set_docling(docling_doc)
        created = await _store_document_with_chunks(
            writing(rag), document, [], docling_doc
        )

        assert await _patch_picture_descriptions(writing(rag), created) == 0


@pytest.mark.vcr()
async def test_rebuild_warns_when_post_rebuild_vacuum_fails(temp_db_path, monkeypatch):
    """A failing post-rebuild vacuum is logged, not raised — the rebuild itself
    already succeeded."""
    import logging

    from haiku.rag.client import rebuild as rebuild_module

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content="vacuum failure doc")
        assert doc.id is not None
        client._config.storage.auto_vacuum = True

        async def failing_vacuum():
            raise RuntimeError("vacuum exploded")

        monkeypatch.setattr(client.store, "vacuum", failing_vacuum)

        with capture_logs(rebuild_module.logger, logging.WARNING) as records:
            processed = [
                doc_id
                async for doc_id in client.rebuild_database(mode=RebuildMode.RECHUNK)
            ]

        assert doc.id in processed
        assert any("vacuum failed" in r.getMessage() for r in records)


@pytest.mark.vcr()
async def test_rebuild_embed_only_yields_documents_without_chunks(temp_db_path):
    """A document whose chunks were all removed is still reported as processed."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content="doc that loses its chunks")
        assert doc.id is not None
        await client.chunk_repository.delete_by_document_id(doc.id)

        processed = [
            doc_id
            async for doc_id in client.rebuild_database(mode=RebuildMode.EMBED_ONLY)
        ]

        assert processed == [doc.id]


@pytest.mark.vcr()
async def test_rebuild_embed_only_flushes_in_batches(temp_db_path, monkeypatch):
    """Forces a tiny batch size so the mid-loop flush in phase 2 runs."""
    from haiku.rag.client import rebuild as rebuild_module

    monkeypatch.setattr(rebuild_module, "_REBUILD_BATCH_SIZE", 2)

    async with HaikuRAG(temp_db_path, create=True) as client:
        ids = []
        for i in range(3):
            doc = await client.create_document(content=f"embed only batch doc {i}")
            assert doc.id is not None
            ids.append(doc.id)

        # Phase 2 writes straight to the chunks table rather than going
        # through _flush_rebuild_batch, so count the adds it makes. Patch at
        # class level: embed-only recreates the table, discarding any patch
        # applied to the instance that exists now.
        import lancedb

        real_add = lancedb.AsyncTable.add
        adds: list[int] = []

        async def counting_add(self, records, *args, **kwargs):
            if self.name == "chunks":
                adds.append(len(records))
            return await real_add(self, records, *args, **kwargs)

        monkeypatch.setattr(lancedb.AsyncTable, "add", counting_add)

        processed = [
            doc_id
            async for doc_id in client.rebuild_database(mode=RebuildMode.EMBED_ONLY)
        ]

        assert sorted(processed) == sorted(ids)
        for doc_id in ids:
            assert await client.chunk_repository.get_by_document_id(doc_id)

        # 3 docs at batch size 2: one mid-loop write plus the trailing one.
        assert len(adds) == 2


@pytest.mark.vcr()
async def test_rechunk_raises_when_docling_blob_is_missing(temp_db_path):
    """RECHUNK needs the stored docling document; without it the user is told
    to run a full rebuild."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content="doc with a cleared blob")
        assert doc.id is not None

        await client.store.documents_table.update(
            {"docling_document": None}, where=f"id = '{doc.id}'"
        )

        with pytest.raises(ValueError, match="has no stored docling document"):
            async for _ in client.rebuild_database(mode=RebuildMode.RECHUNK):
                pass


async def _data_files(table) -> list[tuple[int, list[str]]]:
    dataset = await table.to_lance()
    return [
        (fragment.fragment_id, [f.path for f in fragment.data_files()])
        for fragment in dataset.get_fragments()
    ]


@pytest.mark.vcr()
async def test_rebuild_rechunk_rewrites_no_document_rows(temp_db_path):
    from haiku.rag.config import AppConfig

    config = AppConfig()
    config.storage.auto_vacuum = False

    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        await client.create_document(content="rechunk leaves documents alone")
        before = await _data_files(client.store.documents_table)

        async for _ in client.rebuild_database(mode=RebuildMode.RECHUNK):
            pass

        assert await _data_files(client.store.documents_table) == before


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "mode", [RebuildMode.RECHUNK, RebuildMode.DESCRIPTIONS], ids=lambda m: m.name
)
async def test_rebuild_holds_no_page_images(temp_db_path, monkeypatch, mode):
    """Batched documents carry no page images, and the stored ones survive."""
    from haiku.rag.client import rebuild as rebuild_module
    from haiku.rag.client.documents import _store_document_with_chunks
    from haiku.rag.config import AppConfig
    from haiku.rag.store.models.document import Document
    from tests.store.test_document_items import _docling_doc_with_picture

    config = AppConfig()
    config.processing.pictures = "description"

    async def fake_describe(image_bytes_by_ref, *, config):
        return {ref: "A red square (mocked)." for ref in image_bytes_by_ref}

    monkeypatch.setattr(
        "haiku.rag.providers.picture_description.describe_pictures", fake_describe
    )

    flushed: list[Document] = []
    real = rebuild_module._flush_rebuild_batch

    async def spy(client, documents, chunks, **kwargs):
        flushed.extend(documents)
        return await real(client, documents, chunks, **kwargs)

    monkeypatch.setattr(rebuild_module, "_flush_rebuild_batch", spy)

    async with HaikuRAG(temp_db_path, config=config, create=True) as rag:
        docling_doc = _docling_doc_with_picture()
        document = Document(content="picture doc", uri="test://doc")
        document.set_docling(docling_doc)
        created = await _store_document_with_chunks(
            writing(rag), document, [], docling_doc
        )
        assert created.id is not None
        pages = b"\x80SENTINEL_PAGE_BYTES"
        await rag.store.documents_table.update(
            {"docling_pages": pages}, where=f"id = '{created.id}'"
        )

        async for _ in rag.rebuild_database(mode=mode):
            pass

        assert [d.id for d in flushed] == [created.id]
        assert flushed[0].docling_pages is None
        after = await rag.document_repository.get_by_id(created.id, include_blobs=True)
        assert after is not None
        assert after.content == "picture doc"
        assert after.docling_pages == pages


@pytest.mark.vcr()
async def test_rebuild_full_flushes_in_batches(temp_db_path, monkeypatch):
    from haiku.rag.client import rebuild as rebuild_module

    monkeypatch.setattr(rebuild_module, "_REBUILD_BATCH_SIZE", 2)
    flushes = _count_flushes(monkeypatch, rebuild_module)

    async with HaikuRAG(temp_db_path, create=True) as client:
        ids = []
        for i in range(3):
            doc = await client.create_document(content=f"full batch doc {i}")
            assert doc.id is not None
            ids.append(doc.id)

        processed = [
            doc_id async for doc_id in client.rebuild_database(mode=RebuildMode.FULL)
        ]

        assert sorted(processed) == sorted(ids)

    assert len(flushes) == 2


@pytest.mark.vcr()
async def test_rebuild_full_warns_when_source_is_missing(temp_db_path):
    """A document whose file source is gone is re-embedded from stored content."""
    import logging

    from haiku.rag.client import rebuild as rebuild_module

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(
            content="content whose source vanished",
            uri="file:///definitely/not/here.txt",
        )
        assert doc.id is not None

        with capture_logs(rebuild_module.logger, logging.WARNING) as records:
            processed = [
                doc_id
                async for doc_id in client.rebuild_database(mode=RebuildMode.FULL)
            ]

        assert doc.id in processed
        assert any("Source missing" in r.getMessage() for r in records)


@pytest.mark.vcr()
async def test_rebuild_full_flushes_pending_before_source_rebuild(temp_db_path):
    """A live source is re-ingested, which creates a new document, so any
    documents pending from the content path must be flushed first."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        content_doc = await client.create_document(content="plain content doc")
        assert content_doc.id is not None

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "live.txt"
            source.write_text("content from a source that still exists")
            source_doc = await client.create_document_from_source(source)
            assert not isinstance(source_doc, list)

            processed = [
                doc_id
                async for doc_id in client.rebuild_database(mode=RebuildMode.FULL)
            ]

            # Both documents keep their ids: the content-path one must survive
            # the flush that precedes the source refresh, and the source one is
            # refreshed in place.
            assert sorted(processed) == sorted([content_doc.id, source_doc.id])
            assert await client.store.documents_table.count_rows() == 2


@pytest.mark.vcr()
async def test_rebuild_descriptions_flushes_in_batches(temp_db_path, monkeypatch):
    """Two picture documents with a batch size of one exercise the mid-loop
    flush in the descriptions path."""
    from haiku.rag.client import rebuild as rebuild_module
    from haiku.rag.client.documents import _store_document_with_chunks
    from haiku.rag.config import AppConfig
    from haiku.rag.store.models.document import Document
    from tests.store.test_document_items import _docling_doc_with_picture

    config = AppConfig()
    config.processing.pictures = "description"

    monkeypatch.setattr(rebuild_module, "_REBUILD_BATCH_SIZE", 1)
    flushes = _count_flushes(monkeypatch, rebuild_module)

    async def fake_describe(image_bytes_by_ref, *, config):
        return {ref: "A red square (mocked)." for ref in image_bytes_by_ref}

    monkeypatch.setattr(
        "haiku.rag.providers.picture_description.describe_pictures", fake_describe
    )

    async with HaikuRAG(temp_db_path, config=config, create=True) as rag:
        ids = []
        for i in range(2):
            docling_doc = _docling_doc_with_picture()
            document = Document(content=f"picture doc {i}", uri=f"test://doc-{i}")
            document.set_docling(docling_doc)
            created = await _store_document_with_chunks(
                writing(rag), document, [], docling_doc
            )
            assert created.id is not None
            ids.append(created.id)

        processed = [
            doc_id
            async for doc_id in rag.rebuild_database(mode=RebuildMode.DESCRIPTIONS)
        ]

        assert sorted(processed) == sorted(ids)

    # 2 docs at batch size 1: one flush each, none left for the trailing pass.
    assert len(flushes) == 2


@pytest.mark.vcr()
async def test_rebuild_full_skips_document_deleted_mid_rebuild(temp_db_path):
    """A document removed between listing and the content-path reload is skipped."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content="doc that disappears")
        assert doc.id is not None

        async def vanished(_document_id, include_blobs=False):
            return None

        client.document_repository.get_by_id = vanished  # type: ignore[method-assign]

        processed = [
            doc_id async for doc_id in client.rebuild_database(mode=RebuildMode.FULL)
        ]

        assert processed == []


@pytest.mark.vcr()
@pytest.mark.parametrize("wipe_bytes", [True, False], ids=["wiped", "recoverable"])
async def test_rebuild_embed_only_recovers_picture_bytes(
    temp_db_path, monkeypatch, wipe_bytes
):
    """Embed-only re-attaches picture bytes from document_items. When they are
    gone the chunk falls back to embedding its caption as text rather than
    failing the rebuild."""
    import logging

    from haiku.rag.client import rebuild as rebuild_module
    from haiku.rag.client.documents import _store_document_with_chunks
    from haiku.rag.store.models.chunk import Chunk
    from haiku.rag.store.models.document import Document
    from tests.store.test_document_items import _docling_doc_with_picture

    docling_doc = _docling_doc_with_picture()
    ref = docling_doc.pictures[0].self_ref

    async with HaikuRAG(temp_db_path, create=True) as rag:
        # The configured ollama embedder is text-only; stand in for a
        # multimodal one so the picture-bytes recovery branch runs.
        embedded_images: list[bytes] = []

        async def fake_embed_image(image):
            embedded_images.append(image)
            return [0.2] * rag.embedder.vector_dim

        monkeypatch.setattr(rag.embedder, "supports_images", True)
        monkeypatch.setattr(rag.embedder, "embed_image", fake_embed_image)

        document = Document(content="picture doc", uri="test://pic")
        document.set_docling(docling_doc)
        picture_chunk = Chunk(
            content="Figure caption",
            metadata={"doc_item_refs": [ref], "labels": ["picture"]},
            order=0,
            embedding=[0.1] * rag.embedder.vector_dim,
        )
        # A sibling text chunk exercises the non-picture skip in the same loop.
        text_chunk = Chunk(
            content="Surrounding prose",
            metadata={"doc_item_refs": ["#/texts/0"], "labels": ["text"]},
            order=1,
            embedding=[0.1] * rag.embedder.vector_dim,
        )
        created = await _store_document_with_chunks(
            writing(rag), document, [picture_chunk, text_chunk], docling_doc
        )
        assert created.id is not None

        if wipe_bytes:
            await rag.store.document_items_table.update(
                {"picture_data": None},
                where=f"document_id = '{created.id}' AND label = 'picture'",
            )

        with capture_logs(rebuild_module.logger, logging.WARNING) as records:
            processed = [
                doc_id
                async for doc_id in rag.rebuild_database(mode=RebuildMode.EMBED_ONLY)
            ]

        assert created.id in processed
        warned = any("no recoverable bytes" in r.getMessage() for r in records)
        assert warned is wipe_bytes

        if wipe_bytes:
            # Nothing to recover, so the caption is text-embedded instead.
            assert embedded_images == []
        else:
            # The stored PNG was re-attached and routed through embed_image.
            assert len(embedded_images) == 1
            assert embedded_images[0].startswith(b"\x89PNG")


async def _import_documents(client: HaikuRAG, names: tuple[str, ...]) -> None:
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel

    from haiku.rag.store.models.chunk import Chunk

    dim = client.store.embedder.vector_dim
    for name in names:
        doc = DoclingDocument(name=name)
        doc.add_text(label=DocItemLabel.TEXT, text=f"document {name}")
        await client.import_document(
            doc,
            [Chunk(content=f"document {name}", embedding=[0.1] * dim, order=0)],
            uri=f"test://{name}",
        )


async def _unindexed_rows(table) -> dict[str, int]:
    return {i.name: i.num_unindexed_rows for i in await table.list_indices()}


@pytest.mark.parametrize("mode", [RebuildMode.EMBED_ONLY, RebuildMode.RECHUNK])
async def test_chunk_writing_rebuild_ends_with_every_row_indexed(
    temp_db_path, monkeypatch, mode
):
    """Coverage does not wait for a vacuum."""
    from haiku.rag import embeddings as embeddings_module
    from haiku.rag.client import rebuild as rebuild_module
    from haiku.rag.config import AppConfig

    monkeypatch.setattr(rebuild_module, "_REBUILD_BATCH_SIZE", 1)

    async def keep_embeddings(chunks, embedder, config=None):
        for chunk in chunks:
            if chunk.embedding is None:
                chunk.embedding = [0.1] * embedder.vector_dim
        return chunks

    monkeypatch.setattr(embeddings_module, "embed_chunks", keep_embeddings)

    config = AppConfig()
    config.storage.auto_vacuum = False
    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        await _import_documents(client, ("one", "two", "three"))

        async for _ in client.rebuild_database(mode=mode):
            pass

        for name, table in client.store._tables().items():
            unindexed = await _unindexed_rows(table)
            assert set(unindexed.values()) <= {0}, (name, unindexed)


async def test_rebuild_title_only_leaves_the_chunk_indexes_alone(
    temp_db_path, monkeypatch
):
    from haiku.rag.config import AppConfig

    async def fake_generate_title(config, doc):
        return "A title"

    monkeypatch.setattr("haiku.rag.client.rebuild.generate_title", fake_generate_title)

    config = AppConfig()
    config.storage.auto_vacuum = False
    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        await _import_documents(client, ("one",))
        before = await client.store.chunks_table.version()

        async for _ in client.rebuild_database(mode=RebuildMode.TITLE_ONLY):
            pass

        assert await client.store.chunks_table.version() == before


async def test_rebuild_reindex_covers_rows_and_writes_nothing_else(temp_db_path):
    from haiku.rag.config import AppConfig

    config = AppConfig()
    config.storage.auto_vacuum = False
    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        await _import_documents(client, ("one", "two"))
        assert (await _unindexed_rows(client.store.chunks_table))[
            "content_fts_idx"
        ] == 1
        settings_version = await client.store.settings_table.version()
        rows_before = (
            await client.store.chunks_table.query().select(["id", "content"]).to_arrow()
        ).sort_by("id")

        yielded = [
            doc_id async for doc_id in client.rebuild_database(mode=RebuildMode.REINDEX)
        ]

        assert yielded == []
        assert set((await _unindexed_rows(client.store.chunks_table)).values()) == {0}
        assert await client.store.settings_table.version() == settings_version
        rows_after = (
            await client.store.chunks_table.query().select(["id", "content"]).to_arrow()
        ).sort_by("id")
        assert rows_after == rows_before


async def test_rebuild_reindex_works_on_empty_database(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        async for _ in client.rebuild_database(mode=RebuildMode.REINDEX):
            pass

        assert {i.name for i in await client.store.chunks_table.list_indices()} == {
            "id_idx",
            "document_id_idx",
        }


async def _vector_indexes(client: HaikuRAG) -> list[str]:
    return [
        i.name
        for i in await client.store.chunks_table.list_indices()
        if "vector" in i.columns
    ]


async def _import_indexed_document(
    client: HaikuRAG, monkeypatch, *, chunks: int, vector_index: bool
) -> None:
    """One document of `chunks` random-vector chunks, re-embedded randomly."""
    import random

    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel

    from haiku.rag import embeddings as embeddings_module
    from haiku.rag.store.models.chunk import Chunk

    rng = random.Random(0)

    async def random_embeddings(chunks, embedder, config=None):
        for chunk in chunks:
            chunk.embedding = [rng.random() for _ in range(embedder.vector_dim)]
        return chunks

    monkeypatch.setattr(embeddings_module, "embed_chunks", random_embeddings)

    dim = client.store.embedder.vector_dim
    doc = DoclingDocument(name="many")
    doc.add_text(label=DocItemLabel.TEXT, text="many chunks")
    await client.import_document(
        doc,
        [
            Chunk(
                content=f"chunk {i}",
                embedding=[rng.random() for _ in range(dim)],
                order=i,
            )
            for i in range(chunks)
        ],
        uri="test://many",
    )
    if vector_index:
        await client.store._ensure_vector_index()
        assert await _vector_indexes(client) == ["vector_idx"]


@pytest.mark.parametrize("had_vector_index", [True, False])
async def test_rebuild_keeps_vector_index_opt_in(
    temp_db_path, monkeypatch, had_vector_index
):
    """A vector index survives the table a rebuild recreates, and a rebuild
    never creates one where there was none."""
    from haiku.rag.config import AppConfig

    config = AppConfig()
    config.storage.auto_vacuum = False
    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        await _import_indexed_document(
            client, monkeypatch, chunks=300, vector_index=had_vector_index
        )

        async for _ in client.rebuild_database(mode=RebuildMode.EMBED_ONLY):
            pass

        assert await _vector_indexes(client) == (
            ["vector_idx"] if had_vector_index else []
        )


def _picture_document(name: str, sections: int = 0):
    """A docling document with a picture, and `sections` headed paragraphs."""
    from docling_core.types.doc.document import DoclingDocument, ImageRef
    from docling_core.types.doc.labels import DocItemLabel
    from PIL import Image as PilImageModule

    doc = DoclingDocument(name=name)
    doc.add_text(label=DocItemLabel.TEXT, text=f"document {name}")
    for i in range(sections):
        heading = doc.add_heading(text=f"{name} section {i}")
        doc.add_text(label=DocItemLabel.TEXT, text=f"{name} body {i}", parent=heading)
    doc.add_picture(
        image=ImageRef.from_pil(PilImageModule.new("RGB", (8, 8), "red"), dpi=72)
    )
    return doc


async def _import_picture_documents(
    client: HaikuRAG, names: tuple[str, ...], *, sections: int = 0, chunks: int = 1
) -> list[str]:
    import random

    from haiku.rag.store.models.chunk import Chunk

    rng = random.Random(0)
    dim = client.store.embedder.vector_dim
    ids = []
    for name in names:
        document = await client.import_document(
            _picture_document(name, sections),
            [
                Chunk(
                    content=f"document {name} {i}",
                    embedding=[rng.random() for _ in range(dim)],
                    order=i,
                )
                for i in range(chunks)
            ],
            uri=f"test://{name}",
        )
        assert document.id is not None
        ids.append(document.id)
    return ids


def _stub_embedder(monkeypatch, *, fail_on_call: int | None = None) -> None:
    """Embed rebuilt chunks randomly; the `fail_on_call`th call raises."""
    import random

    from haiku.rag import embeddings as embeddings_module

    rng = random.Random(1)
    calls = 0

    async def embed(chunks, embedder, config=None):
        nonlocal calls
        calls += 1
        if calls == fail_on_call:
            raise RuntimeError("embedder unavailable")
        for chunk in chunks:
            chunk.embedding = [rng.random() for _ in range(embedder.vector_dim)]
        return chunks

    monkeypatch.setattr(embeddings_module, "embed_chunks", embed)


def _stub_picture_descriptions(monkeypatch) -> None:
    async def fake_describe(image_bytes_by_ref, *, config):
        return {ref: "A red square (mocked)." for ref in image_bytes_by_ref}

    monkeypatch.setattr(
        "haiku.rag.providers.picture_description.describe_pictures", fake_describe
    )


def _rebuild_config():
    from haiku.rag.config import AppConfig

    config = AppConfig()
    config.storage.auto_vacuum = False
    config.processing.pictures = "description"
    return config


async def _chunk_ids(client: HaikuRAG, document_id: str) -> set[str]:
    chunks = await client.chunk_repository.get_by_document_id(document_id)
    return {c.id for c in chunks if c.id is not None}


async def _document_state(client: HaikuRAG, document_id: str):
    """Everything a rebuild writes for a document, in comparable form."""
    items = await client.document_item_repository.get_all_items(document_id)
    structure = await client.document_repository.get_docling_data(document_id)
    assert structure is not None
    return (
        await _chunk_ids(client, document_id),
        [(i.position, i.self_ref, i.text) for i in items],
        await client.document_item_repository.get_all_picture_data(document_id),
        structure.docling_document,
    )


_CHUNK_REWRITING_MODES = [
    RebuildMode.RECHUNK,
    RebuildMode.DESCRIPTIONS,
    RebuildMode.FULL,
]


@pytest.mark.parametrize("mode", _CHUNK_REWRITING_MODES, ids=lambda m: m.name)
async def test_failed_rebuild_keeps_the_chunks_of_unprocessed_documents(
    temp_db_path, monkeypatch, mode
):
    from haiku.rag.client import rebuild as rebuild_module

    monkeypatch.setattr(rebuild_module, "_REBUILD_BATCH_SIZE", 1)
    _stub_picture_descriptions(monkeypatch)
    _stub_embedder(monkeypatch, fail_on_call=3)

    async with HaikuRAG(temp_db_path, config=_rebuild_config(), create=True) as client:
        ids = await _import_picture_documents(client, ("one", "two", "three"))
        before = {i: await _chunk_ids(client, i) for i in ids}

        with pytest.raises(RuntimeError, match="embedder unavailable"):
            async for _ in client.rebuild_database(mode=mode):
                pass

        after = {i: await _chunk_ids(client, i) for i in ids}
        assert all(after.values())
        # Two documents were rewritten before the failure, one keeps what it had.
        assert sorted(after[i] == before[i] for i in ids) == [False, False, True]
        assert len(await client.search("document", search_type="fts")) == 3


@pytest.mark.parametrize("mode", _CHUNK_REWRITING_MODES, ids=lambda m: m.name)
async def test_rebuild_batch_that_fails_to_write_leaves_the_database_as_it_was(
    temp_db_path, monkeypatch, mode
):
    from haiku.rag.store.repositories.document_item import DocumentItemRepository

    _stub_picture_descriptions(monkeypatch)
    _stub_embedder(monkeypatch)

    async with HaikuRAG(temp_db_path, config=_rebuild_config(), create=True) as client:
        ids = await _import_picture_documents(client, ("one", "two", "three"))
        before = {i: await _document_state(client, i) for i in ids}

        real = DocumentItemRepository.create_items
        writes = 0

        async def fail_on_second(self, document_id, items):
            nonlocal writes
            writes += 1
            if writes == 2:
                raise RuntimeError("disk full")
            await real(self, document_id, items)

        monkeypatch.setattr(DocumentItemRepository, "create_items", fail_on_second)

        with pytest.raises(RuntimeError, match="disk full"):
            async for _ in client.rebuild_database(mode=mode):
                pass

        assert writes == 2
        assert {i: await _document_state(client, i) for i in ids} == before


@pytest.mark.parametrize("mode", _CHUNK_REWRITING_MODES, ids=lambda m: m.name)
async def test_closed_rebuild_has_written_the_documents_it_reported(
    temp_db_path, monkeypatch, mode
):
    from haiku.rag.client import rebuild as rebuild_module

    monkeypatch.setattr(rebuild_module, "_REBUILD_BATCH_SIZE", 1)
    _stub_picture_descriptions(monkeypatch)
    _stub_embedder(monkeypatch)

    async with HaikuRAG(temp_db_path, config=_rebuild_config(), create=True) as client:
        ids = await _import_picture_documents(client, ("one", "two", "three"))
        before = {i: await _chunk_ids(client, i) for i in ids}

        rebuild = client.rebuild_database(mode=mode)
        reported = await anext(rebuild)
        await rebuild.aclose()

        for document_id in ids:
            after = await _chunk_ids(client, document_id)
            assert after
            assert (after != before[document_id]) == (document_id == reported)


@pytest.mark.parametrize("mode", _CHUNK_REWRITING_MODES, ids=lambda m: m.name)
async def test_interrupted_rebuild_restores_the_vector_index_when_run_again(
    temp_db_path, monkeypatch, mode
):
    """A rebuild that stops after its first batch keeps the vector index, and
    the run that finishes it retrains it over every row."""
    from haiku.rag.client import rebuild as rebuild_module

    monkeypatch.setattr(rebuild_module, "_REBUILD_BATCH_SIZE", 1)
    _stub_picture_descriptions(monkeypatch)
    _stub_embedder(monkeypatch)

    async with HaikuRAG(temp_db_path, config=_rebuild_config(), create=True) as client:
        # Rechunking has to leave the 256 rows a vector index needs.
        await _import_picture_documents(
            client, ("one", "two"), sections=200, chunks=200
        )
        await client.store._ensure_vector_index()
        assert await _vector_indexes(client) == ["vector_idx"]

        rebuild = client.rebuild_database(mode=mode)
        await anext(rebuild)
        await rebuild.aclose()

        assert await _vector_indexes(client) == ["vector_idx"]

        async for _ in client.rebuild_database(mode=mode):
            pass

        assert await client.store.chunks_table.count_rows() >= 256
        assert await _vector_indexes(client) == ["vector_idx"]
        stats = await client.store.chunks_table.index_stats("vector_idx")
        assert stats is not None
        assert stats.num_unindexed_rows == 0


@pytest.mark.parametrize("mode", _CHUNK_REWRITING_MODES, ids=lambda m: m.name)
async def test_rebuild_to_another_vector_dimension_recreates_the_chunks(
    temp_db_path, monkeypatch, mode
):
    async with HaikuRAG(temp_db_path, create=True) as client:
        await _import_documents(client, ("one", "two"))
        assert client.store.embedder.vector_dim == 2560

    config = _rebuild_config()
    config.embeddings.model.vector_dim = 8
    _stub_embedder(monkeypatch)
    _stub_picture_descriptions(monkeypatch)

    async with HaikuRAG(temp_db_path, config=config, skip_validation=True) as client:
        rebuilt = [doc_id async for doc_id in client.rebuild_database(mode=mode)]

        assert len(rebuilt) == 2
        chunks = (await client.store.chunks_table.query().to_arrow()).to_pylist()
        assert len(chunks) == 2
        assert {len(chunk["vector"]) for chunk in chunks} == {8}


# --- the settings record the embedder once the chunks are its own (#673) ---

_RECORDING_MODES = [*_CHUNK_REWRITING_MODES, RebuildMode.EMBED_ONLY]


def _other_embedder_config(*, vector_dim: int | None = None):
    config = _rebuild_config()
    config.embeddings.model.name = "another-embedder"
    if vector_dim is not None:
        config.embeddings.model.vector_dim = vector_dim
    return config


async def _recorded_model(temp_db_path) -> dict:
    from haiku.rag.store.repositories.settings import SettingsRepository

    async with HaikuRAG(
        temp_db_path, config=_rebuild_config(), skip_validation=True, read_only=True
    ) as client:
        settings = await SettingsRepository(client.store).get_current_settings()
    return settings["embeddings"]["model"]


async def _interrupt_after_first_document(temp_db_path, config, mode) -> None:
    async with HaikuRAG(temp_db_path, config=config, skip_validation=True) as client:
        rebuild = client.rebuild_database(mode=mode)
        await anext(rebuild)
        await rebuild.aclose()


@pytest.mark.parametrize("mode", _CHUNK_REWRITING_MODES, ids=lambda m: m.name)
async def test_interrupted_rebuild_to_another_embedder_keeps_the_old_one_recorded(
    temp_db_path, monkeypatch, mode
):
    """The documents it did not reach keep the old embedder's vectors, so the
    database still records it: a writable open refuses the new embedder, a
    read-only open warns, and running the rebuild again finishes it."""
    from haiku.rag.client import rebuild as rebuild_module
    from haiku.rag.store.exceptions import ConfigMismatchError

    monkeypatch.setattr(rebuild_module, "_REBUILD_BATCH_SIZE", 1)
    _stub_embedder(monkeypatch)
    _stub_picture_descriptions(monkeypatch)
    async with HaikuRAG(temp_db_path, config=_rebuild_config(), create=True) as client:
        ids = await _import_picture_documents(client, ("one", "two", "three"))
        old_name = _rebuild_config().embeddings.model.name

    config = _other_embedder_config()
    await _interrupt_after_first_document(temp_db_path, config, mode)

    assert (await _recorded_model(temp_db_path))["name"] == old_name
    with pytest.raises(ConfigMismatchError):
        async with HaikuRAG(temp_db_path, config=config):
            pass
    async with HaikuRAG(temp_db_path, config=config, read_only=True):
        pass

    async with HaikuRAG(temp_db_path, config=config, skip_validation=True) as client:
        async for _ in client.rebuild_database(mode=mode):
            pass

    assert (await _recorded_model(temp_db_path))["name"] == "another-embedder"
    async with HaikuRAG(temp_db_path, config=config) as client:
        for document_id in ids:
            assert await _chunk_ids(client, document_id)


async def test_interrupted_embed_only_to_another_embedder_records_the_new_one(
    temp_db_path, monkeypatch
):
    """Embed-only empties the table before writing, so no vector of the old
    embedder is left for the settings to describe."""
    from haiku.rag.client import rebuild as rebuild_module

    monkeypatch.setattr(rebuild_module, "_REBUILD_BATCH_SIZE", 1)
    _stub_embedder(monkeypatch)
    async with HaikuRAG(temp_db_path, config=_rebuild_config(), create=True) as client:
        ids = await _import_picture_documents(client, ("one", "two", "three"))

    config = _other_embedder_config()
    await _interrupt_after_first_document(temp_db_path, config, RebuildMode.EMBED_ONLY)

    assert (await _recorded_model(temp_db_path))["name"] == "another-embedder"
    async with HaikuRAG(temp_db_path, config=config) as client:
        async for _ in client.rebuild_database(mode=RebuildMode.EMBED_ONLY):
            pass
        for document_id in ids:
            assert await _chunk_ids(client, document_id)


@pytest.mark.parametrize("mode", _RECORDING_MODES, ids=lambda m: m.name)
async def test_interrupted_dimension_change_is_finished_by_running_it_again(
    temp_db_path, monkeypatch, mode
):
    """The recreated table has the new dimension and so do the settings: the
    old configuration is refused, writable or read-only, the new one opens,
    and the rerun writes the documents the first run did not reach."""
    from haiku.rag.client import rebuild as rebuild_module
    from haiku.rag.store.exceptions import ConfigMismatchError

    monkeypatch.setattr(rebuild_module, "_REBUILD_BATCH_SIZE", 1)
    _stub_embedder(monkeypatch)
    _stub_picture_descriptions(monkeypatch)
    async with HaikuRAG(temp_db_path, config=_rebuild_config(), create=True) as client:
        ids = await _import_picture_documents(client, ("one", "two", "three"))

    config = _other_embedder_config(vector_dim=8)
    await _interrupt_after_first_document(temp_db_path, config, mode)

    assert (await _recorded_model(temp_db_path))["vector_dim"] == 8
    for read_only in (False, True):
        with pytest.raises(ConfigMismatchError):
            async with HaikuRAG(
                temp_db_path, config=_rebuild_config(), read_only=read_only
            ):
                pass
    async with HaikuRAG(temp_db_path, config=config):
        pass

    async with HaikuRAG(temp_db_path, config=config, skip_validation=True) as client:
        async for _ in client.rebuild_database(mode=mode):
            pass

    assert (await _recorded_model(temp_db_path))["vector_dim"] == 8
    async with HaikuRAG(temp_db_path, config=config) as client:
        for document_id in ids:
            assert await _chunk_ids(client, document_id)
        rows = (await client.store.chunks_table.query().to_arrow()).to_pylist()
        assert {len(row["vector"]) for row in rows} == {8}


@pytest.mark.parametrize("mode", _CHUNK_REWRITING_MODES, ids=lambda m: m.name)
async def test_rebuild_stopped_before_recording_the_new_dimension_finishes(
    temp_db_path, monkeypatch, mode
):
    """The table already has the new dimension and the settings still the old
    one, so the rerun opens with a chunk model of the old width."""
    from haiku.rag.client import rebuild as rebuild_module

    _stub_embedder(monkeypatch)
    _stub_picture_descriptions(monkeypatch)
    async with HaikuRAG(temp_db_path, config=_rebuild_config(), create=True) as client:
        ids = await _import_picture_documents(client, ("one", "two"))
        old_dim = _rebuild_config().embeddings.model.vector_dim

    async def stopped_before_recording(session):
        await session.store.recreate_embeddings_table()
        raise RuntimeError("stopped")

    config = _other_embedder_config(vector_dim=8)
    with monkeypatch.context() as patch:
        patch.setattr(
            rebuild_module, "_recreate_chunks_table", stopped_before_recording
        )
        async with HaikuRAG(
            temp_db_path, config=config, skip_validation=True
        ) as client:
            with pytest.raises(RuntimeError, match="stopped"):
                async for _ in client.rebuild_database(mode=mode):
                    pass
    assert (await _recorded_model(temp_db_path))["vector_dim"] == old_dim

    async with HaikuRAG(temp_db_path, config=config, skip_validation=True) as client:
        async for _ in client.rebuild_database(mode=mode):
            pass

    assert (await _recorded_model(temp_db_path))["vector_dim"] == 8
    async with HaikuRAG(temp_db_path, config=config) as client:
        for document_id in ids:
            assert await _chunk_ids(client, document_id)


async def test_title_only_leaves_the_recorded_embedder_alone(temp_db_path, monkeypatch):
    """It writes no vectors, so the ones in the table are still the old
    embedder's."""

    async def fake_generate_title(config, doc):
        return "A title"

    monkeypatch.setattr("haiku.rag.client.rebuild.generate_title", fake_generate_title)
    async with HaikuRAG(temp_db_path, config=_rebuild_config(), create=True) as client:
        await _import_documents(client, ("one",))
        old_model = dict(await _recorded_model_of(client))

    async with HaikuRAG(
        temp_db_path, config=_other_embedder_config(), skip_validation=True
    ) as client:
        async for _ in client.rebuild_database(mode=RebuildMode.TITLE_ONLY):
            pass

    assert await _recorded_model(temp_db_path) == old_model


async def _recorded_model_of(client: HaikuRAG) -> dict:
    from haiku.rag.store.repositories.settings import SettingsRepository

    settings = await SettingsRepository(client.store).get_current_settings()
    return settings["embeddings"]["model"]


@pytest.mark.parametrize(
    "mode", [*_RECORDING_MODES, RebuildMode.TITLE_ONLY], ids=lambda m: m.name
)
async def test_read_only_rebuild_writes_nothing(temp_db_path, monkeypatch, mode):
    from haiku.rag.store.exceptions import ReadOnlyError

    _stub_embedder(monkeypatch)
    _stub_picture_descriptions(monkeypatch)
    async with HaikuRAG(temp_db_path, config=_rebuild_config(), create=True) as client:
        await _import_picture_documents(client, ("one", "two"))

    async def versions() -> dict[str, int]:
        async with HaikuRAG(
            temp_db_path, config=_rebuild_config(), read_only=True
        ) as client:
            tables = (await client.store.db.list_tables()).tables
            return {
                name: await (await client.store.db.open_table(name)).version()
                for name in tables
            }

    before = await versions()
    async with HaikuRAG(
        temp_db_path, config=_rebuild_config(), read_only=True
    ) as client:
        with pytest.raises(ReadOnlyError):
            async for _ in client.rebuild_database(mode=mode):
                pass

    assert await versions() == before


_FLUSH_OVER_POOL = """
import asyncio, os, sys
from pathlib import Path

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.client import HaikuRAG
from haiku.rag.client.rebuild import _DESCRIPTION_COLUMNS, _flush_rebuild_batch
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models.document import Document
from tests.conftest import writing


async def main():
    config = AppConfig()
    config.storage.auto_vacuum = False
    async with HaikuRAG(Path(sys.argv[1]), config=config, create=True) as rag:
        repo = rag.document_repository
        target = await repo.create(
            Document(content="target", uri="test://target", docling_pages=b"pages")
        )
        other = await repo.create(
            Document(content="other", uri="test://other", docling_document=b"kept")
        )
        await rag.store.rebuild_indexes()
        stats = await rag.store.documents_table.index_stats("id_idx")
        assert stats is not None and stats.num_indexed_rows == 2
        version = await rag.store.documents_table.version()

        # Hex text compresses to about half, so the blob outgrows the pool.
        docling = DoclingDocument(name="large")
        docling.add_text(label=DocItemLabel.PARAGRAPH, text=os.urandom(int(sys.argv[2])).hex())
        target.set_docling(docling)
        blob = target.docling_document
        assert blob is not None and len(blob) > int(sys.argv[2]) // 2
        target.docling_pages = b"not written"

        await _flush_rebuild_batch(
            writing(rag), [target], [], document_columns=_DESCRIPTION_COLUMNS
        )

        assert await rag.store.documents_table.version() == version + 1
        stored = await repo.get_by_id(target.id, include_blobs=True)
        assert stored is not None
        assert stored.docling_document == blob
        assert stored.content == "target"
        assert stored.docling_pages == b"pages"
        untouched = await repo.get_by_id(other.id, include_blobs=True)
        assert untouched is not None
        assert untouched.content == "other"
        assert untouched.docling_document == b"kept"


asyncio.run(main())
"""


def test_flush_rebuild_batch_writes_a_row_larger_than_the_lance_memory_pool(
    tmp_path,
):
    """A rebuild batch rewriting an indexed row with blobs larger than lance's pool."""
    result = run_with_lance_memory_pool(
        _FLUSH_OVER_POOL, 1 << 20, str(tmp_path / "db.lancedb"), str(4 << 20)
    )
    assert result.returncode == 0, result.stderr
