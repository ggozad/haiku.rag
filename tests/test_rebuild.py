import tempfile
from pathlib import Path
from typing import TypedDict

import pytest
from datasets import Dataset

from haiku.rag.client import HaikuRAG, RebuildMode


class ChunkData(TypedDict):
    id: str
    document_id: str
    content: str
    content_fts: str
    metadata: str
    order: int


@pytest.mark.vcr()
async def test_rebuild_full(qa_corpus: Dataset, temp_db_path):
    """Test full rebuild: converts, chunks, and embeds all documents."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content=qa_corpus["document_extracted"][0])
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
async def test_rebuild_embed_only(qa_corpus: Dataset, temp_db_path):
    """Test embed-only rebuild: keeps chunks, only regenerates embeddings."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content=qa_corpus["document_extracted"][0])
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
async def test_rebuild_embed_only_skips_unchanged(qa_corpus: Dataset, temp_db_path):
    """Test embed-only rebuild skips chunks with unchanged embeddings."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content=qa_corpus["document_extracted"][0])
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
    qa_corpus: Dataset, temp_db_path
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
        doc = await client.create_document(content=qa_corpus["document_extracted"][0])
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
async def test_rebuild_rechunk(qa_corpus: Dataset, temp_db_path):
    """Test rechunk rebuild: re-chunks from content without accessing source files."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content=qa_corpus["document_extracted"][0])
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
