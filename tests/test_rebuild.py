import pytest
from datasets import Dataset

from haiku.rag.client import HaikuRAG, RebuildMode


@pytest.mark.vcr()
async def test_rebuild_full(qa_corpus: Dataset, temp_db_path):
    """Test full rebuild: converts, chunks, and embeds all documents."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content=qa_corpus["document_extracted"][0])
        assert doc.id is not None
        assert doc.docling_document_json is not None

        chunks_before = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(chunks_before) > 0
        chunk_ids_before = {c.id for c in chunks_before}

        processed_ids = [doc_id async for doc_id in client.rebuild_database()]

        assert doc.id in processed_ids

        # Verify DoclingDocument JSON is preserved after rebuild
        doc_after = await client.document_repository.get_by_id(doc.id)
        assert doc_after is not None
        assert doc_after.docling_document_json is not None
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
        original_docling_json = doc.docling_document_json

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
        assert doc_after.docling_document_json == original_docling_json

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
        records_before = list(
            client.store.chunks_table.search()
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
        records_after = list(
            client.store.chunks_table.search()
            .where(f"document_id = '{doc.id}'")
            .to_pydantic(client.store.ChunkRecord)
        )
        embeddings_after = {rec.id: rec.vector for rec in records_after}

        # Embeddings should be identical (same content, same embedder)
        assert embeddings_before.keys() == embeddings_after.keys()
        for chunk_id in embeddings_before:
            assert embeddings_before[chunk_id] == embeddings_after[chunk_id]


@pytest.mark.vcr()
async def test_rebuild_rechunk(qa_corpus: Dataset, temp_db_path):
    """Test rechunk rebuild: re-chunks from content without accessing source files."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content=qa_corpus["document_extracted"][0])
        assert doc.id is not None
        assert doc.docling_document_json is not None

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
        assert doc_after.docling_document_json is not None
        assert doc_after.docling_version is not None

        chunks_after = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(chunks_after) > 0
        chunk_ids_after = {c.id for c in chunks_after}

        # Chunk IDs should change (chunks are recreated)
        assert chunk_ids_before.isdisjoint(chunk_ids_after)
