import pytest
from datasets import Dataset

from haiku.rag.client import HaikuRAG, RebuildMode


@pytest.mark.asyncio
async def test_rebuild_full(qa_corpus: Dataset, temp_db_path):
    """Test full rebuild: converts, chunks, and embeds all documents."""
    async with HaikuRAG(temp_db_path) as client:
        doc = await client.create_document(content=qa_corpus["document_extracted"][0])
        assert doc.id is not None

        chunks_before = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(chunks_before) > 0
        chunk_ids_before = {c.id for c in chunks_before}

        processed_ids = [doc_id async for doc_id in client.rebuild_database()]

        assert doc.id in processed_ids

        chunks_after = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(chunks_after) > 0
        chunk_ids_after = {c.id for c in chunks_after}

        # Chunk IDs should change (chunks are recreated)
        assert chunk_ids_before.isdisjoint(chunk_ids_after)


@pytest.mark.asyncio
async def test_rebuild_embed_only(qa_corpus: Dataset, temp_db_path):
    """Test embed-only rebuild: keeps chunks, only regenerates embeddings."""
    async with HaikuRAG(temp_db_path) as client:
        doc = await client.create_document(content=qa_corpus["document_extracted"][0])
        assert doc.id is not None

        chunks_before = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(chunks_before) > 0
        chunk_ids_before = {c.id for c in chunks_before}
        chunk_contents_before = {c.id: c.content for c in chunks_before}

        processed_ids = [
            doc_id
            async for doc_id in client.rebuild_database(mode=RebuildMode.EMBED_ONLY)
        ]

        assert doc.id in processed_ids

        chunks_after = await client.chunk_repository.get_by_document_id(doc.id)
        chunk_ids_after = {c.id for c in chunks_after}

        # Chunk IDs should be preserved (same chunks, just re-embedded)
        assert chunk_ids_before == chunk_ids_after

        # Content should be identical
        for chunk in chunks_after:
            assert chunk.content == chunk_contents_before[chunk.id]


@pytest.mark.asyncio
async def test_rebuild_rechunk(qa_corpus: Dataset, temp_db_path):
    """Test rechunk rebuild: re-chunks from content without accessing source files."""
    async with HaikuRAG(temp_db_path) as client:
        doc = await client.create_document(content=qa_corpus["document_extracted"][0])
        assert doc.id is not None

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

        # Document content should be unchanged
        doc_after = await client.document_repository.get_by_id(doc.id)
        assert doc_after is not None
        assert doc_after.content == content_before

        chunks_after = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(chunks_after) > 0
        chunk_ids_after = {c.id for c in chunks_after}

        # Chunk IDs should change (chunks are recreated)
        assert chunk_ids_before.isdisjoint(chunk_ids_after)
