import pytest
from datasets import Dataset

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models.document import Document


@pytest.mark.asyncio
async def test_rebuild_database(qa_corpus: Dataset, temp_db_path):
    """Test rebuild functionality with existing documents."""
    async with HaikuRAG(temp_db_path) as client:
        created_docs: list[Document] = []
        for content in qa_corpus["document_extracted"][:3]:
            doc = await client.create_document(
                content=content,
            )
            created_docs.append(doc)

        documents_before = await client.list_documents()
        assert len(documents_before) == 3

        chunks_before = []
        for doc in created_docs:
            assert doc.id is not None
            doc_chunks = await client.chunk_repository.get_by_document_id(doc.id)
            chunks_before.extend(doc_chunks)

        assert len(chunks_before) > 0

        # Perform rebuild
        processed_doc_ids = []
        async for doc_id in client.rebuild_database():
            processed_doc_ids.append(doc_id)

        # Verify all documents were processed
        expected_doc_ids = [doc.id for doc in created_docs]
        assert set(processed_doc_ids) == set(expected_doc_ids)

        documents_after = await client.list_documents()
        assert len(documents_after) == 3

        # Verify chunks were recreated
        chunks_after = []
        for doc in documents_after:
            if doc.id is not None:
                doc_chunks = await client.chunk_repository.get_by_document_id(doc.id)
                chunks_after.extend(doc_chunks)

        assert len(chunks_after) > 0


@pytest.mark.asyncio
async def test_rebuild_with_missing_source(qa_corpus: Dataset, temp_db_path):
    """Test rebuild functionality when document source is missing."""
    async with HaikuRAG(temp_db_path) as client:
        # Create document with content
        content = qa_corpus["document_extracted"][0]
        doc = await client.create_document(content=content)

        # Manually set a URI that doesn't exist
        assert doc.id is not None
        doc_with_uri = await client.document_repository.get_by_id(doc.id)
        assert doc_with_uri is not None
        doc_with_uri.uri = "file:///nonexistent/path.txt"
        await client.document_repository.update(doc_with_uri)

        # Verify chunks exist before rebuild
        chunks_before = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(chunks_before) > 0

        # Perform rebuild
        processed_doc_ids = []
        async for doc_id in client.rebuild_database():
            processed_doc_ids.append(doc_id)

        # Document should still be processed (not skipped)
        assert doc.id in processed_doc_ids

        # Verify document still exists
        doc_after = await client.document_repository.get_by_id(doc.id)
        assert doc_after is not None
        assert doc_after.content == content

        # Verify chunks were recreated from content
        chunks_after = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(chunks_after) > 0
