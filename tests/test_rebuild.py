import pytest
from datasets import Dataset

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models.document import Document


@pytest.mark.asyncio
async def test_rebuild_database(qa_corpus: Dataset):
    """Test rebuild functionality with existing documents."""
    async with HaikuRAG(":memory:") as client:
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
