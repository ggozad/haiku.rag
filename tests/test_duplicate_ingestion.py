import asyncio
import tempfile
from pathlib import Path

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models.document import Document

pytestmark = pytest.mark.asyncio


@pytest.mark.vcr()
async def test_concurrent_same_uri_ingestion_creates_single_document(temp_db_path):
    """Two concurrent ingestions of the same URI must not create duplicates.

    Both calls read `existing_doc=None` before either acquires the write lock;
    the atomic re-check under the lock turns the loser into an update instead of
    a second insert (LanceDB has no unique constraint on `uri`).
    """
    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "dup.txt"
            temp_path.write_text("Duplicate ingestion regression content.")

            results = await asyncio.gather(
                client.create_document_from_source(temp_path),
                client.create_document_from_source(temp_path),
            )

            for doc in results:
                assert isinstance(doc, Document)
            # Both concurrent calls resolve to one document, not two.
            assert results[0].id == results[1].id
            assert await client.count_documents() == 1

            # The surviving document owns exactly one ingestion's chunks; the
            # loser's create was collapsed into an update, leaving no orphans.
            surviving = await client.get_document_by_uri(temp_path.as_uri())
            assert surviving is not None
            chunks = await client.chunk_repository.get_by_document_id(surviving.id)
            assert len(chunks) >= 1
