import json

import pytest

from haiku.rag.store.engine import Store
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.upgrades.v0_57_0 import _apply_split_document_meta
from tests.store.legacy_documents import (
    LegacyDocumentRecord,
    seed_legacy_documents,
)

_LEGACY_COLUMNS = {"uri", "title", "metadata", "created_at", "updated_at"}


@pytest.mark.asyncio
class TestV0_57_0Migration:
    """v0.57.0 moves mutable attributes out of the documents row into
    document_meta so metadata/title updates stop rewriting the docling blob."""

    async def test_moves_attributes_and_drops_columns(self, temp_db_path):
        async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
            await seed_legacy_documents(
                store,
                [
                    LegacyDocumentRecord(
                        id="doc-1",
                        content="body one",
                        uri="s3://b/one",
                        title="One",
                        metadata=json.dumps({"source_revision": "r1", "md5": "a"}),
                        docling_document=b"structure-blob-1",
                        docling_pages=b"pages-blob-1",
                        docling_version="1.10.0",
                        created_at="2026-01-01T00:00:00",
                        updated_at="2026-01-02T00:00:00",
                    )
                ],
            )
            await store.set_haiku_version("0.56.0")

        async with Store(temp_db_path, skip_migration_check=True) as store:
            applied = await store.migrate()
            assert any("0.57.0" in d for d in applied)

            # Legacy columns dropped from documents; blobs stay.
            doc_names = {f.name for f in await store.documents_table.schema()}
            assert _LEGACY_COLUMNS.isdisjoint(doc_names)
            assert {"id", "content", "docling_document", "docling_pages"} <= doc_names

            # Attributes landed in document_meta.
            meta_rows = await store.document_meta_table.query().to_list()
            assert len(meta_rows) == 1
            row = meta_rows[0]
            assert row["document_id"] == "doc-1"
            assert row["uri"] == "s3://b/one"
            assert row["title"] == "One"
            assert json.loads(row["metadata"]) == {"source_revision": "r1", "md5": "a"}

            # Full hydration still works (content + metadata + blobs intact).
            repo = DocumentRepository(store)
            doc = await repo.get_by_id("doc-1")
            assert doc is not None
            assert doc.content == "body one"
            assert doc.uri == "s3://b/one"
            assert doc.title == "One"
            assert doc.metadata == {"source_revision": "r1", "md5": "a"}
            assert doc.docling_document == b"structure-blob-1"
            assert doc.docling_pages == b"pages-blob-1"
            assert doc.docling_version == "1.10.0"

            # Lookup by uri (resolved via document_meta) works too.
            by_uri = await repo.get_by_uri("s3://b/one")
            assert by_uri is not None and by_uri.id == "doc-1"

    async def test_idempotent_on_already_split(self, temp_db_path):
        async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
            await seed_legacy_documents(
                store,
                [
                    LegacyDocumentRecord(
                        id="doc-1",
                        content="body",
                        uri="u1",
                        metadata=json.dumps({"source_revision": "r1"}),
                    )
                ],
            )
            await store.set_haiku_version("0.56.0")

        async with Store(temp_db_path, skip_migration_check=True) as store:
            await store.migrate()
            # Re-applying must be a no-op (documents already split).
            await _apply_split_document_meta(store)

            meta_rows = await store.document_meta_table.query().to_list()
            assert len(meta_rows) == 1
            assert meta_rows[0]["document_id"] == "doc-1"
