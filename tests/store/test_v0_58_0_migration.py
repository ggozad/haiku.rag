import json

import pytest

from haiku.rag.store.engine import Store
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.upgrades.v0_58_0 import _apply_split_document_meta
from tests.store.legacy_documents import (
    LegacyDocumentRecord,
    seed_legacy_documents,
)

_LEGACY_COLUMNS = {"uri", "title", "metadata", "created_at", "updated_at"}


@pytest.mark.asyncio
class TestV0_58_0Migration:
    """v0.58.0 moves mutable attributes out of the documents row into
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
            await store.set_haiku_version("0.57.0")

        async with Store(temp_db_path, skip_migration_check=True) as store:
            applied = await store.migrate()
            assert any("0.58.0" in d for d in applied)

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
            await store.set_haiku_version("0.57.0")

        async with Store(temp_db_path, skip_migration_check=True) as store:
            await store.migrate()
            # Re-applying must be a no-op (documents already split).
            await _apply_split_document_meta(store)

            meta_rows = await store.document_meta_table.query().to_list()
            assert len(meta_rows) == 1
            assert meta_rows[0]["document_id"] == "doc-1"


@pytest.mark.asyncio
class TestV0_58_0MigrationEdgeCases:
    async def test_resume_skips_already_migrated_rows(self, temp_db_path):
        """A half-finished prior run leaves some document_meta rows; re-running
        migrates only the rest and never duplicates."""
        from haiku.rag.store.engine import DocumentMetaRecord

        async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
            await seed_legacy_documents(
                store,
                [
                    LegacyDocumentRecord(id="a", content="x", uri="u-a", metadata="{}"),
                    LegacyDocumentRecord(id="b", content="y", uri="u-b", metadata="{}"),
                ],
            )
            # Pretend a prior run already moved doc "a".
            await store.document_meta_table.add(
                [DocumentMetaRecord(document_id="a", uri="u-a", metadata="{}")]
            )
            await store.set_haiku_version("0.57.0")

        async with Store(temp_db_path, skip_migration_check=True) as store:
            await store.migrate()
            rows = await store.document_meta_table.query().to_list()
            by_id = {r["document_id"]: r for r in rows}
            assert set(by_id) == {"a", "b"}  # exactly one row each, no duplicates
            assert len(rows) == 2

    async def test_skips_vacuum_when_disk_is_tight(self, temp_db_path, monkeypatch):
        """When free disk can't cover one compacted copy, the split still
        completes but the reclaim vacuum is skipped."""
        from types import SimpleNamespace

        from haiku.rag.store.upgrades import v0_58_0

        async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
            await seed_legacy_documents(
                store,
                [LegacyDocumentRecord(id="a", content="x", uri="u", metadata="{}")],
            )
            await store.set_haiku_version("0.57.0")

        # Pretend almost no free disk so the reclaim vacuum is skipped.
        monkeypatch.setattr(
            v0_58_0.shutil,
            "disk_usage",
            lambda _p: SimpleNamespace(total=1, used=1, free=1),
        )

        vacuum_calls: list[int] = []

        async with Store(temp_db_path, skip_migration_check=True) as store:
            # Force a known nonzero live size so the free<live check is
            # deterministic (real stats().total_bytes can be 0 for a tiny table).
            async def fake_stats():
                return {"total_bytes": 10_000_000}

            monkeypatch.setattr(store.documents_table, "stats", fake_stats)

            orig_vacuum = store.vacuum

            async def tracking_vacuum(*args, **kwargs):
                vacuum_calls.append(1)
                return await orig_vacuum(*args, **kwargs)

            monkeypatch.setattr(store, "vacuum", tracking_vacuum)

            await store.migrate()
            # Split still happened despite the skipped vacuum.
            names = {f.name for f in await store.documents_table.schema()}
            assert _LEGACY_COLUMNS.isdisjoint(names)
            repo = DocumentRepository(store)
            migrated = await repo.get_by_id("a")
            assert migrated is not None and migrated.uri == "u"

        assert vacuum_calls == []  # reclaim vacuum was skipped


@pytest.mark.asyncio
async def test_delete_all_clears_document_meta(temp_db_path):
    """delete_all drops and recreates both documents and document_meta."""
    from haiku.rag.store.models.document import Document

    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="x", uri="u1"))
        await repo.create(Document(content="y", uri="u2"))
        assert await store.document_meta_table.count_rows() == 2

        await repo.delete_all()

        assert await store.documents_table.count_rows() == 0
        assert await store.document_meta_table.count_rows() == 0
        # Tables are usable again after recreation.
        await repo.create(Document(content="z", uri="u3"))
        assert await store.document_meta_table.count_rows() == 1
