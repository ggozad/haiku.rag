import pytest
from lancedb.pydantic import LanceModel

from haiku.rag.store.engine import Store
from haiku.rag.store.upgrades.v0_64_0 import _apply_rename_document_meta_id


class LegacyMetaRecord(LanceModel):
    """The pre-0.63.2 `document_meta` record (identity column `document_id`)."""

    document_id: str
    uri: str | None = None
    title: str | None = None
    metadata: str = "{}"
    created_at: str = ""
    updated_at: str = ""


async def _seed_legacy_meta(store: Store, doc_id: str) -> None:
    await store.db.drop_table("document_meta")
    store.document_meta_table = await store.db.create_table(
        "document_meta", schema=LegacyMetaRecord
    )
    await store.document_meta_table.add(
        [LegacyMetaRecord(document_id=doc_id, uri="u", metadata="{}")]
    )


@pytest.mark.asyncio
async def test_renames_document_id_to_id_and_keeps_rows(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_legacy_meta(store, "doc-1")

        await _apply_rename_document_meta_id(store)

        names = {f.name for f in await store.document_meta_table.schema()}
        assert "id" in names and "document_id" not in names
        rows = await store.document_meta_table.query().to_list()
        assert [r["id"] for r in rows] == ["doc-1"]


@pytest.mark.asyncio
async def test_idempotent_when_already_renamed(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_legacy_meta(store, "doc-1")
        await _apply_rename_document_meta_id(store)
        # Second run must be a no-op, not an error.
        await _apply_rename_document_meta_id(store)

        names = {f.name for f in await store.document_meta_table.schema()}
        assert "id" in names and "document_id" not in names
