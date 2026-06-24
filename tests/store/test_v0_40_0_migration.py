"""Tests for the v0.40.0 document_items population migration.

The migration walks every document with a docling blob, extracts items, and
populates the ``document_items`` table. It must stay independent of columns
introduced by later migrations (``picture_data`` in v0.45.0,
``heading_level`` / ``tree_depth`` in v0.48.0).
"""

import pytest

from haiku.rag.store import Store
from haiku.rag.store.compression import compress_docling_split
from haiku.rag.store.engine import DocumentRecord
from haiku.rag.store.upgrades.v0_40_0 import _apply_populate_document_items


def _simple_docling_doc():
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel

    doc = DoclingDocument(name="simple")
    doc.add_heading(text="Intro", level=1)
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Hello there.")
    return doc


@pytest.mark.asyncio
async def test_populate_handles_extra_columns_on_items_table(temp_db_path):
    """Regression: v0.40.0 must not fail when the live ``document_items``
    table carries columns added by later migrations.

    Mirrors what happens in practice: ``_init_tables`` always creates
    ``document_items`` with the latest schema (picture_data, heading_level,
    tree_depth). v0.40.0 then runs against that table — its input must be
    accepted even though it only writes the original 6 columns.
    """
    docling_doc = _simple_docling_doc()
    structure, pages = compress_docling_split(docling_doc.model_dump(mode="json"))

    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        # _init_tables already created document_items with the latest schema.
        names = {f.name for f in await store.document_items_table.schema()}
        assert {"picture_data", "heading_level", "tree_depth"} <= names

        await store.documents_table.add(
            [
                DocumentRecord(
                    id="doc-1",
                    content="x",
                    docling_document=structure,
                    docling_pages=pages,
                    docling_version=docling_doc.version,
                )
            ]
        )

        await _apply_populate_document_items(store)

        rows = await (
            store.document_items_table.query().where("document_id = 'doc-1'").to_list()
        )
        assert len(rows) >= 2
        # Columns we didn't write should be null / default-typed.
        for row in rows:
            assert row.get("picture_data") is None
