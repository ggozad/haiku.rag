"""Tests for the v0.40.0 document_items population migration.

The migration walks every document with a docling blob, extracts items, and
populates the ``document_items`` table. It must stay independent of columns
introduced by later migrations (``picture_data`` in v0.45.0,
``heading_level`` / ``tree_depth`` in v0.48.0).
"""

from haiku.rag.store.compression import compress_docling_split
from haiku.rag.store.engine import Store
from haiku.rag.store.schema import DocumentRecord
from haiku.rag.store.upgrades.v0_40_0 import _apply_populate_document_items


def _mixed_docling_doc():
    from docling_core.types.doc.document import DoclingDocument, TableData
    from docling_core.types.doc.labels import DocItemLabel

    doc = DoclingDocument(name="test")
    doc.add_text(label=DocItemLabel.SECTION_HEADER, text="Introduction")
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="This is the first paragraph.")
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="This is the second paragraph.")
    doc.add_table(data=TableData(num_rows=2, num_cols=2, table_cells=[]))
    doc.add_text(label=DocItemLabel.SECTION_HEADER, text="Conclusion")
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Final thoughts here.")
    return doc


async def test_populate_handles_extra_columns_on_items_table(temp_db_path):
    """v0.40.0 writes document items into the current schema and leaves
    picture_data null."""
    docling_doc = _mixed_docling_doc()
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
        assert len(rows) == 6
        assert {row["label"] for row in rows} == {
            "section_header",
            "paragraph",
            "table",
        }
        # Columns we didn't write should be null / default-typed.
        for row in rows:
            assert row.get("picture_data") is None


async def test_populate_skips_documents_without_docling(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await store.set_haiku_version("0.39.0")
        await store.documents_table.add(
            [DocumentRecord(id="no-docling", content="plain text document")]
        )

    async with Store(temp_db_path, skip_migration_check=True) as store:
        await _apply_populate_document_items(store)

        assert await store.document_items_table.count_rows() == 0
