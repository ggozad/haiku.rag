import json

import pytest

from haiku.rag.store.compression import compress_docling_split
from haiku.rag.store.engine import DocumentItemRecord, DocumentRecord, Store


def _docling_with_levels():
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel

    doc = DoclingDocument(name="leveled")
    doc.add_heading(text="Introduction", level=1)
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Opening.")
    doc.add_heading(text="Background", level=2)
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Bg paragraph.")
    doc.add_heading(text="Methods", level=1)
    return doc


@pytest.mark.asyncio
class TestV0_48_0Migration:
    """v0.48.0 backfills heading_level + tree_depth on existing items rows."""

    async def test_backfill_populates_levels(self, temp_db_path):
        docling_doc = _docling_with_levels()
        structure, pages = compress_docling_split(docling_doc.model_dump_json())

        async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
            await store.set_haiku_version("0.45.0")
            await store.documents_table.add(
                [
                    DocumentRecord(
                        id="doc-1",
                        content="legacy",
                        docling_document=structure,
                        docling_pages=pages,
                        docling_version=docling_doc.version,
                    )
                ]
            )
            # Pre-migration items: heading_level / tree_depth default to 0.
            # Match what v0.40.0 produced (label, text, page_numbers, no levels).
            items = list(docling_doc.iterate_items())
            await store.document_items_table.add(
                [
                    DocumentItemRecord(
                        document_id="doc-1",
                        position=pos,
                        self_ref=item.self_ref,
                        label=str(getattr(item.label, "value", item.label) or ""),
                        text=getattr(item, "text", "") or "",
                        page_numbers="[]",
                    )
                    for pos, (item, _depth) in enumerate(items)
                ]
            )

        async with Store(temp_db_path, skip_migration_check=True) as store:
            applied = await store.migrate()
            assert any("0.48.0" in d for d in applied)

            rows = await (
                store.document_items_table.query()
                .where("document_id = 'doc-1'")
                .to_list()
            )
            rows.sort(key=lambda r: r["position"])
            headers = [r for r in rows if r["label"] == "section_header"]
            assert [r["heading_level"] for r in headers] == [1, 2, 1]
            non_headers = [r for r in rows if r["label"] != "section_header"]
            assert non_headers
            assert all(r["heading_level"] == 0 for r in non_headers)
            assert all(r["tree_depth"] > 0 for r in rows)

    async def test_backfill_idempotent(self, temp_db_path):
        docling_doc = _docling_with_levels()
        structure, pages = compress_docling_split(docling_doc.model_dump_json())

        async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
            await store.set_haiku_version("0.45.0")
            await store.documents_table.add(
                [
                    DocumentRecord(
                        id="doc-1",
                        content="legacy",
                        docling_document=structure,
                        docling_pages=pages,
                        docling_version=docling_doc.version,
                    )
                ]
            )
            items = list(docling_doc.iterate_items())
            await store.document_items_table.add(
                [
                    DocumentItemRecord(
                        document_id="doc-1",
                        position=pos,
                        self_ref=item.self_ref,
                        label=str(getattr(item.label, "value", item.label) or ""),
                        text=getattr(item, "text", "") or "",
                        page_numbers="[]",
                    )
                    for pos, (item, _depth) in enumerate(items)
                ]
            )

        async with Store(temp_db_path, skip_migration_check=True) as store:
            await store.migrate()

            from haiku.rag.store.upgrades.v0_48_0 import (
                _apply_backfill_heading_hierarchy,
            )

            await _apply_backfill_heading_hierarchy(store)

            rows = await (
                store.document_items_table.query()
                .where("document_id = 'doc-1'")
                .to_list()
            )
            rows.sort(key=lambda r: r["position"])
            headers = [r for r in rows if r["label"] == "section_header"]
            assert [r["heading_level"] for r in headers] == [1, 2, 1]

    async def test_skips_documents_without_docling(self, temp_db_path):
        async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
            await store.set_haiku_version("0.45.0")
            await store.documents_table.add(
                [DocumentRecord(id="plain", content="no docling")]
            )
            await store.document_items_table.add(
                [
                    DocumentItemRecord(
                        document_id="plain",
                        position=0,
                        self_ref="#/texts/0",
                        label="paragraph",
                        text="Hi",
                        page_numbers="[]",
                    )
                ]
            )

        async with Store(temp_db_path, skip_migration_check=True) as store:
            await store.migrate()
            rows = await (
                store.document_items_table.query()
                .where("document_id = 'plain'")
                .to_list()
            )
            # Plain doc — no levels to backfill; defaults stay zero.
            assert rows[0]["heading_level"] == 0
            assert rows[0]["tree_depth"] == 0


@pytest.mark.asyncio
class TestV0_48_0FreshSchema:
    """A newly-created DB has the new columns from the start."""

    async def test_fresh_db_has_heading_level_and_tree_depth(self, temp_db_path):
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as rag:
            schema = await rag.store.document_items_table.schema()
            names = {f.name for f in schema}
            assert "heading_level" in names
            assert "tree_depth" in names


def test_module_imports():
    from haiku.rag.store.upgrades import v0_48_0  # noqa: F401

    assert hasattr(v0_48_0, "upgrade_backfill_heading_hierarchy")
    assert json.dumps  # silence unused-import flake on json
