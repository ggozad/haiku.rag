"""Tests for the per-document toc.json view and the heading_level / tree_depth
fields surfaced in items.jsonl.

The TOC is derived from `DocumentItem.heading_level` (positive only) in
position order. PDF-style corpora (all section_headers at level 1) get a flat
list of siblings; HTML/markdown corpora with real heading hierarchy get a
nested tree. Items with no section_header at all produce `tree: []`.
"""

import json
from pathlib import PurePosixPath

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.sandbox import AnalysisContext, Sandbox
from haiku.rag.store.models.document import Document
from haiku.rag.store.models.document_item import DocumentItem


async def _empty_doc(client, *, uri: str, title: str) -> str:
    """Create a Document row directly via the repository (no chunking, no
    embedder) so these tests can run without a reachable embedding endpoint.
    Returns the document id."""
    doc = await client.document_repository.create(
        Document(content="x", uri=uri, title=title)
    )
    return doc.id


def _para(doc_id: str, pos: int, depth: int = 1) -> DocumentItem:
    return DocumentItem(
        document_id=doc_id,
        position=pos,
        self_ref=f"#/texts/{pos}",
        label="paragraph",
        text=f"para{pos}",
        page_numbers=[1],
        tree_depth=depth,
    )


def _header(
    doc_id: str, pos: int, level: int, text: str, depth: int = 1, page: int = 1
) -> DocumentItem:
    return DocumentItem(
        document_id=doc_id,
        position=pos,
        self_ref=f"#/texts/{pos}",
        label="section_header",
        text=text,
        page_numbers=[page],
        heading_level=level,
        tree_depth=depth,
    )


async def _read_toc(sandbox: Sandbox, doc_id: str) -> dict:
    vfs = await sandbox._build_vfs()
    raw = vfs.path_read_text(PurePosixPath(f"/documents/{doc_id}/toc.json"))
    return json.loads(raw)


async def _read_items_jsonl(sandbox: Sandbox, doc_id: str) -> list[dict]:
    vfs = await sandbox._build_vfs()
    raw = vfs.path_read_text(PurePosixPath(f"/documents/{doc_id}/items.jsonl"))
    return [json.loads(line) for line in raw.strip().splitlines()] if raw else []


def _flatten(tree: list[dict]) -> list[dict]:
    out = []
    for node in tree:
        out.append(node)
        out.extend(_flatten(node["children"]))
    return out


@pytest.mark.asyncio
class TestTocShape:
    """toc.json builds a section tree from heading_level + position."""

    async def test_multilevel_tree(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc_id = await _empty_doc(client, uri="test://multi", title="TOC Test Doc")
            items = [
                _header(doc_id, 0, 1, "Intro"),
                _para(doc_id, 1),
                _header(doc_id, 2, 2, "Background"),
                _para(doc_id, 3),
                _header(doc_id, 4, 3, "Prior Work"),
                _para(doc_id, 5),
                _header(doc_id, 6, 2, "Approach"),
                _para(doc_id, 7),
                _header(doc_id, 8, 1, "Methods"),
                _para(doc_id, 9),
            ]
            await client.document_item_repository.create_items(doc_id, items)

        sandbox = Sandbox(temp_db_path, AppConfig(), AnalysisContext())
        toc = await _read_toc(sandbox, doc_id)

        assert toc["doc_id"] == doc_id
        assert toc["title"] == "TOC Test Doc"
        tree = toc["tree"]
        # Two roots: Intro (children: Background>{Prior Work}, Approach) and Methods
        assert [n["title"] for n in tree] == ["Intro", "Methods"]
        intro = tree[0]
        assert intro["level"] == 1
        assert intro["item_range"] == [0, 8]  # ends at "Methods" position
        assert [c["title"] for c in intro["children"]] == ["Background", "Approach"]

        background = intro["children"][0]
        assert background["level"] == 2
        # Background covers positions 2..5; "Approach" begins at 6 (same-level sibling)
        assert background["item_range"] == [2, 6]
        assert [c["title"] for c in background["children"]] == ["Prior Work"]

        prior = background["children"][0]
        assert prior["level"] == 3
        # Prior Work has no descendants and the next same-or-shallower header is
        # "Approach" at level 2, position 6.
        assert prior["item_range"] == [4, 6]
        assert prior["children"] == []

        approach = intro["children"][1]
        assert approach["item_range"] == [6, 8]

        methods = tree[1]
        assert methods["item_range"] == [8, 10]  # to end of items
        assert methods["children"] == []

    async def test_flat_pdf_style(self, temp_db_path):
        """All section_headers at level 1 (PDF reality) -> flat sibling list."""
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc_id = await _empty_doc(client, uri="test://pdf-style", title="Flat PDF")
            items = [
                _header(doc_id, 0, 1, "Chapter 1"),
                _para(doc_id, 1),
                _para(doc_id, 2),
                _header(doc_id, 3, 1, "Chapter 2"),
                _para(doc_id, 4),
                _header(doc_id, 5, 1, "Chapter 3"),
                _para(doc_id, 6),
            ]
            await client.document_item_repository.create_items(doc_id, items)

        sandbox = Sandbox(temp_db_path, AppConfig(), AnalysisContext())
        toc = await _read_toc(sandbox, doc_id)

        tree = toc["tree"]
        assert [n["title"] for n in tree] == ["Chapter 1", "Chapter 2", "Chapter 3"]
        assert all(n["level"] == 1 and n["children"] == [] for n in tree)
        assert tree[0]["item_range"] == [0, 3]
        assert tree[1]["item_range"] == [3, 5]
        assert tree[2]["item_range"] == [5, 7]

    async def test_no_headers(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc_id = await _empty_doc(
                client, uri="test://no-headers", title="No Headers"
            )
            await client.document_item_repository.create_items(
                doc_id, [_para(doc_id, i) for i in range(5)]
            )

        sandbox = Sandbox(temp_db_path, AppConfig(), AnalysisContext())
        toc = await _read_toc(sandbox, doc_id)
        assert toc["tree"] == []

    async def test_skip_header_with_zero_level(self, temp_db_path):
        """A section_header with ``heading_level == 0`` is skipped."""
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc_id = await _empty_doc(client, uri="test://zero", title="Zero Level")
            items = [
                _header(doc_id, 0, 1, "Real H1"),
                _para(doc_id, 1),
                _header(doc_id, 2, 0, "Pre-migration ghost"),
                _para(doc_id, 3),
            ]
            await client.document_item_repository.create_items(doc_id, items)

        sandbox = Sandbox(temp_db_path, AppConfig(), AnalysisContext())
        toc = await _read_toc(sandbox, doc_id)
        titles = [n["title"] for n in _flatten(toc["tree"])]
        assert titles == ["Real H1"]
        assert toc["tree"][0]["item_range"] == [0, 4]


@pytest.mark.asyncio
class TestTocCaching:
    """toc.json is cached across reads — the items query runs once per sandbox."""

    async def test_cached_across_reads(self, temp_db_path, monkeypatch):
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc_id = await _empty_doc(client, uri="test://cache", title="Cache")
            await client.document_item_repository.create_items(
                doc_id,
                [_header(doc_id, 0, 1, "Only"), _para(doc_id, 1), _para(doc_id, 2)],
            )

        sandbox = Sandbox(temp_db_path, AppConfig(), AnalysisContext())

        # Patch the repository call that backs both items.jsonl and toc.json so we
        # can count how many bulk fetches happen. The sandbox opens its own
        # HaikuRAG client(s) lazily; patch the class method.
        from haiku.rag.store.repositories.document_item import DocumentItemRepository

        call_count = {"n": 0}
        original = DocumentItemRepository.get_all_items_grouped

        async def counting(self, document_ids=None):
            call_count["n"] += 1
            return await original(self, document_ids)

        monkeypatch.setattr(DocumentItemRepository, "get_all_items_grouped", counting)

        # First read of either file triggers ONE bulk fetch.
        _ = await _read_toc(sandbox, doc_id)
        _ = await _read_items_jsonl(sandbox, doc_id)
        _ = await _read_toc(sandbox, doc_id)
        _ = await _read_items_jsonl(sandbox, doc_id)

        assert call_count["n"] == 1


@pytest.mark.asyncio
class TestItemsJsonlSurfacesNewFields:
    """items.jsonl row shape: heading_level is always present (0 on non-headers);
    chunk_ids surfaces each item's containing chunks; position and tree_depth
    are not exposed."""

    async def test_jsonl_row_shape(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc_id = await _empty_doc(client, uri="test://jsonl-fields", title="Fields")
            items = [
                _header(doc_id, 0, 1, "H1", depth=2),
                _para(doc_id, 1, depth=3),
                _header(doc_id, 2, 2, "H2", depth=4),
            ]
            await client.document_item_repository.create_items(doc_id, items)

        sandbox = Sandbox(temp_db_path, AppConfig(), AnalysisContext())
        rows = await _read_items_jsonl(sandbox, doc_id)

        assert len(rows) == 3
        assert rows[0]["heading_level"] == 1
        assert rows[1]["heading_level"] == 0
        assert rows[2]["heading_level"] == 2
        for r in rows:
            expected = {
                "self_ref",
                "label",
                "text",
                "page_numbers",
                "heading_level",
                "chunk_ids",
            }
            assert expected <= set(r)
            assert "position" not in r
            assert "tree_depth" not in r
