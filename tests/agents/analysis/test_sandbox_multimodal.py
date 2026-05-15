"""Tests for the multimodal sandbox surface: show_image, picture_refs in
search results, binary_attachments on SandboxResult, and the absence of
the old llm() external function.
"""

import base64
from io import BytesIO

import pytest
from PIL import Image

from haiku.rag.agents.analysis.dependencies import AnalysisContext
from haiku.rag.agents.analysis.sandbox import Sandbox
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models.chunk import SearchResult
from haiku.rag.store.models.document_item import DocumentItem


def _png_bytes(color: str = "red", size: tuple[int, int] = (8, 8)) -> bytes:
    """Generate a real PNG so PIL.Image.verify() accepts it."""
    img = Image.new("RGB", size, color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def _seed_doc_with_picture(client, *, png: bytes) -> tuple[str, str]:
    """Create a Document row, replace its items with one picture row carrying
    the given bytes. Returns (doc_id, self_ref)."""
    doc = await client.create_document(content="x", uri="test://pic", title="Pic")
    await client.document_item_repository.delete_by_document_id(doc.id)
    self_ref = "#/pictures/0"
    items = [
        DocumentItem(
            document_id=doc.id,
            position=0,
            self_ref=self_ref,
            label="picture",
            text="",
            page_numbers=[1],
            picture_data=png,
        )
    ]
    await client.document_item_repository.create_items(doc.id, items)
    return doc.id, self_ref


@pytest.mark.asyncio
class TestShowImage:
    """show_image() appends a BinaryContent attachment when bytes verify."""

    async def test_appends_binary_attachment(self, temp_db_path):
        png = _png_bytes("red")
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc_id, ref = await _seed_doc_with_picture(client, png=png)

        sandbox = Sandbox(temp_db_path, AppConfig(), AnalysisContext())
        result = await sandbox.execute(
            f"await show_image({doc_id!r}, {ref!r})\nprint('done')"
        )
        assert result.success, result.stderr
        assert len(result.binary_attachments) == 1
        att = result.binary_attachments[0]
        assert att.media_type == "image/png"
        assert att.identifier == ref
        assert att.data == png

    async def test_missing_picture_is_silent_noop(self, temp_db_path):
        png = _png_bytes("red")
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc_id, _ = await _seed_doc_with_picture(client, png=png)

        sandbox = Sandbox(temp_db_path, AppConfig(), AnalysisContext())
        result = await sandbox.execute(
            f"await show_image({doc_id!r}, '#/pictures/999')\nprint('ok')"
        )
        assert result.success, result.stderr
        assert result.binary_attachments == []

    async def test_invalid_bytes_rejected(self, temp_db_path):
        # Garbage bytes — PIL.verify() should refuse, no attachment emitted.
        garbage = b"this is not a PNG"
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc_id, ref = await _seed_doc_with_picture(client, png=garbage)

        sandbox = Sandbox(temp_db_path, AppConfig(), AnalysisContext())
        result = await sandbox.execute(
            f"await show_image({doc_id!r}, {ref!r})\nprint('checked')"
        )
        assert result.success, result.stderr
        assert result.binary_attachments == []

    async def test_attachments_reset_across_executes(self, temp_db_path):
        png = _png_bytes("red")
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc_id, ref = await _seed_doc_with_picture(client, png=png)

        sandbox = Sandbox(temp_db_path, AppConfig(), AnalysisContext())
        first = await sandbox.execute(
            f"await show_image({doc_id!r}, {ref!r})\nprint('first')"
        )
        second = await sandbox.execute("print('second')")
        assert first.success and len(first.binary_attachments) == 1
        assert second.success and second.binary_attachments == []


@pytest.mark.asyncio
class TestSearchPictureRefs:
    """search() result dicts carry a `picture_refs` list (subset of
    doc_item_refs labeled 'picture'). No `image_data` base64 in the dict."""

    async def test_picture_refs_extracted_from_labels(self, temp_db_path, monkeypatch):
        # Build a fake SearchResult with mixed labels so we don't need an embedder.
        synthetic = [
            SearchResult(
                chunk_id="c1",
                content="hit",
                document_id="d1",
                document_uri="test://d1",
                document_title="D1",
                score=1.0,
                page_numbers=[1],
                headings=None,
                doc_item_refs=["#/texts/0", "#/pictures/0", "#/pictures/1"],
                labels=["text", "picture", "picture"],
            ),
            SearchResult(
                chunk_id="c2",
                content="text only",
                document_id="d1",
                document_uri="test://d1",
                document_title="D1",
                score=0.5,
                page_numbers=[2],
                headings=None,
                doc_item_refs=["#/texts/5"],
                labels=["text"],
            ),
        ]

        async def fake_search(self, *args, **kwargs):
            return synthetic

        async def fake_expand_context(self, results):
            return results

        # Patch HaikuRAG.search and expand_context so the sandbox closure runs
        # without an embedder. The sandbox opens its own HaikuRAG instance, so
        # we patch on the class.
        monkeypatch.setattr(HaikuRAG, "search", fake_search)
        monkeypatch.setattr(HaikuRAG, "expand_context", fake_expand_context)

        async with HaikuRAG(temp_db_path, create=True):
            pass  # ensure the DB exists so the sandbox can open it read-only

        sandbox = Sandbox(temp_db_path, AppConfig(), AnalysisContext())
        external = sandbox._build_external_functions()
        results = await external["search"]("anything")

        assert len(results) == 2
        assert results[0]["picture_refs"] == ["#/pictures/0", "#/pictures/1"]
        assert results[1]["picture_refs"] == []
        # No raw base64 garbage in the dict.
        assert "image_data" not in results[0]


@pytest.mark.asyncio
class TestExternalFunctionsShape:
    """llm() is gone; show_image() is present."""

    async def test_llm_gone_show_image_present(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True):
            pass
        sandbox = Sandbox(temp_db_path, AppConfig(), AnalysisContext())
        external = sandbox._build_external_functions()
        assert "llm" not in external
        assert "show_image" in external
        assert "search" in external
        assert "list_documents" in external


# Silence unused-import flake — base64 is reserved for follow-up tests that
# decode attachment.data and compare. Kept eagerly imported for parity with
# the QA binary-content tests.
_ = base64
