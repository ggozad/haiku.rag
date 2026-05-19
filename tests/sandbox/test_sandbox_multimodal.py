"""Tests for the multimodal sandbox surface: picture_refs on search dicts
and the set of external functions exposed to the interpreter.
"""

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.sandbox import AnalysisContext, Sandbox
from haiku.rag.store.models.chunk import SearchResult


@pytest.mark.asyncio
class TestSearchPictureRefs:
    """search() result dicts carry a `picture_refs` list (subset of
    doc_item_refs labeled 'picture'). No `image_data` base64 in the dict."""

    async def test_picture_refs_extracted_from_self_refs(
        self, temp_db_path, monkeypatch
    ):
        # ``labels`` is a deduplicated set after expand_context and is not
        # aligned with ``doc_item_refs``, so picture refs must be recovered
        # from the docling ``#/pictures/...`` self_ref convention.
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
                labels=["picture", "text"],
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
        assert "image_data" not in results[0]


@pytest.mark.asyncio
class TestExternalFunctionsShape:
    """Sandbox externals expose exactly ``search`` and ``list_documents``.

    Pictures reach the driving model through the skill's own ``search``
    tool (BinaryContent auto-attach for picture chunks) and surface in
    the UI as citations with ``picture_refs`` populated by
    ``resolve_citations``.
    """

    async def test_externals_are_search_and_list_documents_only(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True):
            pass
        sandbox = Sandbox(temp_db_path, AppConfig(), AnalysisContext())
        external = sandbox._build_external_functions()
        assert set(external) == {"search", "list_documents"}
