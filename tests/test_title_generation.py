import random

import pytest
from docling_core.types.doc.document import ContentLayer, DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig
from haiku.rag.config.models import ProcessingConfig
from haiku.rag.embeddings import EmbedderWrapper


@pytest.fixture(autouse=True)
def mock_embedder(monkeypatch):
    """Monkeypatch the embedder to return deterministic vectors."""

    async def fake_embed_query(self, text):
        random.seed(hash(text) % (2**32))
        return [random.random() for _ in range(2560)]

    async def fake_embed_documents(self, texts):
        result = []
        for t in texts:
            random.seed(hash(t) % (2**32))
            result.append([random.random() for _ in range(2560)])
        return result

    monkeypatch.setattr(EmbedderWrapper, "embed_query", fake_embed_query)
    monkeypatch.setattr(EmbedderWrapper, "embed_documents", fake_embed_documents)


# =========================================================================
# Structural title extraction
# =========================================================================


class TestExtractStructuralTitle:
    def _make_client(self, tmp_path):
        config = AppConfig(processing=ProcessingConfig(auto_title=True))
        return HaikuRAG(tmp_path / "test.lancedb", config=config, create=True)

    def test_furniture_title(self, tmp_path):
        """TITLE on FURNITURE layer (HTML <title>) is extracted."""
        doc = DoclingDocument(name="test")
        doc.add_text(
            label=DocItemLabel.TITLE,
            text="Website Page Title",
            content_layer=ContentLayer.FURNITURE,
        )
        doc.add_text(label=DocItemLabel.PARAGRAPH, text="Body text")

        client = self._make_client(tmp_path)
        result = client._extract_structural_title(doc)
        assert result == "Website Page Title"

    def test_body_title(self, tmp_path):
        """TITLE on BODY layer (h1, PDF title) is extracted."""
        doc = DoclingDocument(name="test")
        doc.add_text(
            label=DocItemLabel.TITLE,
            text="Document Heading",
            content_layer=ContentLayer.BODY,
        )
        doc.add_text(label=DocItemLabel.PARAGRAPH, text="Body text")

        client = self._make_client(tmp_path)
        result = client._extract_structural_title(doc)
        assert result == "Document Heading"

    def test_section_header_fallback(self, tmp_path):
        """First SECTION_HEADER is used when no TITLE exists."""
        doc = DoclingDocument(name="test")
        doc.add_text(label=DocItemLabel.SECTION_HEADER, text="Introduction")
        doc.add_text(label=DocItemLabel.SECTION_HEADER, text="Background")
        doc.add_text(label=DocItemLabel.PARAGRAPH, text="Body text")

        client = self._make_client(tmp_path)
        result = client._extract_structural_title(doc)
        assert result == "Introduction"

    def test_no_title_or_headers(self, tmp_path):
        """Returns None when no TITLE or SECTION_HEADER exists."""
        doc = DoclingDocument(name="test")
        doc.add_text(label=DocItemLabel.PARAGRAPH, text="Just a paragraph")

        client = self._make_client(tmp_path)
        result = client._extract_structural_title(doc)
        assert result is None

    def test_furniture_title_preferred_over_body_title(self, tmp_path):
        """FURNITURE TITLE takes priority over BODY TITLE."""
        doc = DoclingDocument(name="test")
        doc.add_text(
            label=DocItemLabel.TITLE,
            text="Body H1 Title",
            content_layer=ContentLayer.BODY,
        )
        doc.add_text(
            label=DocItemLabel.TITLE,
            text="HTML Page Title",
            content_layer=ContentLayer.FURNITURE,
        )

        client = self._make_client(tmp_path)
        result = client._extract_structural_title(doc)
        assert result == "HTML Page Title"

    def test_whitespace_stripped(self, tmp_path):
        """Whitespace is stripped from extracted titles."""
        doc = DoclingDocument(name="test")
        doc.add_text(
            label=DocItemLabel.TITLE,
            text="  Padded Title  ",
            content_layer=ContentLayer.BODY,
        )

        client = self._make_client(tmp_path)
        result = client._extract_structural_title(doc)
        assert result == "Padded Title"

    def test_empty_title_text_skipped(self, tmp_path):
        """Empty or whitespace-only TITLE text is skipped."""
        doc = DoclingDocument(name="test")
        doc.add_text(
            label=DocItemLabel.TITLE,
            text="   ",
            content_layer=ContentLayer.BODY,
        )
        doc.add_text(label=DocItemLabel.SECTION_HEADER, text="Actual Heading")

        client = self._make_client(tmp_path)
        result = client._extract_structural_title(doc)
        assert result == "Actual Heading"


# =========================================================================
# _resolve_title
# =========================================================================


class TestResolveTitle:
    def _make_client(self, tmp_path, auto_title=True):
        config = AppConfig(processing=ProcessingConfig(auto_title=auto_title))
        return HaikuRAG(tmp_path / "test.lancedb", config=config, create=True)

    @pytest.mark.asyncio
    async def test_explicit_title_always_wins(self, tmp_path):
        """Caller-supplied title is never overridden."""
        doc = DoclingDocument(name="test")
        doc.add_text(label=DocItemLabel.TITLE, text="Structural Title")

        client = self._make_client(tmp_path)
        result = await client._resolve_title("My Explicit Title", doc, "some content")
        assert result == "My Explicit Title"

    @pytest.mark.asyncio
    async def test_auto_title_disabled_returns_none(self, tmp_path):
        """When auto_title is False, returns None (no title generation)."""
        doc = DoclingDocument(name="test")
        doc.add_text(label=DocItemLabel.TITLE, text="Structural Title")

        client = self._make_client(tmp_path, auto_title=False)
        result = await client._resolve_title(None, doc, "some content")
        assert result is None

    @pytest.mark.asyncio
    async def test_structural_title_extracted(self, tmp_path):
        """Structural title is extracted when auto_title is enabled."""
        doc = DoclingDocument(name="test")
        doc.add_text(label=DocItemLabel.TITLE, text="Auto Extracted Title")

        client = self._make_client(tmp_path)
        result = await client._resolve_title(None, doc, "some content")
        assert result == "Auto Extracted Title"

    @pytest.mark.asyncio
    async def test_no_structural_title_no_llm_returns_none(self, tmp_path):
        """Returns None when no structural title and no LLM available."""
        doc = DoclingDocument(name="test")
        doc.add_text(label=DocItemLabel.PARAGRAPH, text="Just text")

        client = self._make_client(tmp_path)
        result = await client._resolve_title(None, doc, "some content")
        # LLM call will fail without allow_model_requests, so we get None
        assert result is None


# =========================================================================
# Integration: create_document with auto_title
# =========================================================================


class TestCreateDocumentAutoTitle:
    @pytest.mark.asyncio
    async def test_auto_title_from_structural(self, temp_db_path):
        """create_document with auto_title=True extracts title from docling."""
        config = AppConfig(processing=ProcessingConfig(auto_title=True))
        async with HaikuRAG(temp_db_path, config=config, create=True) as client:
            doc = await client.create_document(
                "# My Document\n\nSome content here.", uri="test://auto-title"
            )
            assert doc.title == "My Document"

    @pytest.mark.asyncio
    async def test_auto_title_disabled(self, temp_db_path):
        """create_document with auto_title=False leaves title as None."""
        config = AppConfig(processing=ProcessingConfig(auto_title=False))
        async with HaikuRAG(temp_db_path, config=config, create=True) as client:
            doc = await client.create_document(
                "# My Document\n\nSome content here.", uri="test://no-auto-title"
            )
            assert doc.title is None

    @pytest.mark.asyncio
    async def test_explicit_title_not_overridden(self, temp_db_path):
        """Explicit title is never overridden by auto-generation."""
        config = AppConfig(processing=ProcessingConfig(auto_title=True))
        async with HaikuRAG(temp_db_path, config=config, create=True) as client:
            doc = await client.create_document(
                "# Auto Title\n\nSome content here.",
                uri="test://explicit-title",
                title="My Explicit Title",
            )
            assert doc.title == "My Explicit Title"


# =========================================================================
# Integration: import_document with auto_title
# =========================================================================


class TestImportDocumentAutoTitle:
    @pytest.mark.asyncio
    async def test_auto_title_from_structural(self, temp_db_path):
        """import_document with auto_title=True extracts title from docling."""
        config = AppConfig(processing=ProcessingConfig(auto_title=True))
        async with HaikuRAG(temp_db_path, config=config, create=True) as client:
            docling_doc = await client.convert("# Imported Doc\n\nContent here.")
            chunks = await client.chunk(docling_doc)
            doc = await client.import_document(
                docling_doc, chunks, uri="test://import-auto-title"
            )
            assert doc.title == "Imported Doc"

    @pytest.mark.asyncio
    async def test_explicit_title_preserved(self, temp_db_path):
        """import_document explicit title is not overridden."""
        config = AppConfig(processing=ProcessingConfig(auto_title=True))
        async with HaikuRAG(temp_db_path, config=config, create=True) as client:
            docling_doc = await client.convert("# Auto Title\n\nContent here.")
            chunks = await client.chunk(docling_doc)
            doc = await client.import_document(
                docling_doc,
                chunks,
                uri="test://import-explicit",
                title="Keep This Title",
            )
            assert doc.title == "Keep This Title"
