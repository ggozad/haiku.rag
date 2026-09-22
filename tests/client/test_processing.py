import logging
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.client.processing import _warn_if_descriptions_missing, convert
from haiku.rag.config import AppConfig
from haiku.rag.store.models.chunk import Chunk
from tests.conftest import capture_logs


def _doc_with_pictures(*, with_descriptions: bool):
    """Build a tiny DoclingDocument carrying one PictureItem.

    When ``with_descriptions=True`` the picture's ``meta.description.text``
    is populated, simulating a successful VLM call. Otherwise it's left
    empty, simulating a silent VLM failure.
    """
    from docling_core.types.doc.document import (
        DescriptionMetaField,
        DoclingDocument,
        PictureItem,
        PictureMeta,
    )
    from docling_core.types.doc.labels import DocItemLabel

    doc = DoclingDocument(name="test")
    pic = PictureItem(
        self_ref="#/pictures/0",
        label=DocItemLabel.PICTURE,
    )
    if with_descriptions:
        pic.meta = PictureMeta(
            description=DescriptionMetaField(text="A red square."),
        )
    doc.pictures.append(pic)
    return doc


def _doc_without_pictures():
    from docling_core.types.doc.document import DoclingDocument

    return DoclingDocument(name="test")


@pytest.fixture
def caplog_warnings():
    """Capture WARNING-level records from the processing logger."""
    from haiku.rag.client.processing import logger as proc_logger

    with capture_logs(proc_logger, logging.WARNING) as records:
        yield records


def test_no_warning_when_picture_description_disabled(caplog_warnings):
    """Default config has pictures='image' — even a document full of
    pictures with no descriptions should not warn."""
    config = AppConfig()
    assert config.processing.pictures == "image"
    doc = _doc_with_pictures(with_descriptions=False)

    _warn_if_descriptions_missing(config, doc, "fake.pdf")

    assert caplog_warnings == []


def test_no_warning_when_doc_has_no_pictures(caplog_warnings):
    """A picture-less document under enabled=True shouldn't warn — there
    was simply nothing to describe."""
    config = AppConfig()
    config.processing.pictures = "description"
    doc = _doc_without_pictures()

    _warn_if_descriptions_missing(config, doc, "no-pictures.txt")

    assert caplog_warnings == []


def test_warns_when_pictures_present_but_no_descriptions(caplog_warnings):
    """VLM was requested via ``processing.pictures='description'``, the
    doc has pictures, but the converter returned zero descriptions
    (docling-serve swallows VLM errors). Warn loudly so the user can fix
    their VLM config before a long ingest."""
    config = AppConfig()
    config.processing.pictures = "description"
    config.processing.conversion_options.picture_description.model.name = "qwen3.6"
    config.processing.conversion_options.picture_description.model.base_url = (
        "http://host.docker.internal:11434"
    )
    doc = _doc_with_pictures(with_descriptions=False)

    _warn_if_descriptions_missing(config, doc, "doclaynet.pdf")

    assert len(caplog_warnings) == 1
    msg = caplog_warnings[0].getMessage()
    assert "doclaynet.pdf" in msg
    assert "1 pictures" in msg
    assert "0 described" in msg
    assert "qwen3.6" in msg
    assert "host.docker.internal" in msg


def test_no_warning_when_at_least_one_description_came_back(caplog_warnings):
    """Partial coverage (some pictures described, some not) is acceptable
    and does not warn — the user might have area-threshold filtering or
    classification gating."""
    config = AppConfig()
    config.processing.pictures = "description"
    doc = _doc_with_pictures(with_descriptions=True)

    _warn_if_descriptions_missing(config, doc, "fine.pdf")

    assert caplog_warnings == []


async def test_convert_emits_warning_via_chokepoint(
    monkeypatch, tmp_path, caplog_warnings
):
    """End-to-end: ``convert(...)`` runs the description-missing check
    after the converter returns, so a VLM error swallowed inside
    docling-serve still surfaces as a warning at the haiku.rag layer
    regardless of which converter (local vs serve) ran."""
    from haiku.rag.converters.base import DocumentConverter

    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    class StubConverter(DocumentConverter):
        @property
        def supported_extensions(self) -> list[str]:
            return [".pdf"]

        async def convert_file(self, path: Path, source_uri: str | None = None):
            return _doc_with_pictures(with_descriptions=False)

        async def convert_text(
            self,
            text: str,
            name: str = "content.md",
            format: str = "md",
            source_uri: str | None = None,
        ):
            return _doc_without_pictures()

    monkeypatch.setattr(
        "haiku.rag.client.processing.get_converter", lambda config: StubConverter()
    )

    config = AppConfig()
    config.processing.pictures = "description"

    await convert(config, pdf)

    assert any(
        "0 described" in r.getMessage() and "fake.pdf" in r.getMessage()
        for r in caplog_warnings
    )


async def test_convert_text_path_also_warns(monkeypatch, caplog_warnings):
    """Raw text input (HTML, markdown) can still produce pictures via
    docling, so the description-missing check must run on the
    convert_text branch too — otherwise an HTML-with-images source
    would never trigger the warning even when picture_description is
    enabled and the VLM didn't actually run."""
    from haiku.rag.converters.base import DocumentConverter

    class StubConverter(DocumentConverter):
        @property
        def supported_extensions(self) -> list[str]:
            return [".html"]

        async def convert_file(self, path: Path, source_uri: str | None = None):
            return _doc_without_pictures()

        async def convert_text(
            self,
            text: str,
            name: str = "content.md",
            format: str = "md",
            source_uri: str | None = None,
        ):
            return _doc_with_pictures(with_descriptions=False)

    monkeypatch.setattr(
        "haiku.rag.client.processing.get_converter", lambda config: StubConverter()
    )

    config = AppConfig()
    config.processing.pictures = "description"

    # No URL scheme, no Path → drops into the convert_text branch.
    await convert(config, "<html><img src='...'/></html>")

    assert any("0 described" in r.getMessage() for r in caplog_warnings)


def test_merge_picture_chunks_no_pictures_returns_text_chunks():
    """When there are no picture chunks, _merge_picture_chunks returns
    text chunks with order set."""
    from haiku.rag.client.processing import _merge_picture_chunks
    from haiku.rag.store.models.chunk import Chunk

    doc = _doc_without_pictures()
    text_chunks = [Chunk(content="a"), Chunk(content="b")]

    result = _merge_picture_chunks(doc, text_chunks, None, None, 0)

    assert result is text_chunks
    assert [c.order for c in result] == [0, 1]


async def test_convert_dispatches_large_pdfs_through_split_and_merge(
    tmp_path, monkeypatch
):
    """With split_pages configured, PDF conversion routes through the
    split-and-merge helper rather than the converter directly."""
    from docling_core.types.doc.document import DoclingDocument

    config = AppConfig()
    config.processing.split_pages = 2

    pdf = tmp_path / "big.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    called: dict = {}

    async def fake_split(converter, path, uri, slice_size):
        called["slice_size"] = slice_size
        called["path"] = path
        return DoclingDocument(name="merged")

    monkeypatch.setattr(
        "haiku.rag.converters.pdf_split.convert_pdf_with_splitting", fake_split
    )

    doc = await convert(config, pdf)

    assert doc.name == "merged"
    assert called["slice_size"] == 2
    assert called["path"] == pdf


def _write_unsupported(directory):
    target = directory / "thing.sqlite3"
    target.write_bytes(b"binary")
    return target.as_uri()


@pytest.mark.parametrize(
    "make_source,match",
    [
        (lambda d: (d / "missing.md").as_uri(), "File does not exist"),
        (_write_unsupported, "Unsupported file extension"),
    ],
    ids=["missing_file", "unsupported_extension"],
)
async def test_convert_rejects_bad_file_uris(tmp_path, make_source, match):
    from haiku.rag.client.exceptions import UnsupportedSourceError

    with pytest.raises(UnsupportedSourceError, match=match):
        await convert(AppConfig(), make_source(tmp_path))


async def test_convert_percent_encoded_file_uri(tmp_path):
    """`Path.as_uri()` encodes brackets and spaces; convert must decode them."""
    target = tmp_path / "a[b] c.md"
    target.write_text("# Heading")

    doc = await convert(AppConfig(), target.as_uri())

    assert "Heading" in doc.export_to_markdown()


@pytest.mark.vcr()
async def test_client_convert_text(temp_db_path):
    from docling_core.types.doc.document import DoclingDocument

    async with HaikuRAG(temp_db_path, create=True) as client:
        text = "This is some test content for conversion."
        docling_doc = await client.convert(text)

        assert isinstance(docling_doc, DoclingDocument)
        markdown = docling_doc.export_to_markdown()
        assert "test content" in markdown


@pytest.mark.vcr()
async def test_client_convert_file(temp_db_path):
    from docling_core.types.doc.document import DoclingDocument

    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test.txt"
            temp_path.write_text("File content for conversion test.")

            docling_doc = await client.convert(temp_path)

            assert isinstance(docling_doc, DoclingDocument)
            markdown = docling_doc.export_to_markdown()
            assert "File content" in markdown


@pytest.mark.vcr()
async def test_client_convert_file_not_found(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        with pytest.raises(ValueError, match="File does not exist"):
            await client.convert(Path("/nonexistent/path/file.txt"))


async def test_client_convert_from_url(temp_db_path):
    from docling_core.types.doc.document import DoclingDocument

    async with HaikuRAG(temp_db_path, create=True) as client:
        mock_response = AsyncMock()
        mock_response.content = (
            b"<html><body><p>URL convert path content.</p></body></html>"
        )
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = AsyncMock()

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            docling_doc = await client.convert("https://example.com/page.html")

        assert isinstance(docling_doc, DoclingDocument)
        markdown = docling_doc.export_to_markdown()
        assert "URL convert path content" in markdown


async def test_client_convert_from_url_unsupported_content_type(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        mock_response = AsyncMock()
        mock_response.content = b"\x00\x01\x02binary"
        mock_response.headers = {"content-type": "application/octet-stream"}
        mock_response.raise_for_status = AsyncMock()

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            with pytest.raises(ValueError, match="Unsupported content type"):
                await client.convert("https://example.com/blob.bin")


@pytest.mark.vcr()
async def test_client_convert_unsupported_extension(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test.xyz"
            temp_path.write_text("content")

            with pytest.raises(ValueError, match="Unsupported file extension"):
                await client.convert(temp_path)


@pytest.mark.vcr()
async def test_client_convert_file_uri(temp_db_path):
    from docling_core.types.doc.document import DoclingDocument

    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test.txt"
            temp_path.write_text("URI file content.")
            file_uri = temp_path.as_uri()

            docling_doc = await client.convert(file_uri)

            assert isinstance(docling_doc, DoclingDocument)
            markdown = docling_doc.export_to_markdown()
            assert "URI file content" in markdown


@pytest.mark.vcr()
async def test_client_chunk_basic(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        docling_doc = await client.convert("This is test content for chunking.")
        chunks = await client.chunk(docling_doc)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.content for c in chunks)
        assert all(c.embedding is None for c in chunks)
        assert all(c.document_id is None for c in chunks)


@pytest.mark.vcr()
async def test_client_chunk_preserves_metadata(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        markdown = """# Chapter 1

This is the first paragraph.

## Section 1.1

This is a subsection.
"""
        docling_doc = await client.convert(markdown)
        chunks = await client.chunk(docling_doc)

        assert len(chunks) > 0

        has_metadata = False
        for chunk in chunks:
            meta = chunk.get_chunk_metadata()
            if meta.doc_item_refs or meta.headings:
                has_metadata = True
                break

        assert has_metadata, "Chunks should have structured metadata"


@pytest.mark.vcr()
async def test_client_chunk_empty_document(temp_db_path):
    from docling_core.types.doc.document import DoclingDocument

    async with HaikuRAG(temp_db_path, create=True) as client:
        empty_doc = DoclingDocument(name="empty")

        chunks = await client.chunk(empty_doc)

        assert isinstance(chunks, list)
        assert len(chunks) == 0


@pytest.mark.vcr()
async def test_import_document_embeds_chunks_without_embeddings(temp_db_path):
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel

    async with HaikuRAG(temp_db_path, create=True) as client:
        docling_doc = DoclingDocument(name="test")
        docling_doc.add_text(
            label=DocItemLabel.TEXT, text="Document with unembedded chunks"
        )

        chunks = [
            Chunk(content="First chunk without embedding", order=0),
            Chunk(content="Second chunk without embedding", order=1),
        ]

        doc = await client.import_document(
            docling_document=docling_doc,
            chunks=chunks,
        )
        assert doc.id is not None

        stored_chunks = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(stored_chunks) == 2

        results = await client.search("First chunk", search_type="vector")
        assert len(results) > 0
        assert results[0].content == "First chunk without embedding"


@pytest.mark.vcr()
async def test_update_document_embeds_chunks_without_embeddings(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content="Initial content")
        assert doc.id is not None

        new_chunks = [
            Chunk(content="Updated chunk without embedding", order=0),
        ]
        await client.update_document(
            document_id=doc.id,
            content="Updated content",
            chunks=new_chunks,
        )

        stored_chunks = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(stored_chunks) == 1
        assert stored_chunks[0].content == "Updated chunk without embedding"

        results = await client.search("Updated chunk", search_type="vector")
        assert len(results) > 0
        assert results[0].content == "Updated chunk without embedding"


@pytest.mark.vcr()
async def test_client_create_document_with_html_format(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        html_content = """
        <h1>Main Title</h1>
        <p>Introduction paragraph.</p>
        <h2>Section Header</h2>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
        """

        doc = await client.create_document(
            content=html_content,
            uri="test://html-doc",
            format="html",
        )

        assert doc.id is not None
        assert doc.docling_document is not None

        docling_doc = doc.get_docling_document()
        assert docling_doc is not None

        items = list(docling_doc.iterate_items())
        labels = [str(getattr(item, "label", "")) for item, _ in items]

        assert "title" in labels or "section_header" in labels
        assert "list_item" in labels


@pytest.mark.vcr()
async def test_client_convert_with_html_format(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        html_content = "<h1>Title</h1><p>Text</p>"

        docling_doc = await client.convert(html_content, format="html")

        items = list(docling_doc.iterate_items())
        labels = [str(getattr(item, "label", "")) for item, _ in items]

        assert "title" in labels
