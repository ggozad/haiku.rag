"""Tests for document converters."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from docling_core.types.doc.document import DoclingDocument

from haiku.rag.config import AppConfig
from haiku.rag.converters import get_converter
from haiku.rag.converters.docling_local import DoclingLocalConverter
from haiku.rag.converters.docling_serve import DoclingServeConverter
from haiku.rag.converters.text_utils import TextFileHandler


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent / "cassettes" / "test_converters")


def create_mock_docling_document(name: str = "test") -> dict:
    """Create a minimal valid DoclingDocument JSON structure for mocking."""
    return {
        "schema_name": "DoclingDocument",
        "version": "1.10.0",
        "name": name,
        "origin": {
            "mimetype": "text/markdown",
            "binary_hash": 12345,
            "filename": f"{name}.md",
        },
        "furniture": {
            "self_ref": "#/furniture",
            "parent": None,
            "children": [],
            "content_layer": "furniture",
            "name": "_root_",
            "label": "unspecified",
        },
        "body": {
            "self_ref": "#/body",
            "parent": None,
            "children": [],
            "content_layer": "body",
            "name": "_root_",
            "label": "unspecified",
        },
        "groups": [],
        "texts": [],
        "pictures": [],
        "tables": [],
    }


def create_async_workflow_zip_mocks(
    doc_json: dict,
    artifacts: dict[str, bytes] | None = None,
    task_id: str = "test-task-zip",
) -> tuple[Mock, Mock, Mock]:
    """Mock responses for the ``target_type=zip`` async workflow.

    Builds a real zip in memory containing the document JSON at the archive
    root and each ``artifacts[name] = bytes`` entry under ``artifacts/<name>``.
    The result response exposes the zip via ``.content`` (raw bytes) so the
    converter's zip-parsing path is exercised end-to-end.
    """
    import io
    import json as _json
    import zipfile

    submit_response = Mock()
    submit_response.status_code = 200
    submit_response.json.return_value = {"task_id": task_id, "task_status": "pending"}
    submit_response.raise_for_status = Mock()

    poll_response = Mock()
    poll_response.status_code = 200
    poll_response.json.return_value = {"task_id": task_id, "task_status": "success"}
    poll_response.raise_for_status = Mock()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        zf.writestr(f"{doc_json.get('name', 'document')}.json", _json.dumps(doc_json))
        for filename, blob in (artifacts or {}).items():
            zf.writestr(f"artifacts/{filename}", blob)

    result_response = Mock()
    result_response.status_code = 200
    result_response.content = buf.getvalue()
    result_response.raise_for_status = Mock()

    return submit_response, poll_response, result_response


@pytest.mark.asyncio
async def test_parse_zip_runs_off_event_loop_thread():
    """_parse_zip_to_docling does zip decompress, per-image base64 re-encoding,
    and DoclingDocument.model_validate — all synchronous and CPU-heavy (full-
    resolution page rasters when generate_page_images is on). It must run off
    the event-loop thread, or it stalls every other worker's coroutine. Capture
    the thread it runs on and assert it is not the main thread."""
    import threading

    config = AppConfig()
    config.processing.converter = "docling-serve"
    converter = get_converter(config)
    assert isinstance(converter, DoclingServeConverter)

    converter.client.submit_and_poll_zip = AsyncMock(return_value=b"zip-bytes")

    called_from: list[threading.Thread] = []

    def spy(zip_bytes, name):
        called_from.append(threading.current_thread())
        return Mock()

    converter._parse_zip_to_docling = spy  # type: ignore[method-assign]

    files = {"files": ("doc.pdf", b"pdf", "application/octet-stream")}
    await converter._make_request(files, "doc.pdf")

    assert called_from, "_parse_zip_to_docling was never called"
    assert called_from[0] is not threading.main_thread(), (
        "_parse_zip_to_docling ran on the event-loop thread; it must be "
        "dispatched via asyncio.to_thread"
    )


class TestTextFileHandler:
    """Tests for TextFileHandler utility class."""

    def test_text_extensions_defined(self):
        """Test that text extensions list is defined."""
        assert len(TextFileHandler.text_extensions) > 0
        assert ".py" in TextFileHandler.text_extensions
        assert ".js" in TextFileHandler.text_extensions
        assert ".txt" in TextFileHandler.text_extensions

    def test_plantuml_extensions_supported(self):
        """Test that PlantUML extensions are in text_extensions."""
        assert ".puml" in TextFileHandler.text_extensions
        assert ".plantuml" in TextFileHandler.text_extensions
        assert ".pu" in TextFileHandler.text_extensions

    def test_code_markdown_identifiers(self):
        """Test code language identifiers mapping."""
        assert TextFileHandler.code_markdown_identifier[".py"] == "python"
        assert TextFileHandler.code_markdown_identifier[".js"] == "javascript"
        assert TextFileHandler.code_markdown_identifier[".ts"] == "typescript"

    def test_plantuml_markdown_identifiers(self):
        """Test PlantUML language identifiers mapping."""
        assert TextFileHandler.code_markdown_identifier[".puml"] == "plantuml"
        assert TextFileHandler.code_markdown_identifier[".plantuml"] == "plantuml"
        assert TextFileHandler.code_markdown_identifier[".pu"] == "plantuml"

    def test_prepare_text_content_with_code(self):
        """Test that code files are wrapped in markdown code blocks."""
        code = "def hello():\n    pass"
        result = TextFileHandler.prepare_text_content(code, ".py")
        assert result.startswith("```python\n")
        assert result.endswith("\n```")
        assert "def hello():" in result

    def test_prepare_text_content_with_plantuml(self):
        """Test that PlantUML files are wrapped in plantuml code blocks."""
        puml = "@startuml\nAlice -> Bob: Hello\n@enduml"
        result = TextFileHandler.prepare_text_content(puml, ".puml")
        assert result.startswith("```plantuml\n")
        assert result.endswith("\n```")
        assert "@startuml" in result

    def test_prepare_text_content_without_code(self):
        """Test that plain text files are not wrapped."""
        text = "Hello world"
        result = TextFileHandler.prepare_text_content(text, ".txt")
        assert result == text
        assert not result.startswith("```")


class TestConverterFactory:
    """Tests for converter factory function."""

    def test_get_docling_local_converter(self):
        """Test getting docling-local converter."""
        config = AppConfig()
        config.processing.converter = "docling-local"
        converter = get_converter(config)
        assert isinstance(converter, DoclingLocalConverter)

    def test_get_docling_serve_converter(self):
        """Test getting docling-serve converter."""
        config = AppConfig()
        config.processing.converter = "docling-serve"
        converter = get_converter(config)
        assert isinstance(converter, DoclingServeConverter)

    def test_invalid_converter_raises_error(self):
        """Test that invalid converter name raises ValueError."""
        config = AppConfig()
        config.processing.converter = "invalid-converter"
        with pytest.raises(ValueError, match="Unsupported converter provider"):
            get_converter(config)


class TestTextToDoclingWithFormat:
    """Tests for format parameter in text to DoclingDocument conversion."""

    @pytest.mark.asyncio
    async def test_html_format_preserves_structure(self):
        """Test that HTML content parsed with html format preserves document structure."""
        html_content = """
        <h1>Main Title</h1>
        <p>Introduction paragraph.</p>
        <h2>Section Header</h2>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
        """
        config = AppConfig()
        converter = DoclingLocalConverter(config)

        # With html format, should get proper structure
        doc = await converter.convert_text(
            html_content, name="content.html", format="html"
        )

        items = list(doc.iterate_items())
        labels = [str(getattr(item, "label", "")) for item, _ in items]

        assert "title" in labels or "section_header" in labels
        assert "list_item" in labels
        assert len(items) > 3

    @pytest.mark.asyncio
    async def test_md_format_is_default(self):
        """Test that md format is used by default."""
        config = AppConfig()
        converter = DoclingLocalConverter(config)

        # Plain text should work with default format
        doc = await converter.convert_text("# Heading\n\nParagraph text.")
        items = list(doc.iterate_items())

        assert len(items) >= 2

    @pytest.mark.asyncio
    async def test_html_as_md_loses_structure(self):
        """Test that HTML parsed as markdown loses semantic structure."""
        html_content = "<h1>Title</h1><p>Text</p><ul><li>Item</li></ul>"
        config = AppConfig()
        converter = DoclingLocalConverter(config)

        # With md format (default), HTML tags are treated as text
        doc = await converter.convert_text(html_content, format="md")
        items = list(doc.iterate_items())

        # Should still parse but with different structure
        # (markdown parser will interpret some HTML)
        assert len(items) >= 1

    @pytest.mark.asyncio
    async def test_invalid_format_raises_error(self):
        """Test that invalid format raises ValueError."""
        config = AppConfig()
        converter = DoclingLocalConverter(config)

        with pytest.raises(ValueError, match="Unsupported format"):
            await converter.convert_text("content", format="invalid")

    @pytest.mark.asyncio
    async def test_plain_format(self):
        """Test that format='plain' creates DoclingDocument directly."""
        config = AppConfig()
        converter = DoclingLocalConverter(config)

        plain_text = (
            "MZ Wallace is an American company which designs, manufactures "
            "and markets handbags and fashion accessories."
        )
        doc = await converter.convert_text(plain_text, format="plain")
        assert doc is not None
        exported = doc.export_to_markdown()
        assert "MZ Wallace" in exported

    @pytest.mark.asyncio
    async def test_plain_text_without_markdown_syntax_fallback(self):
        """Test that plain text without markdown syntax falls back gracefully.

        Docling's format detection fails for plain text that doesn't contain
        markdown syntax (headers, lists, etc.). The converter should fall back
        to creating a simple DoclingDocument directly.
        """
        config = AppConfig()
        converter = DoclingLocalConverter(config)

        # Plain text without any markdown syntax
        plain_text = (
            "MZ Wallace is an American company which designs, manufactures "
            "and markets handbags and fashion accessories. The company was "
            "founded in 1999 by Monica Zwirner and Lucy Wallace Eustice."
        )
        doc = await converter.convert_text(plain_text, format="md")
        assert doc is not None
        exported = doc.export_to_markdown()
        assert "MZ Wallace" in exported


class TestDoclingLocalConverter:
    """Tests for DoclingLocalConverter."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AppConfig()

    @pytest.fixture
    def converter(self, config):
        """Create DoclingLocalConverter instance."""
        return DoclingLocalConverter(config)

    @pytest.mark.asyncio
    async def test_docling_document_serialization_roundtrip(self, converter):
        """Test that DoclingDocument can be serialized and parsed back.

        This catches version mismatches between docling (which creates documents)
        and docling-core (which parses them). If their schema versions differ,
        model_validate_json() will raise a ValidationError.
        """
        doc = await converter.convert_text("# Test\n\nContent here", name="test.md")

        json_str = doc.model_dump_json()
        parsed = DoclingDocument.model_validate_json(json_str)

        assert parsed.name == doc.name
        assert parsed.version == doc.version

    def test_supported_extensions(self, converter):
        """Test that converter reports correct supported extensions."""
        extensions = converter.supported_extensions
        assert ".pdf" in extensions
        assert ".docx" in extensions
        assert ".py" in extensions
        assert ".txt" in extensions

    @pytest.mark.asyncio
    async def test_convert_text(self, converter):
        """Test converting text to DoclingDocument."""
        doc = await converter.convert_text("# Test\n\nContent here", name="test.md")
        assert isinstance(doc, DoclingDocument)
        assert doc.name == "test"

    @pytest.mark.asyncio
    async def test_convert_code_file(self, converter):
        """Test that code files are wrapped in code blocks."""
        python_code = "def hello():\n    print('Hello')"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
            f.write(python_code)
            f.flush()
            temp_path = Path(f.name)
            doc = await converter.convert_file(temp_path)
            result = doc.export_to_markdown()

            assert "```" in result
            assert "def hello():" in result

    def test_conversion_options_applied_to_local_converter(self, config):
        """Test that conversion options are applied to local docling converter."""
        config.processing.conversion_options.do_ocr = False
        config.processing.conversion_options.table_mode = "fast"
        config.processing.conversion_options.images_scale = 3.0
        config.processing.conversion_options.generate_page_images = False
        converter = DoclingLocalConverter(config)

        assert converter.config.processing.conversion_options.do_ocr is False
        assert converter.config.processing.conversion_options.table_mode == "fast"
        assert converter.config.processing.conversion_options.images_scale == 3.0
        assert (
            converter.config.processing.conversion_options.generate_page_images is False
        )

    @pytest.mark.asyncio
    async def test_convert_text_html_fetches_data_uri_image(self, config):
        """`fetch_remote_images=True` decodes inline `data:` URIs into picture
        bytes via the HTML backend. Default behavior."""
        png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        html = f'<html><body><p>before</p><img src="data:image/png;base64,{png_b64}" alt="dot"/><p>after</p></body></html>'
        converter = DoclingLocalConverter(config)

        doc = await converter.convert_text(html, format="html")

        assert doc.pictures, "HTML with <img> should yield picture items"
        pics_with_image = [p for p in doc.pictures if p.image is not None]
        assert len(pics_with_image) == len(doc.pictures), (
            "All <img> with valid data: URIs should have decoded bytes"
        )

    @pytest.mark.asyncio
    async def test_convert_text_html_no_fetch_when_disabled(self, config):
        """`fetch_remote_images=False` produces placeholder pictures with no
        bytes — even for inline `data:` URIs (docling's `fetch_images` gates
        all image decoding, not just remote fetches)."""
        png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        html = f'<html><body><img src="data:image/png;base64,{png_b64}"/></body></html>'
        config.processing.conversion_options.fetch_remote_images = False
        converter = DoclingLocalConverter(config)

        doc = await converter.convert_text(html, format="html")

        assert doc.pictures, "Picture placeholders are still emitted"
        for pic in doc.pictures:
            assert pic.image is None, (
                "fetch_remote_images=False must leave picture.image=None"
            )

    @pytest.mark.asyncio
    async def test_convert_text_md_html_block_fetches_data_uri_image(self, config):
        """Markdown with an embedded `<img>` HTML block produces picture bytes
        — proves the MarkdownBackendOptions wiring delegates to the HTML
        backend with our `fetch_images` / `enable_remote_fetch` settings.

        Note: docling's md backend does NOT fetch images from native
        `![alt](url)` syntax — only from embedded HTML blocks. That's an
        upstream limitation, not something this PR can address.
        """
        png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        md = (
            f"# Title\n\nIntro paragraph.\n\n"
            f'<img src="data:image/png;base64,{png_b64}" alt="dot"/>\n\n'
            f"Trailing paragraph.\n"
        )
        converter = DoclingLocalConverter(config)

        doc = await converter.convert_text(md, format="md")

        assert doc.pictures, "MD with <img> HTML block should yield picture items"
        pics_with_image = [p for p in doc.pictures if p.image is not None]
        assert len(pics_with_image) == len(doc.pictures)

    def test_build_format_options_covers_pdf_image_html_md_docx_pptx(self, config):
        """`_build_format_options()` registers every format we care about and
        shares the same `PdfPipelineOptions` instance across them so
        picture-description / classification / chart settings apply uniformly."""
        from docling.datamodel.base_models import InputFormat

        converter = DoclingLocalConverter(config)
        options = converter._build_format_options()

        wired = {
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.HTML,
            InputFormat.MD,
            InputFormat.DOCX,
            InputFormat.PPTX,
        }
        assert wired <= set(options.keys()), (
            f"Missing format options: {wired - set(options.keys())}"
        )

        pdf_opts = options[InputFormat.PDF].pipeline_options
        for fmt in wired:
            assert options[fmt].pipeline_options is pdf_opts, (
                f"{fmt} must share the PDF pipeline_options instance"
            )

    def test_build_format_options_propagates_fetch_remote_images(self, config):
        """HTML and Markdown FormatOptions reflect `fetch_remote_images`."""
        from docling.datamodel.backend_options import (
            HTMLBackendOptions,
            MarkdownBackendOptions,
        )
        from docling.datamodel.base_models import InputFormat

        config.processing.conversion_options.fetch_remote_images = True
        opts = DoclingLocalConverter(config)._build_format_options()
        html_bo = opts[InputFormat.HTML].backend_options
        md_bo = opts[InputFormat.MD].backend_options
        assert isinstance(html_bo, HTMLBackendOptions)
        assert html_bo.fetch_images is True
        assert html_bo.enable_remote_fetch is True
        assert isinstance(md_bo, MarkdownBackendOptions)
        assert md_bo.fetch_images is True
        assert md_bo.enable_remote_fetch is True

        config.processing.conversion_options.fetch_remote_images = False
        opts = DoclingLocalConverter(config)._build_format_options()
        html_bo = opts[InputFormat.HTML].backend_options
        md_bo = opts[InputFormat.MD].backend_options
        assert isinstance(html_bo, HTMLBackendOptions)
        assert html_bo.fetch_images is False
        assert html_bo.enable_remote_fetch is False
        assert isinstance(md_bo, MarkdownBackendOptions)
        assert md_bo.fetch_images is False
        assert md_bo.enable_remote_fetch is False

    def test_build_format_options_threads_source_uri(self, config):
        """`_build_format_options(source_uri=...)` plumbs the URI into the
        HTML and Markdown backend options so docling can resolve relative
        `<img src="/foo.jpg">` paths during URL ingest."""
        from docling.datamodel.backend_options import (
            HTMLBackendOptions,
            MarkdownBackendOptions,
        )
        from docling.datamodel.base_models import InputFormat

        converter = DoclingLocalConverter(config)
        opts = converter._build_format_options(source_uri="https://example.com/article")

        html_bo = opts[InputFormat.HTML].backend_options
        md_bo = opts[InputFormat.MD].backend_options
        assert isinstance(html_bo, HTMLBackendOptions)
        assert str(html_bo.source_uri) == "https://example.com/article"
        assert isinstance(md_bo, MarkdownBackendOptions)
        assert str(md_bo.source_uri) == "https://example.com/article"

        # No source_uri ⇒ both stay None
        opts_no_uri = converter._build_format_options()
        html_bo = opts_no_uri[InputFormat.HTML].backend_options
        md_bo = opts_no_uri[InputFormat.MD].backend_options
        assert isinstance(html_bo, HTMLBackendOptions)
        assert html_bo.source_uri is None
        assert isinstance(md_bo, MarkdownBackendOptions)
        assert md_bo.source_uri is None

    @pytest.mark.asyncio
    async def test_convert_text_html_mixed_img_sources(self, config, monkeypatch):
        """End-to-end: HTML with a mix of remote http, data:, broken http, and
        file:// `<img>` sources. Remote and data: URIs land as picture bytes;
        broken URLs and file:// stay as placeholder pictures. Models the
        wix-style ingest where most images are remote URLs with a handful of
        broken or local-only references mixed in."""
        import base64

        from docling.backend import html_backend as html_backend_module

        canned_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        good_url = "https://cdn.example.com/static/cat.png"
        broken_url = "https://cdn.example.com/missing.png"

        def fake_load_image_data(self, src_loc: str):
            if src_loc == good_url:
                return canned_png
            if src_loc == broken_url:
                # Simulate a 404: docling's _create_image_ref swallows HTTPError
                # via its except clause and returns None for the picture.
                import requests

                resp = requests.Response()
                resp.status_code = 404
                raise requests.HTTPError(response=resp)
            return None  # data: and file:// fall back to docling's own path

        # Wrap rather than replace so data: URIs still decode through the real
        # `_load_image_data`. Only intercept when src is one of our test URLs.
        original = html_backend_module.HTMLDocumentBackend._load_image_data

        def wrapped(self, src_loc: str):
            if src_loc in (good_url, broken_url):
                return fake_load_image_data(self, src_loc)
            return original(self, src_loc)

        monkeypatch.setattr(
            html_backend_module.HTMLDocumentBackend, "_load_image_data", wrapped
        )

        png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        html = (
            "<html><body>"
            f'<img src="{good_url}" alt="cat"/>'
            f'<img src="data:image/png;base64,{png_b64}" alt="dot"/>'
            f'<img src="{broken_url}" alt="missing"/>'
            '<img src="file:///etc/passwd" alt="local"/>'
            "</body></html>"
        )
        converter = DoclingLocalConverter(config)

        doc = await converter.convert_text(html, format="html")

        assert len(doc.pictures) == 4, f"expected 4 pictures, got {len(doc.pictures)}"
        with_bytes = sum(1 for p in doc.pictures if p.image is not None)
        without_bytes = sum(1 for p in doc.pictures if p.image is None)
        assert with_bytes == 2, (
            f"good remote + data: should produce bytes; got {with_bytes}"
        )
        assert without_bytes == 2, (
            f"broken http + file:// should stay placeholder; got {without_bytes}"
        )

    def test_image_format_shares_pdf_pipeline_options(self, config):
        """IMAGE FormatOption shares the same PdfPipelineOptions instance as
        PDF. Without this, `do_ocr` / `processing.pictures='description'` / etc.
        silently no-op when ingesting raw `.png` / `.jpg` files (which run
        through StandardPdfPipeline). End-to-end image conversion is covered
        by the PDF picture test — both paths feed the same pipeline class."""
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        config.processing.conversion_options.do_ocr = False
        config.processing.conversion_options.images_scale = 3.5
        fmt_opts = DoclingLocalConverter(config)._build_format_options()

        pdf_pipe = fmt_opts[InputFormat.PDF].pipeline_options
        image_pipe = fmt_opts[InputFormat.IMAGE].pipeline_options
        assert image_pipe is pdf_pipe
        assert isinstance(pdf_pipe, PdfPipelineOptions)
        assert pdf_pipe.do_ocr is False
        assert pdf_pipe.images_scale == 3.5

    @pytest.mark.asyncio
    async def test_convert_text_html_source_uri_resolves_relative_img(
        self, config, monkeypatch
    ):
        """`convert_text(..., source_uri=...)` lets docling resolve a relative
        `<img src="/path">` against the source URL. We patch the docling HTML
        backend's `_load_image_data` to capture the resolved absolute URL
        instead of doing a real network fetch."""
        captured: list[str] = []

        def fake_load_image_data(self, src_loc: str):
            captured.append(src_loc)
            return None  # docling treats as a fetch failure → placeholder

        from docling.backend import html_backend as html_backend_module

        monkeypatch.setattr(
            html_backend_module.HTMLDocumentBackend,
            "_load_image_data",
            fake_load_image_data,
        )

        html = '<html><body><img src="/static/cat.jpg"/></body></html>'
        converter = DoclingLocalConverter(config)

        await converter.convert_text(
            html, format="html", source_uri="https://example.com/article"
        )

        assert captured, "_load_image_data should have been invoked"
        assert captured[0] == "https://example.com/static/cat.jpg", (
            f"Expected absolute URL resolved via source_uri, got {captured[0]!r}"
        )

    @pytest.mark.asyncio
    async def test_convert_pdf_with_picture_images(
        self, config, doclaynet_first_page_pdf
    ):
        """Picture bytes are produced by the local converter for PDFs that
        contain figures."""
        pdf_path = doclaynet_first_page_pdf
        converter = DoclingLocalConverter(config)

        doc = await converter.convert_file(pdf_path)
        assert isinstance(doc, DoclingDocument)

        pictures_with_images = [p for p in doc.pictures if p.image is not None]
        if doc.pictures:
            assert len(pictures_with_images) > 0, (
                "Pictures should carry image data after conversion"
            )

    @pytest.mark.asyncio
    async def test_split_and_merge_matches_single_pass(self, config):
        """Real-PDF integration test for split_pages: convert the full
        9-page DocLayNet arXiv paper single-pass, then again via
        ``convert_pdf_with_splitting`` with slice_size=1 (one slice per
        page), and assert the merged result is equivalent to single-pass
        on totals + per-page-number coverage + self_ref uniqueness +
        markdown export.

        Slow — runs docling-local 10 times against a multi-page PDF. The
        contract this pins is the highest-risk one: that splitting at the
        byte level and merging via DoclingDocument.concatenate produces a
        document semantically indistinguishable from a single-pass convert.
        """
        from haiku.rag.converters.pdf_split import convert_pdf_with_splitting

        pdf_path = Path("tests/data/doclaynet.pdf")
        config.processing.conversion_options.do_ocr = False
        converter = DoclingLocalConverter(config)

        baseline = await converter.convert_file(pdf_path)
        merged = await convert_pdf_with_splitting(
            converter, pdf_path, source_uri=None, slice_size=1
        )

        # Same totals across every list the consumer cares about.
        assert len(merged.texts) == len(baseline.texts)
        assert len(merged.pictures) == len(baseline.pictures)
        assert len(merged.tables) == len(baseline.tables)
        assert sorted(merged.pages.keys()) == sorted(baseline.pages.keys())

        # Page numbers cover the same range — this is the key thing
        # concatenate handles via its internal page_delta.
        def _page_nos(doc):
            return {p.page_no for t in doc.texts for p in t.prov}

        assert _page_nos(merged) == _page_nos(baseline)

        # self_refs unique across the merged doc — concatenate re-indexes
        # them per-slice, so a duplicate here is a real merger bug.
        merged_refs = [t.self_ref for t in merged.texts]
        assert len(set(merged_refs)) == len(merged_refs)

        # Strongest assertion: rendered markdown matches byte-for-byte.
        # If this fails, the split/merge introduced ordering or content
        # drift the count-based asserts above didn't catch.
        assert merged.export_to_markdown() == baseline.export_to_markdown()

    @pytest.mark.asyncio
    async def test_convert_pdf_without_page_images(
        self, config, doclaynet_first_page_pdf
    ):
        """Test PDF conversion excludes page images when disabled."""
        pdf_path = doclaynet_first_page_pdf
        config.processing.conversion_options.generate_page_images = False
        converter = DoclingLocalConverter(config)

        doc = await converter.convert_file(pdf_path)
        assert isinstance(doc, DoclingDocument)

        # Check that pages don't have image data
        for page in doc.pages.values():
            assert page.image is None, (
                "Pages should not have image data when generate_page_images=False"
            )

    @pytest.mark.asyncio
    async def test_convert_pdf_with_page_images(self, config, doclaynet_first_page_pdf):
        """Test PDF conversion includes page images when enabled."""
        pdf_path = doclaynet_first_page_pdf
        config.processing.conversion_options.generate_page_images = True
        converter = DoclingLocalConverter(config)

        doc = await converter.convert_file(pdf_path)
        assert isinstance(doc, DoclingDocument)

        # Check that pages have image data
        pages_with_images = [p for p in doc.pages.values() if p.image is not None]
        assert len(pages_with_images) > 0, (
            "Pages should have image data when generate_page_images=True"
        )

    def test_get_vlm_api_url_with_ollama(self, config):
        """Test VLM API URL construction for Ollama provider."""
        converter = DoclingLocalConverter(config)
        from haiku.rag.config.models import ModelConfig

        model = ModelConfig(provider="ollama", name="ministral-3")
        url = converter._get_vlm_api_url(model)
        assert url == "http://localhost:11434/v1/chat/completions"

    def test_get_vlm_api_url_with_custom_base_url(self, config):
        """Test VLM API URL construction with custom base_url."""
        converter = DoclingLocalConverter(config)
        from haiku.rag.config.models import ModelConfig

        model = ModelConfig(
            provider="openai", name="gpt-4-vision", base_url="http://my-vllm:8000"
        )
        url = converter._get_vlm_api_url(model)
        assert url == "http://my-vllm:8000/v1/chat/completions"

    def test_get_vlm_api_url_with_openai(self, config):
        """Test VLM API URL construction for OpenAI provider."""
        converter = DoclingLocalConverter(config)
        from haiku.rag.config.models import ModelConfig

        model = ModelConfig(provider="openai", name="gpt-4-vision")
        url = converter._get_vlm_api_url(model)
        assert url == "https://api.openai.com/v1/chat/completions"

    def test_get_vlm_api_url_unsupported_provider(self, config):
        """Test VLM API URL construction raises error for unsupported provider."""
        converter = DoclingLocalConverter(config)
        from haiku.rag.config.models import ModelConfig

        model = ModelConfig(provider="unsupported", name="test")
        with pytest.raises(ValueError, match="Unsupported VLM provider"):
            converter._get_vlm_api_url(model)

    def test_ocr_engine_config_applied(self, config):
        """Test that ocr_engine config is stored correctly."""
        config.processing.conversion_options.ocr_engine = "rapidocr"
        converter = DoclingLocalConverter(config)
        assert converter.config.processing.conversion_options.ocr_engine == "rapidocr"

    def test_get_ocr_options_auto(self, config):
        """Test that _get_ocr_options returns OcrAutoOptions for 'auto'."""
        from docling.datamodel.pipeline_options import OcrAutoOptions

        config.processing.conversion_options.ocr_engine = "auto"
        converter = DoclingLocalConverter(config)
        opts = converter._get_ocr_options(config.processing.conversion_options)
        assert isinstance(opts, OcrAutoOptions)

    def test_get_ocr_options_rapidocr(self, config):
        """Test that _get_ocr_options returns RapidOcrOptions for 'rapidocr'."""
        from docling.datamodel.pipeline_options import RapidOcrOptions

        config.processing.conversion_options.ocr_engine = "rapidocr"
        converter = DoclingLocalConverter(config)
        opts = converter._get_ocr_options(config.processing.conversion_options)
        assert isinstance(opts, RapidOcrOptions)

    def test_get_ocr_options_easyocr(self, config):
        """Test that _get_ocr_options returns EasyOcrOptions for 'easyocr'."""
        from docling.datamodel.pipeline_options import EasyOcrOptions

        config.processing.conversion_options.ocr_engine = "easyocr"
        converter = DoclingLocalConverter(config)
        opts = converter._get_ocr_options(config.processing.conversion_options)
        assert isinstance(opts, EasyOcrOptions)

    def test_get_ocr_options_tesseract(self, config):
        """Test that _get_ocr_options returns TesseractOcrOptions for 'tesseract'."""
        from docling.datamodel.pipeline_options import TesseractOcrOptions

        config.processing.conversion_options.ocr_engine = "tesseract"
        converter = DoclingLocalConverter(config)
        opts = converter._get_ocr_options(config.processing.conversion_options)
        assert isinstance(opts, TesseractOcrOptions)

    def test_get_ocr_options_tesserocr(self, config):
        """Test that _get_ocr_options returns TesseractCliOcrOptions for 'tesserocr'."""
        from docling.datamodel.pipeline_options import TesseractCliOcrOptions

        config.processing.conversion_options.ocr_engine = "tesserocr"
        converter = DoclingLocalConverter(config)
        opts = converter._get_ocr_options(config.processing.conversion_options)
        assert isinstance(opts, TesseractCliOcrOptions)

    def test_get_ocr_options_ocrmac(self, config):
        """Test that _get_ocr_options returns OcrMacOptions for 'ocrmac'."""
        from docling.datamodel.pipeline_options import OcrMacOptions

        config.processing.conversion_options.ocr_engine = "ocrmac"
        converter = DoclingLocalConverter(config)
        opts = converter._get_ocr_options(config.processing.conversion_options)
        assert isinstance(opts, OcrMacOptions)

    def test_get_ocr_options_passes_force_ocr_and_lang(self, config):
        """Test that _get_ocr_options passes force_ocr and ocr_lang."""
        config.processing.conversion_options.ocr_engine = "rapidocr"
        config.processing.conversion_options.force_ocr = True
        config.processing.conversion_options.ocr_lang = ["en", "de"]
        converter = DoclingLocalConverter(config)
        opts = converter._get_ocr_options(config.processing.conversion_options)
        assert opts.force_full_page_ocr is True
        assert opts.lang == ["en", "de"]

    def test_picture_description_config_defaults(self, config):
        """Test that picture description config has correct defaults."""
        pic_desc = config.processing.conversion_options.picture_description
        assert config.processing.pictures == "image"
        assert pic_desc.model.provider == "ollama"
        assert pic_desc.model.name == "ministral-3"
        assert pic_desc.timeout == 90
        assert pic_desc.max_tokens == 200
        # Default prompt is in PromptsConfig
        assert "blind user" in config.prompts.picture_description

    def test_picture_description_config_applied(self, config):
        """Test that picture description config is applied to converter."""
        config.processing.pictures = "description"
        config.processing.conversion_options.picture_description.timeout = 120
        converter = DoclingLocalConverter(config)

        assert converter.config.processing.pictures == "description"
        pic_desc = converter.config.processing.conversion_options.picture_description
        assert pic_desc.timeout == 120

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_picture_description_end_to_end(
        self, config, doclaynet_first_page_pdf
    ):
        """End-to-end test: convert PDF with VLM picture descriptions."""
        pdf_path = doclaynet_first_page_pdf

        # Disable OCR (not needed for native PDF, avoids model downloads)
        config.processing.conversion_options.do_ocr = False
        # Enable picture description with Ollama
        config.processing.pictures = "description"
        config.processing.conversion_options.picture_description.model.provider = (
            "ollama"
        )
        config.processing.conversion_options.picture_description.model.name = (
            "ministral-3"
        )

        converter = DoclingLocalConverter(config)
        doc = await converter.convert_file(pdf_path)

        # Export to markdown and check for picture descriptions
        markdown = doc.export_to_markdown()

        # The document should have pictures with descriptions
        assert doc.pictures, "Document should have pictures"

        # Check that at least one picture carries a VLM description.
        pictures_with_descriptions = []
        for pic in doc.pictures:
            if pic.meta and pic.meta.description and pic.meta.description.text:
                pictures_with_descriptions.append(pic)
                assert pic.meta.description.text in markdown, (
                    f"Picture description '{pic.meta.description.text[:50]}...' should be in markdown"
                )

        assert pictures_with_descriptions, (
            "At least one picture should have a VLM description"
        )


class TestDoclingServeConverter:
    """Tests for DoclingServeConverter (mocked)."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = AppConfig()
        config.providers.docling_serve.base_url = "http://localhost:5001"
        config.providers.docling_serve.api_key = ""
        return config

    @pytest.fixture
    def converter(self, config):
        """Create DoclingServeConverter instance."""
        return DoclingServeConverter(config)

    def test_initialization(self, converter):
        """Test converter initialization."""
        assert converter.client.base_url == "http://localhost:5001"

    def test_supported_extensions(self, converter):
        """Test that converter reports correct supported extensions."""
        extensions = converter.supported_extensions
        assert ".pdf" in extensions
        assert ".docx" in extensions
        assert ".py" in extensions
        assert ".md" in extensions

    @pytest.mark.asyncio
    async def test_convert_text_success(self, converter):
        """Test successful text conversion via docling-serve async workflow."""
        doc_json = create_mock_docling_document("test")
        submit_resp, poll_resp, result_resp = create_async_workflow_zip_mocks(doc_json)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=submit_resp)
            mock_client.get = AsyncMock(side_effect=[poll_resp, result_resp])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            doc = await converter.convert_text("# Test", name="test.md")
            assert isinstance(doc, DoclingDocument)
            assert doc.version == "1.10.0"
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_convert_text_with_api_key(self, config):
        """Test that API key is included in request headers."""
        config.providers.docling_serve.api_key = "test-key"
        converter = DoclingServeConverter(config)

        doc_json = create_mock_docling_document("test")
        submit_resp, poll_resp, result_resp = create_async_workflow_zip_mocks(doc_json)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=submit_resp)
            mock_client.get = AsyncMock(side_effect=[poll_resp, result_resp])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await converter.convert_text("# Test")

            call_kwargs = mock_client.post.call_args.kwargs
            assert "headers" in call_kwargs
            assert call_kwargs["headers"]["X-Api-Key"] == "test-key"

    @pytest.mark.asyncio
    async def test_conversion_options_passed_to_api(self, config):
        """Test that conversion options are passed to docling-serve API."""
        config.processing.conversion_options.do_ocr = False
        config.processing.conversion_options.force_ocr = True
        config.processing.conversion_options.ocr_lang = ["en", "fr"]
        config.processing.conversion_options.table_mode = "fast"
        config.processing.conversion_options.table_cell_matching = False
        config.processing.conversion_options.do_table_structure = False
        config.processing.conversion_options.images_scale = 3.0
        converter = DoclingServeConverter(config)

        doc_json = create_mock_docling_document("test")
        submit_resp, poll_resp, result_resp = create_async_workflow_zip_mocks(doc_json)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=submit_resp)
            mock_client.get = AsyncMock(side_effect=[poll_resp, result_resp])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await converter.convert_text("# Test")

            call_kwargs = mock_client.post.call_args.kwargs
            assert "data" in call_kwargs
            data = call_kwargs["data"]
            assert data["do_ocr"] == "false"
            assert data["force_ocr"] == "true"
            assert data["ocr_lang"] == ["en", "fr"]
            assert data["table_mode"] == "fast"
            assert data["table_cell_matching"] == "false"
            assert data["do_table_structure"] == "false"
            assert data["images_scale"] == "3.0"
            assert data["include_images"] == "true"
            assert data["image_export_mode"] == "referenced"
            assert data["target_type"] == "zip"

    @pytest.mark.asyncio
    async def test_ocr_engine_passed_to_api(self, config):
        """Test that ocr_engine is passed to docling-serve API."""
        config.processing.conversion_options.ocr_engine = "rapidocr"
        converter = DoclingServeConverter(config)

        doc_json = create_mock_docling_document("test")
        submit_resp, poll_resp, result_resp = create_async_workflow_zip_mocks(doc_json)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=submit_resp)
            mock_client.get = AsyncMock(side_effect=[poll_resp, result_resp])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await converter.convert_text("# Test")

            call_kwargs = mock_client.post.call_args.kwargs
            data = call_kwargs["data"]
            assert data["ocr_engine"] == "rapidocr"

    def test_parse_zip_rehydrates_picture_uri(self, config):
        """Zip path inlines artifact bytes as data: URIs on PictureItem.image.

        target_type=zip mode emits the URI as ``artifacts/<filename>`` — the
        same string we use to read the entry out of the archive.
        """
        import base64
        import io
        import json as _json
        import zipfile

        converter = DoclingServeConverter(config)

        doc_json = create_mock_docling_document("test")
        doc_json["pictures"] = [
            {
                "self_ref": "#/pictures/0",
                "parent": {"cref": "#/body"},
                "children": [],
                "content_layer": "body",
                "label": "picture",
                "prov": [],
                "captions": [],
                "references": [],
                "footnotes": [],
                "annotations": [],
                "image": {
                    "mimetype": "image/png",
                    "dpi": 144,
                    "size": {"width": 1.0, "height": 1.0},
                    "uri": "artifacts/image_000000_test.png",
                },
            }
        ]
        fake_png = b"\x89PNG\r\n\x1a\nfake-bytes-for-test"

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            zf.writestr("test.json", _json.dumps(doc_json))
            zf.writestr("artifacts/image_000000_test.png", fake_png)

        doc = converter._parse_zip_to_docling(buf.getvalue(), "test")

        assert len(doc.pictures) == 1
        image = doc.pictures[0].image
        assert image is not None
        uri = str(image.uri)
        assert uri.startswith("data:image/png;base64,")
        decoded = base64.b64decode(uri.split(",", 1)[1])
        assert decoded == fake_png

    def test_parse_zip_leaves_unknown_artifact_uri_unchanged(self, config):
        """If a PictureItem references an artifact that is not in the zip,
        the URI is left as-is (no crash, no truncation)."""
        import io
        import json as _json
        import zipfile

        converter = DoclingServeConverter(config)
        doc_json = create_mock_docling_document("test")
        doc_json["pictures"] = [
            {
                "self_ref": "#/pictures/0",
                "parent": {"cref": "#/body"},
                "children": [],
                "content_layer": "body",
                "label": "picture",
                "prov": [],
                "captions": [],
                "references": [],
                "footnotes": [],
                "annotations": [],
                "image": {
                    "mimetype": "image/png",
                    "dpi": 144,
                    "size": {"width": 1.0, "height": 1.0},
                    "uri": "artifacts/missing.png",
                },
            }
        ]

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            zf.writestr("test.json", _json.dumps(doc_json))

        doc = converter._parse_zip_to_docling(buf.getvalue(), "test")
        assert doc.pictures[0].image is not None
        assert str(doc.pictures[0].image.uri) == "artifacts/missing.png"

    @pytest.mark.asyncio
    async def test_convert_text_connection_error(self, converter):
        """Test handling of connection errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection failed")
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(httpx.ConnectError):
                await converter.convert_text("# Test")

    @pytest.mark.asyncio
    async def test_convert_text_timeout_error(self, converter):
        """Test handling of timeout errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(httpx.TimeoutException):
                await converter.convert_text("# Test")

    @pytest.mark.asyncio
    async def test_convert_text_auth_error(self, converter):
        """Auth failures surface as httpx.HTTPStatusError(401) so the
        ingester's pipeline classifier can route them to PermanentError —
        retrying a bad token is wasted work."""
        mock_response = Mock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Auth failed", request=Mock(), response=mock_response
                )
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await converter.convert_text("# Test")
            assert exc_info.value.response.status_code == 401

    @pytest.mark.asyncio
    async def test_convert_file_pdf(self, converter):
        """Test converting PDF file via docling-serve async workflow."""
        doc_json = create_mock_docling_document("test")
        submit_resp, poll_resp, result_resp = create_async_workflow_zip_mocks(doc_json)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=submit_resp)
            mock_client.get = AsyncMock(side_effect=[poll_resp, result_resp])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
                f.write(b"fake pdf content")
                f.flush()
                temp_path = Path(f.name)
                doc = await converter.convert_file(temp_path)

            assert isinstance(doc, DoclingDocument)
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_convert_file_text(self, converter):
        """Test converting text file (reads locally, sends to docling-serve)."""
        doc_json = create_mock_docling_document("test")
        submit_resp, poll_resp, result_resp = create_async_workflow_zip_mocks(doc_json)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=submit_resp)
            mock_client.get = AsyncMock(side_effect=[poll_resp, result_resp])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
                f.write("def hello():\n    pass")
                f.flush()
                temp_path = Path(f.name)
                doc = await converter.convert_file(temp_path)

            assert isinstance(doc, DoclingDocument)
            mock_client.post.assert_called_once()
            call_kwargs = mock_client.post.call_args.kwargs
            assert "files" in call_kwargs


class TestDoclingServeConverterPictureDescription:
    """Tests for DoclingServeConverter picture description support."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = AppConfig()
        config.providers.docling_serve.base_url = "http://localhost:5001"
        config.providers.docling_serve.api_key = ""
        return config

    def test_get_vlm_api_url_with_ollama(self, config):
        """Test VLM API URL construction for Ollama provider."""
        converter = DoclingServeConverter(config)
        from haiku.rag.config.models import ModelConfig

        model = ModelConfig(provider="ollama", name="ministral-3")
        url = converter._get_vlm_api_url(model)
        assert url == "http://localhost:11434/v1/chat/completions"

    def test_get_vlm_api_url_with_custom_base_url(self, config):
        """Test VLM API URL construction with custom base_url."""
        converter = DoclingServeConverter(config)
        from haiku.rag.config.models import ModelConfig

        model = ModelConfig(
            provider="openai", name="gpt-4-vision", base_url="http://my-vllm:8000"
        )
        url = converter._get_vlm_api_url(model)
        assert url == "http://my-vllm:8000/v1/chat/completions"

    def test_get_vlm_api_url_with_openai(self, config):
        """Test VLM API URL construction for OpenAI provider."""
        converter = DoclingServeConverter(config)
        from haiku.rag.config.models import ModelConfig

        model = ModelConfig(provider="openai", name="gpt-4-vision")
        url = converter._get_vlm_api_url(model)
        assert url == "https://api.openai.com/v1/chat/completions"

    def test_get_vlm_api_url_unsupported_provider(self, config):
        """Test VLM API URL construction raises error for unsupported provider."""
        converter = DoclingServeConverter(config)
        from haiku.rag.config.models import ModelConfig

        model = ModelConfig(provider="unsupported", name="test")
        with pytest.raises(ValueError, match="Unsupported VLM provider"):
            converter._get_vlm_api_url(model)

    @pytest.mark.asyncio
    async def test_picture_description_options_passed_to_api(self, config):
        """Picture-description options reach the docling-serve API when the
        VLM is enabled."""
        import json

        config.processing.pictures = "description"
        config.processing.conversion_options.picture_description.model.provider = (
            "ollama"
        )
        config.processing.conversion_options.picture_description.model.name = (
            "ministral-3"
        )
        config.processing.conversion_options.picture_description.timeout = 120
        config.processing.conversion_options.picture_description.max_tokens = 300
        config.prompts.picture_description = "Test prompt for picture description"
        converter = DoclingServeConverter(config)

        doc_json = create_mock_docling_document("test")
        submit_resp, poll_resp, result_resp = create_async_workflow_zip_mocks(doc_json)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=submit_resp)
            mock_client.get = AsyncMock(side_effect=[poll_resp, result_resp])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await converter.convert_text("# Test")

            call_kwargs = mock_client.post.call_args.kwargs
            assert "data" in call_kwargs
            data = call_kwargs["data"]

            assert data["do_picture_description"] == "true"
            assert data["include_images"] == "true"
            assert data["image_export_mode"] == "referenced"
            assert data["target_type"] == "zip"
            assert "picture_description_api" in data

            api_config = json.loads(data["picture_description_api"])
            assert api_config["url"] == "http://localhost:11434/v1/chat/completions"
            assert api_config["params"]["model"] == "ministral-3"
            assert api_config["params"]["max_completion_tokens"] == 300
            assert api_config["prompt"] == "Test prompt for picture description"
            assert api_config["timeout"] == 120

    @pytest.mark.asyncio
    async def test_picture_description_disabled_by_default(self, config):
        """Test that picture description is disabled by default."""
        converter = DoclingServeConverter(config)

        doc_json = create_mock_docling_document("test")
        submit_resp, poll_resp, result_resp = create_async_workflow_zip_mocks(doc_json)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=submit_resp)
            mock_client.get = AsyncMock(side_effect=[poll_resp, result_resp])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await converter.convert_text("# Test")

            call_kwargs = mock_client.post.call_args.kwargs
            data = call_kwargs["data"]
            assert data["do_picture_description"] == "false"
            assert "picture_description_api" not in data


async def _skip_without_ollama_model(model: str) -> None:
    """Skip when the host Ollama isn't serving `model`. docling-serve calls it
    for VLM picture descriptions, so without it the test would fail with an
    opaque error from inside the container rather than skip."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            names = [m["name"] for m in response.json().get("models", [])]
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Ollama not reachable on localhost:11434 ({exc})")
    if not any(name == model or name.startswith(f"{model}:") for name in names):
        pytest.skip(f"Ollama model '{model}' not pulled (run `ollama pull {model}`)")


class TestDoclingServeConverterIntegration:
    """Integration tests with real docling-serve recorded via VCR."""

    @pytest.fixture
    def config(self):
        """Create configuration for integration tests."""
        config = AppConfig()
        config.providers.docling_serve.base_url = "http://localhost:5001"
        return config

    @pytest.fixture
    def converter(self, config):
        """Create converter for integration tests."""
        return DoclingServeConverter(config)

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_convert_text_real_service(self, converter):
        """Test text conversion with real docling-serve."""
        doc = await converter.convert_text("# Test Document\n\nThis is a test.")
        assert isinstance(doc, DoclingDocument)

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_convert_code_file_real_service(self, converter):
        """Test code file conversion with real docling-serve."""
        code = "def test():\n    return 42"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)
        doc = await converter.convert_file(temp_path)
        temp_path.unlink()

        assert isinstance(doc, DoclingDocument)
        result = doc.export_to_markdown()
        assert "def test():" in result

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_picture_description_end_to_end(
        self, config, docling_serve_url, doclaynet_first_page_pdf
    ):
        """End-to-end test: convert PDF with VLM picture descriptions via docling-serve.

        Note: Not using VCR because this test involves polling with changing task IDs.
        """
        await _skip_without_ollama_model("ministral-3")
        config.providers.docling_serve.base_url = docling_serve_url
        pdf_path = doclaynet_first_page_pdf
        config.processing.pictures = "description"
        config.processing.conversion_options.picture_description.model.provider = (
            "ollama"
        )
        config.processing.conversion_options.picture_description.model.name = (
            "ministral-3"
        )
        # Use host.docker.internal so docling-serve in Docker can reach host's Ollama
        config.processing.conversion_options.picture_description.model.base_url = (
            "http://host.docker.internal:11434"
        )
        converter = DoclingServeConverter(config)

        doc = await converter.convert_file(pdf_path)

        assert doc.pictures, "Document should have pictures"

        pictures_with_descriptions = []
        markdown = doc.export_to_markdown()
        for pic in doc.pictures:
            if pic.meta and pic.meta.description and pic.meta.description.text:
                pictures_with_descriptions.append(pic)
                assert pic.meta.description.text in markdown, (
                    f"Picture description '{pic.meta.description.text[:50]}...' should be in markdown"
                )

        assert pictures_with_descriptions, (
            "At least one picture should have a VLM description"
        )

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_convert_pdf_without_page_images(
        self, config, doclaynet_first_page_pdf
    ):
        """Test PDF conversion excludes page images when disabled."""
        pdf_path = doclaynet_first_page_pdf
        config.processing.conversion_options.generate_page_images = False
        converter = DoclingServeConverter(config)

        doc = await converter.convert_file(pdf_path)
        assert isinstance(doc, DoclingDocument)

        # Check that pages don't have image data
        for page in doc.pages.values():
            assert page.image is None, (
                "Pages should not have image data when generate_page_images=False"
            )

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_convert_pdf_with_page_images(self, config, doclaynet_first_page_pdf):
        """Test PDF conversion includes page images when enabled."""
        pdf_path = doclaynet_first_page_pdf
        config.processing.conversion_options.generate_page_images = True
        converter = DoclingServeConverter(config)

        doc = await converter.convert_file(pdf_path)
        assert isinstance(doc, DoclingDocument)

        # Check that pages have image data
        pages_with_images = [p for p in doc.pages.values() if p.image is not None]
        assert len(pages_with_images) > 0, (
            "Pages should have image data when generate_page_images=True"
        )

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_convert_pdf_with_ocr_engine(self, config, doclaynet_first_page_pdf):
        """Test PDF conversion with explicit OCR engine selection."""
        pdf_path = doclaynet_first_page_pdf
        config.processing.conversion_options.ocr_engine = "easyocr"
        converter = DoclingServeConverter(config)

        doc = await converter.convert_file(pdf_path)
        assert isinstance(doc, DoclingDocument)
        assert len(doc.pages) > 0
        assert len(doc.export_to_markdown().strip()) > 100

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_convert_pdf_with_picture_images(
        self, config, doclaynet_first_page_pdf
    ):
        """Picture bytes are produced for PDFs that contain figures.

        docling-serve only emits picture image bytes via the
        ``image_export_mode="referenced"`` + ``target_type="zip"`` path
        (upstream issue docling-project/docling-serve#576). The converter
        always uses that path and rehydrates the bundled artifact files
        into ``data:`` URIs so the result is shape-equivalent to the local
        converter.
        """
        pdf_path = doclaynet_first_page_pdf
        converter = DoclingServeConverter(config)

        doc = await converter.convert_file(pdf_path)
        assert isinstance(doc, DoclingDocument)

        pictures_with_images = [p for p in doc.pictures if p.image is not None]
        assert doc.pictures, "doclaynet.pdf is expected to contain at least one picture"
        assert len(pictures_with_images) > 0, (
            "Pictures should carry image data after conversion"
        )
        sample = pictures_with_images[0]
        assert sample.image is not None
        assert str(sample.image.uri).startswith("data:image/"), (
            "Rehydrated picture URI should be a data: URI, not a bare artifact filename"
        )
