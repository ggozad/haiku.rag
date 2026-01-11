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
        "version": "1.8.0",
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


def create_async_workflow_mocks(
    doc_json: dict, task_id: str = "test-task-123"
) -> tuple[Mock, Mock, Mock]:
    """Create mock responses for docling-serve async workflow.

    Returns tuple of (submit_response, poll_response, result_response).
    """
    submit_response = Mock()
    submit_response.status_code = 200
    submit_response.json.return_value = {"task_id": task_id, "task_status": "pending"}
    submit_response.raise_for_status = Mock()

    poll_response = Mock()
    poll_response.status_code = 200
    poll_response.json.return_value = {"task_id": task_id, "task_status": "success"}
    poll_response.raise_for_status = Mock()

    result_response = Mock()
    result_response.status_code = 200
    result_response.json.return_value = {"document": {"json_content": doc_json}}
    result_response.raise_for_status = Mock()

    return submit_response, poll_response, result_response


class TestTextFileHandler:
    """Tests for TextFileHandler utility class."""

    def test_text_extensions_defined(self):
        """Test that text extensions list is defined."""
        assert len(TextFileHandler.text_extensions) > 0
        assert ".py" in TextFileHandler.text_extensions
        assert ".js" in TextFileHandler.text_extensions
        assert ".txt" in TextFileHandler.text_extensions

    def test_code_markdown_identifiers(self):
        """Test code language identifiers mapping."""
        assert TextFileHandler.code_markdown_identifier[".py"] == "python"
        assert TextFileHandler.code_markdown_identifier[".js"] == "javascript"
        assert TextFileHandler.code_markdown_identifier[".ts"] == "typescript"

    def test_prepare_text_content_with_code(self):
        """Test that code files are wrapped in markdown code blocks."""
        code = "def hello():\n    pass"
        result = TextFileHandler.prepare_text_content(code, ".py")
        assert result.startswith("```python\n")
        assert result.endswith("\n```")
        assert "def hello():" in result

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
        converter = DoclingLocalConverter(config)

        assert converter.config.processing.conversion_options.do_ocr is False
        assert converter.config.processing.conversion_options.table_mode == "fast"
        assert converter.config.processing.conversion_options.images_scale == 3.0

    @pytest.mark.asyncio
    async def test_convert_pdf_without_picture_images(self, config):
        """Test PDF conversion excludes embedded images by default."""
        pdf_path = Path("tests/data/doclaynet.pdf")
        if not pdf_path.exists():
            pytest.skip("doclaynet.pdf not found")

        config.processing.conversion_options.generate_picture_images = False
        converter = DoclingLocalConverter(config)

        doc = await converter.convert_file(pdf_path)
        assert isinstance(doc, DoclingDocument)

        # Check that pictures don't have image data
        for picture in doc.pictures:
            assert picture.image is None, (
                "Pictures should not have image data when generate_picture_images=False"
            )

    @pytest.mark.asyncio
    async def test_convert_pdf_with_picture_images(self, config):
        """Test PDF conversion includes embedded images when enabled."""
        pdf_path = Path("tests/data/doclaynet.pdf")
        if not pdf_path.exists():
            pytest.skip("doclaynet.pdf not found")

        config.processing.conversion_options.generate_picture_images = True
        converter = DoclingLocalConverter(config)

        doc = await converter.convert_file(pdf_path)
        assert isinstance(doc, DoclingDocument)

        # Check that at least some pictures have image data
        pictures_with_images = [p for p in doc.pictures if p.image is not None]
        if doc.pictures:
            assert len(pictures_with_images) > 0, (
                "Pictures should have image data when generate_picture_images=True"
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

    def test_picture_description_config_defaults(self, config):
        """Test that picture description config has correct defaults."""
        assert config.processing.conversion_options.picture_description.enabled is False
        assert (
            config.processing.conversion_options.picture_description.model.provider
            == "ollama"
        )
        assert (
            config.processing.conversion_options.picture_description.model.name
            == "ministral-3"
        )
        assert config.processing.conversion_options.picture_description.timeout == 90
        assert (
            config.processing.conversion_options.picture_description.max_tokens == 200
        )
        # Default prompt is in PromptsConfig
        assert "blind user" in config.prompts.picture_description

    def test_picture_description_config_applied(self, config):
        """Test that picture description config is applied to converter."""
        config.processing.conversion_options.picture_description.enabled = True
        config.processing.conversion_options.picture_description.timeout = 120
        converter = DoclingLocalConverter(config)

        pic_desc = converter.config.processing.conversion_options.picture_description
        assert pic_desc.enabled is True
        assert pic_desc.timeout == 120

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.vcr()
    async def test_picture_description_end_to_end(self, config):
        """End-to-end test: convert PDF with VLM picture descriptions."""
        pdf_path = Path("tests/data/doclaynet.pdf")
        if not pdf_path.exists():
            pytest.skip("doclaynet.pdf not found")

        # Enable picture description with Ollama
        config.processing.conversion_options.picture_description.enabled = True
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

        # Check that at least one picture has a description annotation
        from docling_core.types.doc.document import PictureDescriptionData

        pictures_with_descriptions = []
        for pic in doc.pictures:
            for ann in pic.annotations:
                if isinstance(ann, PictureDescriptionData):
                    pictures_with_descriptions.append(pic)
                    # Description should appear in markdown output
                    assert ann.text in markdown, (
                        f"Picture description '{ann.text[:50]}...' should be in markdown"
                    )
                    break

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
        submit_resp, poll_resp, result_resp = create_async_workflow_mocks(doc_json)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=submit_resp)
            mock_client.get = AsyncMock(side_effect=[poll_resp, result_resp])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            doc = await converter.convert_text("# Test", name="test.md")
            assert isinstance(doc, DoclingDocument)
            assert doc.version == "1.8.0"
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_convert_text_with_api_key(self, config):
        """Test that API key is included in request headers."""
        config.providers.docling_serve.api_key = "test-key"
        converter = DoclingServeConverter(config)

        doc_json = create_mock_docling_document("test")
        submit_resp, poll_resp, result_resp = create_async_workflow_mocks(doc_json)

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
        submit_resp, poll_resp, result_resp = create_async_workflow_mocks(doc_json)

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

            with pytest.raises(ValueError, match="Could not connect to docling-serve"):
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

            with pytest.raises(ValueError, match="timed out"):
                await converter.convert_text("# Test")

    @pytest.mark.asyncio
    async def test_convert_text_auth_error(self, converter):
        """Test handling of authentication errors."""
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

            with pytest.raises(ValueError, match="Authentication failed"):
                await converter.convert_text("# Test")

    @pytest.mark.asyncio
    async def test_convert_text_no_json_content(self, converter):
        """Test handling when docling-serve returns no JSON content."""
        submit_resp, poll_resp, _ = create_async_workflow_mocks({})
        result_resp = Mock()
        result_resp.status_code = 200
        result_resp.json.return_value = {"document": {"json_content": None}}
        result_resp.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=submit_resp)
            mock_client.get = AsyncMock(side_effect=[poll_resp, result_resp])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(ValueError, match="did not return JSON content"):
                await converter.convert_text("# Test")

    @pytest.mark.asyncio
    async def test_convert_file_pdf(self, converter):
        """Test converting PDF file via docling-serve async workflow."""
        doc_json = create_mock_docling_document("test")
        submit_resp, poll_resp, result_resp = create_async_workflow_mocks(doc_json)

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
        submit_resp, poll_resp, result_resp = create_async_workflow_mocks(doc_json)

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
        """Test that picture description options are passed to docling-serve API."""
        import json

        config.processing.conversion_options.picture_description.enabled = True
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
        submit_resp, poll_resp, result_resp = create_async_workflow_mocks(doc_json)

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
            assert data["generate_picture_images"] == "true"
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
        submit_resp, poll_resp, result_resp = create_async_workflow_mocks(doc_json)

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
    async def test_picture_description_end_to_end(self, config):
        """End-to-end test: convert PDF with VLM picture descriptions via docling-serve.

        Note: Not using VCR because this test involves polling with changing task IDs.
        """
        pdf_path = Path("tests/data/doclaynet.pdf")
        if not pdf_path.exists():
            pytest.skip("doclaynet.pdf not found")

        config.processing.conversion_options.picture_description.enabled = True
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

        from docling_core.types.doc.document import PictureDescriptionData

        pictures_with_descriptions = []
        markdown = doc.export_to_markdown()
        for pic in doc.pictures:
            for ann in pic.annotations:
                if isinstance(ann, PictureDescriptionData):
                    pictures_with_descriptions.append(pic)
                    assert ann.text in markdown, (
                        f"Picture description '{ann.text[:50]}...' should be in markdown"
                    )
                    break

        assert pictures_with_descriptions, (
            "At least one picture should have a VLM description"
        )
