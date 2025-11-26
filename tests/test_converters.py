"""Tests for document converters."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
from docling_core.types.doc.document import DoclingDocument

from haiku.rag.config import AppConfig
from haiku.rag.converters import get_converter
from haiku.rag.converters.docling_local import DoclingLocalConverter
from haiku.rag.converters.docling_serve import DoclingServeConverter
from haiku.rag.converters.text_utils import TextFileHandler


def is_docling_serve_available(base_url: str = "http://localhost:5001") -> bool:
    """Check if docling-serve is running and accessible."""
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def create_mock_docling_document_json(name: str = "test") -> dict:
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


class TestDoclingServeConverter:
    """Tests for DoclingServeConverter (mocked)."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = AppConfig()
        config.providers.docling_serve.base_url = "http://localhost:5001"
        config.providers.docling_serve.api_key = ""
        config.providers.docling_serve.timeout = 300
        return config

    @pytest.fixture
    def converter(self, config):
        """Create DoclingServeConverter instance."""
        return DoclingServeConverter(config)

    def test_initialization(self, converter):
        """Test converter initialization."""
        assert converter.base_url == "http://localhost:5001"
        assert converter.timeout == 300

    def test_supported_extensions(self, converter):
        """Test that converter reports correct supported extensions."""
        extensions = converter.supported_extensions
        assert ".pdf" in extensions
        assert ".docx" in extensions
        assert ".py" in extensions
        assert ".md" in extensions

    @pytest.mark.asyncio
    @patch("haiku.rag.converters.docling_serve.requests.post")
    async def test_convert_text_success(self, mock_post, converter):
        """Test successful text conversion via docling-serve."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "document": {"json_content": create_mock_docling_document_json("test")},
        }
        mock_post.return_value = mock_response

        doc = await converter.convert_text("# Test", name="test.md")
        assert isinstance(doc, DoclingDocument)
        assert doc.version == "1.8.0"
        mock_post.assert_called_once()

    @pytest.mark.asyncio
    @patch("haiku.rag.converters.docling_serve.requests.post")
    async def test_convert_text_with_api_key(self, mock_post, config):
        """Test that API key is included in request headers."""
        config.providers.docling_serve.api_key = "test-key"
        converter = DoclingServeConverter(config)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "document": {"json_content": create_mock_docling_document_json("test")},
        }
        mock_post.return_value = mock_response

        await converter.convert_text("# Test")

        call_kwargs = mock_post.call_args.kwargs
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["X-Api-Key"] == "test-key"

    @pytest.mark.asyncio
    @patch("haiku.rag.converters.docling_serve.requests.post")
    async def test_conversion_options_passed_to_api(self, mock_post, config):
        """Test that conversion options are passed to docling-serve API."""
        config.processing.conversion_options.do_ocr = False
        config.processing.conversion_options.force_ocr = True
        config.processing.conversion_options.ocr_lang = ["en", "fr"]
        config.processing.conversion_options.table_mode = "fast"
        config.processing.conversion_options.table_cell_matching = False
        config.processing.conversion_options.do_table_structure = False
        config.processing.conversion_options.images_scale = 3.0
        converter = DoclingServeConverter(config)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "document": {"json_content": create_mock_docling_document_json("test")},
        }
        mock_post.return_value = mock_response

        await converter.convert_text("# Test")

        call_kwargs = mock_post.call_args.kwargs
        assert "data" in call_kwargs
        data = call_kwargs["data"]
        assert data["do_ocr"] is False
        assert data["force_ocr"] is True
        assert data["ocr_lang"] == ["en", "fr"]
        assert "pdf_backend" not in data
        assert data["table_mode"] == "fast"
        assert data["table_cell_matching"] is False
        assert data["do_table_structure"] is False
        assert data["images_scale"] == 3.0

    @pytest.mark.asyncio
    @patch("haiku.rag.converters.docling_serve.requests.post")
    async def test_convert_text_connection_error(self, mock_post, converter):
        """Test handling of connection errors."""
        import requests

        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")

        with pytest.raises(ValueError, match="Could not connect to docling-serve"):
            await converter.convert_text("# Test")

    @pytest.mark.asyncio
    @patch("haiku.rag.converters.docling_serve.requests.post")
    async def test_convert_text_timeout_error(self, mock_post, converter):
        """Test handling of timeout errors."""
        import requests

        mock_post.side_effect = requests.exceptions.Timeout("Timeout")

        with pytest.raises(ValueError, match="timed out"):
            await converter.convert_text("# Test")

    @pytest.mark.asyncio
    @patch("haiku.rag.converters.docling_serve.requests.post")
    async def test_convert_text_auth_error(self, mock_post, converter):
        """Test handling of authentication errors."""
        import requests

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_post.return_value = mock_response

        with pytest.raises(ValueError, match="Authentication failed"):
            await converter.convert_text("# Test")

    @pytest.mark.asyncio
    @patch("haiku.rag.converters.docling_serve.requests.post")
    async def test_convert_text_no_json_content(self, mock_post, converter):
        """Test handling when docling-serve returns no JSON content."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "document": {"json_content": None},
        }
        mock_post.return_value = mock_response

        with pytest.raises(ValueError, match="did not return JSON content"):
            await converter.convert_text("# Test")

    @pytest.mark.asyncio
    @patch("haiku.rag.converters.docling_serve.requests.post")
    async def test_convert_file_pdf(self, mock_post, converter):
        """Test converting PDF file via docling-serve."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "document": {"json_content": create_mock_docling_document_json("test")},
        }
        mock_post.return_value = mock_response

        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(b"fake pdf content")
            f.flush()
            temp_path = Path(f.name)
            doc = await converter.convert_file(temp_path)

        assert isinstance(doc, DoclingDocument)
        mock_post.assert_called_once()

    @pytest.mark.asyncio
    @patch("haiku.rag.converters.docling_serve.requests.post")
    async def test_convert_file_text(self, mock_post, converter):
        """Test converting text file (reads locally, sends to docling-serve)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "document": {"json_content": create_mock_docling_document_json("test")},
        }
        mock_post.return_value = mock_response

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
            f.write("def hello():\n    pass")
            f.flush()
            temp_path = Path(f.name)
            doc = await converter.convert_file(temp_path)

        assert isinstance(doc, DoclingDocument)
        # Should call docling-serve for conversion
        mock_post.assert_called_once()
        # Check that code was wrapped in code block
        call_kwargs = mock_post.call_args.kwargs
        assert "files" in call_kwargs


@pytest.mark.integration
@pytest.mark.skipif(
    not is_docling_serve_available(),
    reason="docling-serve not available at http://localhost:5001",
)
class TestDoclingServeConverterIntegration:
    """Integration tests with real docling-serve (requires service running)."""

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

    @pytest.mark.asyncio
    async def test_convert_text_real_service(self, converter):
        """Test text conversion with real docling-serve (integration)."""
        doc = await converter.convert_text("# Test Document\n\nThis is a test.")
        assert isinstance(doc, DoclingDocument)
        assert doc.version == "1.8.0"

    @pytest.mark.asyncio
    async def test_convert_code_file_real_service(self, converter):
        """Test code file conversion with real docling-serve (integration)."""
        code = "def test():\n    return 42"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
            f.write(code)
            f.flush()
            temp_path = Path(f.name)
            doc = await converter.convert_file(temp_path)

        assert isinstance(doc, DoclingDocument)
        result = doc.export_to_markdown()
        assert "def test():" in result
