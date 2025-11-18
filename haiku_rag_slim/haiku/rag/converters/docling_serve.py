"""docling-serve remote converter implementation."""

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import requests

from haiku.rag.config import AppConfig
from haiku.rag.converters.base import DocumentConverter
from haiku.rag.converters.text_utils import TextFileHandler

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class DoclingServeConverter(DocumentConverter):
    """Converter that uses docling-serve for document conversion.

    This converter offloads document processing to a docling-serve instance,
    which handles heavy operations like PDF parsing, OCR, and table extraction.

    For plain text files, it reads them locally and converts to markdown format
    before sending to docling-serve for DoclingDocument conversion.
    """

    # Extensions that docling-serve can handle
    docling_serve_extensions: ClassVar[list[str]] = [
        ".adoc",
        ".asc",
        ".asciidoc",
        ".bmp",
        ".csv",
        ".docx",
        ".html",
        ".xhtml",
        ".jpeg",
        ".jpg",
        ".md",
        ".pdf",
        ".png",
        ".pptx",
        ".tiff",
        ".xlsx",
        ".xml",
        ".webp",
    ]

    def __init__(self, config: AppConfig):
        """Initialize the converter with configuration.

        Args:
            config: Application configuration containing docling-serve settings.
        """
        self.config = config
        self.base_url = config.providers.docling_serve.base_url.rstrip("/")
        self.api_key = config.providers.docling_serve.api_key
        self.timeout = config.providers.docling_serve.timeout

    @property
    def supported_extensions(self) -> list[str]:
        """Return list of file extensions supported by this converter."""
        return self.docling_serve_extensions + TextFileHandler.text_extensions

    def _make_request(self, files: dict, name: str) -> "DoclingDocument":
        """Make a request to docling-serve and return the DoclingDocument.

        Args:
            files: Dictionary with files parameter for requests
            name: Name of the document being converted (for error messages)

        Returns:
            DoclingDocument representation

        Raises:
            ValueError: If conversion fails or service is unavailable
        """
        from docling_core.types.doc.document import DoclingDocument

        try:
            url = f"{self.base_url}/v1/convert/file"
            opts = self.config.processing.conversion_options

            # Build data dict with conversion options
            data = {
                "to_formats": ["json"],
                # OCR options
                "do_ocr": opts.do_ocr,
                "force_ocr": opts.force_ocr,
                # Table options
                "do_table_structure": opts.do_table_structure,
                "table_mode": opts.table_mode,
                "table_cell_matching": opts.table_cell_matching,
                # Image options
                "images_scale": opts.images_scale,
            }

            # Add OCR language if specified
            if opts.ocr_lang:
                data["ocr_lang"] = opts.ocr_lang

            headers = {}
            if self.api_key:
                headers["X-Api-Key"] = self.api_key

            response = requests.post(
                url,
                files=files,
                data=data,
                headers=headers,
                timeout=self.timeout,
            )

            response.raise_for_status()

            result = response.json()

            if result["status"] not in ("success", "partial_success"):
                errors = result.get("errors", [])
                raise ValueError(f"Conversion failed: {errors}")

            json_content = result["document"]["json_content"]

            if json_content is None:
                raise ValueError(
                    f"docling-serve did not return JSON content for {name}. "
                    "This may indicate an unsupported file format."
                )

            return DoclingDocument.model_validate(json_content)

        except requests.exceptions.ConnectionError as e:
            raise ValueError(
                f"Could not connect to docling-serve at {self.base_url}. "
                f"Ensure the service is running and accessible. Error: {e}"
            )
        except requests.exceptions.Timeout as e:
            raise ValueError(
                f"Request to docling-serve timed out after {self.timeout}s. "
                f"Consider increasing the timeout in configuration. Error: {e}"
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "Authentication failed. Check your API key configuration."
                )
            raise ValueError(f"HTTP error from docling-serve: {e}")
        except Exception as e:
            raise ValueError(f"Failed to convert via docling-serve: {e}")

    def convert_file(self, path: Path) -> "DoclingDocument":
        """Convert a file to DoclingDocument using docling-serve.

        Args:
            path: Path to the file to convert.

        Returns:
            DoclingDocument representation of the file.

        Raises:
            ValueError: If the file cannot be converted or service is unavailable.
        """
        file_extension = path.suffix.lower()

        # For plain text files, read locally and prepare content
        if file_extension in TextFileHandler.text_extensions:
            try:
                content = path.read_text(encoding="utf-8")
                prepared_content = TextFileHandler.prepare_text_content(
                    content, file_extension
                )
                return self.convert_text(prepared_content, name=f"{path.stem}.md")
            except Exception as e:
                raise ValueError(f"Failed to read text file {path}: {e}")

        # For complex formats, send file to docling-serve
        with open(path, "rb") as f:
            files = {"files": f}
            return self._make_request(files, path.name)

    def convert_text(self, text: str, name: str = "content.md") -> "DoclingDocument":
        """Convert text content to DoclingDocument via docling-serve.

        Sends the text as a markdown file to docling-serve for conversion.

        Args:
            text: The text content to convert.
            name: The name to use for the document (defaults to "content.md").

        Returns:
            DoclingDocument representation of the text.

        Raises:
            ValueError: If the text cannot be converted.
        """
        from io import BytesIO

        text_bytes = text.encode("utf-8")
        files = {"files": (name, BytesIO(text_bytes), "text/markdown")}
        return self._make_request(files, name)
