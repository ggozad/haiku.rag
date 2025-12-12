"""docling-serve remote converter implementation."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import httpx

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

    async def _make_request(self, files: dict, name: str) -> "DoclingDocument":
        """Make a request to docling-serve and return the DoclingDocument.

        Args:
            files: Dictionary with files parameter for httpx
            name: Name of the document being converted (for error messages)

        Returns:
            DoclingDocument representation

        Raises:
            ValueError: If conversion fails or service is unavailable
        """
        from docling_core.types.doc.document import DoclingDocument

        try:
            opts = self.config.processing.conversion_options

            data: dict[str, str | list[str]] = {
                "to_formats": "json",
                "do_ocr": str(opts.do_ocr).lower(),
                "force_ocr": str(opts.force_ocr).lower(),
                "do_table_structure": str(opts.do_table_structure).lower(),
                "table_mode": opts.table_mode,
                "table_cell_matching": str(opts.table_cell_matching).lower(),
                "images_scale": str(opts.images_scale),
                "generate_picture_images": str(opts.generate_picture_images).lower(),
            }

            if opts.ocr_lang:
                data["ocr_lang"] = opts.ocr_lang

            headers = {}
            if self.api_key:
                headers["X-Api-Key"] = self.api_key

            url = f"{self.base_url}/v1/convert/file"

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    files=files,
                    data=data,
                    headers=headers,
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

        except httpx.ConnectError as e:
            raise ValueError(
                f"Could not connect to docling-serve at {self.base_url}. "
                f"Ensure the service is running and accessible. Error: {e}"
            )
        except httpx.TimeoutException as e:
            raise ValueError(
                f"Request to docling-serve timed out after {self.timeout}s. "
                f"Consider increasing the timeout in configuration. Error: {e}"
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "Authentication failed. Check your API key configuration."
                )
            raise ValueError(f"HTTP error from docling-serve: {e}")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to convert via docling-serve: {e}")

    async def convert_file(self, path: Path) -> "DoclingDocument":
        """Convert a file to DoclingDocument using docling-serve.

        Args:
            path: Path to the file to convert.

        Returns:
            DoclingDocument representation of the file.

        Raises:
            ValueError: If the file cannot be converted or service is unavailable.
        """
        file_extension = path.suffix.lower()

        if file_extension in TextFileHandler.text_extensions:
            try:
                content = await asyncio.to_thread(path.read_text, encoding="utf-8")
                prepared_content = TextFileHandler.prepare_text_content(
                    content, file_extension
                )
                return await self.convert_text(prepared_content, name=f"{path.stem}.md")
            except Exception as e:
                raise ValueError(f"Failed to read text file {path}: {e}")

        def read_file():
            with open(path, "rb") as f:
                return f.read()

        file_content = await asyncio.to_thread(read_file)
        files = {"files": (path.name, file_content, "application/octet-stream")}
        return await self._make_request(files, path.name)

    SUPPORTED_FORMATS = ("md", "html", "plain")

    async def convert_text(
        self, text: str, name: str = "content.md", format: str = "md"
    ) -> "DoclingDocument":
        """Convert text content to DoclingDocument via docling-serve.

        Sends the text to docling-serve for conversion using the specified format.

        Args:
            text: The text content to convert.
            name: The name to use for the document (defaults to "content.md").
            format: The format of the text content ("md", "html", or "plain").
                Defaults to "md". Use "plain" for plain text without parsing.

        Returns:
            DoclingDocument representation of the text.

        Raises:
            ValueError: If the text cannot be converted or format is unsupported.
        """
        from haiku.rag.converters.text_utils import TextFileHandler

        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        # Derive document name from format to tell docling which parser to use
        doc_name = f"content.{format}" if name == "content.md" else name

        # Plain text doesn't need remote parsing - create document directly
        if format == "plain":
            return TextFileHandler._create_simple_docling_document(text, doc_name)

        mime_type = "text/html" if format == "html" else "text/markdown"

        text_bytes = text.encode("utf-8")
        files = {"files": (doc_name, text_bytes, mime_type)}
        return await self._make_request(files, doc_name)
