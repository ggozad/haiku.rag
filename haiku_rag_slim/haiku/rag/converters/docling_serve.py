"""docling-serve remote converter implementation."""

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from haiku.rag.config import AppConfig
from haiku.rag.converters.base import DocumentConverter
from haiku.rag.converters.text_utils import TextFileHandler
from haiku.rag.providers.docling_serve import DoclingServeClient

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.config.models import ModelConfig


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
        self.client = DoclingServeClient(
            base_url=config.providers.docling_serve.base_url,
            api_key=config.providers.docling_serve.api_key,
        )

    @property
    def supported_extensions(self) -> list[str]:
        """Return list of file extensions supported by this converter."""
        return self.docling_serve_extensions + TextFileHandler.text_extensions

    def _get_vlm_api_url(self, model: "ModelConfig") -> str:
        """Construct VLM API URL from model config."""
        if model.base_url:
            base = model.base_url.rstrip("/")
            return f"{base}/v1/chat/completions"

        if model.provider == "ollama":
            base = self.config.providers.ollama.base_url.rstrip("/")
            return f"{base}/v1/chat/completions"

        if model.provider == "openai":
            return "https://api.openai.com/v1/chat/completions"

        raise ValueError(f"Unsupported VLM provider: {model.provider}")

    def _build_conversion_data(self) -> dict[str, str | list[str]]:
        """Build form data for conversion request."""
        opts = self.config.processing.conversion_options
        pic_desc = opts.picture_description

        data: dict[str, str | list[str]] = {
            "to_formats": "json",
            "do_ocr": str(opts.do_ocr).lower(),
            "force_ocr": str(opts.force_ocr).lower(),
            "do_table_structure": str(opts.do_table_structure).lower(),
            "table_mode": opts.table_mode,
            "table_cell_matching": str(opts.table_cell_matching).lower(),
            "images_scale": str(opts.images_scale),
            "generate_picture_images": str(
                opts.generate_picture_images or pic_desc.enabled
            ).lower(),
            "do_picture_description": str(pic_desc.enabled).lower(),
        }

        if opts.ocr_lang:
            data["ocr_lang"] = opts.ocr_lang

        if pic_desc.enabled:
            prompt = self.config.prompts.picture_description
            picture_description_api = {
                "url": self._get_vlm_api_url(pic_desc.model),
                "params": {
                    "model": pic_desc.model.name,
                    "max_completion_tokens": pic_desc.max_tokens,
                },
                "prompt": prompt,
                "timeout": pic_desc.timeout,
            }
            data["picture_description_api"] = json.dumps(picture_description_api)

        return data

    async def _make_request(self, files: dict, name: str) -> "DoclingDocument":
        """Make an async request to docling-serve and poll for results.

        Args:
            files: Dictionary with files parameter for httpx
            name: Name of the document being converted (for error messages)

        Returns:
            DoclingDocument representation

        Raises:
            ValueError: If conversion fails or service is unavailable
        """
        from docling_core.types.doc.document import DoclingDocument

        data = self._build_conversion_data()
        result = await self.client.submit_and_poll(
            endpoint="/v1/convert/file/async",
            files=files,
            data=data,
            name=name,
        )

        if result.get("status") not in ("success", "partial_success", None):
            errors = result.get("errors", [])
            raise ValueError(f"Conversion failed: {errors}")

        json_content = result.get("document", {}).get("json_content")

        if json_content is None:
            raise ValueError(
                f"docling-serve did not return JSON content for {name}. "
                "This may indicate an unsupported file format."
            )

        return DoclingDocument.model_validate(json_content)

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
