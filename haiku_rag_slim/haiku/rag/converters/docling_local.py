"""Local docling converter implementation."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from haiku.rag.config import AppConfig
from haiku.rag.converters.base import DocumentConverter
from haiku.rag.converters.text_utils import TextFileHandler

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class DoclingLocalConverter(DocumentConverter):
    """Converter that uses local docling for document conversion.

    This converter runs docling locally in-process to convert documents.
    It handles various document formats including PDF, DOCX, HTML, and plain text.
    """

    # Extensions supported by docling
    docling_extensions: ClassVar[list[str]] = [
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
            config: Application configuration containing conversion options.
        """
        self.config = config

    @property
    def supported_extensions(self) -> list[str]:
        """Return list of file extensions supported by this converter."""
        return self.docling_extensions + TextFileHandler.text_extensions

    def _sync_convert_docling_file(self, path: Path) -> "DoclingDocument":
        """Synchronous conversion of docling-supported files."""
        from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            OcrAutoOptions,
            PdfPipelineOptions,
            TableFormerMode,
            TableStructureOptions,
        )
        from docling.document_converter import (
            DocumentConverter as DoclingDocConverter,
        )
        from docling.document_converter import (
            FormatOption,
            PdfFormatOption,
        )

        opts = self.config.processing.conversion_options

        pipeline_options = PdfPipelineOptions(
            do_ocr=opts.do_ocr,
            do_table_structure=opts.do_table_structure,
            images_scale=opts.images_scale,
            generate_page_images=True,
            generate_picture_images=opts.generate_picture_images,
            table_structure_options=TableStructureOptions(
                do_cell_matching=opts.table_cell_matching,
                mode=(
                    TableFormerMode.FAST
                    if opts.table_mode == "fast"
                    else TableFormerMode.ACCURATE
                ),
            ),
            ocr_options=OcrAutoOptions(
                force_full_page_ocr=opts.force_ocr,
                lang=opts.ocr_lang if opts.ocr_lang else [],
            ),
        )

        format_options = cast(
            dict[InputFormat, FormatOption],
            {
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=DoclingParseDocumentBackend,
                )
            },
        )

        converter = DoclingDocConverter(format_options=format_options)
        result = converter.convert(path)
        return result.document

    async def convert_file(self, path: Path) -> "DoclingDocument":
        """Convert a file to DoclingDocument using local docling.

        Args:
            path: Path to the file to convert.

        Returns:
            DoclingDocument representation of the file.

        Raises:
            ValueError: If the file cannot be converted.
        """
        try:
            file_extension = path.suffix.lower()

            if file_extension in self.docling_extensions:
                return await asyncio.to_thread(self._sync_convert_docling_file, path)
            elif file_extension in TextFileHandler.text_extensions:
                content = await asyncio.to_thread(path.read_text, encoding="utf-8")
                prepared_content = TextFileHandler.prepare_text_content(
                    content, file_extension
                )
                return await self.convert_text(prepared_content, name=f"{path.stem}.md")
            else:
                content = await asyncio.to_thread(path.read_text, encoding="utf-8")
                return await self.convert_text(content, name=f"{path.stem}.md")
        except Exception:
            raise ValueError(f"Failed to parse file: {path}")

    async def convert_text(
        self, text: str, name: str = "content.md", format: str = "md"
    ) -> "DoclingDocument":
        """Convert text content to DoclingDocument using local docling.

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
        return await TextFileHandler.text_to_docling_document(text, name, format)
