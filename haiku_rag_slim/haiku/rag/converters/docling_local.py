"""Local docling converter implementation."""

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

    def convert_file(self, path: Path) -> "DoclingDocument":
        """Convert a file to DoclingDocument using local docling.

        Args:
            path: Path to the file to convert.

        Returns:
            DoclingDocument representation of the file.

        Raises:
            ValueError: If the file cannot be converted.
        """
        from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            OcrOptions,
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

        try:
            file_extension = path.suffix.lower()

            if file_extension in self.docling_extensions:
                # Get conversion options from config
                opts = self.config.processing.conversion_options

                # Build pipeline options for PDF conversion
                pipeline_options = PdfPipelineOptions(
                    do_ocr=opts.do_ocr,
                    do_table_structure=opts.do_table_structure,
                    images_scale=opts.images_scale,
                    table_structure_options=TableStructureOptions(
                        do_cell_matching=opts.table_cell_matching,
                        mode=(
                            TableFormerMode.FAST
                            if opts.table_mode == "fast"
                            else TableFormerMode.ACCURATE
                        ),
                    ),
                    ocr_options=OcrOptions(
                        force_full_page_ocr=opts.force_ocr,
                        lang=opts.ocr_lang if opts.ocr_lang else [],
                    ),
                )

                # Create format options for PDF
                format_options = cast(
                    dict[InputFormat, FormatOption],
                    {
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options,
                            backend=DoclingParseDocumentBackend,
                        )
                    },
                )

                # Use docling for complex document formats
                converter = DoclingDocConverter(format_options=format_options)
                result = converter.convert(path)
                return result.document
            elif file_extension in TextFileHandler.text_extensions:
                # Read plain text files directly
                content = path.read_text(encoding="utf-8")
                # Prepare content with code block wrapping if needed
                prepared_content = TextFileHandler.prepare_text_content(
                    content, file_extension
                )
                # Convert text to DoclingDocument by wrapping as markdown
                return self.convert_text(prepared_content, name=f"{path.stem}.md")
            else:
                # Fallback: try to read as text and convert to DoclingDocument
                content = path.read_text(encoding="utf-8")
                return self.convert_text(content, name=f"{path.stem}.md")
        except Exception:
            raise ValueError(f"Failed to parse file: {path}")

    def convert_text(self, text: str, name: str = "content.md") -> "DoclingDocument":
        """Convert text content to DoclingDocument using local docling.

        Args:
            text: The text content to convert.
            name: The name to use for the document (defaults to "content.md").

        Returns:
            DoclingDocument representation of the text.

        Raises:
            ValueError: If the text cannot be converted.
        """
        return TextFileHandler.text_to_docling_document(text, name)
