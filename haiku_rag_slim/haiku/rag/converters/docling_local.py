"""Local docling converter implementation."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from haiku.rag.config import AppConfig
from haiku.rag.converters.base import DocumentConverter
from haiku.rag.converters.text_utils import TextFileHandler

if TYPE_CHECKING:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import FormatOption
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.config.models import ConversionOptions, ModelConfig


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
        ".latex",
        ".md",
        ".pdf",
        ".png",
        ".pptx",
        ".qmd",
        ".rmd",
        ".tex",
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

    def _get_ocr_options(self, opts: "ConversionOptions"):
        """Get OCR options based on configuration."""
        from docling.datamodel.pipeline_options import (
            EasyOcrOptions,
            OcrAutoOptions,
            OcrMacOptions,
            RapidOcrOptions,
            TesseractCliOcrOptions,
            TesseractOcrOptions,
        )

        force_ocr = opts.force_ocr
        lang = opts.ocr_lang if opts.ocr_lang else []

        match opts.ocr_engine:
            case "easyocr":
                return EasyOcrOptions(force_full_page_ocr=force_ocr, lang=lang)
            case "rapidocr":
                return RapidOcrOptions(force_full_page_ocr=force_ocr, lang=lang)
            case "tesseract":
                return TesseractOcrOptions(force_full_page_ocr=force_ocr, lang=lang)
            case "tesserocr":
                return TesseractCliOcrOptions(force_full_page_ocr=force_ocr, lang=lang)
            case "ocrmac":
                return OcrMacOptions(force_full_page_ocr=force_ocr, lang=lang)
            case _:  # "auto" or any other value
                return OcrAutoOptions(force_full_page_ocr=force_ocr, lang=lang)

    def _build_pipeline_options(self):
        """Build the shared PdfPipelineOptions instance applied to every wired
        FormatOption. SimplePipeline-backed formats ignore the PDF-specific
        fields; the ConvertPipelineOptions-level picture description /
        classification / chart-extraction settings apply uniformly."""
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            PictureDescriptionApiOptions,
            TableFormerMode,
            TableStructureOptions,
        )

        opts = self.config.processing.conversion_options
        pic_desc = opts.picture_description
        runs_vlm = pic_desc.enabled

        pipeline_options = PdfPipelineOptions(
            do_ocr=opts.do_ocr,
            do_table_structure=opts.do_table_structure,
            images_scale=opts.images_scale,
            generate_page_images=opts.generate_page_images,
            generate_picture_images=True,
            table_structure_options=TableStructureOptions(
                do_cell_matching=opts.table_cell_matching,
                mode=(
                    TableFormerMode.FAST
                    if opts.table_mode == "fast"
                    else TableFormerMode.ACCURATE
                ),
            ),
            ocr_options=self._get_ocr_options(opts),
            do_picture_description=runs_vlm,
        )

        if runs_vlm:
            from pydantic import AnyUrl

            pipeline_options.enable_remote_services = True
            pipeline_options.picture_description_options = PictureDescriptionApiOptions(
                url=AnyUrl(self._get_vlm_api_url(pic_desc.model)),
                params=dict(
                    model=pic_desc.model.name,
                    max_completion_tokens=pic_desc.max_tokens,
                ),
                prompt=self.config.prompts.picture_description,
                timeout=pic_desc.timeout,
            )

        return pipeline_options

    def _build_format_options(
        self, source_uri: str | None = None
    ) -> "dict[InputFormat, FormatOption]":
        """Per-format options shared between file and text conversion paths.

        Every wired FormatOption gets the same `PdfPipelineOptions` instance so
        picture-description / classification / chart settings apply uniformly
        across PDF, IMAGE, HTML, MD, DOCX, PPTX. HTML and Markdown additionally
        receive backend options gated on `fetch_remote_images`.

        Args:
            source_uri: Origin URI used by the HTML and Markdown backends to
                resolve relative `<img src="/path">` references (e.g. when
                ingesting a downloaded HTML page).
        """
        from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
        from docling.datamodel.backend_options import (
            HTMLBackendOptions,
            MarkdownBackendOptions,
        )
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import (
            HTMLFormatOption,
            ImageFormatOption,
            MarkdownFormatOption,
            PdfFormatOption,
            PowerpointFormatOption,
            WordFormatOption,
        )
        from pydantic import AnyUrl

        opts = self.config.processing.conversion_options
        pipeline_options = self._build_pipeline_options()
        fetch = opts.fetch_remote_images
        source_url = AnyUrl(source_uri) if source_uri else None

        return {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=DoclingParseDocumentBackend,
            ),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
            InputFormat.HTML: HTMLFormatOption(
                pipeline_options=pipeline_options,
                backend_options=HTMLBackendOptions(
                    fetch_images=fetch,
                    enable_remote_fetch=fetch,
                    source_uri=source_url,
                ),
            ),
            InputFormat.MD: MarkdownFormatOption(
                pipeline_options=pipeline_options,
                backend_options=MarkdownBackendOptions(
                    fetch_images=fetch,
                    enable_remote_fetch=fetch,
                    source_uri=source_url,
                ),
            ),
            InputFormat.DOCX: WordFormatOption(pipeline_options=pipeline_options),
            InputFormat.PPTX: PowerpointFormatOption(pipeline_options=pipeline_options),
        }

    def _sync_convert_docling_file(
        self, path: Path, source_uri: str | None = None
    ) -> "DoclingDocument":
        """Synchronous conversion of docling-supported files."""
        from docling.document_converter import (
            DocumentConverter as DoclingDocConverter,
        )

        converter = DoclingDocConverter(
            format_options=self._build_format_options(source_uri=source_uri)
        )
        result = converter.convert(path)
        return result.document

    async def convert_file(
        self, path: Path, source_uri: str | None = None
    ) -> "DoclingDocument":
        """Convert a file to DoclingDocument using local docling.

        Args:
            path: Path to the file to convert.
            source_uri: Optional origin URI used by docling's HTML/Markdown
                backends to resolve relative image references.

        Returns:
            DoclingDocument representation of the file.

        Raises:
            ValueError: If the file cannot be converted.
        """
        try:
            file_extension = path.suffix.lower()

            if file_extension in self.docling_extensions:
                return await asyncio.to_thread(
                    self._sync_convert_docling_file, path, source_uri
                )
            elif file_extension in TextFileHandler.text_extensions:
                content = await asyncio.to_thread(path.read_text, encoding="utf-8")
                prepared_content = TextFileHandler.prepare_text_content(
                    content, file_extension
                )
                return await self.convert_text(
                    prepared_content,
                    name=f"{path.stem}.md",
                    source_uri=source_uri,
                )
            else:
                content = await asyncio.to_thread(path.read_text, encoding="utf-8")
                return await self.convert_text(
                    content, name=f"{path.stem}.md", source_uri=source_uri
                )
        except Exception:
            raise ValueError(f"Failed to parse file: {path}")

    async def convert_text(
        self,
        text: str,
        name: str = "content.md",
        format: str = "md",
        source_uri: str | None = None,
    ) -> "DoclingDocument":
        """Convert text content to DoclingDocument using local docling.

        Args:
            text: The text content to convert.
            name: The name to use for the document (defaults to "content.md").
            format: The format of the text content ("md", "html", or "plain").
                Defaults to "md". Use "plain" for plain text without parsing.
            source_uri: Optional origin URI used by docling's HTML/Markdown
                backends to resolve relative image references.

        Returns:
            DoclingDocument representation of the text.

        Raises:
            ValueError: If the text cannot be converted or format is unsupported.
        """
        if format not in TextFileHandler.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported formats: {', '.join(TextFileHandler.SUPPORTED_FORMATS)}"
            )

        doc_name = f"content.{format}" if name == "content.md" else name

        if format == "plain":
            return TextFileHandler._create_simple_docling_document(text, doc_name)

        try:
            return await asyncio.to_thread(
                self._sync_convert_docling_text, text, doc_name, source_uri
            )
        except Exception as e:
            raise ValueError(f"Failed to convert text to DoclingDocument: {e}")

    def _sync_convert_docling_text(
        self, text: str, doc_name: str, source_uri: str | None = None
    ) -> "DoclingDocument":
        """Synchronous text-to-DoclingDocument using the shared format options."""
        from io import BytesIO

        from docling.document_converter import (
            DocumentConverter as DoclingDocConverter,
        )
        from docling.exceptions import ConversionError
        from docling_core.types.io import DocumentStream

        bytes_io = BytesIO(text.encode("utf-8"))
        doc_stream = DocumentStream(name=doc_name, stream=bytes_io)
        converter = DoclingDocConverter(
            format_options=self._build_format_options(source_uri=source_uri)
        )
        try:
            result = converter.convert(doc_stream)
            return result.document
        except ConversionError:
            return TextFileHandler._create_simple_docling_document(text, doc_name)
