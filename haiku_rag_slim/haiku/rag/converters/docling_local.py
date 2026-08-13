"""Local docling converter implementation."""

import asyncio
import hashlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from haiku.rag.config import AppConfig
from haiku.rag.converters.base import DocumentConverter, vlm_api_url
from haiku.rag.converters.text_utils import TextFileHandler, docling_safe_name

if TYPE_CHECKING:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter as DoclingDocConverter
    from docling.document_converter import FormatOption
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.config.models import ConversionOptions

# Docling builds its layout, table and OCR models per DocumentConverter and
# caches pipelines per instance, so a converter per document reloads every model
# per document. StandardPdfPipeline also keeps per-run state on the instance, so
# the lock spans the conversion, not just the lookup.
_CONVERTER_LOCK = threading.Lock()
_CONVERTERS: dict[str, "DoclingDocConverter"] = {}

# HTML and Markdown backend options carry the per-document source_uri. Both run
# SimplePipeline, which loads no models, so they get a converter per call.
_URI_AWARE_EXTENSIONS = frozenset({".html", ".xhtml", ".md", ".qmd", ".rmd"})


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
        pictures = self.config.processing.pictures
        runs_vlm = pictures == "description"

        pipeline_options = PdfPipelineOptions(
            do_ocr=opts.do_ocr,
            do_table_structure=opts.do_table_structure,
            images_scale=opts.images_scale,
            generate_page_images=opts.generate_page_images,
            generate_picture_images=pictures != "none",
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
                url=AnyUrl(vlm_api_url(self.config, pic_desc.model)),
                params=dict(
                    model=pic_desc.model.name,
                    max_completion_tokens=pic_desc.max_tokens,
                ),
                prompt=self.config.prompts.picture_description,
                timeout=pic_desc.timeout,
            )

        return pipeline_options

    def _build_format_options(
        self,
        source_uri: str | None = None,
        pipeline_options: "PdfPipelineOptions | None" = None,
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
            pipeline_options: Wired into every format option; built from
                configuration when omitted.
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
        if pipeline_options is None:
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

    @contextmanager
    def _shared_converter(self) -> Iterator["DoclingDocConverter"]:
        """Yield the converter shared by every conversion with these pipeline
        options, holding the lock for the caller's conversion.

        `serialize_as_any` is required for the key: without it pydantic
        serializes the nested option models as their declared type, rendering
        them as `{}` and hiding `table_mode` and the OCR engine.
        """
        from docling.document_converter import (
            DocumentConverter as DoclingDocConverter,
        )

        pipeline_options = self._build_pipeline_options()
        key = hashlib.md5(
            pipeline_options.model_dump_json(serialize_as_any=True).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()

        with _CONVERTER_LOCK:
            converter = _CONVERTERS.get(key)
            if converter is None:
                converter = _CONVERTERS[key] = DoclingDocConverter(
                    format_options=self._build_format_options(
                        pipeline_options=pipeline_options
                    )
                )
            yield converter

    def _sync_convert_docling_file(
        self, path: Path, source_uri: str | None = None
    ) -> "DoclingDocument":
        """Synchronous conversion of docling-supported files."""
        if path.suffix.lower() in _URI_AWARE_EXTENSIONS:
            from docling.document_converter import (
                DocumentConverter as DoclingDocConverter,
            )

            converter = DoclingDocConverter(
                format_options=self._build_format_options(source_uri=source_uri)
            )
            return converter.convert(path).document

        with self._shared_converter() as converter:
            return converter.convert(path).document

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

        doc_name = docling_safe_name(
            f"content.{format}" if name == "content.md" else name
        )

        if format == "plain":
            return TextFileHandler._create_simple_docling_document(text, doc_name)

        try:
            return await asyncio.to_thread(
                self._sync_convert_docling_text, text, doc_name, source_uri
            )
        except Exception as e:
            raise ValueError(f"Failed to convert text to DoclingDocument: {e}") from e

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

        # Docling sniffs magic bytes before considering the extension, so text
        # starting with e.g. "BM" (BMP) or "ID3" (MP3) gets routed to a binary
        # backend. A leading newline defeats every magic signature (all match
        # at offset 0) without changing the md/html parse, making docling fall
        # back to the extension in doc_name, which encodes the known format.
        bytes_io = BytesIO(b"\n" + text.encode("utf-8"))
        doc_stream = DocumentStream(name=doc_name, stream=bytes_io)
        converter = DoclingDocConverter(
            format_options=self._build_format_options(source_uri=source_uri)
        )
        try:
            result = converter.convert(doc_stream)
            return result.document
        except ConversionError:
            return TextFileHandler._create_simple_docling_document(text, doc_name)
