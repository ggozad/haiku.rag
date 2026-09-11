"""Local docling converter implementation."""

import asyncio
import contextvars
import hashlib
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from haiku.rag.config import AppConfig
from haiku.rag.converters.base import (
    DocumentConverter,
    flatten_inline_groups,
    vlm_api_headers,
    vlm_api_params,
    vlm_api_url,
)
from haiku.rag.converters.exceptions import (
    ConversionTimeoutError,
    ConverterWedgedError,
)
from haiku.rag.converters.text_utils import TextFileHandler, docling_safe_name

if TYPE_CHECKING:
    from docling.backend.abstract_backend import AbstractDocumentBackend
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
logger = logging.getLogger(__name__)

_CONVERTER_LOCK = threading.Lock()
_CONVERTERS: dict[str, "DoclingDocConverter"] = {}

# Set when a conversion is abandoned on timeout. Its thread cannot be cancelled
# and still holds _CONVERTER_LOCK, so every later shared conversion in this
# process would block on a lock that is never released.
_WEDGED = threading.Event()

_WEDGED_MESSAGE = (
    "A previous conversion timed out and still holds the docling converter; "
    "restart the process to convert again."
)

# How long a queued conversion waits for the converter before looking at
# `_WEDGED` again. A conversion already blocked on the lock when a sibling's
# deadline fires would otherwise never learn the converter is stranded.
_LOCK_POLL_SECONDS = 0.5

# HTML and Markdown backend options carry the per-document source_uri. Both run
# SimplePipeline, which loads no models, so they get a converter per call.
_URI_AWARE_EXTENSIONS = frozenset({".html", ".xhtml", ".md", ".qmd", ".rmd"})


def _pdf_backend(name: str) -> "type[AbstractDocumentBackend]":
    """Resolve `conversion_options.pdf_backend` to its docling backend class."""
    from docling.backend.docling_parse_backend import (
        DoclingParseDocumentBackend,
        ThreadedDoclingParseDocumentBackend,
    )
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

    return {
        "threaded_docling_parse": ThreadedDoclingParseDocumentBackend,
        "docling_parse": DoclingParseDocumentBackend,
        "pypdfium2": PyPdfiumDocumentBackend,
    }[name]


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
        ".eml",
        ".html",
        ".xhtml",
        ".jpeg",
        ".jpg",
        ".latex",
        ".md",
        ".msg",
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
                headers=vlm_api_headers(pic_desc.model),
                params=vlm_api_params(pic_desc.model, pic_desc.max_tokens),
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
        receive backend options gated on `fetch_remote_images`. Only the HTML
        backend takes `headers`.

        Args:
            source_uri: Origin URI used by the HTML and Markdown backends to
                resolve relative `<img src="/path">` references (e.g. when
                ingesting a downloaded HTML page).
            pipeline_options: Wired into every format option; built from
                configuration when omitted.
        """
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
        headers = dict(opts.fetch_headers) or None

        return {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=_pdf_backend(opts.pdf_backend),
            ),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
            InputFormat.HTML: HTMLFormatOption(
                pipeline_options=pipeline_options,
                backend_options=HTMLBackendOptions(
                    fetch_images=fetch,
                    enable_remote_fetch=fetch,
                    source_uri=source_url,
                    headers=headers,
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

    def _refuse_if_wedged(self) -> None:
        """Refuse before doing any work if the shared converter is stranded."""
        if _WEDGED.is_set():
            raise ConverterWedgedError(_WEDGED_MESSAGE)

    def _cached_converter(self) -> "DoclingDocConverter":
        """The converter shared by every conversion with these pipeline
        options. The caller must hold `_CONVERTER_LOCK`.

        The key covers every input to the converter: the pipeline options, and
        `pdf_backend`, which is a format option rather than a pipeline one.
        `serialize_as_any` is required for the key: without it pydantic
        serializes the nested option models as their declared type, rendering
        them as `{}` and hiding `table_mode` and the OCR engine.
        """
        from docling.document_converter import (
            DocumentConverter as DoclingDocConverter,
        )

        pipeline_options = self._build_pipeline_options()
        key = hashlib.md5(
            b"\0".join(
                (
                    pipeline_options.model_dump_json(serialize_as_any=True).encode(),
                    self.config.processing.conversion_options.pdf_backend.encode(),
                )
            ),
            usedforsecurity=False,
        ).hexdigest()

        converter = _CONVERTERS.get(key)
        if converter is None:
            converter = _CONVERTERS[key] = DoclingDocConverter(
                format_options=self._build_format_options(
                    pipeline_options=pipeline_options
                )
            )
        return converter

    def _sync_convert_timed(
        self, path: Path, source_uri: str | None = None
    ) -> "DoclingDocument":
        """The part of a conversion the deadline covers. For shared formats the
        caller holds `_CONVERTER_LOCK`."""
        if path.suffix.lower() in _URI_AWARE_EXTENSIONS:
            from docling.document_converter import (
                DocumentConverter as DoclingDocConverter,
            )

            converter = DoclingDocConverter(
                format_options=self._build_format_options(source_uri=source_uri)
            )
            doc = converter.convert(path).document
        else:
            doc = self._cached_converter().convert(path).document

        flatten_inline_groups(doc)
        return doc

    async def _convert_docling_file(
        self, path: Path, source_uri: str | None
    ) -> "DoclingDocument":
        """Convert through docling under `processing.conversion_timeout`.

        The deadline starts when the conversion has the converter, not when it
        was asked for: `worker_count` documents contend for one shared
        converter, and queueing behind a sibling is not this document's budget.

        The lock belongs to the conversion thread, which cancellation cannot
        reach, and the deadline is shielded from the caller. A caller that goes
        away — a disconnected request, a cancelled tool call, shutdown — must
        not take the watcher with it: a stall would then hold the converter with
        `_WEDGED` clear, and every later conversion in the process would park on
        admission with nothing to raise.
        """
        timeout = self.config.processing.conversion_timeout
        shared = path.suffix.lower() not in _URI_AWARE_EXTENSIONS
        if shared:
            self._refuse_if_wedged()

        loop = asyncio.get_running_loop()
        admitted: asyncio.Future[None] = loop.create_future()
        abandoned = threading.Event()

        def _admit() -> None:
            if not admitted.done():
                admitted.set_result(None)

        def _admit_from_thread() -> None:
            try:
                loop.call_soon_threadsafe(_admit)
            except RuntimeError:  # pragma: no cover - loop already closed
                pass

        # Returned rather than raised: `asyncio.TimeoutError` is `TimeoutError`,
        # so a deadline handler cannot tell docling's own from its own.
        def _convert() -> "DoclingDocument | TimeoutError":
            try:
                return self._sync_convert_timed(path, source_uri)
            except TimeoutError as exc:
                return exc

        def _run() -> "DoclingDocument | TimeoutError | None":
            if not shared:
                _admit_from_thread()
                return _convert()

            # Polled rather than blocking: a conversion parked here when a
            # sibling's deadline fires has to learn that the converter is
            # stranded, and its caller has to hear about it.
            while not _CONVERTER_LOCK.acquire(timeout=_LOCK_POLL_SECONDS):
                if _WEDGED.is_set():
                    raise ConverterWedgedError(_WEDGED_MESSAGE)
                if abandoned.is_set():
                    return None
            try:
                if _WEDGED.is_set():
                    # The holder returned after its deadline. The converter has
                    # been declared stranded and nothing may run on it.
                    raise ConverterWedgedError(_WEDGED_MESSAGE)
                if abandoned.is_set():
                    # Nobody is waiting for this document any more, and
                    # converting it would hold the converter for no one.
                    return None
                _admit_from_thread()
                return _convert()
            finally:
                _CONVERTER_LOCK.release()

        # A daemon thread, not the default executor: `asyncio.run`'s teardown
        # joins that executor and interpreter exit joins its non-daemon
        # workers, so a conversion that never returns would hold the process
        # until something killed it. A stall costs a document, not the run.
        run: asyncio.Future[DoclingDocument | TimeoutError | None] = (
            loop.create_future()
        )

        def _deliver(finish: Callable[[], None]) -> None:
            def _set() -> None:
                # `wait_for` cancels `run` on the deadline, and a cancelled
                # future rejects a result.
                if not run.done():
                    finish()

            try:
                loop.call_soon_threadsafe(_set)
            except RuntimeError:  # pragma: no cover - loop already closed
                pass

        def _thread() -> None:
            try:
                document = _run()
            except BaseException as exc:  # noqa: BLE001
                # Bound outside the handler: `except ... as` unbinds the name
                # when the block ends, and the loop runs the callback later.
                error = exc
                _deliver(lambda: run.set_exception(error))
            else:
                _deliver(lambda: run.set_result(document))

        threading.Thread(
            target=contextvars.copy_context().run,
            args=(_thread,),
            daemon=True,
            name=f"docling-convert-{path.name}",
        ).start()

        def _finished(
            task: "asyncio.Future[DoclingDocument | TimeoutError | None]",
        ) -> None:
            # Also admits, so a thread that dies or declines before signalling
            # cannot leave the await below hanging.
            _admit()
            if not task.cancelled():
                task.exception()

        run.add_done_callback(_finished)

        async def _guard() -> "DoclingDocument":
            await admitted
            try:
                result = await asyncio.wait_for(run, timeout)
            except TimeoutError:
                detail = (
                    "The conversion thread cannot be cancelled and holds the "
                    "shared converter, so this process cannot convert again."
                    if shared
                    else "The conversion thread cannot be cancelled and runs "
                    "on for as long as it lasts."
                )
                if shared:
                    _WEDGED.set()
                if abandoned.is_set():
                    logger.error(
                        "Converting %s exceeded processing.conversion_timeout "
                        "(%ss) after its caller was cancelled; %s",
                        path,
                        timeout,
                        detail,
                    )
                raise ConversionTimeoutError(
                    f"Converting {path} exceeded processing.conversion_timeout "
                    f"({timeout}s). {detail}",
                    converter_wedged=shared,
                ) from None
            if isinstance(result, TimeoutError):
                raise result
            if result is None:
                raise asyncio.CancelledError
            return result

        def _guard_done(task: "asyncio.Task[DoclingDocument]") -> None:
            # An abandoned guard still finishes; retrieving its outcome keeps
            # the loop from reporting it as never retrieved.
            if not task.cancelled():
                task.exception()

        guard = asyncio.ensure_future(_guard())
        guard.add_done_callback(_guard_done)
        try:
            return await asyncio.shield(guard)
        except asyncio.CancelledError:
            abandoned.set()
            raise

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
            ValueError: If the file cannot be converted, chaining the cause.
            TimeoutError: If it exceeds `processing.conversion_timeout`.
        """
        try:
            file_extension = path.suffix.lower()

            if file_extension in self.docling_extensions:
                return await self._convert_docling_file(path, source_uri)
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
        except (TimeoutError, ConverterWedgedError):
            raise
        except Exception as exc:
            raise ValueError(f"Failed to parse file: {path}") from exc

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
        except ConversionError:
            return TextFileHandler._create_simple_docling_document(text, doc_name)
        flatten_inline_groups(result.document)
        return result.document
