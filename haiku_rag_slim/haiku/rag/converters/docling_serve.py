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

    def _picture_images_enabled(self) -> bool:
        """Whether the conversion should produce embedded picture images."""
        return self.config.processing.pictures != "none"

    def _build_conversion_data(self) -> dict[str, str | list[str]]:
        """Build form data for conversion request.

        When picture images are requested, switch to
        ``image_export_mode="referenced"`` + ``target_type="zip"``. Per
        docling-jobkit, ``generate_picture_images=True`` is only set when
        ``image_export_mode == "referenced"``; with ``"embedded"`` the server
        emits page images but leaves ``PictureItem.image=None`` (upstream
        issue docling-project/docling-serve#576). The zip target then ships
        the picture (and page) bytes as files under ``artifacts/`` which the
        caller rehydrates back into ``data:`` URIs for parity with the local
        converter.
        """
        opts = self.config.processing.conversion_options
        pic_desc = opts.picture_description
        pictures_mode = self.config.processing.pictures
        picture_images_enabled = pictures_mode != "none"
        runs_vlm = pictures_mode == "description"

        if picture_images_enabled:
            image_export_mode = "referenced"
        else:
            image_export_mode = (
                "embedded" if opts.generate_page_images else "placeholder"
            )

        data: dict[str, str | list[str]] = {
            "to_formats": "json",
            "do_ocr": str(opts.do_ocr).lower(),
            "force_ocr": str(opts.force_ocr).lower(),
            "ocr_engine": opts.ocr_engine,
            "do_table_structure": str(opts.do_table_structure).lower(),
            "table_mode": opts.table_mode,
            "table_cell_matching": str(opts.table_cell_matching).lower(),
            "images_scale": str(opts.images_scale),
            "image_export_mode": image_export_mode,
            "include_images": str(picture_images_enabled).lower(),
            "do_picture_description": str(runs_vlm).lower(),
        }

        if picture_images_enabled:
            data["target_type"] = "zip"

        if opts.ocr_lang:
            data["ocr_lang"] = opts.ocr_lang

        if runs_vlm:
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

    def _parse_zip_to_docling(self, zip_bytes: bytes, name: str) -> "DoclingDocument":
        """Parse a docling-serve ``target_type=zip`` response.

        The archive contains the document JSON at the root plus an
        ``artifacts/`` directory holding referenced image files (PNG, one per
        picture and per page). ``ImageRef.uri`` in the JSON points at the
        bare filename inside ``artifacts/``; this helper inlines those bytes
        as ``data:<mime>;base64,...`` URIs so the resulting
        :class:`DoclingDocument` is shape-equivalent to what the local
        converter produces.
        """
        import base64
        import io
        import zipfile

        from docling_core.types.doc.document import DoclingDocument

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            top_level_jsons = [
                n for n in zf.namelist() if n.endswith(".json") and "/" not in n
            ]
            if not top_level_jsons:
                raise ValueError(
                    f"docling-serve zip for {name} has no top-level JSON document"
                )
            with zf.open(top_level_jsons[0]) as f:
                doc_json = json.loads(f.read().decode("utf-8"))

            artifact_cache: dict[str, bytes] = {}

            def _inline(image_field: dict | None) -> None:
                if not image_field:
                    return
                uri = image_field.get("uri")
                # In target_type=zip mode docling-serve writes the URI as the
                # in-archive path, e.g. "artifacts/image_000000_<sha>.png".
                # Anything else (data: URI, missing field, weird shape) is left
                # alone.
                if not isinstance(uri, str) or not uri.startswith("artifacts/"):
                    return
                if uri not in artifact_cache:
                    try:
                        artifact_cache[uri] = zf.read(uri)
                    except KeyError:
                        return
                mime = image_field.get("mimetype") or "image/png"
                b64 = base64.b64encode(artifact_cache[uri]).decode("ascii")
                image_field["uri"] = f"data:{mime};base64,{b64}"

            for picture in doc_json.get("pictures") or []:
                _inline(picture.get("image"))
            for page in (doc_json.get("pages") or {}).values():
                if isinstance(page, dict):
                    _inline(page.get("image"))

        return DoclingDocument.model_validate(doc_json)

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

        if self._picture_images_enabled():
            zip_bytes = await self.client.submit_and_poll_zip(
                endpoint="/v1/convert/file/async",
                files=files,
                data=data,
                name=name,
            )
            return self._parse_zip_to_docling(zip_bytes, name)

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
