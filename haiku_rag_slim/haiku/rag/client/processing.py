import asyncio
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from haiku.rag.client.exceptions import UnsupportedSourceError
from haiku.rag.config import AppConfig
from haiku.rag.converters import get_converter
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.models.document_item import _picture_description_text

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.embeddings import EmbedderWrapper


logger = logging.getLogger(__name__)


def _warn_if_descriptions_missing(
    config: AppConfig, doc: "DoclingDocument", source: str
) -> None:
    """Warn when picture-description was requested but produced nothing.

    docling-serve swallows VLM errors (network failures, missing models,
    etc.) and returns a successful conversion with empty descriptions.
    docling-local can do the same when the VLM endpoint is unreachable.
    Surface the silent failure: when ``processing.pictures="description"``
    AND the document has at least one picture AND zero descriptions came
    back, log a clear warning so the user can fix their VLM config before
    a thousand-document ingest produces an empty corpus.
    """
    if config.processing.pictures != "description":
        return
    if not doc.pictures:
        return
    described = sum(1 for p in doc.pictures if _picture_description_text(p))
    if described == 0:
        model = config.processing.conversion_options.picture_description.model
        logger.warning(
            "processing.pictures='description' but no descriptions came back "
            "for %s (%d pictures, 0 described). The VLM call likely failed "
            "silently inside the converter. Check that the VLM at %s is "
            "reachable from the converter and that the model name '%s' "
            "resolves on the server.",
            source,
            len(doc.pictures),
            model.base_url or "<provider default>",
            model.name,
        )


async def convert(
    config: AppConfig,
    source: Path | str,
    *,
    format: str = "md",
    source_uri: str | None = None,
) -> "DoclingDocument":
    """Convert a file, URL, or text to DoclingDocument.

    Args:
        config: Application configuration.
        source: One of:
            - Path: Local file path to convert
            - str (URL): HTTP/HTTPS URL to download and convert
            - str (text): Raw text content to convert
        format: The format of text content ("md", "html", or "plain").
            Defaults to "md". Use "plain" for plain text without parsing.
            Only used when source is raw text (not a file path or URL).
            Files and URLs determine format from extension/content-type.
        source_uri: Origin URI used by docling's HTML/Markdown backends to
            resolve relative `<img src="/path">` references. When omitted,
            defaults to the URL (URL ingest) or `file://` URI (file ingest);
            raw text input has no origin so no default is derived.

    Returns:
        DoclingDocument from the converted source.

    Raises:
        ValueError: If the file doesn't exist or has unsupported extension.
        httpx.RequestError: If URL download fails.
    """
    converter = get_converter(config)

    async def _convert_file(
        file_path: Path, effective_uri: str | None
    ) -> "DoclingDocument":
        """Dispatch through split-and-merge for large PDFs when configured,
        otherwise call the converter directly."""
        if file_path.suffix.lower() == ".pdf" and config.processing.split_pages > 0:
            from haiku.rag.converters.pdf_split import convert_pdf_with_splitting

            return await convert_pdf_with_splitting(
                converter, file_path, effective_uri, config.processing.split_pages
            )
        return await converter.convert_file(file_path, source_uri=effective_uri)

    # Path object - convert file directly
    if isinstance(source, Path):
        if not source.exists():
            raise UnsupportedSourceError(f"File does not exist: {source}")
        if source.suffix.lower() not in converter.supported_extensions:
            raise UnsupportedSourceError(f"Unsupported file extension: {source.suffix}")
        effective_uri = source_uri or source.absolute().as_uri()
        doc = await _convert_file(source, effective_uri)
        _warn_if_descriptions_missing(config, doc, str(source))
        return doc

    # String - check if URL or text
    parsed = urlparse(source)

    if parsed.scheme in ("http", "https"):
        # URL - download and convert
        async with httpx.AsyncClient() as http:
            response = await http.get(source)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            file_extension = get_extension_from_content_type_or_url(
                source, content_type
            )

            if file_extension not in converter.supported_extensions:
                raise UnsupportedSourceError(
                    f"Unsupported content type/extension: {content_type}/{file_extension}"
                )

            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=file_extension, delete=False
            ) as temp_file:
                temp_file.write(response.content)
                temp_file.flush()
                temp_path = Path(temp_file.name)

            try:
                effective_uri = source_uri or source
                doc = await _convert_file(temp_path, effective_uri)
                _warn_if_descriptions_missing(config, doc, source)
                return doc
            finally:
                temp_path.unlink(missing_ok=True)

    elif parsed.scheme == "file":
        # file:// URI
        file_path = Path(parsed.path)
        if not file_path.exists():
            raise UnsupportedSourceError(f"File does not exist: {file_path}")
        if file_path.suffix.lower() not in converter.supported_extensions:
            raise UnsupportedSourceError(
                f"Unsupported file extension: {file_path.suffix}"
            )
        effective_uri = source_uri or file_path.absolute().as_uri()
        doc = await _convert_file(file_path, effective_uri)
        _warn_if_descriptions_missing(config, doc, str(file_path))
        return doc

    else:
        # Raw text content — HTML and markdown can still embed pictures
        # via <img>/![](...) so the same description check applies.
        doc = await converter.convert_text(source, format=format, source_uri=source_uri)
        _warn_if_descriptions_missing(config, doc, "<text input>")
        return doc


def _merge_picture_chunks(
    docling_document: "DoclingDocument",
    text_chunks: list[Chunk],
    document_id: str | None,
    existing_picture_data: dict[str, bytes] | None,
    fast_picture_text: bool,
) -> list[Chunk]:
    picture_chunks = build_picture_chunks(
        docling_document,
        document_id=document_id,
        existing_picture_data=existing_picture_data,
        fast_picture_text=fast_picture_text,
    )

    if not picture_chunks:
        for i, c in enumerate(text_chunks):
            c.order = i
        return text_chunks

    positions = {
        item.self_ref: pos
        for pos, (item, _level) in enumerate(docling_document.iterate_items())
    }

    def first_pos(c: Chunk) -> int:
        refs = (c.metadata or {}).get("doc_item_refs") or []
        return positions.get(refs[0], len(positions)) if refs else len(positions)

    merged = sorted(text_chunks + picture_chunks, key=first_pos)
    for i, c in enumerate(merged):
        c.order = i
    return merged


async def chunk(
    config: AppConfig,
    docling_document: "DoclingDocument",
    *,
    embedder: "EmbedderWrapper",
    existing_picture_data: dict[str, bytes] | None = None,
    document_id: str | None = None,
) -> list[Chunk]:
    """Chunk a DoclingDocument into Chunks.

    When the configured embedder supports images, also emit one synthetic
    Chunk per ``PictureItem`` with available bytes (see ``build_picture_chunks``)
    and merge them with text chunks in structural (``iterate_items()``) order.
    ``chunk.order`` is the index in the merged list.

    ``existing_picture_data`` (snapshot keyed by ``self_ref``) supplies bytes
    for pictures whose ``image.uri`` has been stripped — used by the rebuild
    path where the docling is loaded from the stored blob.
    """
    from haiku.rag.chunkers import get_chunker

    chunker = get_chunker(config)
    text_chunks = await chunker.chunk(docling_document)

    if not embedder.supports_images:
        for i, c in enumerate(text_chunks):
            c.order = i
        return text_chunks

    return await asyncio.to_thread(
        _merge_picture_chunks,
        docling_document,
        text_chunks,
        document_id,
        existing_picture_data,
        config.processing.fast_picture_text,
    )


def build_picture_chunks(
    docling_document: "DoclingDocument",
    *,
    document_id: str | None = None,
    existing_picture_data: dict[str, bytes] | None = None,
    fast_picture_text: bool = True,
) -> list[Chunk]:
    """Emit one synthetic ``Chunk`` per ``PictureItem`` with available bytes.

    Bytes come from ``picture.image.uri`` (live data URI on a freshly-converted
    docling) or from ``existing_picture_data`` keyed by ``self_ref`` (snapshot
    taken before a delete-and-re-extract cycle, when the live docling has had
    its picture URIs stripped). Pictures with no available bytes are skipped.

    The bytes ride on ``Chunk._picture_data`` (a PrivateAttr — not serialized)
    so ``embed_chunks`` can route them through ``embed_image``. The
    ``order`` field is left at its default (0); the caller (``chunk()``)
    reassigns it after merging with text chunks in structural order.
    """
    from haiku.rag.store.models.document_item import (
        _decode_picture_bytes,
        extract_item_text,
    )

    existing = existing_picture_data or {}
    chunks: list[Chunk] = []

    for picture in docling_document.pictures:
        picture_data = _decode_picture_bytes(picture)
        if picture_data is None:
            picture_data = existing.get(picture.self_ref)
        if picture_data is None:
            continue

        text = (
            extract_item_text(
                picture, docling_document, fast_picture_text=fast_picture_text
            )
            or ""
        )

        page_numbers: list[int] = []
        for p in picture.prov:
            if p.page_no not in page_numbers:
                page_numbers.append(p.page_no)

        metadata = {
            "doc_item_refs": [picture.self_ref],
            "labels": ["picture"],
            "page_numbers": sorted(page_numbers),
            "headings": None,
        }
        chunk = Chunk(
            document_id=document_id,
            content=text,
            metadata=metadata,
        )
        chunk._picture_data = picture_data
        chunks.append(chunk)

    return chunks


async def ensure_chunks_embedded(
    config: AppConfig, chunks: list[Chunk], embedder: "EmbedderWrapper"
) -> list[Chunk]:
    """Ensure all chunks have embeddings, embedding any that don't.

    Chunks that already have embeddings are passed through unchanged; missing
    embeddings are filled in in-place in the returned list (preserving order).
    """
    from haiku.rag.embeddings import embed_chunks

    chunks_to_embed = [c for c in chunks if c.embedding is None]

    if not chunks_to_embed:
        return chunks

    embedded = await embed_chunks(chunks_to_embed, embedder, config)

    # Build result maintaining original order
    embedded_map = {(c.content, c.order): c for c in embedded}
    result = []
    for ch in chunks:
        if ch.embedding is not None:
            result.append(ch)
        else:
            result.append(embedded_map[(ch.content, ch.order)])

    return result


def get_extension_from_content_type_or_url(url: str, content_type: str) -> str:
    """Determine file extension from HTTP Content-Type header or URL suffix.

    Returns the mapped extension for known content types, falling back to the
    URL path suffix, and finally `.html` for generic web content.
    """
    content_type_map = {
        "text/html": ".html",
        "text/plain": ".txt",
        "text/markdown": ".md",
        "application/pdf": ".pdf",
        "application/json": ".json",
        "text/csv": ".csv",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    }

    for ct, ext in content_type_map.items():
        if ct in content_type:
            return ext

    parsed_url = urlparse(url)
    path = Path(parsed_url.path)
    if path.suffix:
        return path.suffix.lower()

    return ".html"
