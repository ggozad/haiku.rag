import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from haiku.rag.config import AppConfig
from haiku.rag.converters import get_converter
from haiku.rag.store.models.chunk import Chunk

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


async def convert(
    config: AppConfig, source: Path | str, *, format: str = "md"
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

    Returns:
        DoclingDocument from the converted source.

    Raises:
        ValueError: If the file doesn't exist or has unsupported extension.
        httpx.RequestError: If URL download fails.
    """
    converter = get_converter(config)

    # Path object - convert file directly
    if isinstance(source, Path):
        if not source.exists():
            raise ValueError(f"File does not exist: {source}")
        if source.suffix.lower() not in converter.supported_extensions:
            raise ValueError(f"Unsupported file extension: {source.suffix}")
        return await converter.convert_file(source)

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
                raise ValueError(
                    f"Unsupported content type/extension: {content_type}/{file_extension}"
                )

            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=file_extension, delete=False
            ) as temp_file:
                temp_file.write(response.content)
                temp_file.flush()
                temp_path = Path(temp_file.name)

            try:
                return await converter.convert_file(temp_path)
            finally:
                temp_path.unlink(missing_ok=True)

    elif parsed.scheme == "file":
        # file:// URI
        file_path = Path(parsed.path)
        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        if file_path.suffix.lower() not in converter.supported_extensions:
            raise ValueError(f"Unsupported file extension: {file_path.suffix}")
        return await converter.convert_file(file_path)

    else:
        # Treat as text content
        return await converter.convert_text(source, format=format)


async def chunk(config: AppConfig, docling_document: "DoclingDocument") -> list[Chunk]:
    """Chunk a DoclingDocument into Chunks.

    Returns chunks without embeddings or document_id. Each chunk's `order`
    field is set to its position in the list.
    """
    from haiku.rag.chunkers import get_chunker

    chunker = get_chunker(config)
    return await chunker.chunk(docling_document)


async def ensure_chunks_embedded(config: AppConfig, chunks: list[Chunk]) -> list[Chunk]:
    """Ensure all chunks have embeddings, embedding any that don't.

    Chunks that already have embeddings are passed through unchanged; missing
    embeddings are filled in in-place in the returned list (preserving order).
    """
    from haiku.rag.embeddings import embed_chunks

    chunks_to_embed = [c for c in chunks if c.embedding is None]

    if not chunks_to_embed:
        return chunks

    embedded = await embed_chunks(chunks_to_embed, config)

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
