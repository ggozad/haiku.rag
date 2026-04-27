import hashlib
import mimetypes
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from haiku.rag.client.processing import ensure_chunks_embedded
from haiku.rag.client.titles import resolve_title
from haiku.rag.converters import get_converter
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.models.document import Document
from haiku.rag.store.models.document_item import extract_items

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.client import HaikuRAG


async def _store_document_with_chunks(
    client: "HaikuRAG",
    document: Document,
    chunks: list[Chunk],
    docling_document: "DoclingDocument",
) -> Document:
    """Store a document with chunks, embedding any that lack embeddings.

    Handles versioning/rollback on failure.
    """
    # Ensure all chunks have embeddings before storing
    chunks = await ensure_chunks_embedded(client._config, chunks)

    # Snapshot table versions for versioned rollback (if supported)
    versions = await client.store.current_table_versions()

    # Create the document
    created_doc = await client.document_repository.create(document)

    try:
        assert created_doc.id is not None, (
            "Document ID should not be None after creation"
        )
        # Set document_id and order for all chunks
        for order, chunk in enumerate(chunks):
            chunk.document_id = created_doc.id
            chunk.order = order

        # Batch create all chunks in a single operation
        await client.chunk_repository.create(chunks)

        # Extract and store document items for context expansion
        items = extract_items(created_doc.id, docling_document)
        await client.document_item_repository.create_items(created_doc.id, items)

        # Vacuum old versions in background (non-blocking) if auto_vacuum enabled
        if client._config.storage.auto_vacuum:
            client._schedule_vacuum()

        return created_doc
    except Exception:
        # Roll back to the captured versions and re-raise
        await client.store.restore_table_versions(versions)
        raise


async def _update_document_with_chunks(
    client: "HaikuRAG",
    document: Document,
    chunks: list[Chunk],
    docling_document: "DoclingDocument | None" = None,
) -> Document:
    """Update a document and replace its chunks, embedding any that lack embeddings.

    Handles versioning/rollback on failure. When `docling_document` is None,
    existing items are preserved.
    """
    assert document.id is not None, "Document ID is required for update"

    chunks = await ensure_chunks_embedded(client._config, chunks)

    versions = await client.store.current_table_versions()

    # Delete existing chunks before writing new ones
    await client.chunk_repository.delete_by_document_id(document.id)

    try:
        updated_doc = await client.document_repository.update(document)

        assert updated_doc.id is not None
        for order, chunk in enumerate(chunks):
            chunk.document_id = updated_doc.id
            chunk.order = order

        await client.chunk_repository.create(chunks)

        # Replace document items when a new DoclingDocument is provided.
        # Snapshot existing picture bytes first so they survive the
        # delete-and-re-extract cycle when the live docling has already had
        # its picture URIs stripped (rebuild / round-trip scenarios).
        if docling_document is not None:
            existing_picture_data = (
                await client.document_item_repository.get_all_picture_data(
                    updated_doc.id
                )
            )
            await client.document_item_repository.delete_by_document_id(updated_doc.id)
            items = extract_items(
                updated_doc.id,
                docling_document,
                existing_picture_data=existing_picture_data,
            )
            await client.document_item_repository.create_items(updated_doc.id, items)

        if client._config.storage.auto_vacuum:
            client._schedule_vacuum()

        return updated_doc
    except Exception:
        await client.store.restore_table_versions(versions)
        raise


async def create_document(
    client: "HaikuRAG",
    content: str,
    uri: str | None = None,
    title: str | None = None,
    metadata: dict | None = None,
    format: str = "md",
) -> Document:
    """Create a new document from text content.

    Converts the content, chunks it, and generates embeddings.
    """
    from haiku.rag.embeddings import embed_chunks

    # Convert → Chunk → Embed using primitives
    converter = get_converter(client._config)
    docling_document = await converter.convert_text(content, format=format)
    chunks = await client.chunk(docling_document)
    embedded_chunks = await embed_chunks(chunks, client._config)

    # Store markdown export as content for better display/readability.
    # The original is preserved in docling_document.
    stored_content = docling_document.export_to_markdown()

    if title is None:
        title = await resolve_title(client._config, docling_document, stored_content)

    document = Document(
        content=stored_content,
        uri=uri,
        title=title,
        metadata=metadata or {},
    )
    document.set_docling(docling_document)

    return await _store_document_with_chunks(
        client, document, embedded_chunks, docling_document
    )


async def import_document(
    client: "HaikuRAG",
    docling_document: "DoclingDocument",
    chunks: list[Chunk],
    uri: str | None = None,
    title: str | None = None,
    metadata: dict | None = None,
) -> Document:
    """Import a pre-processed document with chunks.

    Use this when conversion, chunking, and embedding were done externally.
    Chunks without embeddings will be automatically embedded.
    """
    content = docling_document.export_to_markdown()
    if title is None:
        title = await resolve_title(client._config, docling_document, content)

    document = Document(
        content=content,
        uri=uri,
        title=title,
        metadata=metadata or {},
    )
    document.set_docling(docling_document)

    return await _store_document_with_chunks(client, document, chunks, docling_document)


async def create_document_from_source(
    client: "HaikuRAG",
    source: str | Path,
    title: str | None = None,
    metadata: dict | None = None,
) -> Document | list[Document]:
    """Create or update document(s) from a file path, directory, or URL.

    Checks if a document with the same URI already exists:
    - If MD5 is unchanged, returns existing document
    - If MD5 changed, updates the document
    - If no document exists, creates a new one

    Returns a single Document for files/URLs, a list for directories.
    """
    metadata = metadata or {}

    source_str = str(source)
    parsed_url = urlparse(source_str)
    if parsed_url.scheme in ("http", "https"):
        return await _create_or_update_document_from_url(
            client, source_str, title=title, metadata=metadata
        )
    elif parsed_url.scheme == "file":
        source_path = Path(parsed_url.path)
    else:
        source_path = Path(source) if isinstance(source, str) else source

    if source_path.is_dir():
        from haiku.rag.monitor import FileFilter

        documents = []
        filter = FileFilter(
            ignore_patterns=client._config.monitor.ignore_patterns or None,
            include_patterns=client._config.monitor.include_patterns or None,
        )
        for path in source_path.rglob("*"):
            if path.is_file() and filter.include_file(str(path)):
                doc = await _create_document_from_file(
                    client, path, title=None, metadata=metadata
                )
                documents.append(doc)
        return documents

    return await _create_document_from_file(
        client, source_path, title=title, metadata=metadata
    )


async def _create_document_from_file(
    client: "HaikuRAG",
    source_path: Path,
    title: str | None = None,
    metadata: dict | None = None,
) -> Document:
    """Create or update a document from a single file path."""
    from haiku.rag.embeddings import embed_chunks

    metadata = metadata or {}

    converter = get_converter(client._config)
    if source_path.suffix.lower() not in converter.supported_extensions:
        raise ValueError(f"Unsupported file extension: {source_path.suffix}")

    if not source_path.exists():
        raise ValueError(f"File does not exist: {source_path}")

    uri = source_path.absolute().as_uri()
    md5_hash = hashlib.md5(source_path.read_bytes(), usedforsecurity=False).hexdigest()

    content_type, _ = mimetypes.guess_type(str(source_path))
    if not content_type:
        content_type = "application/octet-stream"
    metadata.update({"contentType": content_type, "md5": md5_hash})

    # Check if document already exists
    existing_doc = await client.get_document_by_uri(uri)
    if existing_doc and existing_doc.metadata.get("md5") == md5_hash:
        # MD5 unchanged; update title/metadata if provided
        updated = False
        if title is not None and title != existing_doc.title:
            existing_doc.title = title
            updated = True

        merged_metadata = {**(existing_doc.metadata or {}), **metadata}
        if merged_metadata != existing_doc.metadata:
            existing_doc.metadata = merged_metadata
            updated = True

        if updated:
            return await client.document_repository.update(existing_doc)
        return existing_doc

    # Convert → Chunk → Embed
    docling_document = await client.convert(source_path)
    chunks = await client.chunk(docling_document)
    embedded_chunks = await embed_chunks(chunks, client._config)

    stored_content = docling_document.export_to_markdown()

    if existing_doc:
        # Update existing document and rechunk
        existing_doc.content = stored_content
        existing_doc.metadata = metadata
        existing_doc.set_docling(docling_document)
        if title is not None:
            existing_doc.title = title
        elif existing_doc.title is None:
            existing_doc.title = await resolve_title(
                client._config, docling_document, stored_content
            )
        return await _update_document_with_chunks(
            client, existing_doc, embedded_chunks, docling_document
        )
    else:
        if title is None:
            title = await resolve_title(
                client._config, docling_document, stored_content
            )
        document = Document(
            content=stored_content,
            uri=uri,
            title=title,
            metadata=metadata,
        )
        document.set_docling(docling_document)
        return await _store_document_with_chunks(
            client, document, embedded_chunks, docling_document
        )


async def _create_or_update_document_from_url(
    client: "HaikuRAG",
    url: str,
    title: str | None = None,
    metadata: dict | None = None,
) -> Document:
    """Create or update a document from a URL by downloading and parsing the content."""
    from haiku.rag.client.processing import get_extension_from_content_type_or_url
    from haiku.rag.embeddings import embed_chunks

    metadata = metadata or {}

    converter = get_converter(client._config)
    supported_extensions = converter.supported_extensions

    async with httpx.AsyncClient() as http:
        response = await http.get(url)
        response.raise_for_status()

        md5_hash = hashlib.md5(response.content).hexdigest()

        content_type = response.headers.get("content-type", "").lower()

        # Check if document already exists
        existing_doc = await client.get_document_by_uri(url)
        if existing_doc and existing_doc.metadata.get("md5") == md5_hash:
            updated = False
            if title is not None and title != existing_doc.title:
                existing_doc.title = title
                updated = True

            metadata.update({"contentType": content_type, "md5": md5_hash})
            merged_metadata = {**(existing_doc.metadata or {}), **metadata}
            if merged_metadata != existing_doc.metadata:
                existing_doc.metadata = merged_metadata
                updated = True

            if updated:
                return await client.document_repository.update(existing_doc)
            return existing_doc

        file_extension = get_extension_from_content_type_or_url(url, content_type)

        if file_extension not in supported_extensions:
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
            docling_document = await client.convert(temp_path)
            chunks = await client.chunk(docling_document)
            embedded_chunks = await embed_chunks(chunks, client._config)
        finally:
            temp_path.unlink(missing_ok=True)

        metadata.update({"contentType": content_type, "md5": md5_hash})

        stored_content = docling_document.export_to_markdown()

        if existing_doc:
            existing_doc.content = stored_content
            existing_doc.metadata = metadata
            existing_doc.set_docling(docling_document)
            if title is not None:
                existing_doc.title = title
            elif existing_doc.title is None:
                existing_doc.title = await resolve_title(
                    client._config, docling_document, stored_content
                )
            return await _update_document_with_chunks(
                client, existing_doc, embedded_chunks, docling_document
            )
        else:
            if title is None:
                title = await resolve_title(
                    client._config, docling_document, stored_content
                )
            document = Document(
                content=stored_content,
                uri=url,
                title=title,
                metadata=metadata,
            )
            document.set_docling(docling_document)
            return await _store_document_with_chunks(
                client, document, embedded_chunks, docling_document
            )


async def update_document(
    client: "HaikuRAG",
    document_id: str,
    content: str | None = None,
    metadata: dict | None = None,
    chunks: list[Chunk] | None = None,
    title: str | None = None,
    docling_document: "DoclingDocument | None" = None,
) -> Document:
    """Update a document by ID.

    Updates specified fields. When content or docling_document is provided, the
    document is rechunked and re-embedded. Updates to only metadata or title
    skip rechunking for efficiency.

    Raises:
        ValueError: If document not found, or if both content and
            docling_document are provided.
    """
    from haiku.rag.embeddings import embed_chunks

    # Validate: content and docling_document are mutually exclusive
    if content is not None and docling_document is not None:
        raise ValueError(
            "content and docling_document are mutually exclusive. "
            "Provide one or the other, not both."
        )

    existing_doc = await client.get_document_by_id(document_id)
    if existing_doc is None:
        raise ValueError(f"Document with ID {document_id} not found")

    if title is not None:
        existing_doc.title = title
    if metadata is not None:
        existing_doc.metadata = metadata

    # Only metadata/title update - no rechunking needed
    if content is None and chunks is None and docling_document is None:
        return await client.document_repository.update(existing_doc)

    # Custom chunks provided - use them as-is
    if chunks is not None:
        if docling_document is not None:
            existing_doc.content = docling_document.export_to_markdown()
            existing_doc.set_docling(docling_document)
        elif content is not None:
            existing_doc.content = content

        return await _update_document_with_chunks(
            client, existing_doc, chunks, docling_document
        )

    # DoclingDocument provided without chunks - chunk and embed
    if docling_document is not None:
        existing_doc.content = docling_document.export_to_markdown()
        existing_doc.set_docling(docling_document)

        new_chunks = await client.chunk(docling_document)
        embedded_chunks = await embed_chunks(new_chunks, client._config)
        return await _update_document_with_chunks(
            client, existing_doc, embedded_chunks, docling_document
        )

    # Content provided without chunks - convert, chunk, and embed
    assert content is not None
    existing_doc.content = content
    converter = get_converter(client._config)
    converted_docling = await converter.convert_text(existing_doc.content, format="md")
    existing_doc.set_docling(converted_docling)

    new_chunks = await client.chunk(converted_docling)
    embedded_chunks = await embed_chunks(new_chunks, client._config)
    return await _update_document_with_chunks(
        client, existing_doc, embedded_chunks, converted_docling
    )


def check_source_accessible(uri: str) -> bool:
    """Check if a document's source URI is accessible."""
    parsed_url = urlparse(uri)
    try:
        if parsed_url.scheme == "file":
            return Path(parsed_url.path).exists()
        elif parsed_url.scheme in ("http", "https"):
            return True
        return False
    except Exception:
        return False
