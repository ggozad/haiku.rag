import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

from haiku.rag.client.exceptions import UnsupportedSourceError
from haiku.rag.client.processing import (
    ensure_chunks_embedded,
    get_extension_from_content_type_or_url,
)
from haiku.rag.client.titles import resolve_title
from haiku.rag.converters import get_converter
from haiku.rag.ingester.sources import (
    FetchResult,
    resolve_adhoc_fetcher,
    resolve_configured_source,
)
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.models.document import Document
from haiku.rag.store.models.document_item import extract_items
from haiku.rag.telemetry import logfire

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.client import HaikuRAG
    from haiku.rag.ingester.sources.base import Source


def parent_uri_filter(parent_uri: str) -> str:
    """SQL `WHERE` clause matching documents whose ``metadata.parent_uri``
    equals ``parent_uri``. ``metadata`` is stored as a JSON string produced by
    the standard library's ``json.dumps`` (which inserts ``": "`` between key
    and value), so the match is a substring search over that serialized form —
    escape JSON-meaningful chars in the URI, then SQL-escape single quotes."""
    json_fragment = json.dumps(parent_uri)[1:-1].replace("'", "''")
    return f'metadata LIKE \'%"parent_uri": "{json_fragment}"%\''


async def _store_document_with_chunks(
    client: "HaikuRAG",
    document: Document,
    chunks: list[Chunk],
    docling_document: "DoclingDocument",
) -> Document:
    """Store a document with chunks, embedding any that lack embeddings.

    Handles versioning/rollback on failure.
    """
    chunks = await ensure_chunks_embedded(client._config, chunks)

    versions = await client.store.current_table_versions()

    created_doc = await client.document_repository.create(document)

    try:
        assert created_doc.id is not None, (
            "Document ID should not be None after creation"
        )
        for order, chunk in enumerate(chunks):
            chunk.document_id = created_doc.id
            chunk.order = order

        await client.chunk_repository.create(chunks)

        items = extract_items(created_doc.id, docling_document)
        await client.document_item_repository.create_items(created_doc.id, items)

        if client._config.storage.auto_vacuum:
            client._schedule_vacuum()

        return created_doc
    except Exception:
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

    # Snapshot existing picture bytes before deleting items so the post-delete
    # extract_items can merge them back when the live docling has had its
    # picture URIs stripped (rebuild / re-extract via the stored blob).
    existing_picture_data: dict[str, bytes] | None = None
    if docling_document is not None:
        existing_picture_data = (
            await client.document_item_repository.get_all_picture_data(document.id)
        )

    chunks = await ensure_chunks_embedded(client._config, chunks)

    versions = await client.store.current_table_versions()

    await client.chunk_repository.delete_by_document_id(document.id)

    try:
        updated_doc = await client.document_repository.update(document)

        assert updated_doc.id is not None
        for order, chunk in enumerate(chunks):
            chunk.document_id = updated_doc.id
            chunk.order = order

        await client.chunk_repository.create(chunks)

        if docling_document is not None:
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

    converter = get_converter(client._config)
    docling_document = await converter.convert_text(content, format=format)
    chunks = await client.chunk(docling_document)
    embedded_chunks = await embed_chunks(chunks, client._config)

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


async def _refresh_doc_metadata(
    client: "HaikuRAG",
    doc: Document,
    *,
    title: str | None,
    user_metadata: dict,
    source_metadata: dict | None,
) -> Document:
    """Update a document's title + metadata without re-chunking. Used by the
    cheap revision and MD5 short-circuits in create_document_from_source."""
    updated = False
    if title is not None and title != doc.title:
        doc.title = title
        updated = True

    merged = {**(doc.metadata or {}), **user_metadata}
    if source_metadata:
        merged.update(source_metadata)
    if merged != doc.metadata:
        doc.metadata = merged
        updated = True

    if updated:
        return await client.document_repository.update(doc)
    return doc


async def _ingest_fetch_result(
    client: "HaikuRAG",
    result: FetchResult,
    *,
    title: str | None,
    user_metadata: dict,
    stored_uri: str,
    existing_doc: Document | None,
) -> Document:
    """Convert / chunk / embed / store a fetched document. Replaces an
    existing document if one is supplied."""
    from haiku.rag.embeddings import embed_chunks

    converter = get_converter(client._config)
    file_extension = get_extension_from_content_type_or_url(
        result.uri, result.content_type
    )
    if file_extension not in converter.supported_extensions:
        raise UnsupportedSourceError(
            f"Unsupported content type/extension: {result.content_type}/{file_extension}"
        )

    source_metadata: dict = {
        "content_type": result.content_type,
        "md5": result.content_hash,
        **result.extra_metadata,
    }
    if result.revision is not None:
        source_metadata["source_revision"] = result.revision

    if result.disk_path is not None:
        target_path = result.disk_path
        cleanup_path: Path | None = None
    else:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=file_extension, delete=False
        ) as temp_file:
            temp_file.write(result.body)
            temp_file.flush()
            target_path = Path(temp_file.name)
            cleanup_path = target_path

    try:
        with logfire.span("document.convert", uri=result.uri):
            docling_document = await client.convert(target_path, source_uri=result.uri)
        with logfire.span("document.chunk", uri=result.uri) as chunk_span:
            chunks = await client.chunk(docling_document)
            chunk_span.set_attribute("chunks_created", len(chunks))
        with logfire.span("document.embed", uri=result.uri):
            embedded_chunks = await embed_chunks(chunks, client._config)
    finally:
        if cleanup_path is not None:
            cleanup_path.unlink(missing_ok=True)

    stored_content = docling_document.export_to_markdown()
    final_metadata = {**user_metadata, **source_metadata}

    if existing_doc:
        existing_doc.content = stored_content
        existing_doc.metadata = final_metadata
        existing_doc.set_docling(docling_document)
        if title is not None:
            existing_doc.title = title
        elif existing_doc.title is None:
            existing_doc.title = await resolve_title(
                client._config, docling_document, stored_content
            )
        with logfire.span("document.store", uri=result.uri, op="update") as store_span:
            updated = await _update_document_with_chunks(
                client, existing_doc, embedded_chunks, docling_document
            )
            store_span.set_attribute("document_id", updated.id)
        return updated

    if title is None:
        title = await resolve_title(client._config, docling_document, stored_content)
    document = Document(
        content=stored_content,
        uri=stored_uri,
        title=title,
        metadata=final_metadata,
    )
    document.set_docling(docling_document)
    with logfire.span("document.store", uri=result.uri, op="create") as store_span:
        created = await _store_document_with_chunks(
            client, document, embedded_chunks, docling_document
        )
        store_span.set_attribute("document_id", created.id)
    return created


async def create_document_from_source(
    client: "HaikuRAG",
    source: str | Path,
    title: str | None = None,
    metadata: dict | None = None,
    uri: str | None = None,
    storage_options: dict[str, str] | None = None,
    sources: "list[Source] | None" = None,
    source_id: str | None = None,
) -> Document | list[Document]:
    """Create or update document(s) from a file path, directory, or URL.

    Checks if a document with the same URI already exists:
    - If MD5 is unchanged, returns existing document
    - If MD5 changed, updates the document
    - If no document exists, creates a new one

    If ``uri`` is provided, it overrides the URI auto-derived from the source
    (which is normally ``file://`` for local files or the URL for remote
    sources). Not supported for directory sources, which produce one document
    per file.

    Returns a single Document for files/URLs, a list for directories.
    """
    metadata = metadata or {}

    source_str = str(source)
    parsed_url = urlparse(source_str)

    # Directory case: recurse with the existing FS filter and produce one
    # document per file. Remote schemes (http/s3) never hit this branch.
    if parsed_url.scheme in ("", "file"):
        # file:// URIs URL-encode special characters ([, ], spaces, etc.);
        # unquote to get the real filesystem path before any stat/rglob.
        local_path = (
            Path(unquote(parsed_url.path))
            if parsed_url.scheme == "file"
            else (Path(source) if isinstance(source, str) else source)
        )
        if local_path.is_dir():
            if uri is not None:
                raise UnsupportedSourceError(
                    "uri override is not supported for directory sources; each file "
                    "produces its own document with its own auto-derived URI."
                )
            from haiku.rag.ingester.sources.filter import FileFilter

            # One-shot CLI directory ingest uses the converter's supported
            # extensions but no include/ignore patterns. For pattern-based
            # filtering use `haiku-ingester serve` with an FS source.
            documents: list[Document] = []
            filter = FileFilter()
            for child in local_path.rglob("*"):
                if child.is_file() and filter.include_file(str(child)):
                    doc = await create_document_from_source(
                        client, child, title=None, metadata=metadata
                    )
                    assert isinstance(doc, Document)
                    documents.append(doc)
            return documents

        if not local_path.exists():
            raise UnsupportedSourceError(f"File does not exist: {local_path}")

        # Match the old _create_document_from_file behaviour: fail fast on
        # unsupported extension before reading any bytes.
        converter = get_converter(client._config)
        if local_path.suffix.lower() not in converter.supported_extensions:
            raise UnsupportedSourceError(
                f"Unsupported file extension: {local_path.suffix}"
            )

    # Worker jobs carry source_id from the poller; strict lookup so a
    # renamed/removed source surfaces as a DLQ instead of silently dropping
    # credentials. Ad-hoc CLI calls (no source_id) fall back to scheme-based
    # adapters when no configured source matches.
    if source_id is not None:
        fetcher = resolve_configured_source(source_str, source_id, sources)
    else:
        fetcher = resolve_adhoc_fetcher(
            source_str, sources=sources, storage_options=storage_options
        )

    # The stored URI is what we look up + persist by. For an explicit uri
    # override, use it as-is. For a file:// input the source string is
    # already canonical (URL-encoded); round-tripping via Path.as_uri()
    # would double-encode any escapes like %5B. For bare paths, canonicalize.
    if uri is not None:
        stored_uri = uri
    elif parsed_url.scheme == "file":
        stored_uri = source_str
    elif parsed_url.scheme == "":
        stored_uri = Path(source_str).absolute().as_uri()
    else:
        stored_uri = source_str

    existing_doc = await client.get_document_by_uri(stored_uri)

    # Cheap revision-based short-circuit: only worth a HEAD when we have a
    # stored revision to compare against. All sources persist their native
    # revision (mtime_ns for FS, ETag for S3, ETag/Last-Modified for HTTP)
    # under the canonical "source_revision" metadata key.
    stored_revision = (
        (existing_doc.metadata or {}).get("source_revision") if existing_doc else None
    )
    if existing_doc and stored_revision:
        current_revision = await fetcher.head(source_str)
        if current_revision == stored_revision:
            return await _refresh_doc_metadata(
                client,
                existing_doc,
                title=title,
                user_metadata=metadata,
                source_metadata=None,
            )

    with logfire.span("document.fetch", uri=source_str) as fetch_span:
        result = await fetcher.fetch(source_str)
        fetch_span.set_attribute("bytes", len(result.body))
        fetch_span.set_attribute("content_hash", result.content_hash)

    # MD5 short-circuit: the bytes are unchanged even if the revision wasn't.
    # Refresh the source-derived metadata (revision may have rolled) but skip
    # convert/embed/store entirely.
    if existing_doc and existing_doc.metadata.get("md5") == result.content_hash:
        source_meta: dict = {
            "content_type": result.content_type,
            "md5": result.content_hash,
            **result.extra_metadata,
        }
        if result.revision is not None:
            source_meta["source_revision"] = result.revision
        return await _refresh_doc_metadata(
            client,
            existing_doc,
            title=title,
            user_metadata=metadata,
            source_metadata=source_meta,
        )

    return await _ingest_fetch_result(
        client,
        result,
        title=title,
        user_metadata=metadata,
        stored_uri=stored_uri,
        existing_doc=existing_doc,
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

    if content is None and chunks is None and docling_document is None:
        return await client.document_repository.update(existing_doc)

    if chunks is not None:
        if docling_document is not None:
            existing_doc.content = docling_document.export_to_markdown()
            existing_doc.set_docling(docling_document)
        elif content is not None:
            existing_doc.content = content

        return await _update_document_with_chunks(
            client, existing_doc, chunks, docling_document
        )

    if docling_document is not None:
        existing_doc.content = docling_document.export_to_markdown()
        existing_doc.set_docling(docling_document)

        new_chunks = await client.chunk(docling_document)
        embedded_chunks = await embed_chunks(new_chunks, client._config)
        return await _update_document_with_chunks(
            client, existing_doc, embedded_chunks, docling_document
        )

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
        elif parsed_url.scheme in ("http", "https", "s3"):
            return True
        return False
    except Exception:
        return False
