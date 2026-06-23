import asyncio
import hashlib
import json
import logging
import mimetypes
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote, unquote, urlparse

from haiku.rag.client.exceptions import UnsupportedSourceError
from haiku.rag.client.processing import (
    ensure_chunks_embedded,
    get_extension_from_content_type_or_url,
)
from haiku.rag.client.titles import resolve_title
from haiku.rag.converters import get_converter
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.models.document import Document
from haiku.rag.store.models.document_item import DocumentItem, extract_items
from haiku.rag.telemetry import logfire

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.client import HaikuRAG
    from haiku.rag.ingester.metadata import MetadataProvider
    from haiku.rag.ingester.sources.base import FetchResult, Source

logger = logging.getLogger(__name__)


@dataclass
class DocumentImport:
    """A prepared document for batch import via ``import_documents``.

    Carries the same inputs as ``import_document``: a converted
    ``DoclingDocument``, its chunks (embeddings filled in if missing), and
    optional uri/title/metadata.
    """

    docling_document: "DoclingDocument"
    chunks: list[Chunk]
    uri: str | None = None
    title: str | None = None
    metadata: dict | None = field(default=None)


# Maximum length of an attachment chain rooted at a top-level ingest. With
# value 3, a PDF whose attachments contain PDFs which themselves contain
# PDFs is fully ingested (3 levels); a fourth nested level logs a warning
# and is skipped.
MAX_ATTACHMENT_DEPTH = 3

# Keys the source pipeline owns (content_type/md5/source_revision, which drive
# sync_state). A provider must not set them, or the metadata-only refresh path
# would let provider values overwrite the real source-derived ones. Stripped
# before provider metadata is merged into the document.
_RESERVED_METADATA_KEYS = frozenset({"content_type", "md5", "source_revision"})


def _prepare_document_from_docling_sync(
    document: Document, docling_document: "DoclingDocument"
) -> str:
    """Populate content/docling blobs from a DoclingDocument.

    This performs size-proportional serialization, JSON splitting, and
    compression via ``Document.set_docling``. Async ingestion paths should call
    it through ``_prepare_document_from_docling`` so large image-bearing
    documents do not block the event loop.
    """
    content = docling_document.export_to_markdown()
    document.content = content
    document.set_docling(docling_document)
    return content


async def _prepare_document_from_docling(
    document: Document, docling_document: "DoclingDocument"
) -> str:
    return await asyncio.to_thread(
        _prepare_document_from_docling_sync, document, docling_document
    )


def _write_fetch_body_sync(body: bytes, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=suffix, delete=False
    ) as temp_file:
        temp_file.write(body)
        temp_file.flush()
        return Path(temp_file.name)


async def _write_fetch_body(body: bytes, suffix: str) -> Path:
    return await asyncio.to_thread(_write_fetch_body_sync, body, suffix)


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
    chunks = await ensure_chunks_embedded(client._config, chunks, client.embedder)
    items = await asyncio.to_thread(extract_items, "", docling_document)

    async with client.store._write_lock:
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

            for item in items:
                item.document_id = created_doc.id
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

    chunks = await ensure_chunks_embedded(client._config, chunks, client.embedder)

    items: list[DocumentItem] | None = None
    if docling_document is not None:
        items = await asyncio.to_thread(
            extract_items, document.id, docling_document, existing_picture_data
        )

    async with client.store._write_lock:
        versions = await client.store.current_table_versions()

        try:
            updated_doc = await client.document_repository.update(document)

            assert updated_doc.id is not None
            for order, chunk in enumerate(chunks):
                chunk.document_id = updated_doc.id
                chunk.order = order

            await client.chunk_repository.replace_for_document(updated_doc.id, chunks)

            if items is not None:
                await client.document_item_repository.replace_for_document(
                    updated_doc.id, items
                )

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
    embedded_chunks = await embed_chunks(chunks, client.embedder, client._config)

    document = Document(
        content="",
        uri=uri,
        title=title,
        metadata=metadata or {},
    )
    stored_content = await _prepare_document_from_docling(document, docling_document)

    if title is None:
        document.title = await resolve_title(
            client._config, docling_document, stored_content
        )

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
    document = Document(
        content="",
        uri=uri,
        title=title,
        metadata=metadata or {},
    )
    content = await _prepare_document_from_docling(document, docling_document)
    if title is None:
        document.title = await resolve_title(client._config, docling_document, content)

    return await _store_document_with_chunks(client, document, chunks, docling_document)


async def _store_documents_with_chunks(
    client: "HaikuRAG",
    prepared: list[tuple[Document, list[Chunk], "DoclingDocument"]],
) -> list[Document]:
    """Store many documents with their chunks in a single table version each.

    Embeds any chunks that lack embeddings, then writes the documents, chunks,
    and document_items tables once apiece. Restores all tables on any failure.
    """
    embedded: list[list[Chunk]] = [
        await ensure_chunks_embedded(client._config, chunks, client.embedder)
        for _, chunks, _ in prepared
    ]

    def _extract_all_items():
        return [extract_items("", d) for _, _, d in prepared]

    all_item_lists = await asyncio.to_thread(_extract_all_items)

    async with client.store._write_lock:
        versions = await client.store.current_table_versions()

        created = await client.document_repository.create(
            [doc for doc, _, _ in prepared]
        )

        try:
            all_chunks: list[Chunk] = []
            all_items = []
            for doc, doc_chunks, item_list in zip(created, embedded, all_item_lists):
                assert doc.id is not None
                for order, chunk in enumerate(doc_chunks):
                    chunk.document_id = doc.id
                    chunk.order = order
                all_chunks.extend(doc_chunks)
                for item in item_list:
                    item.document_id = doc.id
                all_items.extend(item_list)

            await client.chunk_repository.create(all_chunks)
            await client.document_item_repository.create_all(all_items)

            if client._config.storage.auto_vacuum:
                client._schedule_vacuum()

            return created
        except Exception:
            await client.store.restore_table_versions(versions)
            raise


async def import_documents(
    client: "HaikuRAG",
    imports: list[DocumentImport],
) -> list[Document]:
    """Batch-import pre-processed documents with their chunks.

    The batch analog of ``import_document``: writes the documents, chunks, and
    document_items tables once each regardless of how many documents are
    imported. Chunks without embeddings are embedded automatically.
    """
    if not imports:
        return []

    prepared: list[tuple[Document, list[Chunk], DoclingDocument]] = []
    for item in imports:
        document = Document(
            content="",
            uri=item.uri,
            title=item.title,
            metadata=item.metadata or {},
        )
        content = await _prepare_document_from_docling(document, item.docling_document)
        if document.title is None:
            document.title = await resolve_title(
                client._config, item.docling_document, content
            )
        prepared.append((document, item.chunks, item.docling_document))

    return await _store_documents_with_chunks(client, prepared)


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
        result = await client.document_repository.update_meta(doc)
        # Reclaim the document_meta churn from rolling source_revision sweeps.
        # The vacuum is debounced, and document_meta is tiny, so this is cheap.
        if client._config.storage.auto_vacuum:
            client._schedule_vacuum()
        return result
    return doc


async def _provider_metadata(
    provider: "MetadataProvider | None",
    source_id: str,
    uri: str,
    result: "FetchResult",
) -> dict:
    if provider is None:
        return {}
    # Hand the provider an isolated copy: mutating the live FetchResult
    # (e.g. result.content_hash or result.extra_metadata) would feed the
    # MD5 short-circuit and source_meta, bypassing the reserved-key filter
    # that only guards the returned dict.
    provider_result = result.model_copy(deep=True)
    return {
        k: v
        for k, v in (await provider(source_id, uri, provider_result)).items()
        if k not in _RESERVED_METADATA_KEYS
    }


async def _ingest_fetch_result(
    client: "HaikuRAG",
    result: "FetchResult",
    *,
    title: str | None,
    user_metadata: dict,
    stored_uri: str,
    existing_doc: Document | None,
    depth: int = 0,
    filename: str | None = None,
) -> Document:
    """Convert / chunk / embed / store a fetched document. Replaces an
    existing document if one is supplied. ``depth`` tracks position in an
    attachment chain so the reconciliation step can bound recursion.

    ``filename``, when given, makes its suffix authoritative for the file
    extension (and thus the docling format), overriding the URI/content-type
    fallback. Callers pass it when ``result.uri`` cannot yield the right
    extension, e.g. embedded attachments whose name lives in a URI fragment."""
    from haiku.rag.embeddings import embed_chunks

    converter = get_converter(client._config)
    if filename is not None:
        file_extension = Path(filename).suffix.lower()
    else:
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
        target_path = await _write_fetch_body(result.body, file_extension)
        cleanup_path = target_path

    try:
        with logfire.span("document.convert", uri=result.uri):
            docling_document = await client.convert(target_path, source_uri=result.uri)
        with logfire.span("document.chunk", uri=result.uri) as chunk_span:
            chunks = await client.chunk(docling_document)
            chunk_span.set_attribute("chunks_created", len(chunks))
        with logfire.span("document.embed", uri=result.uri):
            embedded_chunks = await embed_chunks(
                chunks, client.embedder, client._config
            )
    finally:
        if cleanup_path is not None:
            cleanup_path.unlink(missing_ok=True)

    final_metadata = {**user_metadata, **source_metadata}

    if existing_doc:
        existing_doc.metadata = final_metadata
        stored_content = await _prepare_document_from_docling(
            existing_doc, docling_document
        )
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
        await _reconcile_pdf_attachments(client, updated, result.body, depth=depth)
        return updated

    document = Document(
        content="",
        uri=stored_uri,
        title=title,
        metadata=final_metadata,
    )
    stored_content = await _prepare_document_from_docling(document, docling_document)
    if document.title is None:
        document.title = await resolve_title(
            client._config, docling_document, stored_content
        )
    with logfire.span("document.store", uri=result.uri, op="create") as store_span:
        created = await _store_document_with_chunks(
            client, document, embedded_chunks, docling_document
        )
        store_span.set_attribute("document_id", created.id)
    await _reconcile_pdf_attachments(client, created, result.body, depth=depth)
    return created


def _extract_pdf_attachments(
    parent_body: bytes, parent_uri: str, *, depth: int
) -> dict[str, tuple[str, bytes, str, str]] | None:
    """Open the parent PDF and return its embedded attachments keyed by child
    URI. Returns ``None`` when the PDF can't be opened or the recursion depth
    cap is reached — in both cases the caller skips reconciliation entirely.

    Every pdfium call is held under ``PDFIUM_LOCK`` (shared with page slicing)
    because libpdfium's global C state is not thread-safe; concurrent access
    from another worker corrupts it and then fails valid PDFs with "Data format
    error" until the process restarts.
    """
    import pypdfium2 as pdfium

    from haiku.rag.converters.pdf_split import PDFIUM_LOCK

    with PDFIUM_LOCK:
        try:
            pdf = pdfium.PdfDocument(parent_body)
        except pdfium.PdfiumError as exc:
            logger.warning(
                "Cannot scan %s for embedded attachments: %s", parent_uri, exc
            )
            return None
        try:
            attachment_count = pdf.count_attachments()

            if depth + 1 >= MAX_ATTACHMENT_DEPTH:
                if attachment_count > 0:
                    logger.warning(
                        "Attachment depth cap (%d) reached at %s; skipping %d nested "
                        "attachment(s).",
                        MAX_ATTACHMENT_DEPTH,
                        parent_uri,
                        attachment_count,
                    )
                return None

            new_attachments: dict[str, tuple[str, bytes, str, str]] = {}
            for i in range(attachment_count):
                att = pdf.get_attachment(i)
                name = att.get_name()
                if not name:
                    continue
                data = bytes(att.get_data())
                child_uri = f"{parent_uri}#attachment={quote(name, safe='')}"
                content_type = (
                    mimetypes.guess_type(name)[0] or "application/octet-stream"
                )
                content_hash = hashlib.md5(data, usedforsecurity=False).hexdigest()
                new_attachments[child_uri] = (name, data, content_type, content_hash)
            return new_attachments
        finally:
            pdf.close()


async def _reconcile_pdf_attachments(
    client: "HaikuRAG",
    parent_doc: Document,
    parent_body: bytes,
    *,
    depth: int,
) -> None:
    """Diff the parent PDF's ``/EmbeddedFiles`` table against any children
    already linked via ``metadata.parent_uri`` and bring the child set in line:
    ingest additions, update changed bytes, cascade-delete removed names.

    Re-uses ``_ingest_fetch_result`` for each child so the standard conversion
    path runs uniformly — child PDFs recurse into this helper one level deeper,
    bounded by ``MAX_ATTACHMENT_DEPTH``.
    """
    if not client._config.processing.extract_pdf_attachments:
        return
    if not parent_doc.uri:
        return
    if (parent_doc.metadata or {}).get("content_type") != "application/pdf":
        return

    new_attachments = await asyncio.to_thread(
        _extract_pdf_attachments, parent_body, parent_doc.uri, depth=depth
    )
    if new_attachments is None:
        return

    existing = await client.list_documents(filter=parent_uri_filter(parent_doc.uri))
    existing_by_uri: dict[str, Document] = {d.uri: d for d in existing if d.uri}

    for child_uri, (name, data, content_type, content_hash) in new_attachments.items():
        existing_child = existing_by_uri.get(child_uri)
        if (
            existing_child
            and (existing_child.metadata or {}).get("md5") == content_hash
        ):
            continue

        from haiku.rag.ingester.sources.base import FetchResult

        child_fr = FetchResult(
            uri=child_uri,
            body=data,
            content_type=content_type,
            content_hash=content_hash,
            extra_metadata={"parent_uri": parent_doc.uri},
        )
        try:
            await _ingest_fetch_result(
                client,
                child_fr,
                title=None,
                user_metadata={},
                stored_uri=child_uri,
                existing_doc=existing_child,
                depth=depth + 1,
                filename=name,
            )
        except UnsupportedSourceError:
            logger.warning(
                "Skipping attachment %r in %s: unsupported extension %r "
                "(content type %r)",
                name,
                parent_doc.uri,
                Path(name).suffix.lower(),
                content_type,
            )

    for child_uri, child in existing_by_uri.items():
        if child_uri not in new_attachments and child.id:
            await client.delete_document(child.id)


async def create_document_from_source(
    client: "HaikuRAG",
    source: str | Path,
    title: str | None = None,
    metadata: dict | None = None,
    uri: str | None = None,
    storage_options: dict[str, str] | None = None,
    sources: "list[Source] | None" = None,
    source_id: str | None = None,
    metadata_provider: "MetadataProvider | None" = None,
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
                        client,
                        child,
                        title=None,
                        metadata=metadata,
                        sources=sources,
                        source_id=source_id,
                        metadata_provider=metadata_provider,
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
    from haiku.rag.ingester.sources import (
        resolve_adhoc_fetcher,
        resolve_configured_source,
    )

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

    provider_metadata = await _provider_metadata(
        metadata_provider, source_id or fetcher.source_id, source_str, result
    )
    user_metadata = {**metadata, **provider_metadata}

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
            user_metadata=user_metadata,
            source_metadata=source_meta,
        )

    return await _ingest_fetch_result(
        client,
        result,
        title=title,
        user_metadata=user_metadata,
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
        updated = await client.document_repository.update_meta(existing_doc)
        if client._config.storage.auto_vacuum:
            client._schedule_vacuum()
        return updated

    if chunks is not None:
        if docling_document is not None:
            await _prepare_document_from_docling(existing_doc, docling_document)
        elif content is not None:
            existing_doc.content = content

        return await _update_document_with_chunks(
            client, existing_doc, chunks, docling_document
        )

    if docling_document is not None:
        await _prepare_document_from_docling(existing_doc, docling_document)

        new_chunks = await client.chunk(docling_document)
        embedded_chunks = await embed_chunks(
            new_chunks, client.embedder, client._config
        )
        return await _update_document_with_chunks(
            client, existing_doc, embedded_chunks, docling_document
        )

    assert content is not None
    existing_doc.content = content
    converter = get_converter(client._config)
    converted_docling = await converter.convert_text(existing_doc.content, format="md")
    await _prepare_document_from_docling(existing_doc, converted_docling)

    new_chunks = await client.chunk(converted_docling)
    embedded_chunks = await embed_chunks(new_chunks, client.embedder, client._config)
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
