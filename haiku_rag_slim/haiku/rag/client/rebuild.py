import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import TYPE_CHECKING

from haiku.rag.client.documents import check_source_accessible
from haiku.rag.converters import get_converter
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.models.document import Document
from haiku.rag.store.models.document_item import extract_items
from haiku.rag.store.repositories.settings import SettingsRepository

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG, RebuildMode

logger = logging.getLogger(__name__)

_REBUILD_BATCH_SIZE = 50


async def rebuild_database(
    client: "HaikuRAG", mode: "RebuildMode | None" = None
) -> AsyncGenerator[str, None]:
    """Rebuild the database with the specified mode.

    Yields the ID of each document as it is processed.
    """
    from haiku.rag.client import RebuildMode

    if mode is None:
        mode = RebuildMode.FULL

    # Wait for any already-scheduled background vacuum before the destructive
    # table operations at the top of RECHUNK / FULL. Rebuild drops and
    # recreates tables (and creates indices); a concurrent optimize on the
    # same table fails with "CreateIndex transaction was preempted" from
    # lance. Note: FULL calls create_document_from_source inside its loop,
    # which may schedule *new* background vacuums — those run after the
    # destructive phase and are fine.
    await client._await_vacuum_tasks()

    # Update settings to current config
    settings_repo = SettingsRepository(client.store)
    await settings_repo.save_current_settings()

    # Light listing — id/uri/title/metadata only. Each rebuild function
    # fetches full content (including the multi-MB docling_pages blob) one
    # document at a time so a 1000-doc database doesn't pull ~15 GB of
    # blobs into memory before the loop starts.
    documents = await client.list_documents(include_content=False)

    if mode == RebuildMode.TITLE_ONLY:
        async for doc_id in _rebuild_title_only(client, documents):
            yield doc_id
    elif mode == RebuildMode.EMBED_ONLY:
        async for doc_id in _rebuild_embed_only(client, documents):
            yield doc_id
    elif mode == RebuildMode.RECHUNK:
        await client.chunk_repository.delete_all()
        await client.store.recreate_embeddings_table()
        async for doc_id in _rebuild_rechunk(client, documents):
            yield doc_id
    elif mode == RebuildMode.DESCRIPTIONS:
        await client.chunk_repository.delete_all()
        await client.store.recreate_embeddings_table()
        async for doc_id in _rebuild_descriptions(client, documents):
            yield doc_id
    else:  # FULL
        await client.chunk_repository.delete_all()
        await client.store.recreate_embeddings_table()
        async for doc_id in _rebuild_full(client, documents):
            yield doc_id

    # Final maintenance if auto_vacuum enabled. Swallowing only so that a
    # failed post-rebuild optimize doesn't mask a successful rebuild — but
    # log it so the failure is visible in the output.
    if client._config.storage.auto_vacuum:
        try:
            await client.store.vacuum()
        except Exception:
            logger.warning("Post-rebuild vacuum failed", exc_info=True)


async def _hydrate(
    client: "HaikuRAG", light_docs: list[Document]
) -> AsyncGenerator[Document, None]:
    """Yield fully-loaded documents one at a time from a light listing.

    The light listing in ``rebuild_database`` skips the multi-MB
    ``docling_document``/``docling_pages`` blobs; this helper fetches each
    full record on demand so peak memory stays at ~one document. Documents
    that disappeared between listing and processing are silently skipped.
    """
    for light_doc in light_docs:
        assert light_doc.id is not None
        doc = await client.get_document_by_id(light_doc.id)
        if doc is None:
            continue
        assert doc.id is not None
        yield doc


async def _rebuild_title_only(
    client: "HaikuRAG", documents: list[Document]
) -> AsyncGenerator[str, None]:
    """Generate titles for documents that don't have one."""
    untitled = [d for d in documents if d.title is None]
    async for doc in _hydrate(client, untitled):
        try:
            title = await client.generate_title(doc)
        except Exception:
            logger.warning(
                "Failed to generate title for document %s", doc.id, exc_info=True
            )
            continue
        if title is not None:
            doc.title = title
            await client.document_repository.update(doc)
            assert doc.id is not None
            yield doc.id


async def _rebuild_embed_only(
    client: "HaikuRAG", documents: list[Document]
) -> AsyncGenerator[str, None]:
    """Re-embed all chunks without changing chunk boundaries."""
    from haiku.rag.embeddings import contextualize

    # Collect all chunks with new embeddings
    all_chunk_data: list[tuple[str, dict]] = []

    for doc in documents:
        assert doc.id is not None
        chunks = await client.chunk_repository.get_by_document_id(doc.id)
        if not chunks:
            continue

        texts = contextualize(chunks)
        embeddings = await client.chunk_repository.embedder.embed_documents(texts)

        for chunk, content_fts, embedding in zip(chunks, texts, embeddings):
            all_chunk_data.append(
                (
                    doc.id,
                    {
                        "id": chunk.id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "content_fts": content_fts,
                        "metadata": json.dumps(chunk.metadata),
                        "order": chunk.order,
                        "vector": embedding,
                    },
                )
            )

    # Recreate chunks table (handles dimension changes)
    await client.store.recreate_embeddings_table()

    # Insert all chunks
    if all_chunk_data:
        records = [client.store.ChunkRecord(**data) for _, data in all_chunk_data]
        await client.store.chunks_table.add(records)

    # Yield all processed doc IDs
    yielded_docs: set[str] = set()
    for doc_id, _ in all_chunk_data:
        if doc_id not in yielded_docs:
            yielded_docs.add(doc_id)
            yield doc_id

    # Yield docs with no chunks
    for doc in documents:
        if doc.id and doc.id not in yielded_docs:
            yield doc.id


async def _flush_rebuild_batch(
    client: "HaikuRAG", documents: list[Document], chunks: list[Chunk]
) -> None:
    """Batch write documents and chunks during rebuild.

    Performs two writes: one for all document updates (via merge_insert), one
    for all chunks. Also repopulates document items from the stored docling
    document. Used by RECHUNK and FULL modes after the chunks table has been
    cleared.
    """
    from haiku.rag.store.engine import DocumentRecord

    if not documents:
        return

    now = datetime.now().isoformat()

    # Batch update documents using merge_insert (single LanceDB version)
    doc_records = []
    for doc in documents:
        assert doc.id is not None
        doc_records.append(
            DocumentRecord(
                id=doc.id,
                content=doc.content,
                uri=doc.uri,
                title=doc.title,
                metadata=json.dumps(doc.metadata),
                docling_document=doc.docling_document,
                docling_pages=doc.docling_pages,
                docling_version=doc.docling_version,
                created_at=doc.created_at.isoformat() if doc.created_at else now,
                updated_at=now,
            )
        )

    await (
        client.store.documents_table.merge_insert("id")
        .when_matched_update_all()
        .execute(doc_records)
    )

    # Batch create all chunks (single LanceDB version)
    if chunks:
        await client.chunk_repository.create(chunks)

    # Repopulate document items from stored docling data. The stored docling
    # blob has had its picture URIs stripped (compress_docling_split), so
    # re-extracting from it would lose picture_data — snapshot the existing
    # bytes per document and merge them back.
    for doc in documents:
        assert doc.id is not None
        docling_doc = doc.get_docling_document()
        if docling_doc is not None:
            existing_picture_data = (
                await client.document_item_repository.get_all_picture_data(doc.id)
            )
            await client.document_item_repository.delete_by_document_id(doc.id)
            items = extract_items(
                doc.id,
                docling_doc,
                existing_picture_data=existing_picture_data,
            )
            await client.document_item_repository.create_items(doc.id, items)


async def _rebuild_rechunk(
    client: "HaikuRAG", documents: list[Document]
) -> AsyncGenerator[str, None]:
    """Re-chunk and re-embed each document from its stored docling blob."""
    from haiku.rag.embeddings import embed_chunks, get_embedder

    pending_chunks: list[Chunk] = []
    pending_docs: list[Document] = []
    embedder = get_embedder(client._config)

    async for doc in _hydrate(client, documents):
        assert doc.id is not None
        docling_document = doc.get_docling_document()
        if docling_document is None:
            raise ValueError(
                f"Document {doc.id} has no stored docling document; rechunk "
                "requires it. Run a full rebuild (without --rechunk) instead."
            )

        # Stored blob has stripped picture URIs; pass the snapshot so
        # build_picture_chunks (inside chunk()) can recover the bytes.
        existing_picture_data = (
            await client.document_item_repository.get_all_picture_data(doc.id)
            if embedder.supports_images
            else None
        )
        chunks = await client.chunk(
            docling_document,
            existing_picture_data=existing_picture_data,
            document_id=doc.id,
        )
        embedded_chunks = await embed_chunks(chunks, client._config)

        # Prepare chunks with document_id and order
        for order, chunk in enumerate(embedded_chunks):
            chunk.document_id = doc.id
            chunk.order = order

        pending_chunks.extend(embedded_chunks)
        pending_docs.append(doc)
        # Yield per-doc so progress reporting moves immediately. The actual
        # write batches up to _REBUILD_BATCH_SIZE for throughput; if the
        # process is interrupted between yield and flush, up to one
        # batch's worth of trailing yields aren't persisted, which is
        # consistent with the rebuild already being non-atomic.
        yield doc.id

        # Flush batch when size reached
        if len(pending_docs) >= _REBUILD_BATCH_SIZE:
            await _flush_rebuild_batch(client, pending_docs, pending_chunks)
            pending_chunks = []
            pending_docs = []

    # Flush remaining
    if pending_docs:
        await _flush_rebuild_batch(client, pending_docs, pending_chunks)


async def _patch_picture_descriptions(client: "HaikuRAG", doc: Document) -> int:
    """Run the VLM against pictures lacking a description, patch the docling
    blob in-place. Returns the number of newly described pictures.
    Pictures that already carry ``meta.description.text`` are skipped, so the
    operation is safe to re-run after a partial failure.
    """
    from haiku.rag.providers.picture_description import describe_pictures

    assert doc.id is not None
    docling_doc = doc.get_docling_document()
    if docling_doc is None or not docling_doc.pictures:
        return 0

    needs_description: list[str] = []
    for pic in docling_doc.pictures:
        meta = getattr(pic, "meta", None)
        existing = (
            getattr(getattr(meta, "description", None), "text", None) if meta else None
        )
        if not (isinstance(existing, str) and existing.strip()):
            needs_description.append(pic.self_ref)

    if not needs_description:
        return 0

    bytes_by_ref = await client.document_item_repository.get_pictures_for_chunk(
        doc.id, needs_description
    )
    if not bytes_by_ref:
        logger.warning(
            "Document %s has %d pictures missing descriptions but no stored "
            "picture bytes — skipping. Run a full rebuild from source to "
            "recover the bytes.",
            doc.id,
            len(needs_description),
        )
        return 0

    descriptions = await describe_pictures(bytes_by_ref, config=client._config)

    if not descriptions:
        return 0

    # Patch the docling document in-place. PictureMeta + DescriptionMetaField
    # are pydantic models; build them and assign.
    from docling_core.types.doc.document import (
        DescriptionMetaField,
        PictureMeta,
    )

    for pic in docling_doc.pictures:
        text = descriptions.get(pic.self_ref)
        if not text:
            continue
        if pic.meta is None:
            pic.meta = PictureMeta()
        pic.meta.description = DescriptionMetaField(text=text)

    # Update only docling_document — set_docling would also overwrite
    # docling_pages by routing through compress_docling_split, which
    # extracts pages from the in-memory JSON and finds none (the pages
    # blob is stored separately and is not loaded by get_docling_document).
    # That would silently destroy page rasters for every doc with at
    # least one undescribed picture.
    from haiku.rag.store.compression import compress_docling_split

    structure_bytes, _ = compress_docling_split(docling_doc.model_dump_json())
    doc.docling_document = structure_bytes
    doc.docling_version = docling_doc.version
    return len(descriptions)


async def _rebuild_descriptions(
    client: "HaikuRAG", documents: list[Document]
) -> AsyncGenerator[str, None]:
    """Run the VLM over already-stored picture bytes, patch descriptions into
    the docling blob, then re-chunk + re-embed.

    Skips the docling parse entirely (the blob is already there); only the VLM
    cost remains. Idempotent: pictures whose ``meta.description.text`` is
    already populated are not re-described.
    """
    from haiku.rag.embeddings import embed_chunks, get_embedder

    if not client._config.processing.conversion_options.picture_description.enabled:
        raise ValueError(
            "rebuild --descriptions requires "
            "processing.conversion_options.picture_description.enabled = true "
            "in your config."
        )

    pending_chunks: list[Chunk] = []
    pending_docs: list[Document] = []
    embedder = get_embedder(client._config)

    described_total = 0
    async for doc in _hydrate(client, documents):
        assert doc.id is not None
        docling_document = doc.get_docling_document()
        if docling_document is None:
            raise ValueError(
                f"Document {doc.id} has no stored docling document; "
                "rebuild --descriptions requires it. Run a full rebuild instead."
            )

        n = await _patch_picture_descriptions(client, doc)
        described_total += n
        # Use the (possibly patched) docling document for chunking.
        docling_document = doc.get_docling_document()
        assert docling_document is not None

        existing_picture_data = (
            await client.document_item_repository.get_all_picture_data(doc.id)
            if embedder.supports_images
            else None
        )
        chunks = await client.chunk(
            docling_document,
            existing_picture_data=existing_picture_data,
            document_id=doc.id,
        )
        embedded_chunks = await embed_chunks(chunks, client._config)

        for order, chunk in enumerate(embedded_chunks):
            chunk.document_id = doc.id
            chunk.order = order

        pending_chunks.extend(embedded_chunks)
        pending_docs.append(doc)
        yield doc.id

        if len(pending_docs) >= _REBUILD_BATCH_SIZE:
            await _flush_rebuild_batch(client, pending_docs, pending_chunks)
            pending_chunks = []
            pending_docs = []

    if pending_docs:
        await _flush_rebuild_batch(client, pending_docs, pending_chunks)

    logger.info(
        "rebuild --descriptions: %d new picture descriptions added across %d documents",
        described_total,
        len(documents),
    )


async def _rebuild_full(
    client: "HaikuRAG", documents: list[Document]
) -> AsyncGenerator[str, None]:
    """Full rebuild: re-convert from source, re-chunk, re-embed."""
    from haiku.rag.embeddings import embed_chunks

    pending_chunks: list[Chunk] = []
    pending_docs: list[Document] = []
    converter = get_converter(client._config)

    for light_doc in documents:
        assert light_doc.id is not None

        # Try to rebuild from source if available — uses the light listing
        # directly, no need to load the stored content/blobs first.
        if light_doc.uri and check_source_accessible(light_doc.uri):
            try:
                # Flush pending batch before source rebuild (creates new doc)
                if pending_docs:
                    await _flush_rebuild_batch(client, pending_docs, pending_chunks)
                    pending_chunks = []
                    pending_docs = []

                await client.delete_document(light_doc.id)
                new_doc = await client.create_document_from_source(
                    source=light_doc.uri, metadata=light_doc.metadata or {}
                )
                assert isinstance(new_doc, Document)
                assert new_doc.id is not None
                yield new_doc.id
                continue
            except Exception as e:
                logger.error(
                    "Error recreating document from source %s: %s",
                    light_doc.uri,
                    e,
                )
                continue

        # Fallback: rebuild from stored content. Now we need the full
        # record (content + docling_pages for the round-trip write).
        doc = await client.get_document_by_id(light_doc.id)
        if doc is None:
            continue
        assert doc.id is not None
        if doc.uri:
            logger.warning("Source missing for %s, re-embedding from content", doc.uri)

        docling_document = await converter.convert_text(doc.content, format="md")
        chunks = await client.chunk(docling_document)
        embedded_chunks = await embed_chunks(chunks, client._config)

        doc.set_docling(docling_document)

        # Prepare chunks with document_id and order
        for order, chunk in enumerate(embedded_chunks):
            chunk.document_id = doc.id
            chunk.order = order

        pending_chunks.extend(embedded_chunks)
        pending_docs.append(doc)
        yield doc.id

        # Flush batch when size reached
        if len(pending_docs) >= _REBUILD_BATCH_SIZE:
            await _flush_rebuild_batch(client, pending_docs, pending_chunks)
            pending_chunks = []
            pending_docs = []

    # Flush remaining
    if pending_docs:
        await _flush_rebuild_batch(client, pending_docs, pending_chunks)
