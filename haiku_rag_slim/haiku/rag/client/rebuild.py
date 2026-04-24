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

    # Wait for any background vacuum before destructive table operations.
    # Rebuild drops and recreates tables (+ creates indices); a concurrent
    # optimize on the same table fails with "CreateIndex transaction was
    # preempted" from lance.
    await client._await_vacuum_tasks()

    # Update settings to current config
    settings_repo = SettingsRepository(client.store)
    await settings_repo.save_current_settings()

    documents = await client.list_documents(include_content=True)

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
    else:  # FULL
        await client.chunk_repository.delete_all()
        await client.store.recreate_embeddings_table()
        async for doc_id in _rebuild_full(client, documents):
            yield doc_id

    # Final maintenance if auto_vacuum enabled
    if client._config.storage.auto_vacuum:
        try:
            await client.store.vacuum()
        except Exception:
            pass


async def _rebuild_title_only(
    client: "HaikuRAG", documents: list[Document]
) -> AsyncGenerator[str, None]:
    """Generate titles for documents that don't have one."""
    for doc in documents:
        if doc.title is not None:
            continue
        assert doc.id is not None
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

    # Repopulate document items from stored docling data
    for doc in documents:
        assert doc.id is not None
        docling_doc = doc.get_docling_document()
        if docling_doc is not None:
            await client.document_item_repository.delete_by_document_id(doc.id)
            items = extract_items(doc.id, docling_doc)
            await client.document_item_repository.create_items(doc.id, items)


async def _rebuild_rechunk(
    client: "HaikuRAG", documents: list[Document]
) -> AsyncGenerator[str, None]:
    """Re-chunk and re-embed from existing document content."""
    from haiku.rag.embeddings import embed_chunks

    pending_chunks: list[Chunk] = []
    pending_docs: list[Document] = []
    pending_doc_ids: list[str] = []

    converter = get_converter(client._config)

    for doc in documents:
        assert doc.id is not None

        # Convert stored markdown to DoclingDocument
        docling_document = await converter.convert_text(doc.content, format="md")

        # Chunk and embed
        chunks = await client.chunk(docling_document)
        embedded_chunks = await embed_chunks(chunks, client._config)

        # Update document fields
        doc.set_docling(docling_document)

        # Prepare chunks with document_id and order
        for order, chunk in enumerate(embedded_chunks):
            chunk.document_id = doc.id
            chunk.order = order

        pending_chunks.extend(embedded_chunks)
        pending_docs.append(doc)
        pending_doc_ids.append(doc.id)

        # Flush batch when size reached
        if len(pending_docs) >= _REBUILD_BATCH_SIZE:
            await _flush_rebuild_batch(client, pending_docs, pending_chunks)
            for doc_id in pending_doc_ids:
                yield doc_id
            pending_chunks = []
            pending_docs = []
            pending_doc_ids = []

    # Flush remaining
    if pending_docs:
        await _flush_rebuild_batch(client, pending_docs, pending_chunks)
        for doc_id in pending_doc_ids:
            yield doc_id


async def _rebuild_full(
    client: "HaikuRAG", documents: list[Document]
) -> AsyncGenerator[str, None]:
    """Full rebuild: re-convert from source, re-chunk, re-embed."""
    from haiku.rag.embeddings import embed_chunks

    pending_chunks: list[Chunk] = []
    pending_docs: list[Document] = []
    pending_doc_ids: list[str] = []
    converter = get_converter(client._config)

    for doc in documents:
        assert doc.id is not None

        # Try to rebuild from source if available
        if doc.uri and check_source_accessible(doc.uri):
            try:
                # Flush pending batch before source rebuild (creates new doc)
                if pending_docs:
                    await _flush_rebuild_batch(client, pending_docs, pending_chunks)
                    for doc_id in pending_doc_ids:
                        yield doc_id
                    pending_chunks = []
                    pending_docs = []
                    pending_doc_ids = []

                await client.delete_document(doc.id)
                new_doc = await client.create_document_from_source(
                    source=doc.uri, metadata=doc.metadata or {}
                )
                assert isinstance(new_doc, Document)
                assert new_doc.id is not None
                yield new_doc.id
                continue
            except Exception as e:
                logger.error(
                    "Error recreating document from source %s: %s",
                    doc.uri,
                    e,
                )
                continue

        # Fallback: rebuild from stored content
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
        pending_doc_ids.append(doc.id)

        # Flush batch when size reached
        if len(pending_docs) >= _REBUILD_BATCH_SIZE:
            await _flush_rebuild_batch(client, pending_docs, pending_chunks)
            for doc_id in pending_doc_ids:
                yield doc_id
            pending_chunks = []
            pending_docs = []
            pending_doc_ids = []

    # Flush remaining
    if pending_docs:
        await _flush_rebuild_batch(client, pending_docs, pending_chunks)
        for doc_id in pending_doc_ids:
            yield doc_id
