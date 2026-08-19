import asyncio
import hashlib
import json
import logging
import mimetypes
import tempfile
from collections.abc import AsyncGenerator, Sequence
from enum import Enum
from functools import cached_property
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, overload
from urllib.parse import urlparse

import httpx

from haiku.rag.client.documents import DocumentImport
from haiku.rag.config import AppConfig, get_config
from haiku.rag.converters import get_converter
from haiku.rag.reranking import get_reranker
from haiku.rag.store.engine import Store
from haiku.rag.store.exceptions import SourceUnavailableError
from haiku.rag.store.models.chunk import Chunk, SearchResult, SearchType
from haiku.rag.store.models.document import Document
from haiku.rag.store.models.document_item import extract_items
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.repositories.document_item import DocumentItemRepository
from haiku.rag.store.repositories.settings import SettingsRepository
from haiku.rag.utils import escape_sql_string

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument
    from PIL import Image as PILImage

    from haiku.rag.embeddings import EmbedderWrapper
    from haiku.rag.ingester.metadata import MetadataProvider
    from haiku.rag.reranking.base import RerankerBase
    from haiku.rag.sandbox import AnalysisResult
    from haiku.rag.sources.base import Source
    from haiku.rag.store.models.citation import Citation

logger = logging.getLogger(__name__)

# Throttle for the background auto-vacuum: under sustained ingestion, scheduling
# a compaction on every write degenerates into back-to-back optimize() passes
# that churn the blob-bearing documents table. Fire at most one per interval; a
# final vacuum on close collapses anything throttled here.
_VACUUM_MIN_INTERVAL_S = 300.0


class RebuildMode(Enum):
    """Mode for rebuilding the database."""

    FULL = "full"  # Re-convert from source, re-chunk, re-embed
    RECHUNK = "rechunk"  # Re-chunk from existing content, re-embed
    EMBED_ONLY = "embed_only"  # Keep chunks, only regenerate embeddings
    TITLE_ONLY = "title_only"  # Only generate titles for untitled documents
    DESCRIPTIONS = "descriptions"  # Run the VLM over already-stored picture
    # bytes, patch descriptions into the docling blob, then re-chunk + re-embed.
    SET_EMBEDDER = "set_embedder"  # Adopt the current embedder identity without
    # re-embedding, when the vector dimension is unchanged.


class HaikuRAG:
    """High-level haiku-rag client."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        config: AppConfig | None = None,
        skip_validation: bool = False,
        create: bool = False,
        read_only: bool = False,
        sources: list[str] | None = None,
    ):
        """Initialize the RAG client with a database path.

        Args:
            db_path: Path or string path to the database file. If None, uses
                config.storage.data_dir.
            config: Configuration to use. Defaults to the current global config.
            skip_validation: Whether to skip configuration validation on database load.
            create: Whether to create the database if it doesn't exist.
            read_only: Whether to open the database in read-only mode.
            sources: Names from ``config.lancedb.databases`` this client covers.
                None means all of them. Ignored when a single ``uri`` is
                configured.
        """
        self._config = config if config is not None else get_config()
        if db_path is None:
            db_path = self._config.storage.data_dir / "haiku.rag.lancedb"

        self._db_path = db_path
        self._skip_validation = skip_validation
        self._create = create
        self._read_only = read_only
        self._vacuum_tasks: set[asyncio.Task] = set()
        self._last_vacuum_at: float | None = None
        self._vacuum_dirty = False
        self._requested_sources = sources
        self._clients: dict[str, HaikuRAG] = {}
        self._federated: dict[str, str] = {}
        self._clients_lock = asyncio.Lock()
        self._source: str | None = None

    @property
    def is_read_only(self) -> bool:
        """Whether the client is in read-only mode."""
        return self.store.is_read_only

    @property
    def embedder(self) -> "EmbedderWrapper":
        """The embedder owned by the Store, reused across all operations."""
        return self.store.embedder

    @cached_property
    def reranker(self) -> "RerankerBase | None":
        """The configured reranker, built once and reused across searches.

        None when reranking is disabled. Local rerankers load model weights on
        construction, so building per search would reload them on every query.
        """
        return get_reranker(config=self._config)

    @staticmethod
    def _locate(location: str) -> tuple[str, "Path | None"]:
        """Split a configured location into (uri, db_path).

        A value with a scheme is a `lancedb.uri`; anything else is a local path.
        Routing a local path through `uri` would have `ConnectionMode` classify it
        as object storage, which opens it without the existence check a local
        database gets.
        """
        if "://" in location:
            return location, None
        return "", Path(location)

    def _selected(self) -> dict[str, str]:
        """The configured databases this client covers, name to location."""
        declared = self._config.lancedb.databases
        if not declared:
            return {}
        if self._requested_sources is not None and not self._requested_sources:
            raise ValueError(
                "sources=[] selects no database; pass None for all of them"
            )
        names = (
            list(declared)
            if self._requested_sources is None
            else list(self._requested_sources)
        )
        missing = [n for n in names if n not in declared]
        if missing:
            raise KeyError(
                f"unknown database(s) {', '.join(sorted(missing))}; "
                f"configured: {', '.join(sorted(declared))}"
            )
        return {n: declared[n] for n in names}

    async def __aenter__(self):
        """Async context manager entry — initializes store and repositories.

        A client covering several databases opens none of them here: which are
        searched is a per-query choice, so they open on first use. `store` and the
        repositories stay unset in that case, since they have no unambiguous
        meaning across a set.
        """
        selected = self._selected()
        if len(selected) > 1:
            self._federated = selected
            return self
        if selected:
            [(self._source, location)] = selected.items()
            uri, db_path = self._locate(location)
            self._config = self._config.model_copy(deep=True)
            self._config.lancedb.databases = {}
            self._config.lancedb.uri = uri
            if db_path is not None:
                self._db_path = db_path

        failure: str | None = None
        try:
            self.store = Store(
                self._db_path,
                config=self._config,
                skip_validation=self._skip_validation,
                create=self._create,
                read_only=self._read_only,
            )
            # If _initialize fails mid-way (e.g. migration check raises after
            # connect), close the store so we don't leak the LanceDB connection —
            # __aexit__ won't run because the `async with` never entered.
            try:
                await self.store._initialize()
            except BaseException:
                self.store.close()
                raise
        except Exception as error:
            # A legacy `uri` or `db_path` client has no name to report instead, so
            # its error passes through as it always has.
            if self._source is None:
                raise
            failure = type(error).__name__
        if failure is not None:
            # Raised outside the except block on purpose. A database named in
            # config is reported by name, and the original spells out the path or
            # the bucket: `from None` would only stop it being *printed*, leaving
            # it on `__context__` for anything that walks the chain.
            raise SourceUnavailableError(
                f"database {self._source!r} could not be opened: {failure}"
            )
        self.document_repository = DocumentRepository(self.store)
        self.chunk_repository = ChunkRepository(self.store)
        self.document_item_repository = DocumentItemRepository(self.store)
        return self

    async def clients_for(self, names: list[str]) -> list["HaikuRAG"]:
        """The clients for these databases, opening any not yet open.

        Opening is per query rather than at entry: a set of 25 configured
        databases is typically queried a few at a time, and a database nobody
        asked for must not be able to fail a query, or be opened for nothing.
        """
        unknown = [n for n in names if n not in self._federated]
        if unknown:
            raise KeyError(
                f"unknown database(s) {', '.join(sorted(unknown))}; configured: "
                f"{', '.join(sorted(self._federated))}"
            )
        async with self._clients_lock:
            for name in names:
                if name not in self._clients:
                    self._clients[name] = await self._open_client(
                        name, self._federated[name]
                    )
        return [self._clients[n] for n in names]

    async def _open_client(self, name: str, location: str) -> "HaikuRAG":
        uri, db_path = self._locate(location)
        config = self._config.model_copy(deep=True)
        config.lancedb.databases = {}
        config.lancedb.uri = uri
        client = HaikuRAG(
            db_path,
            config=config,
            skip_validation=self._skip_validation,
            read_only=self._read_only,
        )
        client._source = name
        return await client.__aenter__()

    async def _close_clients(self) -> None:
        for client in self._clients.values():
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                logger.debug("Closing a database failed on teardown", exc_info=True)
        self._clients.clear()

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ARG002
        """Async context manager exit."""
        # Branch on what this client covers, not on what it happened to open:
        # a federating client that answered no query has nothing open and no
        # store either.
        if self._federated:
            await self._close_clients()
            # The set shares one reranker, this client's, so this is the only
            # place it is closed — and only if a text query ever built it.
            reranker = self.__dict__.get("reranker")
            if reranker is not None:
                try:
                    await reranker.aclose()
                except Exception:
                    logger.debug("Closing the reranker failed", exc_info=True)
            return False
        await self._await_vacuum_tasks()
        # Best-effort: __aexit__ may run during exception unwinding, and a
        # raising close must not mask the original exception. The reranker is
        # a cached_property — close it only if it was materialized.
        try:
            await self.embedder.aclose()
            reranker = self.__dict__.get("reranker")
            if reranker is not None:
                await reranker.aclose()
        except Exception:
            logger.debug("Closing embedder/reranker failed on teardown", exc_info=True)
        self.close()
        return False

    async def _await_vacuum_tasks(self) -> None:
        """Drain background vacuum work and run a final collapse before teardown.

        Writes schedule a throttled background vacuum; many are debounced or skip
        because another vacuum holds the lock. The final pass collapses the
        versions those left behind. It runs whenever writes happened
        (``_vacuum_dirty``) — not gated on in-flight tasks remaining, since a
        debounced run may have scheduled none — but never when nothing was
        written (so opening + closing a store still never writes).
        """
        if self._vacuum_tasks:
            await asyncio.gather(*self._vacuum_tasks, return_exceptions=True)
        if not self._vacuum_dirty:
            return
        self._vacuum_dirty = False
        # __aexit__ runs during exception unwinding; a raising vacuum here would
        # mask the original exception, so the drain stays best-effort.
        try:
            await self.store.vacuum()
        except Exception:
            logger.debug("Final vacuum on close failed", exc_info=True)

    def _schedule_vacuum(self) -> None:
        """Schedule a background vacuum, throttled to at most one per
        ``_VACUUM_MIN_INTERVAL_S``. Sustained writes would otherwise trigger
        back-to-back compaction of the blob-bearing documents table. The throttle
        only skips the background task — ``_vacuum_dirty`` still marks that a
        final vacuum on close is owed."""
        self._vacuum_dirty = True
        now = monotonic()
        if (
            self._last_vacuum_at is not None
            and now - self._last_vacuum_at < _VACUUM_MIN_INTERVAL_S
        ):
            return
        self._last_vacuum_at = now
        task = asyncio.create_task(self.store.vacuum())
        self._vacuum_tasks.add(task)
        task.add_done_callback(self._vacuum_tasks.discard)

    # =========================================================================
    # Processing Primitives
    # =========================================================================

    @overload
    async def convert(
        self, source: Path, *, source_uri: str | None = None
    ) -> "DoclingDocument": ...

    @overload
    async def convert(
        self, source: str, *, format: str = "md", source_uri: str | None = None
    ) -> "DoclingDocument": ...

    async def convert(
        self,
        source: Path | str,
        *,
        format: str = "md",
        source_uri: str | None = None,
    ) -> "DoclingDocument":
        from haiku.rag.client.processing import convert

        return await convert(self._config, source, format=format, source_uri=source_uri)

    async def chunk(
        self,
        docling_document: "DoclingDocument",
        *,
        existing_picture_data: dict[str, bytes] | None = None,
        document_id: str | None = None,
    ) -> list[Chunk]:
        from haiku.rag.client.processing import chunk

        return await chunk(
            self._config,
            docling_document,
            embedder=self.embedder,
            existing_picture_data=existing_picture_data,
            document_id=document_id,
        )

    # =========================================================================
    # Title Generation
    # =========================================================================

    async def generate_title(self, document: Document) -> str | None:
        from haiku.rag.client.titles import generate_title

        return await generate_title(self._config, document)

    async def create_document(
        self,
        content: str,
        uri: str | None = None,
        title: str | None = None,
        metadata: dict | None = None,
        format: str = "md",
    ) -> Document:
        from haiku.rag.client.documents import create_document

        return await create_document(self, content, uri, title, metadata, format)

    async def import_document(
        self,
        docling_document: "DoclingDocument",
        chunks: list[Chunk],
        uri: str | None = None,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> Document:
        from haiku.rag.client.documents import import_document

        return await import_document(
            self, docling_document, chunks, uri, title, metadata
        )

    async def import_documents(
        self,
        imports: "list[DocumentImport]",
    ) -> list[Document]:
        from haiku.rag.client.documents import import_documents

        return await import_documents(self, imports)

    async def create_document_from_source(
        self,
        source: str | Path,
        title: str | None = None,
        metadata: dict | None = None,
        uri: str | None = None,
        storage_options: dict[str, str] | None = None,
        sources: "list[Source] | None" = None,
        source_id: str | None = None,
        metadata_provider: "MetadataProvider | None" = None,
    ) -> Document | list[Document]:
        from haiku.rag.client.documents import create_document_from_source

        return await create_document_from_source(
            self,
            source,
            title,
            metadata,
            uri=uri,
            storage_options=storage_options,
            sources=sources,
            source_id=source_id,
            metadata_provider=metadata_provider,
        )

    async def update_document(
        self,
        document_id: str,
        content: str | None = None,
        metadata: dict | None = None,
        chunks: list[Chunk] | None = None,
        title: str | None = None,
        docling_document: "DoclingDocument | None" = None,
        uri: str | None = None,
    ) -> Document:
        from haiku.rag.client.documents import update_document

        return await update_document(
            self,
            document_id,
            content,
            metadata,
            chunks,
            title,
            docling_document,
            uri,
        )

    async def get_document_by_id(self, document_id: str) -> Document | None:
        """Get a document by its ID.

        Args:
            document_id: The unique identifier of the document.

        Returns:
            The Document instance if found, None otherwise.
        """
        return await self.document_repository.get_by_id(document_id)

    async def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        """Get a chunk by its ID.

        Args:
            chunk_id: The unique identifier of the chunk.

        Returns:
            The Chunk instance if found, None otherwise.
        """
        return await self.chunk_repository.get_by_id(chunk_id)

    async def get_document_by_uri(self, uri: str) -> Document | None:
        """Get a document by its URI.

        Args:
            uri: The URI identifier of the document.

        Returns:
            The Document instance if found, None otherwise.
        """
        return await self.document_repository.get_by_uri(uri)

    async def resolve_document(self, id_or_title: str) -> Document | None:
        """Resolve a document by ID, title, or URI (in that order).

        Args:
            id_or_title: Document ID, title, or URI to look up.

        Returns:
            The Document instance if found, None otherwise.
        """
        doc = await self.get_document_by_id(id_or_title)
        if doc:
            return doc

        safe_input = escape_sql_string(id_or_title)
        docs = await self.list_documents(filter=f"title = '{safe_input}'")
        if docs and docs[0].id:
            return await self.get_document_by_id(docs[0].id)

        docs = await self.list_documents(filter=f"uri = '{safe_input}'")
        if docs and docs[0].id:
            return await self.get_document_by_id(docs[0].id)

        return None

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document by its ID. Cascades to children linked via
        ``metadata.parent_uri``.

        The whole subtree (root + transitive children) is deleted under a single
        write lock and a single version snapshot, so the cascade is atomic: any
        failure restores every table to the pre-delete state, and no other write
        can interleave between deleting a child and its parent.
        """
        from haiku.rag.client.documents import parent_uri_filter

        async with self.store.write_transaction():
            # Resolve existence and collect the subtree under the lock so two
            # concurrent deletes of the same id can't both proceed, and children
            # can't appear or move between collection and deletion. parent_uri
            # links a child to its parent's uri; walk transitively, guarding
            # against cycles.
            ids_to_delete: list[str] = []
            seen: set[str] = set()
            queue = [await self.get_document_by_id(document_id)]
            while queue:
                doc = queue.pop()
                if doc is None or doc.id is None or doc.id in seen:
                    continue
                seen.add(doc.id)
                ids_to_delete.append(doc.id)
                if doc.uri:
                    queue.extend(
                        await self.list_documents(filter=parent_uri_filter(doc.uri))
                    )

            if not ids_to_delete:
                return False

            for doc_id in ids_to_delete:
                await self.document_repository.delete(doc_id)

        if self._config.storage.auto_vacuum:
            self._schedule_vacuum()
        return True

    async def list_documents(
        self,
        limit: int | None = None,
        offset: int | None = None,
        filter: str | None = None,
        include_content: bool = False,
    ) -> list[Document]:
        """List all documents with optional pagination and filtering.

        Args:
            limit: Maximum number of documents to return.
            offset: Number of documents to skip.
            filter: Optional SQL WHERE clause to filter documents.
            include_content: Whether to load the text content. Defaults to
                False. A listing never loads the docling blobs.

        Returns:
            List of Document instances matching the criteria.
        """
        return await self.document_repository.list_all(
            limit=limit, offset=offset, filter=filter, include_content=include_content
        )

    async def count_documents(self, filter: str | None = None) -> int:
        """Count documents with optional filtering.

        Args:
            filter: Optional SQL WHERE clause to filter documents.

        Returns:
            Number of documents matching the criteria.
        """
        return await self.document_repository.count(filter=filter)

    async def search(
        self,
        query: "str | bytes | PILImage.Image",
        limit: int | None = None,
        search_type: SearchType | None = None,
        filter: str | None = None,
        include_images: bool = True,
        sources: list[str] | None = None,
    ) -> list[SearchResult]:
        from haiku.rag.client.search import search, search_sources

        if self._federated:
            return await search_sources(
                self, query, limit, search_type, filter, include_images, sources
            )
        if sources is not None and not sources:
            return []
        if sources is not None and sources != [self._source]:
            raise KeyError(
                f"unknown database(s) {', '.join(sources) or '(none)'}; this "
                f"client covers {self._source or 'a single unnamed database'}"
            )
        results = await search(self, query, limit, search_type, filter, include_images)
        # A database named in config keeps its name even when it is the only one
        # this client covers. Only a legacy single `uri` leaves source unset.
        for result in results:
            result.source = self._source
        return results

    async def expand_context(
        self,
        search_results: list[SearchResult],
    ) -> list[SearchResult]:
        from haiku.rag.client.search import expand_context

        return await expand_context(self, search_results)

    async def ask(
        self,
        question: str,
        filter: str | None = None,
        images: Sequence[bytes] | None = None,
    ) -> "tuple[str, list[Citation]]":
        from haiku.rag.client.agents import ask

        return await ask(self, question, filter, images)

    async def analyze(
        self,
        question: str,
        filter: str | None = None,
        images: Sequence[bytes] | None = None,
    ) -> "AnalysisResult":
        from haiku.rag.client.agents import analyze

        return await analyze(self, question, filter, images)

    async def visualize_chunk(
        self,
        chunk: Chunk | Sequence[Chunk],
        refs: list[str] | None = None,
        expand: bool = True,
    ) -> list:
        from haiku.rag.client.search import visualize_chunk

        return await visualize_chunk(self, chunk, refs, expand)

    async def rebuild_database(
        self, mode: RebuildMode = RebuildMode.FULL
    ) -> AsyncGenerator[str, None]:
        from haiku.rag.client.rebuild import rebuild_database

        async for doc_id in rebuild_database(self, mode):
            yield doc_id

    async def vacuum(self) -> None:
        """Optimize and clean up old versions across all tables."""
        await self.store.vacuum()

    def close(self):
        """Close the underlying store connection."""
        self.store.close()
