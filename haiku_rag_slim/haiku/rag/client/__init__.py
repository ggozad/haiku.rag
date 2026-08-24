import asyncio
import hashlib
import json
import logging
import mimetypes
import tempfile
from collections.abc import AsyncGenerator, Callable, Coroutine, Sequence
from enum import Enum
from functools import cached_property
from itertools import zip_longest
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, overload
from urllib.parse import urlparse

import httpx

from haiku.rag.client.documents import DocumentImport
from haiku.rag.config import AppConfig, get_config
from haiku.rag.converters import get_converter
from haiku.rag.embeddings import get_embedder
from haiku.rag.reranking import get_reranker
from haiku.rag.store.engine import Store
from haiku.rag.store.exceptions import (
    AmbiguousDatabaseError,
    ConfigMismatchError,
    MigrationRequiredError,
    ReadOnlyError,
    SourceUnavailableError,
)
from haiku.rag.store.models.chunk import Chunk, SearchResult, SearchType
from haiku.rag.store.models.document import Document
from haiku.rag.store.models.document_item import extract_items
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.repositories.document_item import DocumentItemRepository
from haiku.rag.store.repositories.settings import SettingsRepository
from haiku.rag.utils import escape_sql_string, locate_database

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


# Failures whose message names the remedy and never the location, so the failing
# database is named alongside it instead of in place of it.
_NAMEABLE_FAILURES = (MigrationRequiredError, ConfigMismatchError, ReadOnlyError)


async def first_found(
    clients: "list[HaikuRAG]",
    lookup: "Callable[[HaikuRAG], Coroutine[Any, Any, Any]]",
) -> "tuple[HaikuRAG, Any] | None":
    """The first of `clients` for which `lookup` finds something, and what it found.

    An id or a URI says nothing about which database holds it, so every one is
    asked at once and the first that has it, in the order given, answers. Asking
    in turn would cost a round trip per database for an identifier that is
    missing or held by the last of them.
    """
    found_by_client = await asyncio.gather(*(lookup(client) for client in clients))
    for client, found in zip(clients, found_by_client, strict=True):
        if found is not None:
            return client, found
    return None


async def _aclose_quietly(closeable: Any, what: str) -> None:
    """Close, reporting failure to the log rather than raising.

    Teardown can run while an exception unwinds, so a raising close must
    neither mask that exception nor stop a sibling from being closed.
    """
    try:
        await closeable.aclose()
    except Exception:
        logger.debug("Closing the %s failed on teardown", what, exc_info=True)


def _spell(embedding: tuple[str | None, str | None, int | None]) -> str:
    """An embedder identity, for an error message."""
    provider, name, vector_dim = embedding
    return f"{provider}/{name} at {vector_dim} dimensions"


def _without_repeats(names: list[str]) -> list[str]:
    """`names` in order, without repeats.

    A database named twice would be searched twice and fused as two rank lists,
    which counts it double.
    """
    return list(dict.fromkeys(names))


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
                None means all of them. Ignored when a single ``uri`` or an
                explicit ``db_path`` is given.
        """
        self._config = config if config is not None else get_config()
        self._db_path_given = db_path is not None
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
        """Whether the client is in read-only mode.

        The mode the client was opened with, which is the mode every database it
        covers is opened with. A client covering a set has no store to ask.
        """
        return self._read_only

    @cached_property
    def embedder(self) -> "EmbedderWrapper":
        """The embedder for the databases this client covers.

        An embedder is a function of configuration rather than of a database,
        and the databases in a selection are required to share one, so a client
        covering a set has an unambiguous embedder without opening any of them.
        Built on first use and owned by this client, which closes it.
        """
        if self._federated:
            return get_embedder(config=self._config)
        return self.store.embedder

    @cached_property
    def reranker(self) -> "RerankerBase | None":
        """The configured reranker, built once and reused across searches.

        None when reranking is disabled. Local rerankers load model weights on
        construction, so building per search would reload them on every query.
        """
        return get_reranker(config=self._config)

    def _selected(self) -> dict[str, str]:
        """The configured databases this client covers, name to location.

        Empty when the caller named a database itself: an explicit `db_path` says
        which one to open, so it is not overridden by a configured set.
        """
        declared = self._config.lancedb.databases
        if not declared or self._db_path_given:
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
            uri, db_path = locate_database(location)
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
        except _NAMEABLE_FAILURES as error:
            # These say what to run and never where the database is, so the name
            # is added to the message rather than replacing it: the operator needs
            # both which database failed and what to do about it.
            if self._source is None:
                raise
            raise type(error)(f"database {self._source!r}: {error}") from error
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

        Missing ones open together: on object storage a serial loop makes the
        first query cost the sum of the opens.
        """
        names = _without_repeats(names)
        unknown = [n for n in names if n not in self._federated]
        if unknown:
            raise KeyError(
                f"unknown database(s) {', '.join(sorted(unknown))}; configured: "
                f"{', '.join(sorted(self._federated))}"
            )
        async with self._clients_lock:
            missing = [n for n in names if n not in self._clients]
            if missing:
                opened = await asyncio.gather(
                    *(self._open_client(n, self._federated[n]) for n in missing),
                    return_exceptions=True,
                )
                # Whatever opened is tracked before the failure is reported, so
                # `__aexit__` closes it: `gather` does not cancel the siblings of
                # the one that raised, and an untracked connection leaks.
                failure: BaseException | None = None
                for name, result in zip(missing, opened, strict=True):
                    if isinstance(result, BaseException):
                        failure = failure or result
                    else:
                        self._clients[name] = result
                if failure is not None:
                    raise failure
        return [self._clients[n] for n in names]

    def _require_one_embedder(self, clients: "list[HaikuRAG]") -> None:
        """Fail when two of these databases were written with different embedders.

        Searching a set embeds the query once, so a database written with another
        model answers from a different vector space: its candidates are noise, and
        rank fusion gives them slots anyway. Only databases searched together have
        to agree, so this is a property of the selection rather than of the set.

        Drift between a database and the *config* is a separate, softer matter —
        the same model served by another stack is spelled differently — which
        `SettingsRepository` reports on open.
        """
        recorded = [
            (client._source, client.store.stored_embedding)
            for client in clients
            if client.store.stored_embedding is not None
        ]
        if len(recorded) < 2:
            return
        (first_name, first), *rest = recorded
        for name, embedding in rest:
            if embedding != first:
                raise ConfigMismatchError(
                    f"databases '{first_name}' and '{name}' were written with "
                    f"different embedders ({_spell(first)} and "
                    f"{_spell(embedding)}); searching them together embeds the "
                    "query once, so their vectors are not comparable"
                )

    async def _open_client(self, name: str, location: str) -> "HaikuRAG":
        uri, db_path = locate_database(location)
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
            # The set shares this client's embedder and reranker, so this is the
            # only place they are closed — and only if anything built them.
            await self._aclose_cached("embedder")
            await self._aclose_cached("reranker")
            return False
        await self._await_vacuum_tasks()
        # Accessed so the store's embedder is closed even where nothing used it;
        # `cached_property` stores it, which is what `_aclose_cached` discards.
        _ = self.embedder
        await self._aclose_cached("embedder")
        await self._aclose_cached("reranker")
        self.close()
        return False

    async def _aclose_cached(self, name: str) -> None:
        """Close a cached_property this client materialized, and discard it.

        Discarded rather than left in place so that re-entering the client
        builds a fresh one instead of reusing something already closed.
        """
        cached = self.__dict__.pop(name, None)
        if cached is not None:
            await _aclose_quietly(cached, name)

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

        self._require_one_database("create_document")

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

        self._require_one_database("import_document")

        return await import_document(
            self, docling_document, chunks, uri, title, metadata
        )

    async def import_documents(
        self,
        imports: "list[DocumentImport]",
    ) -> list[Document]:
        from haiku.rag.client.documents import import_documents

        self._require_one_database("import_documents")

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

        self._require_one_database("create_document_from_source")

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

        self._require_one_database("update_document")

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
        if self._federated:
            return await self._from_any_covered(
                lambda owner: owner.get_document_by_id(document_id)
            )
        return self._name(await self.document_repository.get_by_id(document_id))

    async def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        """Get a chunk by its ID.

        Args:
            chunk_id: The unique identifier of the chunk.

        Returns:
            The Chunk instance if found, None otherwise.
        """
        if self._federated:
            return await self._from_any_covered(
                lambda owner: owner.get_chunk_by_id(chunk_id)
            )
        return await self.chunk_repository.get_by_id(chunk_id)

    async def get_picture_bytes(
        self, document_id: str, self_ref: str, source: str | None = None
    ) -> bytes | None:
        """Get a picture's bytes, from the database named by `source`.

        Args:
            document_id: The document holding the picture.
            self_ref: The picture's `self_ref`.
            source: The database it came from. Required when federating.

        Returns:
            The picture bytes if found, None otherwise.
        """
        if not self._federated:
            return await self.document_item_repository.get_picture_bytes(
                document_id, self_ref
            )
        if source is None:
            raise ValueError(
                "a picture lookup across databases needs the source it came from"
            )
        (owner,) = await self.clients_for([source])
        return await owner.document_item_repository.get_picture_bytes(
            document_id, self_ref
        )

    async def get_document_by_uri(self, uri: str) -> Document | None:
        """Get a document by its URI.

        Args:
            uri: The URI identifier of the document.

        Returns:
            The Document instance if found, None otherwise.
        """
        if self._federated:
            return await self._from_any_covered(
                lambda owner: owner.get_document_by_uri(uri)
            )
        return self._name(await self.document_repository.get_by_uri(uri))

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

        self._require_one_database("delete_document")

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
        if self._federated:
            # Each database is asked for enough rows to satisfy the window, and
            # the window is applied to the merged listing: a limit means that
            # many documents in total, not that many per database.
            wanted = None if limit is None else limit + (offset or 0)
            groups = await asyncio.gather(
                *(
                    owner.list_documents(
                        limit=wanted, filter=filter, include_content=include_content
                    )
                    for owner in await self.clients_covering()
                )
            )
            # Round-robin, so a window shows every database. Concatenating lets
            # the first one fill the whole page and hide the rest.
            merged = [
                doc for row in zip_longest(*groups) for doc in row if doc is not None
            ]
            start = offset or 0
            return merged[start:] if limit is None else merged[start : start + limit]
        documents = await self.document_repository.list_all(
            limit=limit, offset=offset, filter=filter, include_content=include_content
        )
        for document in documents:
            document.source = self._source
        return documents

    async def count_documents(self, filter: str | None = None) -> int:
        """Count documents with optional filtering.

        Args:
            filter: Optional SQL WHERE clause to filter documents.

        Returns:
            Number of documents matching the criteria.
        """
        if self._federated:
            counts = await asyncio.gather(
                *(
                    owner.count_documents(filter=filter)
                    for owner in await self.clients_covering()
                )
            )
            return sum(counts)
        return await self.document_repository.count(filter=filter)

    def _require_one_database(self, operation: str) -> None:
        """Refuse an operation that has no meaning across a set of databases.

        Writing, rebuilding and vacuuming all have to name a database. Raised as
        a domain error rather than surfacing the missing repository, so a caller
        can tell an unsupported selection from a bug.
        """
        if self._federated:
            raise AmbiguousDatabaseError(
                f"{operation} works on one database, and this client covers "
                f"{', '.join(sorted(self._federated))}; select one with "
                "clients_for([name])"
            )

    def _name(self, document: Document | None) -> Document | None:
        """`document`, told which configured database it came from.

        None where no database is named, as with a single ``lancedb.uri``.
        """
        if document is not None:
            document.source = self._source
        return document

    async def _from_any_covered(
        self, lookup: "Callable[[HaikuRAG], Coroutine[Any, Any, Any]]"
    ) -> Any:
        """The first result `lookup` finds in the databases this client covers."""
        found = await first_found(await self.clients_covering(), lookup)
        return None if found is None else found[1]

    async def clients_covering(
        self, sources: list[str] | None = None
    ) -> list["HaikuRAG"]:
        """The clients covering this selection.

        The named subset for a client covering a set, or this one where it covers
        a single database. Empty for a selection of none, which is not the same as
        `None` for all of them. Every read honouring `sources` decides through
        this, so the rule cannot differ between one operation and another.
        """
        if self._federated:
            return await self.clients_for(
                list(self._federated) if sources is None else sources
            )
        if sources is None:
            return [self]
        sources = _without_repeats(sources)
        if not sources:
            return []
        if sources != [self._source]:
            raise KeyError(
                f"unknown database(s) {', '.join(sources) or '(none)'}; this "
                f"client covers {self._source or 'a single unnamed database'}"
            )
        return [self]

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
        if not await self.clients_covering(sources):
            return []
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
        sources: list[str] | None = None,
    ) -> "tuple[str, list[Citation]]":
        from haiku.rag.client.agents import ask

        return await ask(self, question, filter, images, sources)

    async def analyze(
        self,
        question: str,
        filter: str | None = None,
        images: Sequence[bytes] | None = None,
        sources: list[str] | None = None,
    ) -> "AnalysisResult":
        from haiku.rag.client.agents import analyze

        return await analyze(self, question, filter, images, sources)

    async def visualize_chunk(
        self,
        chunk: Chunk | Sequence[Chunk],
        refs: list[str] | None = None,
        expand: bool = True,
    ) -> list:
        from haiku.rag.client.search import visualize_chunk

        self._require_one_database("visualize_chunk")

        return await visualize_chunk(self, chunk, refs, expand)

    async def rebuild_database(
        self, mode: RebuildMode = RebuildMode.FULL
    ) -> AsyncGenerator[str, None]:
        from haiku.rag.client.rebuild import rebuild_database

        self._require_one_database("rebuild_database")

        async for doc_id in rebuild_database(self, mode):
            yield doc_id

    async def vacuum(self) -> None:
        """Optimize and clean up old versions across all tables."""
        self._require_one_database("vacuum")
        await self.store.vacuum()

    def close(self):
        """Close the underlying store connection."""
        self._require_one_database("close")
        self.store.close()
