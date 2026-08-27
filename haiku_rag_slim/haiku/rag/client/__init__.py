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
from haiku.rag.client.scope import DatabaseRef, DatabaseScope
from haiku.rag.client.session import (
    FederatedSession,
    SingleDatabaseSession,
    aclose_quietly,
    default_db_path,
)
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
    UnknownDatabaseError,
)
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


async def all_found(
    clients: "list[HaikuRAG]",
    lookup: "Callable[[HaikuRAG], Coroutine[Any, Any, Any]]",
) -> "list[tuple[HaikuRAG, Any]]":
    """Every client for which `lookup` finds something, and what each found.

    An id or a URI says nothing about which database holds it, so every one is
    asked at once. Asking in turn would cost a round trip per database for an
    identifier that is missing or held by the last of them.
    """
    found_by_client = await asyncio.gather(*(lookup(client) for client in clients))
    return [
        (client, found)
        for client, found in zip(clients, found_by_client, strict=True)
        if found is not None
    ]


async def first_found(
    clients: "list[HaikuRAG]",
    lookup: "Callable[[HaikuRAG], Coroutine[Any, Any, Any]]",
) -> "tuple[HaikuRAG, Any] | None":
    """The first of `clients` for which `lookup` finds something, and what it found.

    For a lookup that has an answer wherever it is found, as a document read
    does. Where holding the same identifier in two databases means something,
    ask `all_found` and decide.
    """
    found = await all_found(clients, lookup)
    return found[0] if found else None


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
            db_path: Path or string path to the database. When omitted, resolves
                ``lancedb.databases``, then ``lancedb.uri``, then the default
                path under ``storage.data_dir``.
            config: Configuration to use. Defaults to the current global config.
            skip_validation: Whether to skip configuration validation on database load.
            create: Whether to create the database if it doesn't exist.
            read_only: Whether to open the database in read-only mode.
            sources: Names from ``config.lancedb.databases`` this client covers,
                None for all of them. Only that setting names databases, so a
                name raises when ``lancedb.uri`` placed the database, and is
                rejected alongside ``db_path``, which says the same thing
                another way. ``[]`` raises too: a client over no database can do
                nothing, unlike ``sources=[]`` on a search, which is a selection
                of nothing to search.
        """
        self._configured = config if config is not None else get_config()
        # What the caller configured, kept intact: entering derives a
        # single-database configuration from it, and asking again has to see the
        # same set rather than the answer from last time.
        self._config = self._configured
        self._requested_db_path = Path(db_path) if db_path is not None else None
        if self._requested_db_path is not None and sources is not None:
            raise AmbiguousDatabaseError(
                "a path and `sources` both say which databases to open; pass "
                "one of them"
            )
        self._skip_validation = skip_validation
        self._create = create
        self._read_only = read_only
        self._requested_sources = sources
        self._clients: dict[str, HaikuRAG] = {}
        # The client this one covers a database for, whose reranker it borrows.
        self._lender: HaikuRAG | None = None
        self._scope: DatabaseScope | None = None
        self._session: SingleDatabaseSession | FederatedSession | None = None
        self._owns_session = True
        self._closed = False

    @property
    def covers_multiple(self) -> bool:
        """Whether this client reads from more than one database."""
        return isinstance(self._session, FederatedSession)

    @property
    def source_names(self) -> tuple[str, ...]:
        """The configured databases this client covers, in configured order.

        A single database contributes its own name, or nothing where the
        configuration named none.
        """
        if isinstance(self._session, FederatedSession):
            return self._session.names
        return () if self.source is None else (self.source,)

    @property
    def source(self) -> str | None:
        """The configured database this client reads, or None while covering a
        set or reading a database the configuration did not name."""
        if isinstance(self._session, SingleDatabaseSession):
            return self._session.source
        return None

    @property
    def location(self) -> "Path | str | None":
        """Where the database this client reads is, or None while covering a set."""
        if not isinstance(self._session, SingleDatabaseSession):
            return None
        return self._session.location

    async def reader_for(self, source: str | None) -> "HaikuRAG | None":
        """The client that can read `source` — itself, where it reads one
        database.

        None only when a client covering a set is given no name, as for evidence
        recorded before databases could be named. A name this client does not
        cover raises `UnknownDatabaseError`, decided by `clients_covering` so
        that one database answers a wrong name the same way a set does:
        provenance naming another database is wrong rather than absent.
        """
        if source is None:
            return None if self.covers_multiple else self
        (owner,) = await self.clients_covering([source])
        return owner

    def _single_session(self, operation: str) -> SingleDatabaseSession:
        """The one database this operation works on.

        Writing, rebuilding and vacuuming all name a database, so they start
        here: the session they get cannot be a set, and the refusal is the same
        sentence whichever operation asked.
        """
        if isinstance(self._session, SingleDatabaseSession):
            return self._session
        covered = ", ".join(sorted(self.source_names))
        raise AmbiguousDatabaseError(
            f"{operation} works on one database, and this client covers "
            f"{covered}; select one with clients_for([name])"
        )

    @property
    def store(self) -> Store:
        """The store of the database this client opened.

        Absent while covering a set: a store has no unambiguous meaning across
        several, and `clients_for` reaches the one holding a given database.
        """
        if not isinstance(self._session, SingleDatabaseSession):
            raise AttributeError("store")
        return self._session.store

    @property
    def document_repository(self) -> DocumentRepository:
        if not isinstance(self._session, SingleDatabaseSession):
            raise AttributeError("document_repository")
        return self._session.document_repository

    @property
    def chunk_repository(self) -> ChunkRepository:
        if not isinstance(self._session, SingleDatabaseSession):
            raise AttributeError("chunk_repository")
        return self._session.chunk_repository

    @property
    def document_item_repository(self) -> DocumentItemRepository:
        if not isinstance(self._session, SingleDatabaseSession):
            raise AttributeError("document_item_repository")
        return self._session.document_item_repository

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
        if self.covers_multiple:
            return get_embedder(config=self._config)
        return self.store.embedder

    @property
    def reranker(self) -> "RerankerBase | None":
        """The configured reranker, built once and reused across searches.

        None when reranking is disabled. Local rerankers load model weights on
        construction, so one per database in a set would load the same weights
        that many times over: a client covering a database for another borrows
        that one's, built on the first query to reach any of them.
        """
        if self._lender is not None:
            return self._lender.reranker
        return self._own_reranker

    @cached_property
    def _own_reranker(self) -> "RerankerBase | None":
        return get_reranker(config=self._config)

    def _resolve_scope(self) -> DatabaseScope:
        """The databases this client covers.

        Resolved once, here or by whoever handed one in. An explicit `db_path`
        says which database to open, so a configured set does not override it.
        """
        if self._scope is not None:
            return self._scope
        scope = DatabaseScope.resolve(
            self._configured, database_path=self._requested_db_path
        )
        if self._requested_sources is not None and self._requested_db_path is None:
            scope = scope.select(self._requested_sources)
        return scope

    async def __aenter__(self):
        """Async context manager entry — initializes store and repositories.

        A borrowed client reuses its session, which its owner closes. A client
        covering several opens their sessions lazily, so `store` and the
        repositories stay unset until one database is named.
        """
        self._closed = False
        if not self._owns_session:
            assert self._session is not None
            return self

        scope = self._resolve_scope()
        if scope.covers_multiple:
            if self._create:
                raise AmbiguousDatabaseError(
                    "create=True creates one database, and this client covers "
                    f"{', '.join(sorted(scope.names))}; name the one to create "
                    "with sources=[name]"
                )
            self._session = FederatedSession(
                scope,
                self._configured,
                skip_validation=self._skip_validation,
                read_only=self._read_only,
            )
            return self

        [ref] = scope.databases
        self._config, db_path = ref.connection(self._configured)

        self._session = await SingleDatabaseSession(
            db_path if db_path is not None else default_db_path(self._config),
            self._config,
            skip_validation=self._skip_validation,
            create=self._create,
            read_only=self._read_only,
            source=ref.name,
        ).open()
        return self

    async def clients_for(self, names: list[str]) -> list["HaikuRAG"]:
        """The clients for these databases, opening any not yet open.

        Opening is per query rather than at entry: a set of 25 configured
        databases is typically queried a few at a time, and a database nobody
        asked for must not be able to fail a query, or be opened for nothing.

        The clients returned borrow their databases from this one and are valid
        only while it is open. Closing one, or entering it as a context manager,
        leaves the database alone; this client closes them all on teardown.
        """
        assert isinstance(self._session, FederatedSession)
        names = _without_repeats(names)
        sessions = await self._session.sessions_for(names)
        return [
            self._facade_for(name, session)
            for name, session in zip(names, sessions, strict=True)
        ]

    def _facade_for(self, name: str, session: SingleDatabaseSession) -> "HaikuRAG":
        """The cached client borrowing this session, made once and kept."""
        facade = self._clients.get(name)
        if facade is None:
            facade = HaikuRAG._from_session(session, lender=self)
            self._clients[name] = facade
        return facade

    @classmethod
    def _covering(
        cls,
        scope: DatabaseScope,
        config: AppConfig | None = None,
        *,
        read_only: bool = False,
        create: bool = False,
        skip_validation: bool = False,
    ) -> "HaikuRAG":
        """A client over databases someone already resolved.

        Internal: the public constructor takes a path or names, and resolving
        those is its own job. This is for callers that did the resolving.
        """
        client = cls(
            config=config,
            read_only=read_only,
            create=create,
            skip_validation=skip_validation,
        )
        client._scope = scope
        return client

    @classmethod
    def _from_session(
        cls, session: SingleDatabaseSession, lender: "HaikuRAG | None" = None
    ) -> "HaikuRAG":
        """A client over a database another session opened and will close.

        `lender` is the client that opened it, whose reranker this one borrows
        rather than building a second copy of the same model.
        """
        client = cls(
            session.db_path, config=session.config, read_only=session.read_only
        )
        client._session = session
        client._owns_session = False
        client._lender = lender
        return client

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
            (client.source, client.store.stored_embedding)
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

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ARG002
        """Async context manager exit.

        Nothing to release is not an error: exiting before entering and exiting
        twice both do nothing. The session stays readable afterwards, so what a
        client covered can still be asked.
        """
        if self._session is None or self._closed:
            return False
        # Branch on what this client covers, not on what it happened to open:
        # a federating client that answered no query has nothing open and no
        # store either.
        if isinstance(self._session, FederatedSession):
            # The wrappers only discard what they cached: the databases are the
            # federated session's to close, and the reranker they searched with
            # is this client's, closed below.
            for facade in self._clients.values():
                await facade._release_own()
            self._clients.clear()
            await self._session.aclose()
            # The set shares this client's embedder and reranker, so this is the
            # only place they are closed — and only if anything built them.
            await self._aclose_cached("embedder")
            await self._aclose_cached("_own_reranker")
            self._closed = True
            return False
        if not self._owns_session:
            await self._release_own()
            return False
        # The session drains, releases its store's embedder and closes; the
        # cached reference here is only discarded, never closed twice.
        await self._release_own()
        await self._session.aclose()
        self._closed = True
        return False

    async def _release_own(self) -> None:
        """Release what this client built, leaving the database to its owner.

        The embedder belongs to the store and is closed with it, so the cached
        reference is only discarded. A borrowed reranker belongs to its lender,
        so only one built here is closed.
        """
        self.__dict__.pop("embedder", None)
        await self._aclose_cached("_own_reranker")

    async def _aclose_cached(self, name: str) -> None:
        """Close a cached_property this client materialized, and discard it.

        Discarded rather than left in place so that re-entering the client
        builds a fresh one instead of reusing something already closed.
        """
        cached = self.__dict__.pop(name, None)
        if cached is not None:
            await aclose_quietly(cached, name)

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

        session = self._single_session("create_document")

        return await create_document(session, content, uri, title, metadata, format)

    async def import_document(
        self,
        docling_document: "DoclingDocument",
        chunks: list[Chunk],
        uri: str | None = None,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> Document:
        from haiku.rag.client.documents import import_document

        session = self._single_session("import_document")

        return await import_document(
            session, docling_document, chunks, uri, title, metadata
        )

    async def import_documents(
        self,
        imports: "list[DocumentImport]",
    ) -> list[Document]:
        from haiku.rag.client.documents import import_documents

        session = self._single_session("import_documents")

        return await import_documents(session, imports)

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

        session = self._single_session("create_document_from_source")

        return await create_document_from_source(
            session,
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

        session = self._single_session("update_document")

        return await update_document(
            session,
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
        if self.covers_multiple:
            return await self._from_any_covered(
                lambda owner: owner.get_document_by_id(document_id)
            )
        return await self._single_session("get_document_by_id").get_document_by_id(
            document_id
        )

    async def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        """Get a chunk by its ID.

        Args:
            chunk_id: The unique identifier of the chunk.

        Returns:
            The Chunk instance if found, None otherwise.
        """
        if self.covers_multiple:
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
        if not self.covers_multiple:
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
        if self.covers_multiple:
            return await self._from_any_covered(
                lambda owner: owner.get_document_by_uri(uri)
            )
        return await self._single_session("get_document_by_uri").get_document_by_uri(
            uri
        )

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
        ``metadata.parent_uri``."""
        return await self._single_session("delete_document").delete_document(
            document_id
        )

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
        if self.covers_multiple:
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
        return await self._single_session("list_documents").list_documents(
            limit=limit, offset=offset, filter=filter, include_content=include_content
        )

    async def count_documents(self, filter: str | None = None) -> int:
        """Count documents with optional filtering.

        Args:
            filter: Optional SQL WHERE clause to filter documents.

        Returns:
            Number of documents matching the criteria.
        """
        if self.covers_multiple:
            counts = await asyncio.gather(
                *(
                    owner.count_documents(filter=filter)
                    for owner in await self.clients_covering()
                )
            )
            return sum(counts)
        return await self.document_repository.count(filter=filter)

    async def _from_any_covered(
        self, lookup: "Callable[[HaikuRAG], Coroutine[Any, Any, Any]]"
    ) -> Any:
        """The first result `lookup` finds in the databases this client covers."""
        found = await first_found(await self.clients_covering(), lookup)
        return None if found is None else found[1]

    def _require_known_sources(self, sources: "list[str] | None") -> None:
        """Fail on a name this client does not cover, opening nothing.

        `clients_covering` answers the same question by opening the databases,
        and a name is wrong whether or not what it names can be opened. `[]`
        passes: a selection of nothing to search names nothing wrong.
        """
        if sources is None:
            return
        covered = set(self.source_names)
        unknown = [name for name in sources if name not in covered]
        if unknown:
            raise UnknownDatabaseError(
                f"unknown database(s) {', '.join(sorted(set(unknown)))}; this "
                f"client covers {', '.join(sorted(covered)) or 'a single unnamed database'}"
            )

    async def clients_covering(
        self, sources: list[str] | None = None
    ) -> list["HaikuRAG"]:
        """The clients covering this selection.

        The named subset for a client covering a set, or this one where it covers
        a single database. Empty for a selection of none, which is not the same as
        `None` for all of them. Every read honouring `sources` decides through
        this, so the rule cannot differ between one operation and another.
        """
        if self.covers_multiple:
            return await self.clients_for(
                list(self.source_names) if sources is None else sources
            )
        if sources is None:
            return [self]
        sources = _without_repeats(sources)
        if not sources:
            return []
        if sources != [self.source]:
            raise UnknownDatabaseError(
                f"unknown database(s) {', '.join(sources) or '(none)'}; this "
                f"client covers {self.source or 'a single unnamed database'}"
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

        if self.covers_multiple:
            return await search_sources(
                self, query, limit, search_type, filter, include_images, sources
            )
        if not await self.clients_covering(sources):
            return []
        results = await search(self, query, limit, search_type, filter, include_images)
        # A database named in config keeps its name even when it is the only one
        # this client covers. Only a legacy single `uri` leaves source unset.
        for result in results:
            result.source = self.source
        return results

    async def expand_context(
        self,
        search_results: list[SearchResult],
    ) -> list[SearchResult]:
        from haiku.rag.client.search import expand_context, expand_sources

        if isinstance(self._session, FederatedSession):
            return await expand_sources(self._session, search_results)
        return await expand_context(
            self._single_session("expand_context"), search_results
        )

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

        return await visualize_chunk(
            self._single_session("visualize_chunk"), chunk, refs, expand
        )

    async def rebuild_database(
        self, mode: RebuildMode = RebuildMode.FULL
    ) -> AsyncGenerator[str, None]:
        from haiku.rag.client.rebuild import rebuild_database

        session = self._single_session("rebuild_database")

        async for doc_id in rebuild_database(session, mode):
            yield doc_id

    async def vacuum(self) -> None:
        """Optimize and clean up old versions across all tables."""
        await self._single_session("vacuum").store.vacuum()

    async def aclose(self) -> None:
        """Release everything this client opened, whatever it covers.

        The teardown `async with` runs, for a caller that owns the client's
        lifetime some other way. Nothing to release is not an error, so this is
        safe before entering and after closing.
        """
        await self.__aexit__(None, None, None)

    def close(self) -> None:
        """Close the connection to the one database this client opened.

        The connection and nothing else: draining the background vacuum and
        releasing the embedder and reranker are awaitable, so `aclose` is what
        does all of it, and `async with` is the usual way to ask for it.

        A client covering one of a set borrows that database and never closes
        it: the set opened it and the set closes it. A client covering a set has
        no single connection to close and refuses.
        """
        if not isinstance(self._session, SingleDatabaseSession):
            raise AmbiguousDatabaseError(
                "close works on one connection, and this client covers "
                f"{', '.join(sorted(self.source_names))}; await aclose() to "
                "release every database it opened"
            )
        if not self._owns_session:
            return
        self._session.close()
