import asyncio
import logging
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any

from haiku.rag.client.scope import DatabaseRef, DatabaseScope
from haiku.rag.config import AppConfig
from haiku.rag.store.engine import Store
from haiku.rag.store.exceptions import (
    ConfigMismatchError,
    MigrationRequiredError,
    ReadOnlyError,
    SourceUnavailableError,
    UnknownDatabaseError,
)
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.repositories.document_item import DocumentItemRepository

if TYPE_CHECKING:
    from haiku.rag.store.models.document import Document

logger = logging.getLogger(__name__)

# Throttle for the background auto-vacuum: under sustained ingestion, scheduling
# a compaction on every write degenerates into back-to-back optimize() passes
# that churn the blob-bearing documents table. Fire at most one per interval; a
# final vacuum on close collapses anything throttled here.
_VACUUM_MIN_INTERVAL_S = 300.0


# Failures whose message names the remedy and never the location. `open()`
# prefixes the failing database's name.
_NAMEABLE_FAILURES = (MigrationRequiredError, ConfigMismatchError, ReadOnlyError)


async def aclose_quietly(closeable: Any, what: str) -> None:
    """Close; a failure is logged, never raised."""
    try:
        await closeable.aclose()
    except Exception:
        logger.debug("Closing the %s failed on teardown", what, exc_info=True)


class SingleDatabaseSession:
    """One database: its store, its repositories, and their lifecycle.

    Everything that needs a store lives here, so nothing above has to ask whether
    it has one. Built from the resolved reference: ``source`` is the name it
    answers to, and the store receives its location.

    ``ref``, ``config``, ``read_only`` and ``source`` are readable: a client
    borrowing this session reports them as its own.
    """

    def __init__(
        self,
        ref: DatabaseRef,
        config: AppConfig,
        *,
        skip_validation: bool = False,
        create: bool = False,
        read_only: bool = False,
    ) -> None:
        self.ref = ref
        self.config = config
        self.read_only = read_only
        self._skip_validation = skip_validation
        self._create = create
        self._vacuum_tasks: set[asyncio.Task] = set()
        self._last_vacuum_at: float | None = None
        self._vacuum_dirty = False

    @property
    def source(self) -> str:
        return self.ref.name

    @property
    def location(self) -> Path | str:
        """Where this database is: its path, or its URI."""
        return self.ref.location

    @property
    def db_path(self) -> Path | None:
        """The local path, or None for a database behind a URI."""
        return self.ref.db_path

    async def open(self) -> "SingleDatabaseSession":
        """Connect, validate, and build the repositories."""
        failure: str | None = None
        try:
            self.store = Store(
                self.location,
                config=self.config,
                skip_validation=self._skip_validation,
                create=self._create,
                read_only=self.read_only,
            )
            # Close a partially initialized store: the caller's `async with`
            # never entered, so its exit will not run.
            try:
                await self.store._initialize()
            except BaseException:
                self.store.close()
                raise
        except _NAMEABLE_FAILURES as error:
            # The message keeps its remedy and gains the database's name.
            if self.ref.given:
                raise
            raise type(error)(f"database {self.source!r}: {error}") from error
        except Exception as error:
            # A path the caller gave may be named: the caller knows it already.
            if self.ref.given:
                raise
            failure = (
                "does not exist; create it with `haiku-rag init` or `create=True`"
                if isinstance(error, FileNotFoundError)
                else f"could not be opened: {type(error).__name__}"
            )
        if failure is not None:
            # Raised outside the handler: the exception carries neither a cause
            # nor a location-bearing context.
            raise SourceUnavailableError(f"database {self.source!r} {failure}")
        self.document_repository = DocumentRepository(self.store)
        self.chunk_repository = ChunkRepository(self.store)
        self.document_item_repository = DocumentItemRepository(self.store)
        return self

    async def drain_vacuum(self) -> None:
        """Drain background vacuum work and run a final collapse before teardown.

        The final pass runs whenever writes happened, not when tasks remain: a
        debounced run may have scheduled none and still left versions behind. It
        never runs without writes, so opening and closing a store writes nothing.
        """
        if self._vacuum_tasks:
            await asyncio.gather(*self._vacuum_tasks, return_exceptions=True)
        if not self._vacuum_dirty:
            return
        self._vacuum_dirty = False
        # Teardown runs during exception unwinding; a raising vacuum here would
        # mask the original exception, so the drain stays best-effort.
        try:
            await self.store.vacuum()
        except Exception:
            logger.debug("Final vacuum on close failed", exc_info=True)

    def schedule_vacuum(self) -> None:
        """Schedule a background vacuum, throttled to at most one per
        ``_VACUUM_MIN_INTERVAL_S``. The throttle only skips the background task —
        ``_vacuum_dirty`` still marks that a final vacuum on close is owed."""
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

    def name(self, document: "Document | None") -> "Document | None":
        """`document`, told which database it came from."""
        if document is not None:
            document.source = self.source
        return document

    def name_all(self, documents: "list[Document]") -> "list[Document]":
        """`documents`, each told which database it came from."""
        for document in documents:
            document.source = self.source
        return documents

    async def get_document_by_id(self, document_id: str) -> "Document | None":
        return self.name(await self.document_repository.get_by_id(document_id))

    async def get_document_by_uri(self, uri: str) -> "Document | None":
        return self.name(await self.document_repository.get_by_uri(uri))

    async def list_documents(
        self,
        limit: int | None = None,
        offset: int | None = None,
        filter: str | None = None,
        include_content: bool = False,
    ) -> "list[Document]":
        return self.name_all(
            await self.document_repository.list_all(
                limit=limit,
                offset=offset,
                filter=filter,
                include_content=include_content,
            )
        )

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document, cascading to children linked via
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

        if self.config.storage.auto_vacuum:
            self.schedule_vacuum()
        return True

    async def aclose(self) -> None:
        """Drain, release the embedder, and close the connection.

        The store owns the embedder, so this is where it is released.
        """
        await self.drain_vacuum()
        await aclose_quietly(self.store.embedder, "embedder")
        self.close()

    def close(self) -> None:
        """Close the underlying store connection."""
        self.store.close()


class FederatedSession:
    """Several databases, read as one. Reads only.

    Composes single-database sessions and owns their teardown. They open on first
    use: which databases a query covers is a per-query choice, and a database the
    query does not cover stays closed.
    """

    def __init__(
        self,
        scope: DatabaseScope,
        config: AppConfig,
        *,
        skip_validation: bool = False,
        read_only: bool = False,
    ) -> None:
        self._refs: dict[str, DatabaseRef] = {ref.name: ref for ref in scope.databases}
        self._config = config
        self._skip_validation = skip_validation
        self._read_only = read_only
        self._sessions: dict[str, SingleDatabaseSession] = {}
        self._lock = asyncio.Lock()

    @property
    def names(self) -> tuple[str, ...]:
        """The databases covered, in configured order."""
        return tuple(self._refs)

    async def sessions_for(self, names: list[str]) -> list[SingleDatabaseSession]:
        """The sessions for these databases, opening any not yet open.

        Missing ones open together: on object storage a serial loop makes the
        first query cost the sum of the opens.
        """
        unknown = [name for name in names if name not in self._refs]
        if unknown:
            raise UnknownDatabaseError(
                f"unknown database(s) {', '.join(sorted(unknown))}; configured: "
                f"{', '.join(sorted(self._refs))}"
            )
        async with self._lock:
            missing = [name for name in names if name not in self._sessions]
            if missing:
                opened = await asyncio.gather(
                    *(self._open(name) for name in missing),
                    return_exceptions=True,
                )
                for result in opened:
                    if isinstance(result, BaseException):
                        raise result
        return [self._sessions[name] for name in names]

    async def _open(self, name: str) -> None:
        """Open and register one database before returning to the fan-out.

        Registered here because a cancelled `gather` discards its results.
        """
        self._sessions[name] = await SingleDatabaseSession(
            self._refs[name],
            self._config,
            skip_validation=self._skip_validation,
            read_only=self._read_only,
        ).open()

    async def aclose(self) -> None:
        """Close every database this session opened."""
        for session in self._sessions.values():
            await aclose_quietly(session, "database")
        self._sessions.clear()
