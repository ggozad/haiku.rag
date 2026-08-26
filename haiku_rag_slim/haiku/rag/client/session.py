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


# Failures whose message names the remedy and never the location, so the failing
# database is named alongside it instead of in place of it.
_NAMEABLE_FAILURES = (MigrationRequiredError, ConfigMismatchError, ReadOnlyError)


async def aclose_quietly(closeable: Any, what: str) -> None:
    """Close, reporting failure to the log rather than raising.

    Teardown can run while an exception unwinds, so a raising close must
    neither mask that exception nor stop a sibling from being closed.
    """
    try:
        await closeable.aclose()
    except Exception:
        logger.debug("Closing the %s failed on teardown", what, exc_info=True)


def default_db_path(config: AppConfig) -> Path:
    """Where a database lives when its location names no path."""
    return config.storage.data_dir / "haiku.rag.lancedb"


class SingleDatabaseSession:
    """One database: its store, its repositories, and their lifecycle.

    Everything that needs a store lives here, so nothing above has to ask whether
    it has one. ``source`` is the configured name this database answers to, or
    None where nothing names it.

    ``db_path``, ``config``, ``read_only`` and ``source`` are readable so that a
    client borrowing this session can report them as its own.
    """

    def __init__(
        self,
        db_path: Path | str,
        config: AppConfig,
        *,
        skip_validation: bool = False,
        create: bool = False,
        read_only: bool = False,
        source: str | None = None,
    ) -> None:
        self.db_path = db_path
        self.config = config
        self.read_only = read_only
        self.source = source
        self._skip_validation = skip_validation
        self._create = create
        self._vacuum_tasks: set[asyncio.Task] = set()
        self._last_vacuum_at: float | None = None
        self._vacuum_dirty = False

    async def open(self) -> "SingleDatabaseSession":
        """Connect, validate, and build the repositories."""
        failure: str | None = None
        try:
            self.store = Store(
                self.db_path,
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
            # These name the remedy and not the database, so the name is added
            # rather than substituted.
            if self.source is None:
                raise
            raise type(error)(f"database {self.source!r}: {error}") from error
        except Exception as error:
            # Without a name there is nothing to report in the location's place.
            if self.source is None:
                raise
            failure = type(error).__name__
        if failure is not None:
            # Raised outside the handler to discard the location-bearing
            # context, which `from None` would only stop printing.
            raise SourceUnavailableError(
                f"database {self.source!r} could not be opened: {failure}"
            )
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

    def name(self, document: "Document | None") -> "Document | None":
        """`document`, told which database it came from."""
        if document is not None:
            document.source = self.source
        return document

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
        documents = await self.document_repository.list_all(
            limit=limit, offset=offset, filter=filter, include_content=include_content
        )
        for document in documents:
            document.source = self.source
        return documents

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

        The store owns the embedder, so releasing it belongs here rather than
        with whoever happened to hold the session.
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
    use rather than at entry: which databases a query covers is a per-query
    choice, so a database nobody asked for must neither be opened for nothing nor
    be able to fail a query.
    """

    def __init__(
        self,
        scope: DatabaseScope,
        config: AppConfig,
        *,
        skip_validation: bool = False,
        read_only: bool = False,
    ) -> None:
        self._refs: dict[str, DatabaseRef] = {
            ref.name: ref for ref in scope.databases if ref.name is not None
        }
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
            raise KeyError(
                f"unknown database(s) {', '.join(sorted(unknown))}; configured: "
                f"{', '.join(sorted(self._refs))}"
            )
        async with self._lock:
            missing = [name for name in names if name not in self._sessions]
            if missing:
                opened = await asyncio.gather(
                    *(self._open(self._refs[name]) for name in missing),
                    return_exceptions=True,
                )
                # Whatever opened is tracked before the failure is reported, so
                # teardown closes it: `gather` does not cancel the siblings of the
                # one that raised, and an untracked connection leaks.
                failure: BaseException | None = None
                for name, result in zip(missing, opened, strict=True):
                    if isinstance(result, BaseException):
                        failure = failure or result
                    else:
                        self._sessions[name] = result
                if failure is not None:
                    raise failure
        return [self._sessions[name] for name in names]

    async def _open(self, ref: DatabaseRef) -> SingleDatabaseSession:
        one, db_path = ref.connection(self._config)
        return await SingleDatabaseSession(
            db_path if db_path is not None else default_db_path(one),
            one,
            skip_validation=self._skip_validation,
            read_only=self._read_only,
            source=ref.name,
        ).open()

    async def aclose(self) -> None:
        """Close every database this session opened."""
        for session in self._sessions.values():
            await aclose_quietly(session, "database")
        self._sessions.clear()
