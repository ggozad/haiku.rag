import asyncio
import logging
from pathlib import Path
from time import monotonic

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

logger = logging.getLogger(__name__)

# Throttle for the background auto-vacuum: under sustained ingestion, scheduling
# a compaction on every write degenerates into back-to-back optimize() passes
# that churn the blob-bearing documents table. Fire at most one per interval; a
# final vacuum on close collapses anything throttled here.
_VACUUM_MIN_INTERVAL_S = 300.0


# Failures whose message names the remedy and never the location, so the failing
# database is named alongside it instead of in place of it.
_NAMEABLE_FAILURES = (MigrationRequiredError, ConfigMismatchError, ReadOnlyError)


class SingleDatabaseSession:
    """One database: its store, its repositories, and their lifecycle.

    Everything that needs a store lives here, so nothing above has to ask whether
    it has one. ``source`` is the configured name this database answers to, or
    None where nothing names it.
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
        self._db_path = db_path
        self._config = config
        self._skip_validation = skip_validation
        self._create = create
        self._read_only = read_only
        self.source = source
        self._vacuum_tasks: set[asyncio.Task] = set()
        self._last_vacuum_at: float | None = None
        self._vacuum_dirty = False

    async def open(self) -> "SingleDatabaseSession":
        """Connect, validate, and build the repositories."""
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
            # the caller's `async with` never entered, so its exit won't run.
            try:
                await self.store._initialize()
            except BaseException:
                self.store.close()
                raise
        except _NAMEABLE_FAILURES as error:
            # These say what to run and never where the database is, so the name
            # is added to the message rather than replacing it: the operator needs
            # both which database failed and what to do about it.
            if self.source is None:
                raise
            raise type(error)(f"database {self.source!r}: {error}") from error
        except Exception as error:
            # A legacy `uri` or `db_path` session has no name to report instead,
            # so its error passes through as it always has.
            if self.source is None:
                raise
            failure = type(error).__name__
        if failure is not None:
            # Raised outside the except block on purpose. A database named in
            # config is reported by name, and the original spells out the path or
            # the bucket: `from None` would only stop it being *printed*, leaving
            # it on `__context__` for anything that walks the chain.
            raise SourceUnavailableError(
                f"database {self.source!r} could not be opened: {failure}"
            )
        self.document_repository = DocumentRepository(self.store)
        self.chunk_repository = ChunkRepository(self.store)
        self.document_item_repository = DocumentItemRepository(self.store)
        return self

    async def drain_vacuum(self) -> None:
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

    def close(self) -> None:
        """Close the underlying store connection."""
        self.store.close()
