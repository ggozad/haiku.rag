import asyncio
import json
import logging
from collections.abc import Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import lancedb
import pyarrow as pa
from lancedb.index import FTS, Bitmap, BTree, IvfPq
from lancedb.pydantic import LanceModel, Vector
from packaging.version import parse
from pydantic import BaseModel, Field

from haiku.rag.config import AppConfig, Config
from haiku.rag.embeddings import get_embedder
from haiku.rag.store.exceptions import MigrationRequiredError, ReadOnlyError

if TYPE_CHECKING:
    from lancedb.query import AsyncQueryBase

logger = logging.getLogger(__name__)


async def query_to_pydantic[T: LanceModel](
    query: "AsyncQueryBase", model: type[T]
) -> list[T]:
    """Typed wrapper around AsyncQueryBase.to_pydantic.

    The upstream stub annotates `.to_pydantic()` as returning `list[LanceModel]`
    regardless of the concrete model passed in. This helper narrows the return
    type to the concrete model so attribute access on the results type-checks
    at call sites without needing per-line cast / ignore comments.
    """
    return cast("list[T]", await query.to_pydantic(model))


class ConnectionMode(Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    OBJECT_STORAGE = "object_storage"

    @staticmethod
    def from_config(config: AppConfig) -> "ConnectionMode":
        uri = config.lancedb.uri
        if not uri:
            return ConnectionMode.LOCAL
        if uri.startswith("db://"):
            return ConnectionMode.CLOUD
        return ConnectionMode.OBJECT_STORAGE


async def connect_lancedb(
    config: AppConfig, db_path: Path | None = None
) -> lancedb.AsyncConnection:
    mode = ConnectionMode.from_config(config)
    if mode == ConnectionMode.CLOUD:
        return await lancedb.connect_async(
            uri=config.lancedb.uri,
            api_key=config.lancedb.api_key,
            region=config.lancedb.region,
        )
    elif mode == ConnectionMode.OBJECT_STORAGE:
        kwargs: dict[str, Any] = {"uri": config.lancedb.uri}
        if config.lancedb.storage_options:
            kwargs["storage_options"] = config.lancedb.storage_options
        return await lancedb.connect_async(**kwargs)
    else:
        if db_path is None:
            raise ValueError("No lancedb.uri configured and no db_path provided")
        return await lancedb.connect_async(db_path.absolute())


class DocumentRecord(LanceModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    docling_document: bytes | None = None
    docling_pages: bytes | None = None
    docling_version: str | None = None


class DocumentMetaRecord(LanceModel):
    """Mutable, lightweight document attributes, kept separate from the
    write-once content/blobs in `documents`. Updating these (metadata, title,
    source_revision) must not rewrite the multi-MB docling row."""

    id: str
    uri: str | None = None
    title: str | None = None
    metadata: str = Field(default="{}")
    created_at: str = Field(default_factory=lambda: "")
    updated_at: str = Field(default_factory=lambda: "")


def get_documents_arrow_schema() -> pa.Schema:
    """Generate Arrow schema for documents table with large_binary for docling_document.

    LanceDB maps Python `bytes` to Arrow's `binary` type, which uses 32-bit offsets
    and is limited to ~2GB per column in a fragment. When many large documents
    (with embedded page images) are grouped in a single fragment, this limit is
    exceeded, causing "byte array offset overflow" panics.

    This function overrides the default mapping to use `large_binary` instead,
    which has 64-bit offsets and no practical size limit.
    """
    base_schema = DocumentRecord.to_arrow_schema()
    large_binary_columns = {"docling_document", "docling_pages"}
    fields = []
    for field in base_schema:
        if field.name in large_binary_columns:
            fields.append(pa.field(field.name, pa.large_binary()))
        else:
            fields.append(field)
    return pa.schema(fields)


class ChunkRecordBase(LanceModel):
    """Static base for ChunkRecord — declares the fields so attribute access
    and constructor calls type-check. The concrete `vector` field is overridden
    by create_chunk_model() with a Vector(dim) whose fixed-size-list dimension
    is only known at runtime.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str
    content_fts: str = Field(default="")
    metadata: str = Field(default="{}")
    order: int = Field(default=0)
    vector: list[float] = Field(default_factory=list)


def create_chunk_model(vector_dim: int) -> type[ChunkRecordBase]:
    """Create a ChunkRecord model with the specified vector dimension."""

    class ChunkRecord(ChunkRecordBase):
        vector: Vector(vector_dim) = Field(default_factory=lambda: [0.0] * vector_dim)  # type: ignore

    return ChunkRecord


class DocumentItemRecord(LanceModel):
    document_id: str
    position: int
    self_ref: str
    label: str = Field(default="")
    text: str = Field(default="")
    page_numbers: str = Field(default="[]")
    picture_data: bytes | None = None
    heading_level: int = Field(default=0)
    tree_depth: int = Field(default=0)


def get_document_items_arrow_schema() -> pa.Schema:
    """Generate Arrow schema for document_items with large_binary for picture_data.

    LanceDB maps Python `bytes` to Arrow's `binary` type, which uses 32-bit offsets
    and is limited to ~2GB per column in a fragment. Many embedded picture PNGs in
    one fragment can exceed that limit. `large_binary` uses 64-bit offsets and has
    no practical size limit — same reasoning as `docling_document` on the
    documents table.
    """
    base_schema = DocumentItemRecord.to_arrow_schema()
    large_binary_columns = {"picture_data"}
    fields = []
    for field in base_schema:
        if field.name in large_binary_columns:
            fields.append(pa.field(field.name, pa.large_binary()))
        else:
            fields.append(field)
    return pa.schema(fields)


def index_specs(table_name: str) -> list[tuple[str, Bitmap | BTree | FTS]]:
    """The index set a haiku.rag table is expected to carry.

    Single source of truth for every path that creates a table: initialization,
    migration, and the drop-and-recreate paths in the repositories.

    `label` gets a Bitmap rather than a BTree because it holds around ten
    distinct values across every item of every document, and Bitmap is the
    low-cardinality equality case. The FTS options are load-bearing:
    `with_position` enables phrase queries and keeping stop words lets them match.
    """
    match table_name:
        case "documents":
            return [("id", BTree())]
        case "document_meta":
            return [("id", BTree()), ("uri", BTree())]
        case "chunks":
            return [
                ("content_fts", FTS(with_position=True, remove_stop_words=False)),
                ("id", BTree()),
                ("document_id", BTree()),
            ]
        case "document_items":
            return [
                ("document_id", BTree()),
                ("position", BTree()),
                ("self_ref", BTree()),
                ("label", Bitmap()),
            ]
        case _:
            return []


async def ensure_indexes(table: lancedb.AsyncTable, table_name: str) -> None:
    """Create the table's declared indexes, skipping those already correct.

    Correct means the column is indexed *and* carries the declared index type.
    Matching on the column alone would let a wrong-typed index stand: a BTree on
    `label` covers the column while losing the low-cardinality equality lookup a
    Bitmap gives, and no amount of column coverage reveals that.

    Skipping matters as much as creating: `create_index(replace=True)` rebuilds
    an identical index, writing a fresh index and a new table version rather
    than no-oping, and leaves the previous index behind until the next vacuum.
    On a large table over object storage that is a full column sort per pass.

    Columns not declared for the table are left untouched, so an externally
    added index (a vector index on `chunks`, say) survives.
    """
    indexed = {
        column: index.index_type
        for index in await table.list_indices()
        for column in index.columns
    }
    for column, config in index_specs(table_name):
        if indexed.get(column) == type(config).__name__:
            continue
        await table.create_index(column, config=config, replace=True)


class SettingsRecord(LanceModel):
    id: str = Field(default="settings")
    settings: str = Field(default="{}")


REQUIRED_TABLES: tuple[str, ...] = (
    "documents",
    "document_meta",
    "chunks",
    "document_items",
    "settings",
)

# Keeps the vacuum cleanup cutoff safely older than the oldest tagged
# version; guards against timestamp precision at the boundary.
TAG_RETENTION_MARGIN = timedelta(seconds=1)

# Restore order for multi-table restore and its rollback. documents restores
# last: writes land in it last on the ingest path, making it the closest
# available database commit point.
RESTORE_TABLE_ORDER: tuple[str, ...] = tuple(
    name for name in REQUIRED_TABLES if name != "documents"
) + ("documents",)


async def _wait_protected[T](coro: Coroutine[Any, Any, T]) -> tuple[T, bool]:
    """Await a recovery coroutine that a cancellation cannot interrupt.

    Runs the coroutine as a task and keeps waiting for it even if this
    coroutine is cancelled, so a Ctrl-C cannot leave recovery half applied.
    Returns the result and whether a cancellation was absorbed; the caller
    must re-deliver an absorbed cancellation.
    """
    task = asyncio.ensure_future(coro)
    cancelled = False
    while True:
        try:
            return await asyncio.shield(task), cancelled
        except asyncio.CancelledError:
            if task.cancelled():
                # The recovery coroutine itself ended cancelled; there is
                # nothing left to wait for. A task that completed (even in
                # the same tick as the cancellation) still returns its
                # result on the next pass.
                raise
            cancelled = True


def _safety_tag_name(existing: set[str]) -> str:
    """Collision-resistant name for the pre-restore safety tag."""
    base = f"before-restore-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    if base not in existing:
        return base
    n = 2
    while f"{base}-{n}" in existing:
        n += 1
    return f"{base}-{n}"


@dataclass
class TagInfo:
    """A database-level tag aggregated across all tables.

    A complete tag names the same tag on every table; a partial one (created
    outside haiku.rag or left behind by a failure) lists the tables it is
    missing from.
    """

    tables: dict[str, int]
    missing_tables: list[str]

    @property
    def complete(self) -> bool:
        return not self.missing_tables


async def get_database_stats(db: lancedb.AsyncConnection) -> dict:
    """Collect stats for every haiku.rag table on the connection.

    Missing tables return ``{"exists": False}``. Present tables include
    ``num_rows``, ``total_bytes``, and ``num_versions``. The ``chunks``
    entry additionally reports vector index status and, when an index
    exists, ``num_indexed_rows`` and ``num_unindexed_rows``.
    """
    existing = set((await db.list_tables()).tables)
    stats: dict = {}
    tables: dict = {}

    for name in REQUIRED_TABLES:
        if name not in existing:
            stats[name] = {"exists": False}
            continue
        tbl = await db.open_table(name)
        tables[name] = tbl
        # lancedb's .stats() stub claims TableStatistics but returns a plain dict at runtime.
        tbl_stats: dict = await tbl.stats()  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        stats[name] = {
            "exists": True,
            "num_rows": tbl_stats.get("num_rows", 0),
            "total_bytes": tbl_stats.get("total_bytes", 0),
            "num_versions": len(await tbl.list_versions()),
        }

    if stats["chunks"]["exists"]:
        chunks_tbl = tables["chunks"]
        indices = await chunks_tbl.list_indices()
        has_vector_index = any("vector" in str(idx).lower() for idx in indices)
        stats["chunks"]["has_vector_index"] = has_vector_index
        if has_vector_index:
            index_stats = await chunks_tbl.index_stats("vector_idx")
            if index_stats is not None:
                stats["chunks"]["num_indexed_rows"] = index_stats.num_indexed_rows
                stats["chunks"]["num_unindexed_rows"] = index_stats.num_unindexed_rows

    return stats


class EmbeddingsInfo(BaseModel):
    provider: str = "unknown"
    name: str = "unknown"
    vector_dim: int | None = None


class TableInfo(BaseModel):
    name: str
    exists: bool
    num_rows: int = 0
    total_bytes: int = 0
    num_versions: int = 0


class VectorIndexInfo(BaseModel):
    exists: bool = False
    indexed_rows: int = 0
    unindexed_rows: int = 0


class PendingMigration(BaseModel):
    version: str
    description: str = ""


class DatabaseInfo(BaseModel):
    """Structured snapshot of a haiku.rag database, shared by the `info` CLI
    command and the ingester control plane. Read-only; gathered without
    opening a Store."""

    path: str
    exists: bool
    stored_version: str = "unknown"
    embeddings: EmbeddingsInfo = Field(default_factory=EmbeddingsInfo)
    tables: list[TableInfo] = Field(default_factory=list)
    vector_index: VectorIndexInfo = Field(default_factory=VectorIndexInfo)
    pending_migrations: list[PendingMigration] = Field(default_factory=list)
    packages: dict[str, str] = Field(default_factory=dict)


async def gather_database_info(config: AppConfig, db_path: Path) -> DatabaseInfo:
    """Collect read-only database state without going through Store, so a
    database missing tables (e.g. pre-migration) still reports what it can."""
    from haiku.rag.store.upgrades import get_pending_upgrades
    from haiku.rag.utils import get_package_versions

    display_path = config.lancedb.uri or str(db_path)

    db = await connect_lancedb(config, db_path)
    stats = await get_database_stats(db)

    if not any(entry["exists"] for entry in stats.values()):
        return DatabaseInfo(path=display_path, exists=False)

    stored_version = "unknown"
    embeddings = EmbeddingsInfo()
    if stats["settings"]["exists"]:
        settings_tbl = await db.open_table("settings")
        rows = (
            await settings_tbl.query().where("id = 'settings'").limit(1).to_arrow()
        ).to_pylist()
        if rows:
            raw = rows[0].get("settings") or "{}"
            data = json.loads(raw) if isinstance(raw, str) else (raw or {})
            stored_version = str(data.get("version", "unknown"))
            model = data.get("embeddings", {}).get("model", {})
            embeddings = EmbeddingsInfo(
                provider=model.get("provider", "unknown"),
                name=model.get("name", "unknown"),
                vector_dim=model.get("vector_dim"),
            )

    tables = [
        TableInfo(
            name=name,
            exists=stats[name]["exists"],
            num_rows=stats[name].get("num_rows", 0),
            total_bytes=stats[name].get("total_bytes", 0),
            num_versions=stats[name].get("num_versions", 0),
        )
        for name in ("documents", "document_meta", "chunks", "document_items")
    ]

    vector_index = VectorIndexInfo()
    if stats["chunks"]["exists"] and stats["chunks"].get("has_vector_index"):
        vector_index = VectorIndexInfo(
            exists=True,
            indexed_rows=stats["chunks"].get("num_indexed_rows", 0),
            unindexed_rows=stats["chunks"].get("num_unindexed_rows", 0),
        )

    pending = (
        get_pending_upgrades(stored_version) if stored_version != "unknown" else []
    )

    return DatabaseInfo(
        path=display_path,
        exists=True,
        stored_version=stored_version,
        embeddings=embeddings,
        tables=tables,
        vector_index=vector_index,
        pending_migrations=[
            PendingMigration(version=step.version, description=step.description or "")
            for step in pending
        ],
        packages=get_package_versions(),
    )


class Store:
    def __init__(
        self,
        db_path: Path,
        config: AppConfig = Config,
        skip_validation: bool = False,
        create: bool = False,
        read_only: bool = False,
        skip_migration_check: bool = False,
    ):
        self.db_path: Path = db_path
        self._config = config
        self._read_only = read_only
        self._create = create
        self._skip_validation = skip_validation
        self._skip_migration_check = skip_migration_check
        self._vacuum_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        # Held by rebuild_database for its whole run; tag operations check it
        # and fail fast instead of snapshotting a half-rebuilt database.
        self._rebuild_lock = asyncio.Lock()
        self._is_new_db = False

        # Check if database exists (for local filesystem only)
        if self._connection_mode == ConnectionMode.LOCAL:
            if not db_path.exists():
                if not create:
                    raise FileNotFoundError(
                        f"Database does not exist at {self.db_path.absolute()}. "
                        "Use 'haiku-rag init' to create a new database."
                    )
                self._is_new_db = True
                # Ensure parent directories exist for new databases
                if not db_path.parent.exists():
                    Path.mkdir(db_path.parent, parents=True)

        # Create embedder (sync — no LanceDB needed)
        self.embedder = get_embedder(config=self._config)

    async def _initialize(self):
        """Perform async initialization: connect to LanceDB, init tables, validate."""
        # Connect to LanceDB
        self.db: lancedb.AsyncConnection = await connect_lancedb(
            self._config, self.db_path
        )

        # For remote stores (and as a safety net for local paths that exist but
        # have no tables — e.g. a previously failed init), detect new DB by
        # checking whether any tables exist.
        is_new_db = self._is_new_db
        if not is_new_db:
            existing_tables = (await self.db.list_tables()).tables
            if not existing_tables:
                is_new_db = True

        # For existing databases, read stored vector dimension to create ChunkRecord
        # that can read existing chunks. For new databases, use config's dimension.
        stored_vector_dim = None
        if not is_new_db:
            stored_vector_dim = await self._get_stored_vector_dim()

        # Create ChunkRecord with stored dimension (for reading) or config dimension (for new DB)
        chunk_vector_dim = stored_vector_dim or self.embedder._vector_dim
        self.ChunkRecord: type[ChunkRecordBase] = create_chunk_model(chunk_vector_dim)

        # Initialize tables (creates them if they don't exist). For an existing
        # DB this raises MigrationRequiredError up front when migrations are
        # pending, before creating any newly-introduced table.
        await self._init_tables(is_new_db)

        # Set version for new databases.
        if is_new_db and not self._read_only:
            await self._set_initial_version()

        # Validate config compatibility after connection is established
        if not self._skip_validation:
            await self._validate_configuration()

    async def __aenter__(self):
        # If _initialize connects to LanceDB but then fails (e.g. migration
        # check, config validation), close the connection so it doesn't
        # leak — __aexit__ won't run because the `async with` never entered.
        try:
            await self._initialize()
        except BaseException:
            self.close()
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ARG002
        self.close()
        return False

    @property
    def is_read_only(self) -> bool:
        """Whether the store is in read-only mode."""
        return self._read_only

    async def _get_stored_vector_dim(self) -> int | None:
        """Read the stored vector dimension from the settings table.

        Returns:
            The stored vector dimension, or None if not found.
        """
        try:
            existing_tables = (await self.db.list_tables()).tables
            if "settings" not in existing_tables:
                return None

            settings_table = await self.db.open_table("settings")
            rows = (
                await settings_table.query()
                .where("id = 'settings'")
                .limit(1)
                .to_arrow()
            ).to_pylist()
            if not rows or not rows[0].get("settings"):
                return None

            settings = json.loads(rows[0]["settings"])
            embeddings = settings.get("embeddings", {})
            model = embeddings.get("model", {})
            return model.get("vector_dim")
        except Exception:
            return None

    def _assert_writable(self) -> None:
        """Raise ReadOnlyError if the store is in read-only mode."""
        if self._read_only:
            raise ReadOnlyError("Cannot modify database in read-only mode")

    def _assert_not_rebuilding(self) -> None:
        """Raise if a rebuild is in progress in this process."""
        if self._rebuild_lock.locked():
            raise ValueError(
                "Rebuild in progress; tag operations are unavailable until it completes"
            )

    async def vacuum(self, retention_seconds: int | None = None) -> None:
        """Optimize and clean up old versions across all tables to reduce disk usage.

        Args:
            retention_seconds: Retention threshold in seconds. Only versions older
                              than this will be removed. If None, uses config.storage.vacuum_retention_seconds.

        Note:
            If vacuum is already running, this method returns immediately without blocking.
            Use asyncio.create_task(store.vacuum()) for non-blocking background execution.

        Raises:
            ReadOnlyError: If the store is in read-only mode.
            RuntimeError: On lance errors during optimize; only OSError
                (resource pressure) skips the pass.
        """
        self._assert_writable()

        if self._connection_mode == ConnectionMode.CLOUD:
            return

        # Skip if already running (non-blocking)
        if self._vacuum_lock.locked():
            return

        async with self._vacuum_lock, self._write_lock:
            try:
                # Evaluate config at runtime to allow dynamic changes
                if retention_seconds is None:
                    retention_seconds = self._config.storage.vacuum_retention_seconds
                # Perform maintenance per table using optimize() with configurable retention
                retention = timedelta(seconds=retention_seconds)
                for table in self._tables().values():
                    await table.optimize(
                        cleanup_older_than=await self._tag_safe_retention(
                            table, retention
                        )
                    )
            except OSError as e:
                # Resource errors (e.g. disk pressure) skip the pass; lance
                # errors surface as RuntimeError and must not be swallowed —
                # a silently skipped cleanup hides tag-interaction bugs.
                logger.debug(f"Vacuum skipped due to resource constraints: {e}")

    async def _tag_safe_retention(
        self, table: lancedb.AsyncTable, retention: timedelta
    ) -> timedelta:
        """Grow the retention so the cleanup cutoff stays older than the
        table's oldest tagged version.

        Lance hard-errors when a tagged version falls inside the cleanup
        window and the Python API exposes no way to skip tagged versions, so
        the oldest tagged version and everything newer are retained; versions
        older than the oldest tag remain eligible for cleanup.
        """
        tags = await table.tags.list()
        if not tags:
            return retention

        timestamps = {v["version"]: v["timestamp"] for v in await table.list_versions()}
        tagged = [
            timestamps[tag["version"]]
            for tag in tags.values()
            if tag["version"] in timestamps
        ]
        if not tagged:  # pragma: no cover - vacuum never cleans a tagged version
            return retention

        # LanceDB version timestamps are naive datetimes in local time.
        oldest = min(ts.replace(tzinfo=None) for ts in tagged)
        needed = datetime.now() - oldest + TAG_RETENTION_MARGIN
        return max(retention, needed)

    @property
    def _connection_mode(self) -> ConnectionMode:
        return ConnectionMode.from_config(self._config)

    async def _ensure_vector_index(self) -> None:
        """Create or rebuild vector index on chunks table.

        Cloud deployments auto-create indexes, so we skip for those.
        For self-hosted, creates an IVF_PQ index. If an index exists,
        it will be replaced (using replace=True parameter).
        Note: Index creation requires sufficient training data.
        """
        if self._connection_mode == ConnectionMode.CLOUD:
            return

        try:
            # Check if table has enough data (indexes require training data)
            row_count = await self.chunks_table.count_rows()
            if row_count < 256:
                logger.debug(
                    f"Skipping vector index creation: need at least 256 rows, have {row_count}"
                )
                return

            # Create or replace index (replace=True is the default)
            logger.info("Creating vector index on chunks table...")
            await self.chunks_table.create_index(
                "vector",
                config=IvfPq(
                    distance_type=self._config.search.vector_index_metric,
                ),
                replace=True,
            )

            # Wait for index creation to complete
            # Index name is column_name + "_idx"
            await self.chunks_table.wait_for_index(
                ["vector_idx"], timeout=timedelta(hours=1)
            )

            logger.info("Vector index created successfully")
        except Exception as e:
            logger.warning(f"Could not create vector index: {e}")

    async def _validate_configuration(self) -> None:
        """Validate that the configuration is compatible with the database."""
        from haiku.rag.store.repositories.settings import SettingsRepository

        settings_repo = SettingsRepository(self)
        await settings_repo.validate_config_compatibility()

    async def _init_tables(self, is_new_db: bool):
        """Initialize database tables (create if they don't exist)."""
        existing_tables = (await self.db.list_tables()).tables

        # Surface pending migrations BEFORE creating any newly-introduced table.
        # Otherwise opening a legacy DB would either mutate it (creating an empty
        # document_meta on open) or raise the wrong ReadOnlyError instead of
        # telling the user to run `haiku-rag migrate`. The settings table exists
        # on any non-new DB, which is all _check_migrations needs.
        if (
            not is_new_db
            and not self._skip_migration_check
            and "settings" in existing_tables
        ):
            self.settings_table = await self.db.open_table("settings")
            await self._check_migrations()

        missing_tables = set(REQUIRED_TABLES) - set(existing_tables)

        if missing_tables and self._read_only:
            raise ReadOnlyError(
                "Cannot create tables in read-only mode. "
                "Use 'haiku-rag init' to create a new database."
            )

        # Create or open documents table
        if "documents" in existing_tables:
            self.documents_table = await self.db.open_table("documents")
        else:
            self.documents_table = await self.db.create_table(
                "documents", schema=get_documents_arrow_schema()
            )
            await ensure_indexes(self.documents_table, "documents")

        # Create or open document_meta table (mutable attributes kept out of the
        # blob-bearing documents row).
        if "document_meta" in existing_tables:
            self.document_meta_table = await self.db.open_table("document_meta")
        else:
            self.document_meta_table = await self.db.create_table(
                "document_meta", schema=DocumentMetaRecord
            )
            await ensure_indexes(self.document_meta_table, "document_meta")

        # Create or open chunks table
        if "chunks" in existing_tables:
            self.chunks_table = await self.db.open_table("chunks")
        else:
            self.chunks_table = await self.db.create_table(
                "chunks", schema=self.ChunkRecord
            )
            await ensure_indexes(self.chunks_table, "chunks")

        # Create or open document_items table
        if "document_items" in existing_tables:
            self.document_items_table = await self.db.open_table("document_items")
        else:
            self.document_items_table = await self.db.create_table(
                "document_items", schema=get_document_items_arrow_schema()
            )
            await ensure_indexes(self.document_items_table, "document_items")

        # Create or open settings table
        if "settings" in existing_tables:
            self.settings_table = await self.db.open_table("settings")
        else:
            self.settings_table = await self.db.create_table(
                "settings", schema=SettingsRecord
            )
            # Save current settings to the new database
            settings_data = self._config.model_dump(mode="json")
            await self.settings_table.add(
                [SettingsRecord(id="settings", settings=json.dumps(settings_data))]
            )

    async def _set_initial_version(self):
        """Set the initial version for a new database."""
        await self.set_haiku_version(metadata.version("haiku.rag-slim"))

    async def _check_migrations(self) -> None:
        """Raise if migrations are pending. Opening never writes the version.

        Raises:
            MigrationRequiredError: If migrations are pending.
        """
        from haiku.rag.store.upgrades import get_pending_upgrades

        current_version = metadata.version("haiku.rag-slim")
        db_version = await self.get_haiku_version()

        pending = get_pending_upgrades(db_version)

        if pending:
            # Migrations are pending - require explicit migrate command
            raise MigrationRequiredError(
                f"Database requires migration from {db_version} to {current_version}. "
                f"{len(pending)} migration(s) pending. "
                "Run 'haiku-rag migrate' to upgrade."
            )

    async def migrate(self) -> list[str]:
        """Run pending database migrations.

        Returns:
            List of descriptions of applied upgrades.

        Raises:
            ReadOnlyError: If the store is in read-only mode.
        """
        self._assert_writable()

        from haiku.rag.store.upgrades import run_pending_upgrades

        db_version = await self.get_haiku_version()
        current_version = metadata.version("haiku.rag-slim")

        applied = await run_pending_upgrades(self, db_version)

        # Advance the schema marker only forward — never downgrade a database
        # opened with an older build than last stamped it.
        if parse(current_version) > parse(db_version):
            await self.set_haiku_version(current_version)

        return applied

    async def get_haiku_version(self) -> str:
        """Returns the user version stored in settings."""
        settings_records = await query_to_pydantic(
            self.settings_table.query().limit(1), SettingsRecord
        )
        if settings_records:
            settings = (
                json.loads(settings_records[0].settings)
                if settings_records[0].settings
                else {}
            )
            return settings.get("version", "0.0.0")
        return "0.0.0"

    async def set_haiku_version(self, version: str) -> None:
        """Updates the user version in settings.

        Raises:
            ReadOnlyError: If the store is in read-only mode.
        """
        self._assert_writable()
        settings_records = await query_to_pydantic(
            self.settings_table.query().limit(1), SettingsRecord
        )
        if settings_records:
            # Only write if version actually changes to avoid creating new table versions
            current = (
                json.loads(settings_records[0].settings)
                if settings_records[0].settings
                else {}
            )
            if current.get("version") != version:
                current["version"] = version
                await self.settings_table.update(
                    {"settings": json.dumps(current)},
                    where="id = 'settings'",
                )
        else:
            # Create new settings record
            settings_data = self._config.model_dump(mode="json")
            settings_data["version"] = version
            await self.settings_table.add(
                [SettingsRecord(id="settings", settings=json.dumps(settings_data))]
            )

    async def recreate_embeddings_table(self) -> None:
        """Recreate the chunks table with current vector dimensions.

        Raises:
            ReadOnlyError: If the store is in read-only mode.
        """
        self._assert_writable()
        # Drop and recreate chunks table. Check existence first rather than
        # catching-and-swallowing drop_table's errors — a catch-all would
        # hide real failures (permissions, storage-backend errors) and then
        # the subsequent create_table would fail confusingly.
        if "chunks" in (await self.db.list_tables()).tables:
            await self.db.drop_table("chunks")

        # Update the ChunkRecord model with new vector dimension
        self.ChunkRecord = create_chunk_model(self.embedder._vector_dim)
        self.chunks_table = await self.db.create_table(
            "chunks", schema=self.ChunkRecord
        )
        await ensure_indexes(self.chunks_table, "chunks")

    def close(self):
        """Close the database connection."""
        # AsyncConnection.close() is synchronous
        if hasattr(self, "db"):
            self.db.close()

    def _tables(self) -> dict[str, lancedb.AsyncTable]:
        """Map every haiku.rag table name to its open AsyncTable."""
        return {
            "documents": self.documents_table,
            "document_meta": self.document_meta_table,
            "chunks": self.chunks_table,
            "document_items": self.document_items_table,
            "settings": self.settings_table,
        }

    async def current_table_versions(self) -> dict[str, int]:
        """Capture current versions of key tables for rollback using LanceDB's API."""
        return {name: await table.version() for name, table in self._tables().items()}

    async def restore_table_versions(self, versions: dict[str, int]) -> bool:
        """Restore tables to the provided versions using LanceDB's API.

        Raises:
            ReadOnlyError: If the store is in read-only mode.
        """
        self._assert_writable()
        for name, table in self._tables().items():
            await table.restore(int(versions[name]))
        return True

    async def create_tag(self, name: str) -> None:
        """Tag the current version of every table with the given name.

        Serializes with client writes via the write lock so a write cannot
        land between the version snapshot and the per-table tag creation.
        This is in-process coordination only: a writer in another process
        can commit between the per-table version reads, so create tags with
        all other writers stopped when a consistent snapshot matters.

        Raises:
            ReadOnlyError: If the store is in read-only mode.
            ValueError: If a rebuild is in progress, or if the tag already
                exists on any table. A partial tag (present on some tables
                only) must be deleted before the name can be reused.
        """
        self._assert_writable()
        self._assert_not_rebuilding()

        async with self._rebuild_lock, self._write_lock:
            await self._create_tag_locked(name)

    async def _create_tag_locked(self, name: str) -> None:
        """Create a tag on every table; the caller must hold the write lock."""
        tables = self._tables()

        existing = [
            table_name
            for table_name, table in tables.items()
            if name in await table.tags.list()
        ]
        if len(existing) == len(tables):
            raise ValueError(f"Tag '{name}' already exists")
        if existing:
            raise ValueError(
                f"Tag '{name}' already exists on some tables "
                f"({', '.join(existing)}); delete it first with delete_tag"
            )

        versions = await self.current_table_versions()
        try:
            for table_name, table in tables.items():
                await table.tags.create(name, versions[table_name])
        except BaseException as exc:
            # BaseException: cancellation must also trigger cleanup, and the
            # cleanup itself is protected from further cancellation. The
            # sweep covers all tables, not only the recorded ones: a
            # cancellation can land after lance committed a table's tag but
            # before this attempt recorded it, and preflight guarantees the
            # name was unused, so any occurrence belongs to this attempt.
            (_, failed_cleanup), cancelled = await _wait_protected(
                self._delete_tag_locked(name)
            )
            if failed_cleanup:
                raise RuntimeError(
                    f"Tag '{name}' creation failed ({exc!r}) and cleanup "
                    f"failed on: {', '.join(failed_cleanup)}. A partial "
                    "tag may remain; delete it with delete_tag."
                ) from exc
            if cancelled and not isinstance(exc, asyncio.CancelledError):
                raise asyncio.CancelledError()
            raise

    async def _delete_tag_locked(self, name: str) -> tuple[bool, list[str]]:
        """Delete the tag from every table that has it; the caller must
        hold the write lock.

        Returns whether the tag was found anywhere and the tables where
        listing or deletion failed.
        """
        found = False
        failed: list[str] = []
        for table_name, table in self._tables().items():
            try:
                if name in await table.tags.list():
                    found = True
                    await table.tags.delete(name)
            except Exception:
                failed.append(table_name)
        return found, failed

    async def list_tags(self) -> dict[str, TagInfo]:
        """Aggregate per-table tags into database-level tags.

        Returns:
            Tag name mapped to a TagInfo with the tagged version per table
            and the tables the tag is missing from (empty when complete).
        """
        tables = self._tables()
        tags: dict[str, TagInfo] = {}
        for table_name, table in tables.items():
            for tag_name, tag in (await table.tags.list()).items():
                info = tags.setdefault(tag_name, TagInfo(tables={}, missing_tables=[]))
                info.tables[table_name] = tag["version"]
        for info in tags.values():
            info.missing_tables = [t for t in tables if t not in info.tables]
        return tags

    async def delete_tag(self, name: str) -> None:
        """Delete the tag from every table that has it.

        Serializes with create_tag and client writes via the write lock.

        Raises:
            ReadOnlyError: If the store is in read-only mode.
            ValueError: If a rebuild is in progress or no table has the tag.
            RuntimeError: If deletion failed on some tables; remnants remain
                until a retry succeeds.
        """
        self._assert_writable()
        self._assert_not_rebuilding()
        async with self._rebuild_lock, self._write_lock:
            found, failed = await self._delete_tag_locked(name)
            if failed:
                # A listing failure obscures whether the tag exists on that
                # table, so failures take precedence over not-found.
                raise RuntimeError(
                    f"Tag '{name}' deletion failed on: {', '.join(failed)}. "
                    "Remnants may remain; retry delete_tag."
                )
            if not found:
                raise ValueError(f"Tag '{name}' does not exist")

    async def _restore_tables(
        self, versions: dict[str, int], *, best_effort: bool = False
    ) -> list[tuple[str, Exception]]:
        """Restore every table to the given versions, documents last.

        Stops at the first failure by default; with best_effort, continues
        through all tables. Returns the failures either way.
        """
        tables = self._tables()
        failures: list[tuple[str, Exception]] = []
        for table_name in RESTORE_TABLE_ORDER:
            try:
                await tables[table_name].restore(int(versions[table_name]))
            except Exception as exc:
                failures.append((table_name, exc))
                if not best_effort:
                    break
        return failures

    async def _rollback_to_snapshot(
        self, snapshot: dict[str, int]
    ) -> tuple[list[tuple[str, Exception]], bool]:
        """Best-effort rollback that a cancellation cannot interrupt.

        Returns the rollback failures and whether a cancellation was
        absorbed; the caller must re-deliver an absorbed cancellation.
        """
        return await _wait_protected(self._restore_tables(snapshot, best_effort=True))

    async def restore_tag(self, name: str) -> str:
        """Restore every table to the versions of a complete tag.

        Creates a complete safety tag for the pre-restore state before
        changing any table and returns its name. Each table restore writes a
        new latest version; nothing is left checked out read-only.

        In-process coordination only: all other writers must be stopped for
        the duration of the operation.

        Raises:
            ReadOnlyError: If the store is in read-only mode.
            ValueError: If a rebuild is in progress, the tag does not exist,
                or the tag is partial.
            RuntimeError: If the safety tag could not be created (no table
                changed), or a table restore failed (the error states whether
                rollback succeeded).
        """
        self._assert_writable()
        self._assert_not_rebuilding()

        async with self._rebuild_lock, self._write_lock:
            tags = await self.list_tags()
            info = tags.get(name)
            if info is None:
                raise ValueError(f"Tag '{name}' does not exist")
            if not info.complete:
                raise ValueError(
                    f"Tag '{name}' is partial (missing tables: "
                    f"{', '.join(info.missing_tables)}) and cannot be "
                    "restored; delete it with delete_tag"
                )

            snapshot = await self.current_table_versions()
            safety_tag = _safety_tag_name(set(tags))
            try:
                await self._create_tag_locked(safety_tag)
            except Exception as exc:
                raise RuntimeError(
                    f"Restore of tag '{name}' did not begin: safety tag "
                    f"creation failed ({exc}). No table was changed."
                ) from exc

            try:
                failures = await self._restore_tables(info.tables)
            except asyncio.CancelledError:
                # CancelledError is a BaseException and escapes the
                # per-table handler; roll back before re-raising.
                rollback_failures, _ = await self._rollback_to_snapshot(snapshot)
                if rollback_failures:
                    failed_names = ", ".join(t for t, _ in rollback_failures)
                    raise RuntimeError(
                        f"Restore of tag '{name}' was cancelled and rollback "
                        f"failed on: {failed_names}. The database may be "
                        f"cross-table inconsistent; manual recovery is "
                        f"required using safety tag '{safety_tag}'."
                    ) from None
                raise
            if failures:
                failed_table, cause = failures[0]
                rollback_failures, cancelled = await self._rollback_to_snapshot(
                    snapshot
                )
                if rollback_failures:
                    failed_names = ", ".join(t for t, _ in rollback_failures)
                    raise RuntimeError(
                        f"Restore of tag '{name}' failed on table "
                        f"'{failed_table}' and rollback failed on: "
                        f"{failed_names}. The database may be cross-table "
                        f"inconsistent; manual recovery is required using "
                        f"safety tag '{safety_tag}'."
                    ) from cause
                if cancelled:
                    raise asyncio.CancelledError()
                raise RuntimeError(
                    f"Restore of tag '{name}' failed on table "
                    f"'{failed_table}'; all tables were rolled back to the "
                    f"pre-restore state. Safety tag '{safety_tag}' is "
                    "preserved."
                ) from cause

            return safety_tag

    async def list_table_versions(self, table_name: str) -> list[dict[str, Any]]:
        """List version history for a table.

        Args:
            table_name: Name of the table ("documents", "document_meta",
                "chunks", "document_items", or "settings")

        Returns:
            List of version info dicts with "version" and "timestamp" keys
        """
        table = self._tables().get(table_name)
        if table is None:
            raise ValueError(f"Unknown table: {table_name}")

        return list(await table.list_versions())
