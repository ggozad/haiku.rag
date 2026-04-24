import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import lancedb
import pyarrow as pa
from lancedb.index import FTS, BTree, IvfPq
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field

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
        return await lancedb.connect_async(db_path)


class DocumentRecord(LanceModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    uri: str | None = None
    title: str | None = None
    metadata: str = Field(default="{}")
    docling_document: bytes | None = None
    docling_pages: bytes | None = None
    docling_version: str | None = None
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


class SettingsRecord(LanceModel):
    id: str = Field(default="settings")
    settings: str = Field(default="{}")


REQUIRED_TABLES: tuple[str, ...] = ("documents", "chunks", "document_items", "settings")


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


class Store:
    def __init__(
        self,
        db_path: Path,
        config: AppConfig = Config,
        skip_validation: bool = False,
        create: bool = False,
        read_only: bool = False,
        before: datetime | None = None,
        skip_migration_check: bool = False,
    ):
        self.db_path: Path = db_path
        self._config = config
        self._before = before
        # Time-travel mode is always read-only
        self._read_only = read_only or (before is not None)
        self._create = create
        self._skip_validation = skip_validation
        self._skip_migration_check = skip_migration_check
        self._vacuum_lock = asyncio.Lock()
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

        # Initialize tables (creates them if they don't exist)
        await self._init_tables()

        # Checkout tables to historical state if before is specified
        if self._before is not None:
            await self._checkout_tables_before(self._before)

        # Set version for new databases, check migrations for existing ones
        if is_new_db:
            if not self._read_only:
                await self._set_initial_version()
        elif not self._skip_migration_check:
            await self._check_migrations()

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
        """
        self._assert_writable()

        if self._connection_mode == ConnectionMode.CLOUD:
            return

        # Skip if already running (non-blocking)
        if self._vacuum_lock.locked():
            return

        async with self._vacuum_lock:
            try:
                # Evaluate config at runtime to allow dynamic changes
                if retention_seconds is None:
                    retention_seconds = self._config.storage.vacuum_retention_seconds
                # Perform maintenance per table using optimize() with configurable retention
                retention = timedelta(seconds=retention_seconds)
                for table in [
                    self.documents_table,
                    self.chunks_table,
                    self.document_items_table,
                    self.settings_table,
                ]:
                    await table.optimize(cleanup_older_than=retention)
            except (RuntimeError, OSError) as e:
                # Handle resource errors gracefully
                logger.debug(f"Vacuum skipped due to resource constraints: {e}")

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

    async def _init_tables(self):
        """Initialize database tables (create if they don't exist)."""
        existing_tables = (await self.db.list_tables()).tables
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

        # Create or open chunks table
        if "chunks" in existing_tables:
            self.chunks_table = await self.db.open_table("chunks")
        else:
            self.chunks_table = await self.db.create_table(
                "chunks", schema=self.ChunkRecord
            )
            # Create FTS index on content_fts (contextualized content) for better search
            await self.chunks_table.create_index(
                "content_fts",
                config=FTS(with_position=True, remove_stop_words=False),
                replace=True,
            )

        # Create or open document_items table
        if "document_items" in existing_tables:
            self.document_items_table = await self.db.open_table("document_items")
        else:
            self.document_items_table = await self.db.create_table(
                "document_items", schema=DocumentItemRecord
            )
            await self.document_items_table.create_index(
                "document_id", config=BTree(), replace=True
            )
            await self.document_items_table.create_index(
                "position", config=BTree(), replace=True
            )
            await self.document_items_table.create_index(
                "self_ref", config=BTree(), replace=True
            )

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
        """Check if migrations are pending and error or update version accordingly.

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

        # No pending migrations - update version silently if needed (writable only)
        if not self._read_only and db_version != current_version:
            await self.set_haiku_version(current_version)

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

        # Update version after successful migration
        if applied or db_version != current_version:
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
            settings_data = Config.model_dump(mode="json")
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
        if "chunks" in await self.db.table_names():
            await self.db.drop_table("chunks")

        # Update the ChunkRecord model with new vector dimension
        self.ChunkRecord = create_chunk_model(self.embedder._vector_dim)
        self.chunks_table = await self.db.create_table(
            "chunks", schema=self.ChunkRecord
        )

        # Create FTS index on content_fts (contextualized content) for better search
        await self.chunks_table.create_index(
            "content_fts",
            config=FTS(with_position=True, remove_stop_words=False),
            replace=True,
        )

    def close(self):
        """Close the database connection."""
        # AsyncConnection.close() is synchronous
        if hasattr(self, "db"):
            self.db.close()

    async def current_table_versions(self) -> dict[str, int]:
        """Capture current versions of key tables for rollback using LanceDB's API."""
        return {
            "documents": await self.documents_table.version(),
            "chunks": await self.chunks_table.version(),
            "document_items": await self.document_items_table.version(),
            "settings": await self.settings_table.version(),
        }

    async def restore_table_versions(self, versions: dict[str, int]) -> bool:
        """Restore tables to the provided versions using LanceDB's API.

        Raises:
            ReadOnlyError: If the store is in read-only mode.
        """
        self._assert_writable()
        await self.documents_table.restore(int(versions["documents"]))
        await self.chunks_table.restore(int(versions["chunks"]))
        await self.document_items_table.restore(int(versions["document_items"]))
        await self.settings_table.restore(int(versions["settings"]))
        return True

    @property
    def _connection(self):
        """Compatibility property for repositories expecting _connection."""
        return self

    async def _checkout_tables_before(self, before: datetime) -> None:
        """Checkout all tables to their state at or before the given datetime.

        Args:
            before: The datetime to checkout to

        Raises:
            ValueError: If no version exists before the given datetime
        """
        # LanceDB stores timestamps as naive datetimes in local time.
        # Convert 'before' to naive local time for comparison.
        if before.tzinfo is not None:
            # Convert to local time and make naive
            before_local = before.astimezone().replace(tzinfo=None)
        else:
            # Already naive, assume local time
            before_local = before

        tables = [
            ("documents", self.documents_table),
            ("chunks", self.chunks_table),
            ("document_items", self.document_items_table),
            ("settings", self.settings_table),
        ]

        for table_name, table in tables:
            versions = await table.list_versions()
            # Find the latest version at or before the target datetime
            # Versions are sorted by version number, not timestamp, so we need to check all
            best_version = None
            best_timestamp = None

            for v in versions:
                # LanceDB version timestamps are naive datetime objects in local time
                v_timestamp = v["timestamp"]
                # Make sure it's naive for comparison
                if v_timestamp.tzinfo is not None:
                    v_timestamp = v_timestamp.replace(tzinfo=None)

                if v_timestamp <= before_local:
                    if best_timestamp is None or v_timestamp > best_timestamp:
                        best_version = v["version"]
                        best_timestamp = v_timestamp

            if best_version is None:
                # Find the earliest version to report in error message
                if versions:
                    earliest = min(versions, key=lambda v: v["timestamp"])
                    earliest_ts = earliest["timestamp"]
                    raise ValueError(
                        f"No data exists before {before}. "
                        f"Database was created on {earliest_ts}"
                    )
                else:
                    raise ValueError(
                        f"No data exists before {before}. Table has no versions."
                    )

            # Checkout to the found version
            await table.checkout(best_version)

    async def list_table_versions(self, table_name: str) -> list[dict[str, Any]]:
        """List version history for a table.

        Args:
            table_name: Name of the table ("documents", "chunks", or "settings")

        Returns:
            List of version info dicts with "version" and "timestamp" keys
        """
        table_map = {
            "documents": self.documents_table,
            "chunks": self.chunks_table,
            "document_items": self.document_items_table,
            "settings": self.settings_table,
        }
        table = table_map.get(table_name)
        if table is None:
            raise ValueError(f"Unknown table: {table_name}")

        return list(await table.list_versions())
