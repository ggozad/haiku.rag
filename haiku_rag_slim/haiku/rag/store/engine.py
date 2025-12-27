import asyncio
import json
import logging
from datetime import datetime, timedelta
from importlib import metadata
from pathlib import Path
from typing import Any
from uuid import uuid4

import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field

from haiku.rag.config import AppConfig, Config
from haiku.rag.embeddings import get_embedder
from haiku.rag.store.exceptions import ReadOnlyError

logger = logging.getLogger(__name__)


class DocumentRecord(LanceModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    uri: str | None = None
    title: str | None = None
    metadata: str = Field(default="{}")
    docling_document_json: str | None = None
    docling_version: str | None = None
    created_at: str = Field(default_factory=lambda: "")
    updated_at: str = Field(default_factory=lambda: "")


def create_chunk_model(vector_dim: int):
    """Create a ChunkRecord model with the specified vector dimension.

    This creates a model with proper vector typing for LanceDB.
    """

    class ChunkRecord(LanceModel):
        id: str = Field(default_factory=lambda: str(uuid4()))
        document_id: str
        content: str
        content_fts: str = Field(default="")
        metadata: str = Field(default="{}")
        order: int = Field(default=0)
        vector: Vector(vector_dim) = Field(default_factory=lambda: [0.0] * vector_dim)  # type: ignore

    return ChunkRecord


class SettingsRecord(LanceModel):
    id: str = Field(default="settings")
    settings: str = Field(default="{}")


class Store:
    def __init__(
        self,
        db_path: Path,
        config: AppConfig = Config,
        skip_validation: bool = False,
        create: bool = False,
        read_only: bool = False,
        before: datetime | None = None,
    ):
        self.db_path: Path = db_path
        self._config = config
        self._before = before
        # Time-travel mode is always read-only
        self._read_only = read_only or (before is not None)
        self.embedder = get_embedder(config=self._config)
        self._vacuum_lock = asyncio.Lock()

        # Create the ChunkRecord model with the correct vector dimension
        self.ChunkRecord = create_chunk_model(self.embedder._vector_dim)

        # Check if database exists (for local filesystem only)
        is_new_db = False
        if not self._has_cloud_config():
            if not db_path.exists():
                if not create:
                    raise FileNotFoundError(
                        f"Database does not exist at {self.db_path.absolute()}. "
                        "Use 'haiku-rag init' to create a new database."
                    )
                is_new_db = True
                # Ensure parent directories exist for new databases
                if not db_path.parent.exists():
                    Path.mkdir(db_path.parent, parents=True)

        # Connect to LanceDB
        self.db = self._connect_to_lancedb(db_path)

        # Initialize tables (creates them if they don't exist)
        self._init_tables()

        # Checkout tables to historical state if before is specified
        if before is not None:
            self._checkout_tables_before(before)

        # Run upgrades only on existing databases, set version for new ones
        # Skip upgrades in read-only mode (they would fail anyway)
        if not self._read_only:
            if is_new_db:
                self._set_initial_version()
            else:
                self._run_upgrades()

        # Validate config compatibility after connection is established
        if not skip_validation:
            self._validate_configuration()

    @property
    def is_read_only(self) -> bool:
        """Whether the store is in read-only mode."""
        return self._read_only

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

        if self._has_cloud_config() and str(self._config.lancedb.uri).startswith(
            "db://"
        ):
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
                    self.settings_table,
                ]:
                    table.optimize(cleanup_older_than=retention)
            except (RuntimeError, OSError) as e:
                # Handle resource errors gracefully
                logger.debug(f"Vacuum skipped due to resource constraints: {e}")

    def _connect_to_lancedb(self, db_path: Path):
        """Establish connection to LanceDB (local, cloud, or object storage)."""
        # Check if we have cloud configuration
        if self._has_cloud_config():
            return lancedb.connect(
                uri=self._config.lancedb.uri,
                api_key=self._config.lancedb.api_key,
                region=self._config.lancedb.region,
            )
        else:
            # Local file system connection
            return lancedb.connect(db_path)

    def _has_cloud_config(self) -> bool:
        """Check if cloud configuration is complete."""
        return bool(
            self._config.lancedb.uri
            and self._config.lancedb.api_key
            and self._config.lancedb.region
        )

    def get_stats(self) -> dict:
        """Get comprehensive table statistics.

        Returns:
            Dictionary with statistics for documents and chunks tables including:
            - Row counts
            - Storage sizes
            - Vector index status and statistics
        """
        stats_dict: dict = {
            "documents": {"exists": False},
            "chunks": {"exists": False},
        }

        # Documents table stats
        doc_stats: dict = self.documents_table.stats()  # type: ignore[assignment]
        stats_dict["documents"] = {
            "exists": True,
            "num_rows": doc_stats.get("num_rows", 0),
            "total_bytes": doc_stats.get("total_bytes", 0),
        }

        # Chunks table stats
        chunk_stats: dict = self.chunks_table.stats()  # type: ignore[assignment]
        stats_dict["chunks"] = {
            "exists": True,
            "num_rows": chunk_stats.get("num_rows", 0),
            "total_bytes": chunk_stats.get("total_bytes", 0),
        }

        # Vector index stats
        indices = self.chunks_table.list_indices()
        has_vector_index = any("vector" in str(idx).lower() for idx in indices)
        stats_dict["chunks"]["has_vector_index"] = has_vector_index

        if has_vector_index:
            index_stats = self.chunks_table.index_stats("vector_idx")
            if index_stats is not None:
                stats_dict["chunks"]["num_indexed_rows"] = index_stats.num_indexed_rows
                stats_dict["chunks"]["num_unindexed_rows"] = (
                    index_stats.num_unindexed_rows
                )

        return stats_dict

    def _ensure_vector_index(self) -> None:
        """Create or rebuild vector index on chunks table.

        Cloud deployments auto-create indexes, so we skip for those.
        For self-hosted, creates an IVF_PQ index. If an index exists,
        it will be replaced (using replace=True parameter).
        Note: Index creation requires sufficient training data.
        """
        if self._has_cloud_config():
            return

        try:
            # Check if table has enough data (indexes require training data)
            row_count = self.chunks_table.count_rows()
            if row_count < 256:
                logger.debug(
                    f"Skipping vector index creation: need at least 256 rows, have {row_count}"
                )
                return

            # Create or replace index (replace=True is the default)
            logger.info("Creating vector index on chunks table...")
            self.chunks_table.create_index(
                metric=self._config.search.vector_index_metric,
                index_type="IVF_PQ",
                replace=True,  # Explicit: replace existing index
            )

            # Wait for index creation to complete
            # Index name is column_name + "_idx"
            self.chunks_table.wait_for_index(["vector_idx"], timeout=timedelta(hours=1))

            logger.info("Vector index created successfully")
        except Exception as e:
            logger.warning(f"Could not create vector index: {e}")

    def _validate_configuration(self) -> None:
        """Validate that the configuration is compatible with the database."""
        from haiku.rag.store.repositories.settings import SettingsRepository

        settings_repo = SettingsRepository(self)
        settings_repo.validate_config_compatibility()

    def _init_tables(self):
        """Initialize database tables (create if they don't exist)."""
        # Get list of existing tables
        existing_tables = self.db.table_names()

        # Create or get documents table
        if "documents" in existing_tables:
            self.documents_table = self.db.open_table("documents")
        else:
            self.documents_table = self.db.create_table(
                "documents", schema=DocumentRecord
            )

        # Create or get chunks table
        if "chunks" in existing_tables:
            self.chunks_table = self.db.open_table("chunks")
        else:
            self.chunks_table = self.db.create_table("chunks", schema=self.ChunkRecord)
            # Create FTS index on content_fts (contextualized content) for better search
            self.chunks_table.create_fts_index(
                "content_fts", replace=True, with_position=True, remove_stop_words=False
            )

        # Create or get settings table
        if "settings" in existing_tables:
            self.settings_table = self.db.open_table("settings")
        else:
            self.settings_table = self.db.create_table(
                "settings", schema=SettingsRecord
            )
            # Save current settings to the new database
            settings_data = self._config.model_dump(mode="json")
            self.settings_table.add(
                [SettingsRecord(id="settings", settings=json.dumps(settings_data))]
            )

    def _set_initial_version(self):
        """Set the initial version for a new database."""
        self.set_haiku_version(metadata.version("haiku.rag-slim"))

    def _run_upgrades(self):
        """Run pending database upgrades."""
        try:
            from haiku.rag.store.upgrades import run_pending_upgrades

            current_version = metadata.version("haiku.rag-slim")
            db_version = self.get_haiku_version()

            run_pending_upgrades(self, db_version, current_version)

            self.set_haiku_version(current_version)
        except Exception as e:
            # Avoid hard failure on initial connection; log and continue so CLI remains usable.
            logger.warning(
                "Skipping upgrade due to error (db=%s -> pkg=%s): %s",
                self.get_haiku_version(),
                metadata.version("haiku.rag-slim"),
                e,
            )

    def get_haiku_version(self) -> str:
        """Returns the user version stored in settings."""
        settings_records = list(
            self.settings_table.search().limit(1).to_pydantic(SettingsRecord)
        )
        if settings_records:
            settings = (
                json.loads(settings_records[0].settings)
                if settings_records[0].settings
                else {}
            )
            return settings.get("version", "0.0.0")
        return "0.0.0"

    def set_haiku_version(self, version: str) -> None:
        """Updates the user version in settings.

        Raises:
            ReadOnlyError: If the store is in read-only mode.
        """
        self._assert_writable()
        settings_records = list(
            self.settings_table.search().limit(1).to_pydantic(SettingsRecord)
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
                self.settings_table.update(
                    where="id = 'settings'",
                    values={"settings": json.dumps(current)},
                )
        else:
            # Create new settings record
            settings_data = Config.model_dump(mode="json")
            settings_data["version"] = version
            self.settings_table.add(
                [SettingsRecord(id="settings", settings=json.dumps(settings_data))]
            )

    def recreate_embeddings_table(self) -> None:
        """Recreate the chunks table with current vector dimensions.

        Raises:
            ReadOnlyError: If the store is in read-only mode.
        """
        self._assert_writable()
        # Drop and recreate chunks table
        try:
            self.db.drop_table("chunks")
        except Exception:
            pass

        # Update the ChunkRecord model with new vector dimension
        self.ChunkRecord = create_chunk_model(self.embedder._vector_dim)
        self.chunks_table = self.db.create_table("chunks", schema=self.ChunkRecord)

        # Create FTS index on content_fts (contextualized content) for better search
        self.chunks_table.create_fts_index(
            "content_fts", replace=True, with_position=True, remove_stop_words=False
        )

    def close(self):
        """Close the database connection."""
        # LanceDB connections are automatically managed
        pass

    def current_table_versions(self) -> dict[str, int]:
        """Capture current versions of key tables for rollback using LanceDB's API."""
        return {
            "documents": int(self.documents_table.version),
            "chunks": int(self.chunks_table.version),
            "settings": int(self.settings_table.version),
        }

    def restore_table_versions(self, versions: dict[str, int]) -> bool:
        """Restore tables to the provided versions using LanceDB's API.

        Raises:
            ReadOnlyError: If the store is in read-only mode.
        """
        self._assert_writable()
        self.documents_table.restore(int(versions["documents"]))
        self.chunks_table.restore(int(versions["chunks"]))
        self.settings_table.restore(int(versions["settings"]))
        return True

    @property
    def _connection(self):
        """Compatibility property for repositories expecting _connection."""
        return self

    def _checkout_tables_before(self, before: datetime) -> None:
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
            ("settings", self.settings_table),
        ]

        for table_name, table in tables:
            versions = table.list_versions()
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
            table.checkout(best_version)

    def list_table_versions(self, table_name: str) -> list[dict[str, Any]]:
        """List version history for a table.

        Args:
            table_name: Name of the table ("documents", "chunks", or "settings")

        Returns:
            List of version info dicts with "version" and "timestamp" keys
        """
        table_map = {
            "documents": self.documents_table,
            "chunks": self.chunks_table,
            "settings": self.settings_table,
        }
        table = table_map.get(table_name)
        if table is None:
            raise ValueError(f"Unknown table: {table_name}")

        return list(table.list_versions())
