import asyncio
import json
import logging
from datetime import timedelta
from importlib import metadata
from pathlib import Path
from uuid import uuid4

import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field

from haiku.rag.config import AppConfig, Config
from haiku.rag.embeddings import get_embedder

logger = logging.getLogger(__name__)


class DocumentRecord(LanceModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    uri: str | None = None
    title: str | None = None
    metadata: str = Field(default="{}")
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
        allow_create: bool = True,
    ):
        self.db_path: Path = db_path
        self._config = config
        self.embedder = get_embedder(config=self._config)
        self._vacuum_lock = asyncio.Lock()

        # Create the ChunkRecord model with the correct vector dimension
        self.ChunkRecord = create_chunk_model(self.embedder._vector_dim)

        # Local filesystem handling for DB directory
        if not self._has_cloud_config():
            if not allow_create:
                # Read operations should not create the database
                if not db_path.exists():
                    raise FileNotFoundError(
                        f"Database does not exist: {db_path}. Use a write operation (add, add-src) to create it."
                    )
            else:
                # Write operations - ensure parent directories exist
                if not db_path.parent.exists():
                    Path.mkdir(db_path.parent, parents=True)

        # Connect to LanceDB
        self.db = self._connect_to_lancedb(db_path)

        # Initialize tables
        self.create_or_update_db()

        # Validate config compatibility after connection is established
        if not skip_validation:
            self._validate_configuration()

    async def vacuum(self, retention_seconds: int | None = None) -> None:
        """Optimize and clean up old versions across all tables to reduce disk usage.

        Args:
            retention_seconds: Retention threshold in seconds. Only versions older
                              than this will be removed. If None, uses config.storage.vacuum_retention_seconds.

        Note:
            If vacuum is already running, this method returns immediately without blocking.
            Use asyncio.create_task(store.vacuum()) for non-blocking background execution.
        """
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

    def _validate_configuration(self) -> None:
        """Validate that the configuration is compatible with the database."""
        from haiku.rag.store.repositories.settings import SettingsRepository

        settings_repo = SettingsRepository(self)
        settings_repo.validate_config_compatibility()

    def create_or_update_db(self):
        """Create the database tables."""

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
            # Create FTS index on the new table with phrase query support
            self.chunks_table.create_fts_index(
                "content", replace=True, with_position=True, remove_stop_words=False
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

        # Run pending upgrades based on stored version and package version
        try:
            from haiku.rag.store.upgrades import run_pending_upgrades

            current_version = metadata.version("haiku.rag-slim")
            db_version = self.get_haiku_version()

            if db_version != "0.0.0":
                run_pending_upgrades(self, db_version, current_version)

            # After upgrades complete (or if none), set stored version
            # to the greater of the installed package version and the
            # highest available upgrade step version in code.
            try:
                from packaging.version import parse as _v

                from haiku.rag.store.upgrades import upgrades as _steps

                highest_step = max((_v(u.version) for u in _steps), default=None)
                effective_version = (
                    str(max(_v(current_version), highest_step))
                    if highest_step is not None
                    else current_version
                )
            except Exception:
                effective_version = current_version

            self.set_haiku_version(effective_version)
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
        """Updates the user version in settings."""
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
        """Recreate the chunks table with current vector dimensions."""
        # Drop and recreate chunks table
        try:
            self.db.drop_table("chunks")
        except Exception:
            pass

        # Update the ChunkRecord model with new vector dimension
        self.ChunkRecord = create_chunk_model(self.embedder._vector_dim)
        self.chunks_table = self.db.create_table("chunks", schema=self.ChunkRecord)

        # Create FTS index on the new table with phrase query support
        self.chunks_table.create_fts_index(
            "content", replace=True, with_position=True, remove_stop_words=False
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
        """Restore tables to the provided versions using LanceDB's API."""
        self.documents_table.restore(int(versions["documents"]))
        self.chunks_table.restore(int(versions["chunks"]))
        self.settings_table.restore(int(versions["settings"]))
        return True

    @property
    def _connection(self):
        """Compatibility property for repositories expecting _connection."""
        return self
