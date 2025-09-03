import json
import logging
from importlib import metadata
from pathlib import Path
from uuid import uuid4

import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field

from haiku.rag.config import Config
from haiku.rag.embeddings import get_embedder

logger = logging.getLogger(__name__)


class DocumentRecord(LanceModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    uri: str | None = None
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
        vector: Vector(vector_dim) = Field(default_factory=lambda: [0.0] * vector_dim)  # type: ignore

    return ChunkRecord


class SettingsRecord(LanceModel):
    id: str = Field(default="settings")
    settings: str = Field(default="{}")


class Store:
    def __init__(self, db_path: Path, skip_validation: bool = False):
        self.db_path: Path = db_path
        self.embedder = get_embedder()

        # Create the ChunkRecord model with the correct vector dimension
        self.ChunkRecord = create_chunk_model(self.embedder._vector_dim)

        # Connect to LanceDB
        self.db = self._connect_to_lancedb(db_path)

        # Initialize tables
        self.create_or_update_db()

        # Validate config compatibility after connection is established
        if not skip_validation:
            self._validate_configuration()

    def _connect_to_lancedb(self, db_path: Path):
        """Establish connection to LanceDB (local, cloud, or object storage)."""
        # Check if we have cloud configuration
        if self._has_cloud_config():
            return lancedb.connect(
                uri=Config.LANCEDB_URI,
                api_key=Config.LANCEDB_API_KEY,
                region=Config.LANCEDB_REGION,
            )
        else:
            # Local file system connection
            return lancedb.connect(db_path)

    def _has_cloud_config(self) -> bool:
        """Check if cloud configuration is complete."""
        return bool(
            Config.LANCEDB_URI and Config.LANCEDB_API_KEY and Config.LANCEDB_REGION
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
            # Create FTS index on the new table
            self.chunks_table.create_fts_index("content", replace=True)

        # Create or get settings table
        if "settings" in existing_tables:
            self.settings_table = self.db.open_table("settings")
        else:
            self.settings_table = self.db.create_table(
                "settings", schema=SettingsRecord
            )
            # Save current settings to the new database
            settings_data = Config.model_dump(mode="json")
            self.settings_table.add(
                [SettingsRecord(id="settings", settings=json.dumps(settings_data))]
            )

        # Set current version in settings
        current_version = metadata.version("haiku.rag")
        self.set_haiku_version(current_version)

        # Check if we need to perform upgrades
        try:
            existing_settings = list(
                self.settings_table.search().limit(1).to_pydantic(SettingsRecord)
            )
            if existing_settings:
                db_version = self.get_haiku_version()  # noqa: F841
                # TODO: Add upgrade logic here similar to SQLite version when needed
        except Exception:
            # Settings table might not exist yet in fresh databases
            pass

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
            settings = (
                json.loads(settings_records[0].settings)
                if settings_records[0].settings
                else {}
            )
            settings["version"] = version
            # Update the record
            self.settings_table.update(
                where="id = 'settings'", values={"settings": json.dumps(settings)}
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

        # Create FTS index on the new table
        self.chunks_table.create_fts_index("content", replace=True)

    def close(self):
        """Close the database connection."""
        # LanceDB connections are automatically managed
        pass

    @property
    def _connection(self):
        """Compatibility property for repositories expecting _connection."""
        return self
