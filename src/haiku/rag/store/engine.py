import sqlite3
import struct
from pathlib import Path
from typing import Literal

import sqlite_vec

from haiku.rag.embeddings import get_embedder


class Store:
    def __init__(
        self, db_path: Path | Literal[":memory:"], skip_validation: bool = False
    ):
        self.db_path: Path | Literal[":memory:"] = db_path
        self._connection = self.create_db()

        # Validate config compatibility after connection is established
        if not skip_validation:
            from haiku.rag.store.repositories.settings import SettingsRepository

            settings_repo = SettingsRepository(self)
            settings_repo.validate_config_compatibility()

    def create_db(self) -> sqlite3.Connection:
        """Create the database and tables with sqlite-vec support for embeddings."""
        db = sqlite3.connect(self.db_path)
        db.enable_load_extension(True)
        sqlite_vec.load(db)

        # Create documents table
        db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                uri TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create chunks table
        db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            )
        """)

        # Create vector table for chunk embeddings
        embedder = get_embedder()
        db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{embedder._vector_dim}]
            )
        """)

        # Create FTS5 table for full-text search
        db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                content='chunks',
                content_rowid='id'
            )
        """)

        # Create settings table for storing current configuration
        db.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY DEFAULT 1,
                settings TEXT NOT NULL DEFAULT '{}'
            )
        """)

        # Create indexes for better performance
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)"
        )

        # Save current settings to the new database
        from haiku.rag.config import Config

        settings_json = Config.model_dump_json()
        db.execute(
            "INSERT OR IGNORE INTO settings (id, settings) VALUES (1, ?)",
            (settings_json,),
        )

        db.commit()
        return db

    def recreate_embeddings_table(self) -> None:
        """Recreate the embeddings table with current vector dimensions."""
        if self._connection is None:
            raise ValueError("Store connection is not available")

        # Drop existing embeddings table
        self._connection.execute("DROP TABLE IF EXISTS chunk_embeddings")

        # Recreate with current dimensions
        embedder = get_embedder()
        self._connection.execute(f"""
            CREATE VIRTUAL TABLE chunk_embeddings USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{embedder._vector_dim}]
            )
        """)

        self._connection.commit()

    @staticmethod
    def serialize_embedding(embedding: list[float]) -> bytes:
        """Serialize a list of floats to bytes for sqlite-vec storage."""
        return struct.pack(f"{len(embedding)}f", *embedding)

    def close(self):
        """Close the database connection if it's an in-memory database."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
