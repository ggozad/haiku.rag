import json
import sqlite3
import struct
from pathlib import Path
from uuid import uuid4

from rich.console import Console
from rich.progress import Progress, TaskID

from haiku.rag.store.engine import Store


def deserialize_sqlite_embedding(data: bytes) -> list[float]:
    """Deserialize sqlite-vec embedding from bytes."""
    if not data:
        return []
    # sqlite-vec stores embeddings as float32 arrays
    num_floats = len(data) // 4
    return list(struct.unpack(f"{num_floats}f", data))


class SQLiteToLanceDBMigrator:
    """Migrates data from SQLite to LanceDB."""

    def __init__(self, sqlite_path: Path, lancedb_path: Path):
        self.sqlite_path = sqlite_path
        self.lancedb_path = lancedb_path
        self.console = Console()

    def migrate(self) -> bool:
        """Perform the migration."""
        try:
            self.console.print(
                f"[blue]Starting migration from {self.sqlite_path} to {self.lancedb_path}[/blue]"
            )

            # Check if SQLite database exists
            if not self.sqlite_path.exists():
                self.console.print(
                    f"[red]SQLite database not found: {self.sqlite_path}[/red]"
                )
                return False

            # Connect to SQLite database
            sqlite_conn = sqlite3.connect(self.sqlite_path)
            sqlite_conn.row_factory = sqlite3.Row

            # Load the sqlite-vec extension
            try:
                import sqlite_vec  # type: ignore

                sqlite_conn.enable_load_extension(True)
                sqlite_vec.load(sqlite_conn)
                self.console.print("[cyan]Loaded sqlite-vec extension[/cyan]")
            except Exception as e:
                self.console.print(
                    f"[yellow]Warning: Could not load sqlite-vec extension: {e}[/yellow]"
                )
                self.console.print(
                    "[yellow]Install sqlite-vec with[/yellow]\n[green]uv pip install sqlite-vec [/green]"
                )
                exit(1)

            # Create LanceDB store
            lance_store = Store(self.lancedb_path, skip_validation=True)

            with Progress() as progress:
                # Migrate documents
                doc_task = progress.add_task(
                    "[green]Migrating documents...", total=None
                )
                document_id_mapping = self._migrate_documents(
                    sqlite_conn, lance_store, progress, doc_task
                )

                # Migrate chunks and embeddings
                chunk_task = progress.add_task(
                    "[yellow]Migrating chunks and embeddings...", total=None
                )
                self._migrate_chunks(
                    sqlite_conn, lance_store, progress, chunk_task, document_id_mapping
                )

                # Migrate settings
                settings_task = progress.add_task(
                    "[blue]Migrating settings...", total=None
                )
                self._migrate_settings(
                    sqlite_conn, lance_store, progress, settings_task
                )

            sqlite_conn.close()

            # Optimize and cleanup using centralized vacuum
            self.console.print("[cyan]Optimizing LanceDB...[/cyan]")
            try:
                lance_store.vacuum()
                self.console.print("[green]✅ Optimization completed[/green]")
            except Exception as e:
                self.console.print(
                    f"[yellow]Warning: Optimization failed: {e}[/yellow]"
                )

            lance_store.close()

            self.console.print("[green]✅ Migration completed successfully![/green]")
            self.console.print(
                f"[green]✅ Migrated {len(document_id_mapping)} documents[/green]"
            )
            return True

        except Exception as e:
            self.console.print(f"[red]❌ Migration failed: {e}[/red]")
            import traceback

            self.console.print(f"[red]{traceback.format_exc()}[/red]")
            return False

    def _migrate_documents(
        self,
        sqlite_conn: sqlite3.Connection,
        lance_store: Store,
        progress: Progress,
        task: TaskID,
    ) -> dict[int, str]:
        """Migrate documents from SQLite to LanceDB and return ID mapping."""
        cursor = sqlite_conn.cursor()
        cursor.execute(
            "SELECT id, content, uri, metadata, created_at, updated_at FROM documents ORDER BY id"
        )

        documents = []
        id_mapping = {}  # Maps old integer ID to new UUID

        for row in cursor.fetchall():
            new_uuid = str(uuid4())
            id_mapping[row["id"]] = new_uuid

            doc_data = {
                "id": new_uuid,
                "content": row["content"],
                "uri": row["uri"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            documents.append(doc_data)

        # Batch insert documents to LanceDB
        if documents:
            from haiku.rag.store.engine import DocumentRecord

            doc_records = [
                DocumentRecord(
                    id=doc["id"],
                    content=doc["content"],
                    uri=doc["uri"],
                    metadata=json.dumps(doc["metadata"]),
                    created_at=doc["created_at"],
                    updated_at=doc["updated_at"],
                )
                for doc in documents
            ]
            lance_store.documents_table.add(doc_records)

        progress.update(task, completed=len(documents), total=len(documents))
        return id_mapping

    def _migrate_chunks(
        self,
        sqlite_conn: sqlite3.Connection,
        lance_store: Store,
        progress: Progress,
        task: TaskID,
        document_id_mapping: dict[int, str],
    ):
        """Migrate chunks and embeddings from SQLite to LanceDB."""
        cursor = sqlite_conn.cursor()

        # Get chunks first
        cursor.execute("""
            SELECT id, document_id, content, metadata
            FROM chunks
            ORDER BY id
        """)

        chunks_data = cursor.fetchall()

        # Get embeddings using the sqlite-vec virtual table
        embeddings_map = {}
        try:
            # Use the virtual table to get embeddings properly
            cursor.execute("""
                SELECT chunk_id, embedding
                FROM chunk_embeddings
            """)

            for row in cursor.fetchall():
                chunk_id = row[0]
                embedding_blob = row[1]
                if embedding_blob and chunk_id not in embeddings_map:
                    embeddings_map[chunk_id] = embedding_blob

        except sqlite3.OperationalError as e:
            self.console.print(
                f"[yellow]Warning: Could not extract embeddings from virtual table: {e}[/yellow]"
            )

        chunks = []
        for row in chunks_data:
            # Generate new UUID for chunk
            chunk_uuid = str(uuid4())

            # Map the old document_id to new UUID
            document_uuid = document_id_mapping.get(row["document_id"])
            if not document_uuid:
                self.console.print(
                    f"[yellow]Warning: Document ID {row['document_id']} not found in mapping for chunk {row['id']}[/yellow]"
                )
                continue

            # Get embedding for this chunk
            embedding = []
            embedding_blob = embeddings_map.get(row["id"])
            if embedding_blob:
                try:
                    embedding = deserialize_sqlite_embedding(embedding_blob)
                except Exception as e:
                    self.console.print(
                        f"[yellow]Warning: Failed to deserialize embedding for chunk {row['id']}: {e}[/yellow]"
                    )
                    # Generate a zero vector of the expected dimension
                    embedding = [0.0] * lance_store.embedder._vector_dim
            else:
                # No embedding found, generate zero vector
                embedding = [0.0] * lance_store.embedder._vector_dim

            chunk_data = {
                "id": chunk_uuid,
                "document_id": document_uuid,
                "content": row["content"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "vector": embedding,
            }
            chunks.append(chunk_data)

        # Batch insert chunks to LanceDB
        if chunks:
            chunk_records = [
                lance_store.ChunkRecord(
                    id=chunk["id"],
                    document_id=chunk["document_id"],
                    content=chunk["content"],
                    metadata=json.dumps(chunk["metadata"]),
                    vector=chunk["vector"],
                )
                for chunk in chunks
            ]
            lance_store.chunks_table.add(chunk_records)

        progress.update(task, completed=len(chunks), total=len(chunks))

    def _migrate_settings(
        self,
        sqlite_conn: sqlite3.Connection,
        lance_store: Store,
        progress: Progress,
        task: TaskID,
    ):
        """Migrate settings from SQLite to LanceDB."""
        cursor = sqlite_conn.cursor()

        try:
            cursor.execute("SELECT id, settings FROM settings WHERE id = 1")
            row = cursor.fetchone()

            if row:
                settings_data = json.loads(row["settings"]) if row["settings"] else {}

                # Update the existing settings in LanceDB (use string ID)
                lance_store.settings_table.update(
                    where="id = 'settings'",
                    values={"settings": json.dumps(settings_data)},
                )

                progress.update(task, completed=1, total=1)
            else:
                progress.update(task, completed=0, total=0)

        except sqlite3.OperationalError:
            # Settings table doesn't exist in old SQLite database
            self.console.print(
                "[yellow]No settings table found in SQLite database[/yellow]"
            )
            progress.update(task, completed=0, total=0)


async def migrate_sqlite_to_lancedb(
    sqlite_path: Path, lancedb_path: Path | None = None
) -> bool:
    """
    Migrate an existing SQLite database to LanceDB.

    Args:
        sqlite_path: Path to the existing SQLite database
        lancedb_path: Path for the new LanceDB database (optional, will auto-generate if not provided)

    Returns:
        True if migration was successful, False otherwise
    """
    if lancedb_path is None:
        # Auto-generate LanceDB path
        lancedb_path = sqlite_path.parent / (sqlite_path.stem + ".lancedb")

    migrator = SQLiteToLanceDBMigrator(sqlite_path, lancedb_path)
    return migrator.migrate()
