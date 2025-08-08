import json
from typing import TYPE_CHECKING

from docling_core.types.doc.document import DoclingDocument

from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.base import BaseRepository
from haiku.rag.utils import text_to_docling_document

if TYPE_CHECKING:
    from haiku.rag.store.models.chunk import Chunk


class DocumentRepository(BaseRepository[Document]):
    """Repository for Document database operations."""

    def __init__(self, store, chunk_repository=None):
        super().__init__(store)
        # Avoid circular import by using late import if not provided
        if chunk_repository is None:
            from haiku.rag.store.repositories.chunk import ChunkRepository

            chunk_repository = ChunkRepository(store)
        self.chunk_repository = chunk_repository

    async def _create_with_docling(
        self,
        entity: Document,
        docling_document: DoclingDocument,
        chunks: list["Chunk"] | None = None,
    ) -> Document:
        """Create a document with its chunks and embeddings."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()

        # Start transaction
        cursor.execute("BEGIN TRANSACTION")

        try:
            # Insert the document
            cursor.execute(
                """
                INSERT INTO documents (content, uri, metadata, created_at, updated_at)
                VALUES (:content, :uri, :metadata, :created_at, :updated_at)
                """,
                {
                    "content": entity.content,
                    "uri": entity.uri,
                    "metadata": json.dumps(entity.metadata),
                    "created_at": entity.created_at,
                    "updated_at": entity.updated_at,
                },
            )

            document_id = cursor.lastrowid
            assert document_id is not None, "Failed to create document in database"
            entity.id = document_id

            # Create chunks - either use provided chunks or generate from content
            if chunks is not None:
                # Use provided chunks, but update their document_id and set order from list position
                for order, chunk in enumerate(chunks):
                    chunk.document_id = document_id
                    # Ensure order is set from list position
                    chunk.metadata = chunk.metadata.copy() if chunk.metadata else {}
                    chunk.metadata["order"] = order
                    await self.chunk_repository.create(chunk, commit=False)
            else:
                # Create chunks and embeddings using DoclingDocument
                await self.chunk_repository.create_chunks_for_document(
                    document_id, docling_document, commit=False
                )

            cursor.execute("COMMIT")
            return entity

        except Exception:
            cursor.execute("ROLLBACK")
            raise

    async def create(self, entity: Document) -> Document:
        """Create a document with its chunks and embeddings."""
        # Convert content to DoclingDocument
        docling_document = text_to_docling_document(entity.content)

        return await self._create_with_docling(entity, docling_document)

    async def get_by_id(self, entity_id: int) -> Document | None:
        """Get a document by its ID."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()
        cursor.execute(
            """
            SELECT id, content, uri, metadata, created_at, updated_at
            FROM documents WHERE id = :id
            """,
            {"id": entity_id},
        )

        row = cursor.fetchone()
        if row is None:
            return None

        document_id, content, uri, metadata_json, created_at, updated_at = row
        metadata = json.loads(metadata_json) if metadata_json else {}

        return Document(
            id=document_id,
            content=content,
            uri=uri,
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at,
        )

    async def get_by_uri(self, uri: str) -> Document | None:
        """Get a document by its URI."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()
        cursor.execute(
            """
            SELECT id, content, uri, metadata, created_at, updated_at
            FROM documents WHERE uri = :uri
            """,
            {"uri": uri},
        )

        row = cursor.fetchone()
        if row is None:
            return None

        document_id, content, uri, metadata_json, created_at, updated_at = row
        metadata = json.loads(metadata_json) if metadata_json else {}

        return Document(
            id=document_id,
            content=content,
            uri=uri,
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at,
        )

    async def _update_with_docling(
        self, entity: Document, docling_document: DoclingDocument
    ) -> Document:
        """Update an existing document and regenerate its chunks and embeddings."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")
        if entity.id is None:
            raise ValueError("Document ID is required for update")

        cursor = self.store._connection.cursor()

        # Start transaction
        cursor.execute("BEGIN TRANSACTION")

        try:
            # Update the document
            cursor.execute(
                """
                UPDATE documents
                SET content = :content, uri = :uri, metadata = :metadata, updated_at = :updated_at
                WHERE id = :id
                """,
                {
                    "content": entity.content,
                    "uri": entity.uri,
                    "metadata": json.dumps(entity.metadata),
                    "updated_at": entity.updated_at,
                    "id": entity.id,
                },
            )

            # Delete existing chunks and regenerate using DoclingDocument
            await self.chunk_repository.delete_by_document_id(entity.id, commit=False)
            await self.chunk_repository.create_chunks_for_document(
                entity.id, docling_document, commit=False
            )

            cursor.execute("COMMIT")
            return entity

        except Exception:
            cursor.execute("ROLLBACK")
            raise

    async def update(self, entity: Document) -> Document:
        """Update an existing document and regenerate its chunks and embeddings."""
        # Convert content to DoclingDocument
        docling_document = text_to_docling_document(entity.content)

        return await self._update_with_docling(entity, docling_document)

    async def delete(self, entity_id: int) -> bool:
        """Delete a document and all its associated chunks and embeddings."""
        # Delete chunks and embeddings first
        await self.chunk_repository.delete_by_document_id(entity_id)

        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()
        cursor.execute("DELETE FROM documents WHERE id = :id", {"id": entity_id})

        deleted = cursor.rowcount > 0
        self.store._connection.commit()
        return deleted

    async def list_all(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[Document]:
        """List all documents with optional pagination."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()
        query = "SELECT id, content, uri, metadata, created_at, updated_at FROM documents ORDER BY created_at DESC"
        params = {}

        if limit is not None:
            query += " LIMIT :limit"
            params["limit"] = limit

        if offset is not None:
            query += " OFFSET :offset"
            params["offset"] = offset

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [
            Document(
                id=document_id,
                content=content,
                uri=uri,
                metadata=json.loads(metadata_json) if metadata_json else {},
                created_at=created_at,
                updated_at=updated_at,
            )
            for document_id, content, uri, metadata_json, created_at, updated_at in rows
        ]
