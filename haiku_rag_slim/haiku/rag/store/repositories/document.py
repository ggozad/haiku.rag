import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from haiku.rag.store.engine import DocumentRecord, Store
from haiku.rag.store.models.document import Document

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.store.models.chunk import Chunk


class DocumentRepository:
    """Repository for Document operations."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self._chunk_repository = None

    @property
    def chunk_repository(self):
        """Lazy-load ChunkRepository when needed."""
        if self._chunk_repository is None:
            from haiku.rag.store.repositories.chunk import ChunkRepository

            self._chunk_repository = ChunkRepository(self.store)
        return self._chunk_repository

    def _record_to_document(self, record: DocumentRecord) -> Document:
        """Convert a DocumentRecord to a Document model."""
        return Document(
            id=record.id,
            content=record.content,
            uri=record.uri,
            title=record.title,
            metadata=json.loads(record.metadata),
            created_at=datetime.fromisoformat(record.created_at)
            if record.created_at
            else datetime.now(),
            updated_at=datetime.fromisoformat(record.updated_at)
            if record.updated_at
            else datetime.now(),
        )

    async def create(self, entity: Document) -> Document:
        """Create a document in the database."""
        # Generate new UUID
        doc_id = str(uuid4())

        # Create timestamp
        now = datetime.now().isoformat()

        # Create document record
        doc_record = DocumentRecord(
            id=doc_id,
            content=entity.content,
            uri=entity.uri,
            title=entity.title,
            metadata=json.dumps(entity.metadata),
            created_at=now,
            updated_at=now,
        )

        # Add to table
        self.store.documents_table.add([doc_record])

        entity.id = doc_id
        entity.created_at = datetime.fromisoformat(now)
        entity.updated_at = datetime.fromisoformat(now)
        return entity

    async def get_by_id(self, entity_id: str) -> Document | None:
        """Get a document by its ID."""
        results = list(
            self.store.documents_table.search()
            .where(f"id = '{entity_id}'")
            .limit(1)
            .to_pydantic(DocumentRecord)
        )

        if not results:
            return None

        return self._record_to_document(results[0])

    async def update(self, entity: Document) -> Document:
        """Update an existing document."""
        assert entity.id, "Document ID is required for update"

        # Update timestamp
        now = datetime.now().isoformat()
        entity.updated_at = datetime.fromisoformat(now)

        # Update the record
        self.store.documents_table.update(
            where=f"id = '{entity.id}'",
            values={
                "content": entity.content,
                "uri": entity.uri,
                "title": entity.title,
                "metadata": json.dumps(entity.metadata),
                "updated_at": now,
            },
        )

        return entity

    async def delete(self, entity_id: str) -> bool:
        """Delete a document by its ID."""
        # Check if document exists
        doc = await self.get_by_id(entity_id)
        if doc is None:
            return False

        # Delete associated chunks first
        await self.chunk_repository.delete_by_document_id(entity_id)

        # Delete the document
        self.store.documents_table.delete(f"id = '{entity_id}'")
        return True

    async def list_all(
        self,
        limit: int | None = None,
        offset: int | None = None,
        filter: str | None = None,
    ) -> list[Document]:
        """List all documents with optional pagination and filtering.

        Args:
            limit: Maximum number of documents to return.
            offset: Number of documents to skip.
            filter: Optional SQL WHERE clause to filter documents.

        Returns:
            List of Document instances matching the criteria.
        """
        query = self.store.documents_table.search()

        if filter is not None:
            query = query.where(filter)
        if offset is not None:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)

        results = list(query.to_pydantic(DocumentRecord))
        return [self._record_to_document(doc) for doc in results]

    async def get_by_uri(self, uri: str) -> Document | None:
        """Get a document by its URI."""
        results = list(
            self.store.documents_table.search()
            .where(f"uri = '{uri}'")
            .limit(1)
            .to_pydantic(DocumentRecord)
        )

        if not results:
            return None

        return self._record_to_document(results[0])

    async def delete_all(self) -> None:
        """Delete all documents from the database."""
        # Delete all chunks first
        await self.chunk_repository.delete_all()

        # Get count before deletion
        count = len(
            list(
                self.store.documents_table.search().limit(1).to_pydantic(DocumentRecord)
            )
        )
        if count > 0:
            # Drop and recreate table to clear all data
            self.store.db.drop_table("documents")
            self.store.documents_table = self.store.db.create_table(
                "documents", schema=DocumentRecord
            )

    async def _create_and_chunk(
        self,
        entity: Document,
        docling_document: "DoclingDocument | None",
        chunks: list["Chunk"] | None = None,
    ) -> Document:
        """Create a document with its chunks and embeddings."""
        # Snapshot table versions for versioned rollback (if supported)
        versions = self.store.current_table_versions()

        # Create the document
        created_doc = await self.create(entity)

        # Attempt to create chunks; on failure, prefer version rollback
        try:
            # Create chunks if not provided
            if chunks is None:
                assert docling_document is not None, (
                    "docling_document is required when chunks are not provided"
                )
                assert created_doc.id is not None, (
                    "Document ID should not be None after creation"
                )
                await self.chunk_repository.create_chunks_for_document(
                    created_doc.id, docling_document
                )
            else:
                # Use provided chunks, set order from list position
                assert created_doc.id is not None, (
                    "Document ID should not be None after creation"
                )
                for order, chunk in enumerate(chunks):
                    chunk.document_id = created_doc.id
                    chunk.order = order
                    await self.chunk_repository.create(chunk)

            # Vacuum old versions in background (non-blocking)
            asyncio.create_task(self.store.vacuum())

            return created_doc
        except Exception:
            # Roll back to the captured versions and re-raise
            self.store.restore_table_versions(versions)
            raise

    async def _update_and_rechunk(
        self, entity: Document, docling_document: "DoclingDocument"
    ) -> Document:
        """Update a document and regenerate its chunks."""
        assert entity.id is not None, "Document ID is required for update"

        # Snapshot table versions for versioned rollback
        versions = self.store.current_table_versions()

        # Delete existing chunks before writing new ones
        await self.chunk_repository.delete_by_document_id(entity.id)

        try:
            # Update the document
            updated_doc = await self.update(entity)

            # Create new chunks
            assert updated_doc.id is not None, (
                "Document ID should not be None after update"
            )
            await self.chunk_repository.create_chunks_for_document(
                updated_doc.id, docling_document
            )

            # Vacuum old versions in background (non-blocking)
            asyncio.create_task(self.store.vacuum())

            return updated_doc
        except Exception:
            # Roll back to the captured versions and re-raise
            self.store.restore_table_versions(versions)
            raise
