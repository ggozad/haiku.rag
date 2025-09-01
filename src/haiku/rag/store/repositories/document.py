import json
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from docling_core.types.doc.document import DoclingDocument

from haiku.rag.store.engine import DocumentRecord, Store
from haiku.rag.store.models.document import Document

if TYPE_CHECKING:
    from haiku.rag.store.models.chunk import Chunk


class DocumentRepository:
    """Repository for Document operations."""

    def __init__(self, store: Store) -> None:
        self.store = store

        from haiku.rag.store.repositories.chunk import ChunkRepository

        chunk_repository = ChunkRepository(store)
        self.chunk_repository = chunk_repository

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

        doc_record = results[0]
        return Document(
            id=doc_record.id,
            content=doc_record.content,
            uri=doc_record.uri,
            metadata=json.loads(doc_record.metadata) if doc_record.metadata else {},
            created_at=datetime.fromisoformat(doc_record.created_at)
            if doc_record.created_at
            else datetime.now(),
            updated_at=datetime.fromisoformat(doc_record.updated_at)
            if doc_record.updated_at
            else datetime.now(),
        )

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
        from haiku.rag.store.repositories.chunk import ChunkRepository

        chunk_repo = ChunkRepository(self.store)
        await chunk_repo.delete_by_document_id(entity_id)

        # Delete the document
        self.store.documents_table.delete(f"id = '{entity_id}'")
        return True

    async def list_all(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[Document]:
        """List all documents with optional pagination."""
        query = self.store.documents_table.search()

        if offset is not None:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)

        results = list(query.to_pydantic(DocumentRecord))

        return [
            Document(
                id=doc.id,
                content=doc.content,
                uri=doc.uri,
                metadata=json.loads(doc.metadata) if doc.metadata else {},
                created_at=datetime.fromisoformat(doc.created_at)
                if doc.created_at
                else datetime.now(),
                updated_at=datetime.fromisoformat(doc.updated_at)
                if doc.updated_at
                else datetime.now(),
            )
            for doc in results
        ]

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

        doc_record = results[0]
        return Document(
            id=doc_record.id,
            content=doc_record.content,
            uri=doc_record.uri,
            metadata=json.loads(doc_record.metadata) if doc_record.metadata else {},
            created_at=datetime.fromisoformat(doc_record.created_at)
            if doc_record.created_at
            else datetime.now(),
            updated_at=datetime.fromisoformat(doc_record.updated_at)
            if doc_record.updated_at
            else datetime.now(),
        )

    async def delete_all(self) -> None:
        """Delete all documents from the database."""
        # Delete all chunks first
        from haiku.rag.store.repositories.chunk import ChunkRepository

        chunk_repo = ChunkRepository(self.store)
        await chunk_repo.delete_all()

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

    async def _create_with_docling(
        self,
        entity: Document,
        docling_document: DoclingDocument,
        chunks: list["Chunk"] | None = None,
    ) -> Document:
        """Create a document with its chunks and embeddings."""
        # Create the document
        created_doc = await self.create(entity)

        # Create chunks if not provided
        if chunks is None:
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
                chunk.metadata["order"] = order
                await self.chunk_repository.create(chunk)

        return created_doc

    async def _update_with_docling(
        self, entity: Document, docling_document: DoclingDocument
    ) -> Document:
        """Update a document and regenerate its chunks."""
        # Delete existing chunks
        assert entity.id is not None, "Document ID is required for update"
        await self.chunk_repository.delete_by_document_id(entity.id)

        # Update the document
        updated_doc = await self.update(entity)

        # Create new chunks
        assert updated_doc.id is not None, "Document ID should not be None after update"
        await self.chunk_repository.create_chunks_for_document(
            updated_doc.id, docling_document
        )

        return updated_doc
