import json
from datetime import datetime
from uuid import uuid4

from haiku.rag.store.engine import DocumentRecord, Store
from haiku.rag.store.models.document import Document


def _escape_sql_string(value: str) -> str:
    """Escape single quotes in SQL string literals."""
    return value.replace("'", "''")


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
            docling_document_json=record.docling_document_json,
            docling_version=record.docling_version,
            created_at=datetime.fromisoformat(record.created_at)
            if record.created_at
            else datetime.now(),
            updated_at=datetime.fromisoformat(record.updated_at)
            if record.updated_at
            else datetime.now(),
        )

    async def create(self, entity: Document) -> Document:
        """Create a document in the database."""
        self.store._assert_writable()
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
            docling_document_json=entity.docling_document_json,
            docling_version=entity.docling_version,
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
        self.store._assert_writable()
        from haiku.rag.store.models.document import invalidate_docling_document_cache

        assert entity.id, "Document ID is required for update"

        # Invalidate cache before update
        invalidate_docling_document_cache(entity.id)

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
                "docling_document_json": entity.docling_document_json,
                "docling_version": entity.docling_version,
                "updated_at": now,
            },
        )

        return entity

    async def delete(self, entity_id: str) -> bool:
        """Delete a document by its ID."""
        self.store._assert_writable()
        from haiku.rag.store.models.document import invalidate_docling_document_cache

        # Check if document exists
        doc = await self.get_by_id(entity_id)
        if doc is None:
            return False

        # Invalidate cache before delete
        invalidate_docling_document_cache(entity_id)

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
        escaped_uri = _escape_sql_string(uri)
        results = list(
            self.store.documents_table.search()
            .where(f"uri = '{escaped_uri}'")
            .limit(1)
            .to_pydantic(DocumentRecord)
        )

        if not results:
            return None

        return self._record_to_document(results[0])

    async def delete_all(self) -> None:
        """Delete all documents from the database."""
        self.store._assert_writable()
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
