import json
from datetime import datetime
from uuid import uuid4

from lancedb.index import BTree

from haiku.rag.store.engine import (
    DocumentRecord,
    Store,
    get_documents_arrow_schema,
    query_to_pydantic,
)
from haiku.rag.store.models.document import Document
from haiku.rag.utils import escape_sql_string


class DocumentRepository:
    """Repository for Document operations."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self._chunk_repository = None
        self._document_item_repository = None

    @property
    def chunk_repository(self):
        """Lazy-load ChunkRepository when needed."""
        if self._chunk_repository is None:
            from haiku.rag.store.repositories.chunk import ChunkRepository

            self._chunk_repository = ChunkRepository(self.store)
        return self._chunk_repository

    @property
    def document_item_repository(self):
        """Lazy-load DocumentItemRepository when needed."""
        if self._document_item_repository is None:
            from haiku.rag.store.repositories.document_item import (
                DocumentItemRepository,
            )

            self._document_item_repository = DocumentItemRepository(self.store)
        return self._document_item_repository

    def _record_to_document(self, record: DocumentRecord) -> Document:
        """Convert a DocumentRecord to a Document model."""
        return Document(
            id=record.id,
            content=record.content,
            uri=record.uri,
            title=record.title,
            metadata=json.loads(record.metadata),
            docling_document=record.docling_document,
            docling_pages=record.docling_pages,
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
            docling_document=entity.docling_document,
            docling_pages=entity.docling_pages,
            docling_version=entity.docling_version,
            created_at=now,
            updated_at=now,
        )

        # Add to table
        await self.store.documents_table.add([doc_record])

        entity.id = doc_id
        entity.created_at = datetime.fromisoformat(now)
        entity.updated_at = datetime.fromisoformat(now)
        return entity

    async def get_by_id(self, entity_id: str) -> Document | None:
        """Get a document by its ID."""
        safe_id = escape_sql_string(entity_id)
        results = await query_to_pydantic(
            self.store.documents_table.query().where(f"id = '{safe_id}'").limit(1),
            DocumentRecord,
        )

        if not results:
            return None

        return self._record_to_document(results[0])

    async def get_content(self, entity_id: str) -> str | None:
        """Get only the text content of a document (skips docling blobs)."""
        safe_id = escape_sql_string(entity_id)
        results = await (
            self.store.documents_table.query()
            .select(["content"])
            .where(f"id = '{safe_id}'")
            .limit(1)
            .to_list()
        )
        if not results:
            return None
        return results[0]["content"]

    _DOCLING_COLUMNS = ["id", "docling_document", "docling_version"]

    async def get_docling_data(self, entity_id: str) -> Document | None:
        """Get a document with only docling data loaded (skips content blob)."""
        safe_id = escape_sql_string(entity_id)
        results = await (
            self.store.documents_table.query()
            .select(self._DOCLING_COLUMNS)
            .where(f"id = '{safe_id}'")
            .limit(1)
            .to_list()
        )

        if not results:
            return None

        row = results[0]
        return Document(
            id=row["id"],
            content="",
            docling_document=row.get("docling_document"),
            docling_version=row.get("docling_version"),
        )

    async def get_pages_data(self, entity_id: str) -> Document | None:
        """Get a document with only page image data loaded."""
        safe_id = escape_sql_string(entity_id)
        results = await (
            self.store.documents_table.query()
            .select(["id", "docling_pages"])
            .where(f"id = '{safe_id}'")
            .limit(1)
            .to_list()
        )

        if not results:
            return None

        row = results[0]
        return Document(
            id=row["id"],
            content="",
            docling_pages=row.get("docling_pages"),
        )

    async def update(self, entity: Document) -> Document:
        """Update an existing document."""
        self.store._assert_writable()

        assert entity.id, "Document ID is required for update"

        # Update timestamp
        now = datetime.now().isoformat()
        entity.updated_at = datetime.fromisoformat(now)

        # Update the record
        safe_id = escape_sql_string(entity.id)
        await self.store.documents_table.update(
            {
                "content": entity.content,
                "uri": entity.uri,
                "title": entity.title,
                "metadata": json.dumps(entity.metadata),
                "docling_document": entity.docling_document,
                "docling_pages": entity.docling_pages,
                "docling_version": entity.docling_version,
                "updated_at": now,
            },
            where=f"id = '{safe_id}'",
        )

        return entity

    async def delete(self, entity_id: str) -> bool:
        """Delete a document by its ID."""
        self.store._assert_writable()

        # Check if document exists
        doc = await self.get_by_id(entity_id)
        if doc is None:
            return False

        # Delete associated chunks and items first
        await self.chunk_repository.delete_by_document_id(entity_id)
        await self.document_item_repository.delete_by_document_id(entity_id)

        # Delete the document
        safe_id = escape_sql_string(entity_id)
        await self.store.documents_table.delete(f"id = '{safe_id}'")
        return True

    _LISTING_COLUMNS = ["id", "title", "uri", "metadata", "created_at", "updated_at"]

    async def list_all(
        self,
        limit: int | None = None,
        offset: int | None = None,
        filter: str | None = None,
        include_content: bool = False,
    ) -> list[Document]:
        """List all documents with optional pagination and filtering.

        Args:
            limit: Maximum number of documents to return.
            offset: Number of documents to skip.
            filter: Optional SQL WHERE clause to filter documents.
            include_content: Whether to load content and docling_document.
                Defaults to False to avoid loading large blobs for listing.

        Returns:
            List of Document instances matching the criteria.
        """
        query = self.store.documents_table.query()

        if not include_content:
            query = query.select(self._LISTING_COLUMNS)
        if filter is not None:
            query = query.where(filter)
        if offset is not None:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)

        if include_content:
            results = await query_to_pydantic(query, DocumentRecord)
            return [self._record_to_document(doc) for doc in results]

        return [
            Document(
                id=row["id"],
                content="",
                title=row.get("title"),
                uri=row.get("uri"),
                metadata=json.loads(row.get("metadata", "{}")),
                created_at=datetime.fromisoformat(row["created_at"])
                if row.get("created_at")
                else datetime.now(),
                updated_at=datetime.fromisoformat(row["updated_at"])
                if row.get("updated_at")
                else datetime.now(),
            )
            for row in await query.to_list()
        ]

    async def count(self, filter: str | None = None) -> int:
        """Count documents with optional filtering.

        Args:
            filter: Optional SQL WHERE clause to filter documents.

        Returns:
            Number of documents matching the criteria.
        """
        return await self.store.documents_table.count_rows(filter=filter)

    async def get_by_uri(self, uri: str) -> Document | None:
        """Get a document by its URI."""
        escaped_uri = escape_sql_string(uri)
        results = await query_to_pydantic(
            self.store.documents_table.query().where(f"uri = '{escaped_uri}'").limit(1),
            DocumentRecord,
        )

        if not results:
            return None

        return self._record_to_document(results[0])

    async def delete_all(self) -> None:
        """Delete all documents from the database."""
        self.store._assert_writable()
        from haiku.rag.store.engine import DocumentItemRecord

        # Delete all chunks and items first
        await self.chunk_repository.delete_all()
        await self.store.db.drop_table("document_items")
        self.store.document_items_table = await self.store.db.create_table(
            "document_items", schema=DocumentItemRecord
        )
        await self.store.document_items_table.create_index(
            "document_id", config=BTree(), replace=True
        )
        await self.store.document_items_table.create_index(
            "position", config=BTree(), replace=True
        )
        await self.store.document_items_table.create_index(
            "self_ref", config=BTree(), replace=True
        )

        # Get count before deletion
        count = len(
            await query_to_pydantic(
                self.store.documents_table.query().limit(1), DocumentRecord
            )
        )
        if count > 0:
            # Drop and recreate table to clear all data
            await self.store.db.drop_table("documents")
            self.store.documents_table = await self.store.db.create_table(
                "documents", schema=get_documents_arrow_schema()
            )
