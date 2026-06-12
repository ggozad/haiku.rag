import json
from datetime import datetime
from typing import overload
from uuid import uuid4

from lancedb.index import BTree

from haiku.rag.store.engine import (
    DocumentMetaRecord,
    DocumentRecord,
    Store,
    get_documents_arrow_schema,
    query_to_pydantic,
)
from haiku.rag.store.models.document import Document
from haiku.rag.utils import escape_sql_string


class DocumentRepository:
    """Repository for Document operations.

    A document is stored across two tables with a strict invariant: every
    `documents` row (id, content, docling blobs — write-once) has exactly one
    matching `document_meta` row (uri, title, metadata, timestamps — mutable),
    keyed by `document_id`. The mutable attributes never live in `documents`,
    so metadata/title/source_revision updates (`update_meta`) rewrite only the
    small meta row and never the multi-MB docling blob.

    To preserve the invariant: `create` writes meta then documents and deletes
    the meta row if the documents write fails; `update_meta` updates matched
    rows only (no insert — an insert on a missing id would create a ghost
    surfaced by `list_all`/`count`); `delete` removes both rows.
    """

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

    def _merge_to_document(
        self, doc: DocumentRecord, meta: DocumentMetaRecord | None
    ) -> Document:
        """Merge a `documents` record (content+blobs) with its `document_meta`
        record (uri/title/metadata/timestamps) into a Document."""
        created = meta.created_at if meta else ""
        updated = meta.updated_at if meta else ""
        return Document(
            id=doc.id,
            content=doc.content,
            uri=meta.uri if meta else None,
            title=meta.title if meta else None,
            metadata=json.loads(meta.metadata) if meta else {},
            docling_document=doc.docling_document,
            docling_pages=doc.docling_pages,
            docling_version=doc.docling_version,
            created_at=datetime.fromisoformat(created) if created else datetime.now(),
            updated_at=datetime.fromisoformat(updated) if updated else datetime.now(),
        )

    def _to_documents_record(self, entity: Document, doc_id: str) -> DocumentRecord:
        return DocumentRecord(
            id=doc_id,
            content=entity.content,
            docling_document=entity.docling_document,
            docling_pages=entity.docling_pages,
            docling_version=entity.docling_version,
        )

    def _to_meta_record(
        self,
        entity: Document,
        doc_id: str,
        created_at: str,
        updated_at: str,
    ) -> DocumentMetaRecord:
        return DocumentMetaRecord(
            document_id=doc_id,
            uri=entity.uri,
            title=entity.title,
            metadata=json.dumps(entity.metadata),
            created_at=created_at,
            updated_at=updated_at,
        )

    async def _meta_by_id(self, doc_id: str) -> DocumentMetaRecord | None:
        safe_id = escape_sql_string(doc_id)
        results = await query_to_pydantic(
            self.store.document_meta_table.query()
            .where(f"document_id = '{safe_id}'")
            .limit(1),
            DocumentMetaRecord,
        )
        return results[0] if results else None

    @overload
    async def create(self, entity: Document) -> Document: ...

    @overload
    async def create(self, entity: list[Document]) -> list[Document]: ...

    async def create(
        self, entity: Document | list[Document]
    ) -> Document | list[Document]:
        """Create one or more documents in the database.

        A list is written in a single table version regardless of length.
        """
        self.store._assert_writable()

        # document_meta is written before documents so the documents row write
        # is the commit point: time-travel to any documents version always sees
        # the matching (earlier-written) document_meta row. If the documents
        # write then fails, delete just the rows we added (not a table-version
        # restore, which would clobber a concurrent writer's meta write) so a
        # failed create can't leave a ghost row that list_all/count (which read
        # document_meta) would surface.
        if isinstance(entity, Document):
            doc_id = str(uuid4())
            now = datetime.now().isoformat()
            await self.store.document_meta_table.add(
                [self._to_meta_record(entity, doc_id, now, now)]
            )
            try:
                await self.store.documents_table.add(
                    [self._to_documents_record(entity, doc_id)]
                )
            except Exception:
                safe_id = escape_sql_string(doc_id)
                await self.store.document_meta_table.delete(
                    f"document_id = '{safe_id}'"
                )
                raise
            entity.id = doc_id
            entity.created_at = datetime.fromisoformat(now)
            entity.updated_at = datetime.fromisoformat(now)
            return entity

        documents = entity
        if not documents:
            return []

        now = datetime.now().isoformat()
        created_at = datetime.fromisoformat(now)
        doc_records = []
        meta_records = []
        doc_ids = []
        for document in documents:
            doc_id = str(uuid4())
            doc_ids.append(doc_id)
            doc_records.append(self._to_documents_record(document, doc_id))
            meta_records.append(self._to_meta_record(document, doc_id, now, now))
            document.id = doc_id
            document.created_at = created_at
            document.updated_at = created_at

        await self.store.document_meta_table.add(meta_records)
        try:
            await self.store.documents_table.add(doc_records)
        except Exception:
            ids = ", ".join(f"'{escape_sql_string(d)}'" for d in doc_ids)
            await self.store.document_meta_table.delete(f"document_id IN ({ids})")
            raise
        return documents

    async def get_by_id(self, entity_id: str) -> Document | None:
        """Get a document by its ID."""
        safe_id = escape_sql_string(entity_id)
        results = await query_to_pydantic(
            self.store.documents_table.query().where(f"id = '{safe_id}'").limit(1),
            DocumentRecord,
        )

        if not results:
            return None

        meta = await self._meta_by_id(entity_id)
        return self._merge_to_document(results[0], meta)

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

    async def update_meta(self, entity: Document) -> Document:
        """Update only the mutable attributes (uri/title/metadata/updated_at) in
        `document_meta`. Does NOT touch the `documents` row, so the multi-MB
        docling blob is never rewritten — this is the blob-bloat fix for
        metadata/title/source_revision changes."""
        self.store._assert_writable()
        assert entity.id, "Document ID is required for update"

        now = datetime.now().isoformat()
        entity.updated_at = datetime.fromisoformat(now)
        created = entity.created_at.isoformat() if entity.created_at else now
        record = self._to_meta_record(entity, entity.id, created, now)
        # Update only — no insert. Every real document has a document_meta row
        # from create()/migration; inserting on no-match would manufacture a
        # ghost row (visible to list_all/count) for an id with no documents row.
        await (
            self.store.document_meta_table.merge_insert("document_id")
            .when_matched_update_all()
            .execute([record])
        )
        return entity

    async def update(self, entity: Document) -> Document:
        """Update a document's content+blobs (genuine re-conversion) and its
        mutable attributes. Rewrites the `documents` row, so use only when the
        docling content actually changed; for metadata/title-only changes use
        `update_meta`."""
        self.store._assert_writable()
        assert entity.id, "Document ID is required for update"

        doc_record = self._to_documents_record(entity, entity.id)
        await (
            self.store.documents_table.merge_insert("id")
            .when_matched_update_all()
            .execute([doc_record])
        )
        await self.update_meta(entity)
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

        # Delete the document row, its mutable attributes
        safe_id = escape_sql_string(entity_id)
        await self.store.documents_table.delete(f"id = '{safe_id}'")
        await self.store.document_meta_table.delete(f"document_id = '{safe_id}'")
        return True

    async def list_all(
        self,
        limit: int | None = None,
        offset: int | None = None,
        filter: str | None = None,
        include_content: bool = False,
    ) -> list[Document]:
        """List all documents with optional pagination and filtering.

        Listing reads `document_meta` (uri/title/metadata/timestamps); the
        SQL `filter` is evaluated against those columns. When `include_content`
        is set, the content+blob row is loaded from `documents` and merged in.

        Args:
            limit: Maximum number of documents to return.
            offset: Number of documents to skip.
            filter: Optional SQL WHERE clause over document_meta columns.
            include_content: Whether to also load content and docling blobs.

        Returns:
            List of Document instances matching the criteria.
        """
        query = self.store.document_meta_table.query()

        if filter is not None:
            query = query.where(filter)
        if offset is not None:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)

        meta_records = await query_to_pydantic(query, DocumentMetaRecord)

        if not include_content:
            return [
                self._merge_to_document(DocumentRecord(id=m.document_id, content=""), m)
                for m in meta_records
            ]

        documents: list[Document] = []
        for meta in meta_records:
            safe_id = escape_sql_string(meta.document_id)
            doc_results = await query_to_pydantic(
                self.store.documents_table.query().where(f"id = '{safe_id}'").limit(1),
                DocumentRecord,
            )
            doc_record = (
                doc_results[0]
                if doc_results
                else DocumentRecord(id=meta.document_id, content="")
            )
            documents.append(self._merge_to_document(doc_record, meta))
        return documents

    async def count(self, filter: str | None = None) -> int:
        """Count documents with optional filtering (over document_meta columns)."""
        return await self.store.document_meta_table.count_rows(filter=filter)

    async def get_by_uri(self, uri: str) -> Document | None:
        """Get a document by its URI (resolved via document_meta)."""
        escaped_uri = escape_sql_string(uri)
        meta_results = await query_to_pydantic(
            self.store.document_meta_table.query()
            .where(f"uri = '{escaped_uri}'")
            .limit(1),
            DocumentMetaRecord,
        )

        if not meta_results:
            return None

        meta = meta_results[0]
        safe_id = escape_sql_string(meta.document_id)
        doc_results = await query_to_pydantic(
            self.store.documents_table.query().where(f"id = '{safe_id}'").limit(1),
            DocumentRecord,
        )
        if not doc_results:
            return None

        return self._merge_to_document(doc_results[0], meta)

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
            # Drop and recreate tables to clear all data
            await self.store.db.drop_table("documents")
            self.store.documents_table = await self.store.db.create_table(
                "documents", schema=get_documents_arrow_schema()
            )
            await self.store.db.drop_table("document_meta")
            self.store.document_meta_table = await self.store.db.create_table(
                "document_meta", schema=DocumentMetaRecord
            )
            await self.store.document_meta_table.create_index(
                "document_id", config=BTree(), replace=True
            )
            await self.store.document_meta_table.create_index(
                "uri", config=BTree(), replace=True
            )
