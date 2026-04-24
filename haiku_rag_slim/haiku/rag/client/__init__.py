import asyncio
import hashlib
import json
import logging
import mimetypes
import tempfile
from collections.abc import AsyncGenerator
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, overload
from urllib.parse import urlparse

import httpx

from haiku.rag.config import AppConfig, Config
from haiku.rag.converters import get_converter
from haiku.rag.reranking import get_reranker
from haiku.rag.store.engine import Store
from haiku.rag.store.models.chunk import Chunk, SearchResult
from haiku.rag.store.models.document import Document
from haiku.rag.store.models.document_item import extract_items
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.repositories.document_item import DocumentItemRepository
from haiku.rag.store.repositories.settings import SettingsRepository
from haiku.rag.utils import escape_sql_string

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.agents.analysis.models import AnalysisResult
    from haiku.rag.agents.research.models import (
        Citation,
        ResearchReport,
    )

logger = logging.getLogger(__name__)


class RebuildMode(Enum):
    """Mode for rebuilding the database."""

    FULL = "full"  # Re-convert from source, re-chunk, re-embed
    RECHUNK = "rechunk"  # Re-chunk from existing content, re-embed
    EMBED_ONLY = "embed_only"  # Keep chunks, only regenerate embeddings
    TITLE_ONLY = "title_only"  # Only generate titles for untitled documents


class HaikuRAG:
    """High-level haiku-rag client."""

    def __init__(
        self,
        db_path: Path | None = None,
        config: AppConfig = Config,
        skip_validation: bool = False,
        create: bool = False,
        read_only: bool = False,
        before: datetime | None = None,
    ):
        """Initialize the RAG client with a database path.

        Args:
            db_path: Path to the database file. If None, uses config.storage.data_dir.
            config: Configuration to use. Defaults to global Config.
            skip_validation: Whether to skip configuration validation on database load.
            create: Whether to create the database if it doesn't exist.
            read_only: Whether to open the database in read-only mode.
            before: Query the database as it existed at this datetime.
                Implies read_only=True.
        """
        self._config = config
        if db_path is None:
            db_path = self._config.storage.data_dir / "haiku.rag.lancedb"

        self._db_path = db_path
        self._skip_validation = skip_validation
        self._create = create
        self._read_only = read_only
        self._before = before
        self._vacuum_tasks: set[asyncio.Task] = set()

    @property
    def is_read_only(self) -> bool:
        """Whether the client is in read-only mode."""
        return self.store.is_read_only

    async def __aenter__(self):
        """Async context manager entry — initializes store and repositories."""
        self.store = Store(
            self._db_path,
            config=self._config,
            skip_validation=self._skip_validation,
            create=self._create,
            read_only=self._read_only,
            before=self._before,
        )
        await self.store._initialize()
        self.document_repository = DocumentRepository(self.store)
        self.chunk_repository = ChunkRepository(self.store)
        self.document_item_repository = DocumentItemRepository(self.store)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ARG002
        """Async context manager exit."""
        await self._await_vacuum_tasks()
        self.close()
        return False

    async def _await_vacuum_tasks(self) -> None:
        """Wait for all in-flight background vacuum tasks to complete.

        Each create_document / update_document can schedule its own vacuum task;
        all must be awaited before tearing down the connection, not just the
        most recently scheduled one.
        """
        if self._vacuum_tasks:
            await asyncio.gather(*self._vacuum_tasks, return_exceptions=True)

    def _schedule_vacuum(self) -> None:
        """Schedule a background vacuum and track the task for later awaiting."""
        task = asyncio.create_task(self.store.vacuum())
        self._vacuum_tasks.add(task)
        task.add_done_callback(self._vacuum_tasks.discard)

    # =========================================================================
    # Processing Primitives
    # =========================================================================

    @overload
    async def convert(self, source: Path) -> "DoclingDocument": ...

    @overload
    async def convert(
        self, source: str, *, format: str = "md"
    ) -> "DoclingDocument": ...

    async def convert(
        self, source: Path | str, *, format: str = "md"
    ) -> "DoclingDocument":
        from haiku.rag.client.processing import convert

        return await convert(self._config, source, format=format)

    async def chunk(self, docling_document: "DoclingDocument") -> list[Chunk]:
        from haiku.rag.client.processing import chunk

        return await chunk(self._config, docling_document)

    async def _ensure_chunks_embedded(self, chunks: list[Chunk]) -> list[Chunk]:
        from haiku.rag.client.processing import ensure_chunks_embedded

        return await ensure_chunks_embedded(self._config, chunks)

    # =========================================================================
    # Title Generation
    # =========================================================================

    def _extract_structural_title(
        self, docling_document: "DoclingDocument"
    ) -> str | None:
        from haiku.rag.client.titles import extract_structural_title

        return extract_structural_title(docling_document)

    async def _generate_title_with_llm(self, content: str) -> str | None:
        from haiku.rag.client.titles import generate_title_with_llm

        return await generate_title_with_llm(self._config, content)

    async def _resolve_title(
        self,
        docling_document: "DoclingDocument",
        content: str,
    ) -> str | None:
        from haiku.rag.client.titles import resolve_title

        return await resolve_title(self._config, docling_document, content)

    async def generate_title(self, document: Document) -> str | None:
        from haiku.rag.client.titles import generate_title

        return await generate_title(self._config, document)

    async def _store_document_with_chunks(
        self,
        document: Document,
        chunks: list[Chunk],
        docling_document: "DoclingDocument",
    ) -> Document:
        """Store a document with chunks, embedding any that lack embeddings.

        Handles versioning/rollback on failure.

        Args:
            document: The document to store (will be created).
            chunks: Chunks to store (will be embedded if lacking embeddings).
            docling_document: The DoclingDocument to extract items from.

        Returns:
            The created Document instance with ID set.
        """
        # Ensure all chunks have embeddings before storing
        chunks = await self._ensure_chunks_embedded(chunks)

        # Snapshot table versions for versioned rollback (if supported)
        versions = await self.store.current_table_versions()

        # Create the document
        created_doc = await self.document_repository.create(document)

        try:
            assert created_doc.id is not None, (
                "Document ID should not be None after creation"
            )
            # Set document_id and order for all chunks
            for order, chunk in enumerate(chunks):
                chunk.document_id = created_doc.id
                chunk.order = order

            # Batch create all chunks in a single operation
            await self.chunk_repository.create(chunks)

            # Extract and store document items for context expansion
            items = extract_items(created_doc.id, docling_document)
            await self.document_item_repository.create_items(created_doc.id, items)

            # Vacuum old versions in background (non-blocking) if auto_vacuum enabled
            if self._config.storage.auto_vacuum:
                self._schedule_vacuum()

            return created_doc
        except Exception:
            # Roll back to the captured versions and re-raise
            await self.store.restore_table_versions(versions)
            raise

    async def _update_document_with_chunks(
        self,
        document: Document,
        chunks: list[Chunk],
        docling_document: "DoclingDocument | None" = None,
    ) -> Document:
        """Update a document and replace its chunks, embedding any that lack embeddings.

        Handles versioning/rollback on failure.

        Args:
            document: The document to update (must have ID set).
            chunks: Chunks to replace existing (will be embedded if lacking embeddings).
            docling_document: The DoclingDocument to extract items from.
                When None, existing items are preserved.

        Returns:
            The updated Document instance.
        """
        assert document.id is not None, "Document ID is required for update"

        # Ensure all chunks have embeddings before storing
        chunks = await self._ensure_chunks_embedded(chunks)

        # Snapshot table versions for versioned rollback
        versions = await self.store.current_table_versions()

        # Delete existing chunks before writing new ones
        await self.chunk_repository.delete_by_document_id(document.id)

        try:
            # Update the document
            updated_doc = await self.document_repository.update(document)

            # Set document_id and order for all chunks
            assert updated_doc.id is not None
            for order, chunk in enumerate(chunks):
                chunk.document_id = updated_doc.id
                chunk.order = order

            # Batch create all chunks in a single operation
            await self.chunk_repository.create(chunks)

            # Replace document items when a new DoclingDocument is provided
            if docling_document is not None:
                await self.document_item_repository.delete_by_document_id(
                    updated_doc.id
                )
                items = extract_items(updated_doc.id, docling_document)
                await self.document_item_repository.create_items(updated_doc.id, items)

            # Vacuum old versions in background (non-blocking) if auto_vacuum enabled
            if self._config.storage.auto_vacuum:
                self._schedule_vacuum()

            return updated_doc
        except Exception:
            # Roll back to the captured versions and re-raise
            await self.store.restore_table_versions(versions)
            raise

    async def create_document(
        self,
        content: str,
        uri: str | None = None,
        title: str | None = None,
        metadata: dict | None = None,
        format: str = "md",
    ) -> Document:
        """Create a new document from text content.

        Converts the content, chunks it, and generates embeddings.

        Args:
            content: The text content of the document.
            uri: Optional URI identifier for the document.
            title: Optional title for the document.
            metadata: Optional metadata dictionary.
            format: The format of the content ("md", "html", or "plain").
                Defaults to "md". Use "plain" for plain text without parsing.

        Returns:
            The created Document instance.
        """
        from haiku.rag.embeddings import embed_chunks

        # Convert → Chunk → Embed using primitives
        converter = get_converter(self._config)
        docling_document = await converter.convert_text(content, format=format)
        chunks = await self.chunk(docling_document)
        embedded_chunks = await embed_chunks(chunks, self._config)

        # Store markdown export as content for better display/readability
        # The original content is preserved in docling_document
        stored_content = docling_document.export_to_markdown()

        if title is None:
            title = await self._resolve_title(docling_document, stored_content)

        # Create document model
        document = Document(
            content=stored_content,
            uri=uri,
            title=title,
            metadata=metadata or {},
        )
        document.set_docling(docling_document)

        # Store document and chunks
        return await self._store_document_with_chunks(
            document, embedded_chunks, docling_document
        )

    async def import_document(
        self,
        docling_document: "DoclingDocument",
        chunks: list[Chunk],
        uri: str | None = None,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> Document:
        """Import a pre-processed document with chunks.

        Use this when document conversion, chunking, and embedding were done
        externally and you want to store the results in haiku.rag.

        Args:
            docling_document: The DoclingDocument to import.
            chunks: Pre-created chunks. Chunks without embeddings will be
                automatically embedded.
            uri: Optional URI identifier for the document.
            title: Optional title for the document.
            metadata: Optional metadata dictionary.

        Returns:
            The created Document instance.
        """
        content = docling_document.export_to_markdown()
        if title is None:
            title = await self._resolve_title(docling_document, content)

        document = Document(
            content=content,
            uri=uri,
            title=title,
            metadata=metadata or {},
        )
        document.set_docling(docling_document)

        return await self._store_document_with_chunks(
            document, chunks, docling_document
        )

    async def create_document_from_source(
        self, source: str | Path, title: str | None = None, metadata: dict | None = None
    ) -> Document | list[Document]:
        """Create or update document(s) from a file path, directory, or URL.

        Checks if a document with the same URI already exists:
        - If MD5 is unchanged, returns existing document
        - If MD5 changed, updates the document
        - If no document exists, creates a new one

        Args:
            source: File path, directory (as string or Path), or URL to parse
            title: Optional title (only used for single files, not directories)
            metadata: Optional metadata dictionary

        Returns:
            Document instance (created, updated, or existing) for single files/URLs
            List of Document instances for directories

        Raises:
            ValueError: If the file/URL cannot be parsed or doesn't exist
            httpx.RequestError: If URL request fails
        """
        # Normalize metadata
        metadata = metadata or {}

        # Check if it's a URL
        source_str = str(source)
        parsed_url = urlparse(source_str)
        if parsed_url.scheme in ("http", "https"):
            return await self._create_or_update_document_from_url(
                source_str, title=title, metadata=metadata
            )
        elif parsed_url.scheme == "file":
            # Handle file:// URI by converting to path
            source_path = Path(parsed_url.path)
        else:
            # Handle as regular file path
            source_path = Path(source) if isinstance(source, str) else source

        # Handle directories
        if source_path.is_dir():
            from haiku.rag.monitor import FileFilter

            documents = []
            filter = FileFilter(
                ignore_patterns=self._config.monitor.ignore_patterns or None,
                include_patterns=self._config.monitor.include_patterns or None,
            )
            for path in source_path.rglob("*"):
                if path.is_file() and filter.include_file(str(path)):
                    doc = await self._create_document_from_file(
                        path, title=None, metadata=metadata
                    )
                    documents.append(doc)
            return documents

        # Handle single file
        return await self._create_document_from_file(
            source_path, title=title, metadata=metadata
        )

    async def _create_document_from_file(
        self, source_path: Path, title: str | None = None, metadata: dict | None = None
    ) -> Document:
        """Create or update a document from a single file path.

        Args:
            source_path: Path to the file
            title: Optional title
            metadata: Optional metadata dictionary

        Returns:
            Document instance (created, updated, or existing)

        Raises:
            ValueError: If the file cannot be parsed or doesn't exist
        """
        from haiku.rag.embeddings import embed_chunks

        metadata = metadata or {}

        converter = get_converter(self._config)
        if source_path.suffix.lower() not in converter.supported_extensions:
            raise ValueError(f"Unsupported file extension: {source_path.suffix}")

        if not source_path.exists():
            raise ValueError(f"File does not exist: {source_path}")

        uri = source_path.absolute().as_uri()
        md5_hash = hashlib.md5(
            source_path.read_bytes(), usedforsecurity=False
        ).hexdigest()

        # Get content type from file extension (do before early return)
        content_type, _ = mimetypes.guess_type(str(source_path))
        if not content_type:
            content_type = "application/octet-stream"
        # Merge metadata with contentType and md5
        metadata.update({"contentType": content_type, "md5": md5_hash})

        # Check if document already exists
        existing_doc = await self.get_document_by_uri(uri)
        if existing_doc and existing_doc.metadata.get("md5") == md5_hash:
            # MD5 unchanged; update title/metadata if provided
            updated = False
            if title is not None and title != existing_doc.title:
                existing_doc.title = title
                updated = True

            # Check if metadata actually changed (beyond contentType and md5)
            merged_metadata = {**(existing_doc.metadata or {}), **metadata}
            if merged_metadata != existing_doc.metadata:
                existing_doc.metadata = merged_metadata
                updated = True

            if updated:
                return await self.document_repository.update(existing_doc)
            return existing_doc

        # Convert → Chunk → Embed using primitives
        docling_document = await self.convert(source_path)
        chunks = await self.chunk(docling_document)
        embedded_chunks = await embed_chunks(chunks, self._config)

        stored_content = docling_document.export_to_markdown()

        if existing_doc:
            # Update existing document and rechunk
            existing_doc.content = stored_content
            existing_doc.metadata = metadata
            existing_doc.set_docling(docling_document)
            if title is not None:
                existing_doc.title = title
            elif existing_doc.title is None:
                existing_doc.title = await self._resolve_title(
                    docling_document, stored_content
                )
            return await self._update_document_with_chunks(
                existing_doc, embedded_chunks, docling_document
            )
        else:
            # Create new document
            if title is None:
                title = await self._resolve_title(docling_document, stored_content)
            document = Document(
                content=stored_content,
                uri=uri,
                title=title,
                metadata=metadata,
            )
            document.set_docling(docling_document)
            return await self._store_document_with_chunks(
                document, embedded_chunks, docling_document
            )

    async def _create_or_update_document_from_url(
        self, url: str, title: str | None = None, metadata: dict | None = None
    ) -> Document:
        """Create or update a document from a URL by downloading and parsing the content.

        Checks if a document with the same URI already exists:
        - If MD5 is unchanged, returns existing document
        - If MD5 changed, updates the document
        - If no document exists, creates a new one

        Args:
            url: URL to download and parse
            metadata: Optional metadata dictionary

        Returns:
            Document instance (created, updated, or existing)

        Raises:
            ValueError: If the content cannot be parsed
            httpx.RequestError: If URL request fails
        """
        from haiku.rag.embeddings import embed_chunks

        metadata = metadata or {}

        converter = get_converter(self._config)
        supported_extensions = converter.supported_extensions

        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()

            md5_hash = hashlib.md5(response.content).hexdigest()

            # Get content type early (used for potential no-op update)
            content_type = response.headers.get("content-type", "").lower()

            # Check if document already exists
            existing_doc = await self.get_document_by_uri(url)
            if existing_doc and existing_doc.metadata.get("md5") == md5_hash:
                # MD5 unchanged; update title/metadata if provided
                updated = False
                if title is not None and title != existing_doc.title:
                    existing_doc.title = title
                    updated = True

                metadata.update({"contentType": content_type, "md5": md5_hash})
                # Check if metadata actually changed (beyond contentType and md5)
                merged_metadata = {**(existing_doc.metadata or {}), **metadata}
                if merged_metadata != existing_doc.metadata:
                    existing_doc.metadata = merged_metadata
                    updated = True

                if updated:
                    return await self.document_repository.update(existing_doc)
                return existing_doc
            file_extension = self._get_extension_from_content_type_or_url(
                url, content_type
            )

            if file_extension not in supported_extensions:
                raise ValueError(
                    f"Unsupported content type/extension: {content_type}/{file_extension}"
                )

            # Create a temporary file with the appropriate extension
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=file_extension, delete=False
            ) as temp_file:
                temp_file.write(response.content)
                temp_file.flush()
                temp_path = Path(temp_file.name)

            try:
                # Convert → Chunk → Embed using primitives
                docling_document = await self.convert(temp_path)
                chunks = await self.chunk(docling_document)
                embedded_chunks = await embed_chunks(chunks, self._config)
            finally:
                temp_path.unlink(missing_ok=True)

            # Merge metadata with contentType and md5
            metadata.update({"contentType": content_type, "md5": md5_hash})

            stored_content = docling_document.export_to_markdown()

            if existing_doc:
                # Update existing document and rechunk
                existing_doc.content = stored_content
                existing_doc.metadata = metadata
                existing_doc.set_docling(docling_document)
                if title is not None:
                    existing_doc.title = title
                elif existing_doc.title is None:
                    existing_doc.title = await self._resolve_title(
                        docling_document, stored_content
                    )
                return await self._update_document_with_chunks(
                    existing_doc, embedded_chunks, docling_document
                )
            else:
                # Create new document
                if title is None:
                    title = await self._resolve_title(docling_document, stored_content)
                document = Document(
                    content=stored_content,
                    uri=url,
                    title=title,
                    metadata=metadata,
                )
                document.set_docling(docling_document)
                return await self._store_document_with_chunks(
                    document, embedded_chunks, docling_document
                )

    def _get_extension_from_content_type_or_url(
        self, url: str, content_type: str
    ) -> str:
        from haiku.rag.client.processing import get_extension_from_content_type_or_url

        return get_extension_from_content_type_or_url(url, content_type)

    async def get_document_by_id(self, document_id: str) -> Document | None:
        """Get a document by its ID.

        Args:
            document_id: The unique identifier of the document.

        Returns:
            The Document instance if found, None otherwise.
        """
        return await self.document_repository.get_by_id(document_id)

    async def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        """Get a chunk by its ID.

        Args:
            chunk_id: The unique identifier of the chunk.

        Returns:
            The Chunk instance if found, None otherwise.
        """
        return await self.chunk_repository.get_by_id(chunk_id)

    async def get_document_by_uri(self, uri: str) -> Document | None:
        """Get a document by its URI.

        Args:
            uri: The URI identifier of the document.

        Returns:
            The Document instance if found, None otherwise.
        """
        return await self.document_repository.get_by_uri(uri)

    async def resolve_document(self, id_or_title: str) -> Document | None:
        """Resolve a document by ID, title, or URI (in that order).

        Args:
            id_or_title: Document ID, title, or URI to look up.

        Returns:
            The Document instance if found, None otherwise.
        """
        doc = await self.get_document_by_id(id_or_title)
        if doc:
            return doc

        safe_input = escape_sql_string(id_or_title)
        docs = await self.list_documents(filter=f"title = '{safe_input}'")
        if docs and docs[0].id:
            return await self.get_document_by_id(docs[0].id)

        docs = await self.list_documents(filter=f"uri = '{safe_input}'")
        if docs and docs[0].id:
            return await self.get_document_by_id(docs[0].id)

        return None

    async def update_document(
        self,
        document_id: str,
        content: str | None = None,
        metadata: dict | None = None,
        chunks: list[Chunk] | None = None,
        title: str | None = None,
        docling_document: "DoclingDocument | None" = None,
    ) -> Document:
        """Update a document by ID.

        Updates specified fields. When content or docling_document is provided,
        the document is rechunked and re-embedded. Updates to only metadata or title
        skip rechunking for efficiency.

        Args:
            document_id: The ID of the document to update.
            content: New content (mutually exclusive with docling_document).
            metadata: New metadata dict.
            chunks: Custom chunks (will be embedded if missing embeddings).
            title: New title.
            docling_document: DoclingDocument to replace content (mutually exclusive with content).

        Returns:
            The updated Document instance.

        Raises:
            ValueError: If document not found, or if both content and docling_document
                are provided.
        """
        from haiku.rag.embeddings import embed_chunks

        # Validate: content and docling_document are mutually exclusive
        if content is not None and docling_document is not None:
            raise ValueError(
                "content and docling_document are mutually exclusive. "
                "Provide one or the other, not both."
            )

        # Fetch the existing document
        existing_doc = await self.get_document_by_id(document_id)
        if existing_doc is None:
            raise ValueError(f"Document with ID {document_id} not found")

        # Update metadata/title fields
        if title is not None:
            existing_doc.title = title
        if metadata is not None:
            existing_doc.metadata = metadata

        # Only metadata/title update - no rechunking needed
        if content is None and chunks is None and docling_document is None:
            return await self.document_repository.update(existing_doc)

        # Custom chunks provided - use them as-is
        if chunks is not None:
            # Store docling data if provided
            if docling_document is not None:
                existing_doc.content = docling_document.export_to_markdown()
                existing_doc.set_docling(docling_document)
            elif content is not None:
                existing_doc.content = content

            return await self._update_document_with_chunks(
                existing_doc, chunks, docling_document
            )

        # DoclingDocument provided without chunks - chunk and embed using primitives
        if docling_document is not None:
            existing_doc.content = docling_document.export_to_markdown()
            existing_doc.set_docling(docling_document)

            new_chunks = await self.chunk(docling_document)
            embedded_chunks = await embed_chunks(new_chunks, self._config)
            return await self._update_document_with_chunks(
                existing_doc, embedded_chunks, docling_document
            )

        # Content provided without chunks - convert, chunk, and embed using primitives
        assert content is not None
        existing_doc.content = content
        converter = get_converter(self._config)
        converted_docling = await converter.convert_text(
            existing_doc.content, format="md"
        )
        existing_doc.set_docling(converted_docling)

        new_chunks = await self.chunk(converted_docling)
        embedded_chunks = await embed_chunks(new_chunks, self._config)
        return await self._update_document_with_chunks(
            existing_doc, embedded_chunks, converted_docling
        )

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document by its ID."""
        return await self.document_repository.delete(document_id)

    async def list_documents(
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
                Defaults to False to avoid loading large blobs.

        Returns:
            List of Document instances matching the criteria.
        """
        return await self.document_repository.list_all(
            limit=limit, offset=offset, filter=filter, include_content=include_content
        )

    async def count_documents(self, filter: str | None = None) -> int:
        """Count documents with optional filtering.

        Args:
            filter: Optional SQL WHERE clause to filter documents.

        Returns:
            Number of documents matching the criteria.
        """
        return await self.document_repository.count(filter=filter)

    async def search(
        self,
        query: str,
        limit: int | None = None,
        search_type: str = "hybrid",
        filter: str | None = None,
    ) -> list[SearchResult]:
        from haiku.rag.client.search import search

        return await search(self, query, limit, search_type, filter)

    async def expand_context(
        self,
        search_results: list[SearchResult],
    ) -> list[SearchResult]:
        from haiku.rag.client.search import expand_context

        return await expand_context(self, search_results)

    async def ask(
        self,
        question: str,
        system_prompt: str | None = None,
        filter: str | None = None,
    ) -> "tuple[str, list[Citation]]":
        from haiku.rag.client.agents import ask

        return await ask(self, question, system_prompt, filter)

    async def research(
        self,
        question: str,
        *,
        filter: str | None = None,
        max_iterations: int | None = None,
    ) -> "ResearchReport":
        from haiku.rag.client.agents import research

        return await research(
            self, question, filter=filter, max_iterations=max_iterations
        )

    async def analyze(
        self,
        question: str,
        documents: list[str] | None = None,
        filter: str | None = None,
    ) -> "AnalysisResult":
        from haiku.rag.client.agents import analyze

        return await analyze(self, question, documents, filter)

    async def visualize_chunk(self, chunk: Chunk) -> list:
        from haiku.rag.client.search import visualize_chunk

        return await visualize_chunk(self, chunk)

    async def rebuild_database(
        self, mode: RebuildMode = RebuildMode.FULL
    ) -> AsyncGenerator[str, None]:
        from haiku.rag.client.rebuild import rebuild_database

        async for doc_id in rebuild_database(self, mode):
            yield doc_id

    def _check_source_accessible(self, uri: str) -> bool:
        """Check if a document's source URI is accessible."""
        parsed_url = urlparse(uri)
        try:
            if parsed_url.scheme == "file":
                return Path(parsed_url.path).exists()
            elif parsed_url.scheme in ("http", "https"):
                return True
            return False
        except Exception:
            return False

    async def vacuum(self) -> None:
        """Optimize and clean up old versions across all tables."""
        await self.store.vacuum()

    def close(self):
        """Close the underlying store connection."""
        self.store.close()
