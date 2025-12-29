import asyncio
import hashlib
import json
import logging
import mimetypes
import tempfile
from collections.abc import AsyncGenerator
from dataclasses import dataclass
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
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.repositories.settings import SettingsRepository

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.graph.research.models import Citation

logger = logging.getLogger(__name__)


class RebuildMode(Enum):
    """Mode for rebuilding the database."""

    FULL = "full"  # Re-convert from source, re-chunk, re-embed
    RECHUNK = "rechunk"  # Re-chunk from existing content, re-embed
    EMBED_ONLY = "embed_only"  # Keep chunks, only regenerate embeddings


@dataclass
class DownloadProgress:
    """Progress event for model downloads."""

    model: str
    status: str
    completed: int = 0
    total: int = 0
    digest: str = ""


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

        self.store = Store(
            db_path,
            config=self._config,
            skip_validation=skip_validation,
            create=create,
            read_only=read_only,
            before=before,
        )
        self.document_repository = DocumentRepository(self.store)
        self.chunk_repository = ChunkRepository(self.store)

    @property
    def is_read_only(self) -> bool:
        """Whether the client is in read-only mode."""
        return self.store.is_read_only

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ARG002
        """Async context manager exit."""
        # Wait for any pending vacuum to complete before closing
        async with self.store._vacuum_lock:
            pass
        self.close()
        return False

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
        """Convert a file, URL, or text to DoclingDocument.

        Args:
            source: One of:
                - Path: Local file path to convert
                - str (URL): HTTP/HTTPS URL to download and convert
                - str (text): Raw text content to convert
            format: The format of text content ("md", "html", or "plain").
                Defaults to "md". Use "plain" for plain text without parsing.
                Only used when source is raw text (not a file path or URL).
                Files and URLs determine format from extension/content-type.

        Returns:
            DoclingDocument from the converted source.

        Raises:
            ValueError: If the file doesn't exist or has unsupported extension.
            httpx.RequestError: If URL download fails.
        """
        converter = get_converter(self._config)

        # Path object - convert file directly
        if isinstance(source, Path):
            if not source.exists():
                raise ValueError(f"File does not exist: {source}")
            if source.suffix.lower() not in converter.supported_extensions:
                raise ValueError(f"Unsupported file extension: {source.suffix}")
            return await converter.convert_file(source)

        # String - check if URL or text
        parsed = urlparse(source)

        if parsed.scheme in ("http", "https"):
            # URL - download and convert
            async with httpx.AsyncClient() as http:
                response = await http.get(source)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "").lower()
                file_extension = self._get_extension_from_content_type_or_url(
                    source, content_type
                )

                if file_extension not in converter.supported_extensions:
                    raise ValueError(
                        f"Unsupported content type/extension: {content_type}/{file_extension}"
                    )

                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=file_extension, delete=False
                ) as temp_file:
                    temp_file.write(response.content)
                    temp_file.flush()
                    temp_path = Path(temp_file.name)

                try:
                    return await converter.convert_file(temp_path)
                finally:
                    temp_path.unlink(missing_ok=True)

        elif parsed.scheme == "file":
            # file:// URI
            file_path = Path(parsed.path)
            if not file_path.exists():
                raise ValueError(f"File does not exist: {file_path}")
            if file_path.suffix.lower() not in converter.supported_extensions:
                raise ValueError(f"Unsupported file extension: {file_path.suffix}")
            return await converter.convert_file(file_path)

        else:
            # Treat as text content
            return await converter.convert_text(source, format=format)

    async def chunk(self, docling_document: "DoclingDocument") -> list[Chunk]:
        """Chunk a DoclingDocument into Chunks.

        Args:
            docling_document: The DoclingDocument to chunk.

        Returns:
            List of Chunk objects (without embeddings, without document_id).
            Each chunk has its `order` field set to its position in the list.
        """
        from haiku.rag.chunkers import get_chunker

        chunker = get_chunker(self._config)
        return await chunker.chunk(docling_document)

    async def _ensure_chunks_embedded(self, chunks: list[Chunk]) -> list[Chunk]:
        """Ensure all chunks have embeddings, embedding any that don't.

        Args:
            chunks: List of chunks, some may have embeddings already.

        Returns:
            List of chunks with all embeddings populated.
        """
        from haiku.rag.embeddings import embed_chunks

        # Find chunks that need embedding
        chunks_to_embed = [c for c in chunks if c.embedding is None]

        if not chunks_to_embed:
            return chunks

        # Embed chunks that don't have embeddings (returns new Chunk objects)
        embedded = await embed_chunks(chunks_to_embed, self._config)

        # Build result maintaining original order
        embedded_map = {(c.content, c.order): c for c in embedded}
        result = []
        for chunk in chunks:
            if chunk.embedding is not None:
                result.append(chunk)
            else:
                result.append(embedded_map[(chunk.content, chunk.order)])

        return result

    async def _store_document_with_chunks(
        self,
        document: Document,
        chunks: list[Chunk],
    ) -> Document:
        """Store a document with chunks, embedding any that lack embeddings.

        Handles versioning/rollback on failure.

        Args:
            document: The document to store (will be created).
            chunks: Chunks to store (will be embedded if lacking embeddings).

        Returns:
            The created Document instance with ID set.
        """
        import asyncio

        # Ensure all chunks have embeddings before storing
        chunks = await self._ensure_chunks_embedded(chunks)

        # Snapshot table versions for versioned rollback (if supported)
        versions = self.store.current_table_versions()

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

            # Vacuum old versions in background (non-blocking) if auto_vacuum enabled
            if self._config.storage.auto_vacuum:
                asyncio.create_task(self.store.vacuum())

            return created_doc
        except Exception:
            # Roll back to the captured versions and re-raise
            self.store.restore_table_versions(versions)
            raise

    async def _update_document_with_chunks(
        self,
        document: Document,
        chunks: list[Chunk],
    ) -> Document:
        """Update a document and replace its chunks, embedding any that lack embeddings.

        Handles versioning/rollback on failure.

        Args:
            document: The document to update (must have ID set).
            chunks: Chunks to replace existing (will be embedded if lacking embeddings).

        Returns:
            The updated Document instance.
        """
        import asyncio

        assert document.id is not None, "Document ID is required for update"

        # Ensure all chunks have embeddings before storing
        chunks = await self._ensure_chunks_embedded(chunks)

        # Snapshot table versions for versioned rollback
        versions = self.store.current_table_versions()

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

            # Vacuum old versions in background (non-blocking) if auto_vacuum enabled
            if self._config.storage.auto_vacuum:
                asyncio.create_task(self.store.vacuum())

            return updated_doc
        except Exception:
            # Roll back to the captured versions and re-raise
            self.store.restore_table_versions(versions)
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
        docling_document = await self.convert(content, format=format)
        chunks = await self.chunk(docling_document)
        embedded_chunks = await embed_chunks(chunks, self._config)

        # Store markdown export as content for better display/readability
        # The original content is preserved in docling_document_json
        stored_content = docling_document.export_to_markdown()

        # Create document model
        document = Document(
            content=stored_content,
            uri=uri,
            title=title,
            metadata=metadata or {},
            docling_document_json=docling_document.model_dump_json(),
            docling_version=docling_document.version,
        )

        # Store document and chunks
        return await self._store_document_with_chunks(document, embedded_chunks)

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
        document = Document(
            content=docling_document.export_to_markdown(),
            uri=uri,
            title=title,
            metadata=metadata or {},
            docling_document_json=docling_document.model_dump_json(),
            docling_version=docling_document.version,
        )

        return await self._store_document_with_chunks(document, chunks)

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
        md5_hash = hashlib.md5(source_path.read_bytes()).hexdigest()

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

        if existing_doc:
            # Update existing document and rechunk
            existing_doc.content = docling_document.export_to_markdown()
            existing_doc.metadata = metadata
            existing_doc.docling_document_json = docling_document.model_dump_json()
            existing_doc.docling_version = docling_document.version
            if title is not None:
                existing_doc.title = title
            return await self._update_document_with_chunks(
                existing_doc, embedded_chunks
            )
        else:
            # Create new document
            document = Document(
                content=docling_document.export_to_markdown(),
                uri=uri,
                title=title,
                metadata=metadata,
                docling_document_json=docling_document.model_dump_json(),
                docling_version=docling_document.version,
            )
            return await self._store_document_with_chunks(document, embedded_chunks)

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

            if existing_doc:
                # Update existing document and rechunk
                existing_doc.content = docling_document.export_to_markdown()
                existing_doc.metadata = metadata
                existing_doc.docling_document_json = docling_document.model_dump_json()
                existing_doc.docling_version = docling_document.version
                if title is not None:
                    existing_doc.title = title
                return await self._update_document_with_chunks(
                    existing_doc, embedded_chunks
                )
            else:
                # Create new document
                document = Document(
                    content=docling_document.export_to_markdown(),
                    uri=url,
                    title=title,
                    metadata=metadata,
                    docling_document_json=docling_document.model_dump_json(),
                    docling_version=docling_document.version,
                )
                return await self._store_document_with_chunks(document, embedded_chunks)

    def _get_extension_from_content_type_or_url(
        self, url: str, content_type: str
    ) -> str:
        """Determine file extension from content type or URL."""
        # Common content type mappings
        content_type_map = {
            "text/html": ".html",
            "text/plain": ".txt",
            "text/markdown": ".md",
            "application/pdf": ".pdf",
            "application/json": ".json",
            "text/csv": ".csv",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        }

        # Try content type first
        for ct, ext in content_type_map.items():
            if ct in content_type:
                return ext

        # Try URL extension
        parsed_url = urlparse(url)
        path = Path(parsed_url.path)
        if path.suffix:
            return path.suffix.lower()

        # Default to .html for web content
        return ".html"

    async def get_document_by_id(self, document_id: str) -> Document | None:
        """Get a document by its ID.

        Args:
            document_id: The unique identifier of the document.

        Returns:
            The Document instance if found, None otherwise.
        """
        return await self.document_repository.get_by_id(document_id)

    async def get_document_by_uri(self, uri: str) -> Document | None:
        """Get a document by its URI.

        Args:
            uri: The URI identifier of the document.

        Returns:
            The Document instance if found, None otherwise.
        """
        return await self.document_repository.get_by_uri(uri)

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
                existing_doc.docling_document_json = docling_document.model_dump_json()
                existing_doc.docling_version = docling_document.version
            elif content is not None:
                existing_doc.content = content

            return await self._update_document_with_chunks(existing_doc, chunks)

        # DoclingDocument provided without chunks - chunk and embed using primitives
        if docling_document is not None:
            existing_doc.content = docling_document.export_to_markdown()
            existing_doc.docling_document_json = docling_document.model_dump_json()
            existing_doc.docling_version = docling_document.version

            new_chunks = await self.chunk(docling_document)
            embedded_chunks = await embed_chunks(new_chunks, self._config)
            return await self._update_document_with_chunks(
                existing_doc, embedded_chunks
            )

        # Content provided without chunks - convert, chunk, and embed using primitives
        existing_doc.content = content  # type: ignore[assignment]
        converted_docling = await self.convert(existing_doc.content)
        existing_doc.docling_document_json = converted_docling.model_dump_json()
        existing_doc.docling_version = converted_docling.version

        new_chunks = await self.chunk(converted_docling)
        embedded_chunks = await embed_chunks(new_chunks, self._config)
        return await self._update_document_with_chunks(existing_doc, embedded_chunks)

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document by its ID."""
        return await self.document_repository.delete(document_id)

    async def list_documents(
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
        return await self.document_repository.list_all(
            limit=limit, offset=offset, filter=filter
        )

    async def search(
        self,
        query: str,
        limit: int | None = None,
        search_type: str = "hybrid",
        filter: str | None = None,
    ) -> list[SearchResult]:
        """Search for relevant chunks using the specified search method with optional reranking.

        Args:
            query: The search query string.
            limit: Maximum number of results to return. Defaults to config.search.default_limit.
            search_type: Type of search - "vector", "fts", or "hybrid" (default).
            filter: Optional SQL WHERE clause to filter documents before searching chunks.

        Returns:
            List of SearchResult objects ordered by relevance.
        """
        if limit is None:
            limit = self._config.search.limit

        reranker = get_reranker(config=self._config)

        if reranker is None:
            chunk_results = await self.chunk_repository.search(
                query, limit, search_type, filter
            )
        else:
            search_limit = limit * 10
            raw_results = await self.chunk_repository.search(
                query, search_limit, search_type, filter
            )
            chunks = [chunk for chunk, _ in raw_results]
            chunk_results = await reranker.rerank(query, chunks, top_n=limit)

        return [SearchResult.from_chunk(chunk, score) for chunk, score in chunk_results]

    async def expand_context(
        self,
        search_results: list[SearchResult],
    ) -> list[SearchResult]:
        """Expand search results with adjacent content from the source document.

        When DoclingDocument is available and results have doc_item_refs, expands
        by finding adjacent DocItems with accurate bounding boxes and metadata.
        Otherwise, falls back to chunk-based expansion using adjacent chunks.

        Expansion is type-aware based on content:
        - Tables, code blocks, and lists expand to include complete structures
        - Text content uses the configured radius (search.context_radius)
        - Expansion is limited by search.max_context_items and search.max_context_chars

        Args:
            search_results: List of SearchResult objects from search.

        Returns:
            List of SearchResult objects with expanded content and resolved provenance.
        """
        radius = self._config.search.context_radius
        max_items = self._config.search.max_context_items
        max_chars = self._config.search.max_context_chars

        # Group by document_id for efficient processing
        document_groups: dict[str | None, list[SearchResult]] = {}
        for result in search_results:
            doc_id = result.document_id
            if doc_id not in document_groups:
                document_groups[doc_id] = []
            document_groups[doc_id].append(result)

        expanded_results = []

        for doc_id, doc_results in document_groups.items():
            if doc_id is None:
                expanded_results.extend(doc_results)
                continue

            # Fetch the document to get DoclingDocument
            doc = await self.get_document_by_id(doc_id)
            if doc is None:
                expanded_results.extend(doc_results)
                continue

            docling_doc = doc.get_docling_document()

            # Check if we can use DoclingDocument-based expansion
            has_docling = docling_doc is not None
            has_refs = any(r.doc_item_refs for r in doc_results)

            if has_docling and has_refs:
                # Use DoclingDocument-based expansion
                expanded = await self._expand_with_docling(
                    doc_results,
                    docling_doc,
                    radius,
                    max_items,
                    max_chars,
                )
                expanded_results.extend(expanded)
            else:
                # Fall back to chunk-based expansion (always uses fixed radius)
                if radius > 0:
                    expanded = await self._expand_with_chunks(
                        doc_id, doc_results, radius
                    )
                    expanded_results.extend(expanded)
                else:
                    expanded_results.extend(doc_results)

        return expanded_results

    def _merge_ranges(
        self, ranges: list[tuple[int, int, SearchResult]]
    ) -> list[tuple[int, int, list[SearchResult]]]:
        """Merge overlapping or adjacent ranges."""
        if not ranges:
            return []

        sorted_ranges = sorted(ranges, key=lambda x: x[0])
        merged: list[tuple[int, int, list[SearchResult]]] = []
        cur_min, cur_max, cur_results = (
            sorted_ranges[0][0],
            sorted_ranges[0][1],
            [sorted_ranges[0][2]],
        )

        for min_idx, max_idx, result in sorted_ranges[1:]:
            if cur_max >= min_idx - 1:  # Overlapping or adjacent
                cur_max = max(cur_max, max_idx)
                cur_results.append(result)
            else:
                merged.append((cur_min, cur_max, cur_results))
                cur_min, cur_max, cur_results = min_idx, max_idx, [result]

        merged.append((cur_min, cur_max, cur_results))
        return merged

    # Label groups for type-aware expansion
    _STRUCTURAL_LABELS = {"table", "code", "list_item", "form", "key_value_region"}

    def _extract_item_text(self, item, docling_doc) -> str | None:
        """Extract text content from a DocItem.

        Handles different item types:
        - TextItem, SectionHeaderItem, etc.: Use .text attribute
        - TableItem: Use export_to_markdown() for table content
        - PictureItem: Use caption if available
        """
        # Try simple text attribute first (works for most items)
        if text := getattr(item, "text", None):
            return text

        # For tables, export as markdown
        if hasattr(item, "export_to_markdown"):
            try:
                return item.export_to_markdown(docling_doc)
            except Exception:
                pass

        # For pictures/charts, try to get caption
        if caption := getattr(item, "caption", None):
            if hasattr(caption, "text"):
                return caption.text

        return None

    def _get_item_label(self, item) -> str | None:
        """Extract label string from a DocItem."""
        label = getattr(item, "label", None)
        if label is None:
            return None
        return str(label.value) if hasattr(label, "value") else str(label)

    def _compute_type_aware_range(
        self,
        all_items: list,
        indices: list[int],
        radius: int,
        max_items: int,
        max_chars: int,
    ) -> tuple[int, int]:
        """Compute expansion range based on content type with limits.

        For structural content (tables, code, lists), expands to include complete
        structures. For text, uses the configured radius. Applies hybrid limits.
        """
        if not indices:
            return (0, 0)

        min_idx = min(indices)
        max_idx = max(indices)

        # Determine the primary label type from matched items
        labels_in_chunk = set()
        for idx in indices:
            item, _ = all_items[idx]
            if label := self._get_item_label(item):
                labels_in_chunk.add(label)

        # Check if we have structural content
        is_structural = bool(labels_in_chunk & self._STRUCTURAL_LABELS)

        if is_structural:
            # Expand to complete structure boundaries
            # Expand backwards to find structure start
            while min_idx > 0:
                prev_item, _ = all_items[min_idx - 1]
                prev_label = self._get_item_label(prev_item)
                if prev_label in labels_in_chunk & self._STRUCTURAL_LABELS:
                    min_idx -= 1
                else:
                    break

            # Expand forwards to find structure end
            while max_idx < len(all_items) - 1:
                next_item, _ = all_items[max_idx + 1]
                next_label = self._get_item_label(next_item)
                if next_label in labels_in_chunk & self._STRUCTURAL_LABELS:
                    max_idx += 1
                else:
                    break
        else:
            # Text content: use radius-based expansion
            min_idx = max(0, min_idx - radius)
            max_idx = min(len(all_items) - 1, max_idx + radius)

        # Apply hybrid limits
        # First check item count hard limit
        if max_idx - min_idx + 1 > max_items:
            # Center the window around original indices
            original_center = (min(indices) + max(indices)) // 2
            half_items = max_items // 2
            min_idx = max(0, original_center - half_items)
            max_idx = min(len(all_items) - 1, min_idx + max_items - 1)

        # Then check character soft limit (but keep at least original items)
        char_count = 0
        effective_max = min_idx
        for i in range(min_idx, max_idx + 1):
            item, _ = all_items[i]
            text = getattr(item, "text", "") or ""
            char_count += len(text)
            effective_max = i
            # Once we've included original items, check char limit
            if i >= max(indices) and char_count > max_chars:
                break

        max_idx = effective_max

        return (min_idx, max_idx)

    async def _expand_with_docling(
        self,
        results: list[SearchResult],
        docling_doc,
        radius: int,
        max_items: int,
        max_chars: int,
    ) -> list[SearchResult]:
        """Expand results using DoclingDocument structure.

        Structural content (tables, code, lists) expands to complete structures.
        Text content uses radius-based expansion.
        """
        all_items = list(docling_doc.iterate_items())
        ref_to_index = {
            getattr(item, "self_ref", None): i
            for i, (item, _) in enumerate(all_items)
            if getattr(item, "self_ref", None)
        }

        # Compute expanded ranges
        ranges: list[tuple[int, int, SearchResult]] = []
        passthrough: list[SearchResult] = []

        for result in results:
            indices = [
                ref_to_index[r] for r in result.doc_item_refs if r in ref_to_index
            ]
            if not indices:
                passthrough.append(result)
                continue

            min_idx, max_idx = self._compute_type_aware_range(
                all_items, indices, radius, max_items, max_chars
            )

            ranges.append((min_idx, max_idx, result))

        # Merge overlapping ranges
        merged = self._merge_ranges(ranges)

        final_results: list[SearchResult] = []
        for min_idx, max_idx, original_results in merged:
            content_parts: list[str] = []
            refs: list[str] = []
            pages: set[int] = set()
            labels: set[str] = set()

            for i in range(min_idx, max_idx + 1):
                item, _ = all_items[i]
                # Extract text content - handle different item types
                text = self._extract_item_text(item, docling_doc)
                if text:
                    content_parts.append(text)
                if self_ref := getattr(item, "self_ref", None):
                    refs.append(self_ref)
                if label := getattr(item, "label", None):
                    labels.add(
                        str(label.value) if hasattr(label, "value") else str(label)
                    )
                if prov := getattr(item, "prov", None):
                    for p in prov:
                        if (page_no := getattr(p, "page_no", None)) is not None:
                            pages.add(page_no)

            # Merge headings preserving order
            all_headings: list[str] = []
            for r in original_results:
                if r.headings:
                    all_headings.extend(h for h in r.headings if h not in all_headings)

            first = original_results[0]
            final_results.append(
                SearchResult(
                    content="\n\n".join(content_parts),
                    score=max(r.score for r in original_results),
                    chunk_id=first.chunk_id,
                    document_id=first.document_id,
                    document_uri=first.document_uri,
                    document_title=first.document_title,
                    doc_item_refs=refs,
                    page_numbers=sorted(pages),
                    headings=all_headings or None,
                    labels=sorted(labels),
                )
            )

        return final_results + passthrough

    async def _expand_with_chunks(
        self,
        doc_id: str,
        results: list[SearchResult],
        radius: int,
    ) -> list[SearchResult]:
        """Expand results using chunk-based adjacency."""
        all_chunks = await self.chunk_repository.get_by_document_id(doc_id)
        if not all_chunks:
            return results

        content_to_chunk = {c.content: c for c in all_chunks}
        chunk_by_order = {c.order: c for c in all_chunks}
        min_order, max_order = min(chunk_by_order.keys()), max(chunk_by_order.keys())

        # Build ranges
        ranges: list[tuple[int, int, SearchResult]] = []
        passthrough: list[SearchResult] = []

        for result in results:
            chunk = content_to_chunk.get(result.content)
            if chunk is None:
                passthrough.append(result)
                continue
            start = max(min_order, chunk.order - radius)
            end = min(max_order, chunk.order + radius)
            ranges.append((start, end, result))

        # Merge and build results
        final_results: list[SearchResult] = []
        for min_idx, max_idx, original_results in self._merge_ranges(ranges):
            # Collect chunks in order
            chunks_in_range = [
                chunk_by_order[o]
                for o in range(min_idx, max_idx + 1)
                if o in chunk_by_order
            ]
            first = original_results[0]
            final_results.append(
                SearchResult(
                    content="".join(c.content for c in chunks_in_range),
                    score=max(r.score for r in original_results),
                    chunk_id=first.chunk_id,
                    document_id=first.document_id,
                    document_uri=first.document_uri,
                    document_title=first.document_title,
                    doc_item_refs=first.doc_item_refs,
                    page_numbers=first.page_numbers,
                    headings=first.headings,
                    labels=first.labels,
                )
            )

        return final_results + passthrough

    async def ask(
        self,
        question: str,
        system_prompt: str | None = None,
        filter: str | None = None,
    ) -> "tuple[str, list[Citation]]":
        """Ask a question using the configured QA agent.

        Args:
            question: The question to ask.
            system_prompt: Optional custom system prompt for the QA agent.
            filter: SQL WHERE clause to filter documents.

        Returns:
            Tuple of (answer text, list of resolved citations).
        """
        from haiku.rag.qa import get_qa_agent

        qa_agent = get_qa_agent(self, config=self._config, system_prompt=system_prompt)
        return await qa_agent.answer(question, filter=filter)

    async def visualize_chunk(self, chunk: Chunk) -> list:
        """Render page images with bounding box highlights for a chunk.

        Gets the DoclingDocument from the chunk's document, resolves bounding boxes
        from chunk metadata, and renders all pages that contain bounding boxes with
        yellow/orange highlight overlays.

        Args:
            chunk: The chunk to visualize.

        Returns:
            List of PIL Image objects, one per page with bounding boxes.
            Empty list if no bounding boxes or page images available.
        """
        from copy import deepcopy

        from PIL import ImageDraw

        # Get the document
        if not chunk.document_id:
            return []

        doc = await self.document_repository.get_by_id(chunk.document_id)
        if not doc:
            return []

        # Get DoclingDocument
        docling_doc = doc.get_docling_document()
        if not docling_doc:
            return []

        # Resolve bounding boxes from chunk metadata
        chunk_meta = chunk.get_chunk_metadata()
        bounding_boxes = chunk_meta.resolve_bounding_boxes(docling_doc)
        if not bounding_boxes:
            return []

        # Group bounding boxes by page
        boxes_by_page: dict[int, list] = {}
        for bbox in bounding_boxes:
            if bbox.page_no not in boxes_by_page:
                boxes_by_page[bbox.page_no] = []
            boxes_by_page[bbox.page_no].append(bbox)

        # Render each page with its bounding boxes
        images = []
        for page_no in sorted(boxes_by_page.keys()):
            if page_no not in docling_doc.pages:
                continue

            page = docling_doc.pages[page_no]
            if page.image is None or page.image.pil_image is None:
                continue

            pil_image = page.image.pil_image
            page_height = page.size.height

            # Calculate scale factor (image pixels vs document coordinates)
            scale_x = pil_image.width / page.size.width
            scale_y = pil_image.height / page.size.height

            # Draw bounding boxes
            image = deepcopy(pil_image)
            draw = ImageDraw.Draw(image, "RGBA")

            for bbox in boxes_by_page[page_no]:
                # Convert from document coordinates to image coordinates
                # Document coords are bottom-left origin, PIL uses top-left
                x0 = bbox.left * scale_x
                y0 = (page_height - bbox.top) * scale_y
                x1 = bbox.right * scale_x
                y1 = (page_height - bbox.bottom) * scale_y

                # Ensure proper ordering (y0 should be less than y1 for PIL)
                if y0 > y1:
                    y0, y1 = y1, y0

                # Draw filled rectangle with transparency
                fill_color = (255, 255, 0, 80)  # Yellow with transparency
                outline_color = (255, 165, 0, 255)  # Orange outline

                draw.rectangle([(x0, y0), (x1, y1)], fill=fill_color, outline=None)
                draw.rectangle([(x0, y0), (x1, y1)], outline=outline_color, width=3)

            images.append(image)

        return images

    async def rebuild_database(
        self, mode: RebuildMode = RebuildMode.FULL
    ) -> AsyncGenerator[str, None]:
        """Rebuild the database with the specified mode.

        Args:
            mode: The rebuild mode to use:
                - FULL: Re-convert from source files, re-chunk, re-embed (default)
                - RECHUNK: Re-chunk from existing content, re-embed (no source access)
                - EMBED_ONLY: Keep existing chunks, only regenerate embeddings

        Yields:
            The ID of the document currently being processed.
        """
        # Update settings to current config
        settings_repo = SettingsRepository(self.store)
        settings_repo.save_current_settings()

        documents = await self.list_documents()

        if mode == RebuildMode.EMBED_ONLY:
            async for doc_id in self._rebuild_embed_only(documents):
                yield doc_id
        elif mode == RebuildMode.RECHUNK:
            await self.chunk_repository.delete_all()
            self.store.recreate_embeddings_table()
            async for doc_id in self._rebuild_rechunk(documents):
                yield doc_id
        else:  # FULL
            await self.chunk_repository.delete_all()
            self.store.recreate_embeddings_table()
            async for doc_id in self._rebuild_full(documents):
                yield doc_id

        # Final maintenance if auto_vacuum enabled
        if self._config.storage.auto_vacuum:
            try:
                await self.store.vacuum()
            except Exception:
                pass

    async def _rebuild_embed_only(
        self, documents: list[Document]
    ) -> AsyncGenerator[str, None]:
        """Re-embed all chunks without changing chunk boundaries."""
        from haiku.rag.embeddings import contextualize

        batch_size = 50
        pending_records: list = []
        pending_doc_ids: list[str] = []

        for doc in documents:
            assert doc.id is not None

            # Get existing chunks
            chunks = await self.chunk_repository.get_by_document_id(doc.id)
            if not chunks:
                yield doc.id
                continue

            # Generate new embeddings using contextualize for consistency
            texts = contextualize(chunks)
            embeddings = await self.chunk_repository.embedder.embed_documents(texts)

            # Build updated records
            for chunk, content_fts, embedding in zip(chunks, texts, embeddings):
                pending_records.append(
                    self.store.ChunkRecord(
                        id=chunk.id,  # type: ignore[arg-type]
                        document_id=chunk.document_id,  # type: ignore[arg-type]
                        content=chunk.content,
                        content_fts=content_fts,
                        metadata=json.dumps(chunk.metadata),
                        order=chunk.order,
                        vector=embedding,
                    )
                )

            pending_doc_ids.append(doc.id)

            # Flush batch when size reached
            if len(pending_doc_ids) >= batch_size:
                if pending_records:
                    self.store.chunks_table.merge_insert(
                        "id"
                    ).when_matched_update_all().execute(pending_records)
                for doc_id in pending_doc_ids:
                    yield doc_id
                pending_records = []
                pending_doc_ids = []

        # Flush remaining
        if pending_records:
            self.store.chunks_table.merge_insert(
                "id"
            ).when_matched_update_all().execute(pending_records)
        for doc_id in pending_doc_ids:
            yield doc_id

    async def _flush_rebuild_batch(
        self, documents: list[Document], chunks: list[Chunk]
    ) -> None:
        """Batch write documents and chunks during rebuild.

        This performs two writes: one for all document updates, one for all chunks.
        Used by RECHUNK and FULL modes after the chunks table has been cleared.
        """
        from haiku.rag.store.engine import DocumentRecord
        from haiku.rag.store.models.document import invalidate_docling_document_cache

        if not documents:
            return

        now = datetime.now().isoformat()

        # Invalidate cache for all documents being updated
        for doc in documents:
            if doc.id:
                invalidate_docling_document_cache(doc.id)

        # Batch update documents using merge_insert (single LanceDB version)
        doc_records = [
            DocumentRecord(
                id=doc.id,  # type: ignore[arg-type]
                content=doc.content,
                uri=doc.uri,
                title=doc.title,
                metadata=json.dumps(doc.metadata),
                docling_document_json=doc.docling_document_json,
                docling_version=doc.docling_version,
                created_at=doc.created_at.isoformat() if doc.created_at else now,
                updated_at=now,
            )
            for doc in documents
        ]

        self.store.documents_table.merge_insert("id").when_matched_update_all().execute(
            doc_records
        )

        # Batch create all chunks (single LanceDB version)
        if chunks:
            await self.chunk_repository.create(chunks)

    async def _rebuild_rechunk(
        self, documents: list[Document]
    ) -> AsyncGenerator[str, None]:
        """Re-chunk and re-embed from existing document content."""
        from haiku.rag.embeddings import embed_chunks

        batch_size = 50
        pending_chunks: list[Chunk] = []
        pending_docs: list[Document] = []
        pending_doc_ids: list[str] = []

        for doc in documents:
            assert doc.id is not None

            # Convert content to DoclingDocument
            docling_document = await self.convert(doc.content)

            # Chunk and embed
            chunks = await self.chunk(docling_document)
            embedded_chunks = await embed_chunks(chunks, self._config)

            # Update document fields
            doc.docling_document_json = docling_document.model_dump_json()
            doc.docling_version = docling_document.version

            # Prepare chunks with document_id and order
            for order, chunk in enumerate(embedded_chunks):
                chunk.document_id = doc.id
                chunk.order = order

            pending_chunks.extend(embedded_chunks)
            pending_docs.append(doc)
            pending_doc_ids.append(doc.id)

            # Flush batch when size reached
            if len(pending_docs) >= batch_size:
                await self._flush_rebuild_batch(pending_docs, pending_chunks)
                for doc_id in pending_doc_ids:
                    yield doc_id
                pending_chunks = []
                pending_docs = []
                pending_doc_ids = []

        # Flush remaining
        if pending_docs:
            await self._flush_rebuild_batch(pending_docs, pending_chunks)
            for doc_id in pending_doc_ids:
                yield doc_id

    async def _rebuild_full(
        self, documents: list[Document]
    ) -> AsyncGenerator[str, None]:
        """Full rebuild: re-convert from source, re-chunk, re-embed."""
        from haiku.rag.embeddings import embed_chunks

        batch_size = 50
        pending_chunks: list[Chunk] = []
        pending_docs: list[Document] = []
        pending_doc_ids: list[str] = []

        for doc in documents:
            assert doc.id is not None

            # Try to rebuild from source if available
            if doc.uri and self._check_source_accessible(doc.uri):
                try:
                    # Flush pending batch before source rebuild (creates new doc)
                    if pending_docs:
                        await self._flush_rebuild_batch(pending_docs, pending_chunks)
                        for doc_id in pending_doc_ids:
                            yield doc_id
                        pending_chunks = []
                        pending_docs = []
                        pending_doc_ids = []

                    await self.delete_document(doc.id)
                    new_doc = await self.create_document_from_source(
                        source=doc.uri, metadata=doc.metadata or {}
                    )
                    assert isinstance(new_doc, Document)
                    assert new_doc.id is not None
                    yield new_doc.id
                    continue
                except Exception as e:
                    logger.error(
                        "Error recreating document from source %s: %s",
                        doc.uri,
                        e,
                    )
                    continue

            # Fallback: rebuild from stored content
            if doc.uri:
                logger.warning(
                    "Source missing for %s, re-embedding from content", doc.uri
                )

            docling_document = await self.convert(doc.content)
            chunks = await self.chunk(docling_document)
            embedded_chunks = await embed_chunks(chunks, self._config)

            doc.docling_document_json = docling_document.model_dump_json()
            doc.docling_version = docling_document.version

            # Prepare chunks with document_id and order
            for order, chunk in enumerate(embedded_chunks):
                chunk.document_id = doc.id
                chunk.order = order

            pending_chunks.extend(embedded_chunks)
            pending_docs.append(doc)
            pending_doc_ids.append(doc.id)

            # Flush batch when size reached
            if len(pending_docs) >= batch_size:
                await self._flush_rebuild_batch(pending_docs, pending_chunks)
                for doc_id in pending_doc_ids:
                    yield doc_id
                pending_chunks = []
                pending_docs = []
                pending_doc_ids = []

        # Flush remaining
        if pending_docs:
            await self._flush_rebuild_batch(pending_docs, pending_chunks)
            for doc_id in pending_doc_ids:
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

    async def download_models(self) -> AsyncGenerator[DownloadProgress, None]:
        """Download required models, yielding progress events.

        Yields DownloadProgress events for:
        - Docling models (status="docling_start", "docling_done")
        - HuggingFace tokenizer (status="tokenizer_start", "tokenizer_done")
        - Ollama models (status="pulling", "downloading", "done", or other Ollama statuses)
        """
        # Docling models
        try:
            from docling.utils.model_downloader import download_models

            yield DownloadProgress(model="docling", status="start")
            await asyncio.to_thread(download_models)
            yield DownloadProgress(model="docling", status="done")
        except ImportError:
            pass

        # HuggingFace tokenizer
        from transformers import AutoTokenizer

        tokenizer_name = self._config.processing.chunking_tokenizer
        yield DownloadProgress(model=tokenizer_name, status="start")
        await asyncio.to_thread(AutoTokenizer.from_pretrained, tokenizer_name)
        yield DownloadProgress(model=tokenizer_name, status="done")

        # Collect Ollama models from config
        required_models: set[str] = set()
        if self._config.embeddings.model.provider == "ollama":
            required_models.add(self._config.embeddings.model.name)
        if self._config.qa.model.provider == "ollama":
            required_models.add(self._config.qa.model.name)
        if self._config.research.model.provider == "ollama":
            required_models.add(self._config.research.model.name)
        if (
            self._config.reranking.model
            and self._config.reranking.model.provider == "ollama"
        ):
            required_models.add(self._config.reranking.model.name)

        if not required_models:
            return

        base_url = self._config.providers.ollama.base_url

        async with httpx.AsyncClient(timeout=None) as client:
            for model in sorted(required_models):
                yield DownloadProgress(model=model, status="pulling")

                async with client.stream(
                    "POST", f"{base_url}/api/pull", json={"model": model}
                ) as r:
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            status = data.get("status", "")
                            digest = data.get("digest", "")

                            if digest and "total" in data:
                                yield DownloadProgress(
                                    model=model,
                                    status="downloading",
                                    total=data.get("total", 0),
                                    completed=data.get("completed", 0),
                                    digest=digest,
                                )
                            elif status:
                                yield DownloadProgress(model=model, status=status)
                        except json.JSONDecodeError:
                            pass

                yield DownloadProgress(model=model, status="done")

    def close(self):
        """Close the underlying store connection."""
        self.store.close()
