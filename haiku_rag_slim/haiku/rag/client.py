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

    # =========================================================================
    # Title Generation
    # =========================================================================

    def _extract_structural_title(
        self, docling_document: "DoclingDocument"
    ) -> str | None:
        """Extract a title from DoclingDocument structural metadata.

        Priority: FURNITURE TITLE > BODY TITLE > first SECTION_HEADER.
        """
        from docling_core.types.doc.document import ContentLayer
        from docling_core.types.doc.labels import DocItemLabel

        furniture_title = None
        body_title = None
        first_section_header = None

        for item in docling_document.texts:
            if item.label == DocItemLabel.TITLE:
                text = item.text.strip()
                if not text:
                    continue
                if item.content_layer == ContentLayer.FURNITURE:
                    furniture_title = text
                elif body_title is None:
                    body_title = text
            elif (
                item.label == DocItemLabel.SECTION_HEADER
                and first_section_header is None
            ):
                text = item.text.strip()
                if text:
                    first_section_header = text

        return furniture_title or body_title or first_section_header

    async def _generate_title_with_llm(self, content: str) -> str | None:
        """Generate a title using LLM from document content."""
        from pydantic_ai import Agent

        from haiku.rag.utils import get_model

        truncated = content[:2000]

        model = get_model(self._config.processing.title_model, self._config)
        agent: Agent[None, str] = Agent(
            model=model,
            output_type=str,
            instructions=(
                "Generate a concise, descriptive title for the following document. "
                "The title should be at most 10 words. "
                "Return ONLY the title text, nothing else."
            ),
        )
        result = await agent.run(truncated)
        title = result.output.strip()
        return title if title else None

    async def _resolve_title(
        self,
        docling_document: "DoclingDocument",
        content: str,
    ) -> str | None:
        """Auto-generate a title from document structure or LLM.

        Returns None if auto_title is disabled or generation fails.
        """
        if not self._config.processing.auto_title:
            return None

        structural = self._extract_structural_title(docling_document)
        if structural:
            return structural

        try:
            return await self._generate_title_with_llm(content)
        except Exception:
            logger.warning(
                "LLM title generation failed during ingestion", exc_info=True
            )
            return None

    async def generate_title(self, document: Document) -> str | None:
        """Generate a title for a document.

        Attempts structural extraction from the stored DoclingDocument,
        then falls back to LLM generation. Bypasses the auto_title config
        since this is an explicit call.

        Does NOT update the document — caller decides.
        """
        docling_doc = document.get_docling_document()
        content = document.content or ""

        if docling_doc is not None:
            structural = self._extract_structural_title(docling_doc)
            if structural:
                return structural

        return await self._generate_title_with_llm(content)

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
        """Expand search results with surrounding content from the document.

        Uses the document_items table for section-bounded expansion.
        See haiku.rag.context for the algorithm description.

        Results without doc_item_refs pass through unexpanded. This happens
        when chunks were created without docling metadata (e.g., custom chunks
        passed to import_document).

        Args:
            search_results: List of SearchResult objects from search.

        Returns:
            List of SearchResult objects with expanded content.
        """
        from haiku.rag.context import expand_with_items

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

            has_refs = any(r.doc_item_refs for r in doc_results)
            if not has_refs:
                expanded_results.extend(doc_results)
                continue

            expanded = await expand_with_items(
                self.document_item_repository,
                doc_id,
                doc_results,
                max_chars,
            )
            expanded_results.extend(expanded)

        expanded_results.sort(key=lambda r: r.score, reverse=True)
        return expanded_results

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
        from haiku.rag.agents.qa import get_qa_agent

        qa_agent = get_qa_agent(self, config=self._config, system_prompt=system_prompt)
        return await qa_agent.answer(question, filter=filter)

    async def research(
        self,
        question: str,
        *,
        filter: str | None = None,
        max_iterations: int | None = None,
    ) -> "ResearchReport":
        """Run multi-agent research to investigate a question.

        Args:
            question: The research question to investigate.
            filter: SQL WHERE clause to filter documents.
            max_iterations: Override max iterations (None uses config default).

        Returns:
            ResearchReport with structured findings.
        """
        from haiku.rag.agents.research.dependencies import ResearchContext
        from haiku.rag.agents.research.graph import build_research_graph
        from haiku.rag.agents.research.state import ResearchDeps, ResearchState

        graph = build_research_graph(config=self._config)
        context = ResearchContext(original_question=question)
        state = ResearchState.from_config(
            context=context, config=self._config, max_iterations=max_iterations
        )
        state.search_filter = filter
        deps = ResearchDeps(client=self)

        return await graph.run(state=state, deps=deps)

    async def analyze(
        self,
        question: str,
        documents: list[str] | None = None,
        filter: str | None = None,
    ) -> "AnalysisResult":
        """Answer a question using the analysis agent with code execution.

        The analysis agent can write and execute Python code in a sandboxed
        environment to solve problems that require computation, aggregation,
        or complex traversal across documents.

        Args:
            question: The question to answer.
            documents: Optional list of document IDs or titles to pre-load.
            filter: SQL WHERE clause to filter documents during searches.

        Returns:
            AnalysisResult with the answer and the final consolidated program.
        """
        from haiku.rag.agents.analysis import (
            AnalysisContext,
            AnalysisDeps,
            Sandbox,
            create_analysis_agent,
        )

        context = AnalysisContext(filter=filter)

        if documents:
            loaded_docs = []
            for doc_ref in documents:
                doc = await self.resolve_document(doc_ref)
                if doc:
                    loaded_docs.append(doc)
            context.documents = loaded_docs if loaded_docs else None

        sandbox = Sandbox(
            db_path=self.store.db_path,
            config=self._config,
            context=context,
        )
        deps = AnalysisDeps(
            sandbox=sandbox,
            context=context,
        )

        from haiku.rag.agents.analysis.models import AnalysisResult
        from haiku.rag.agents.research.models import Citation

        agent = create_analysis_agent(self._config)
        result = await agent.run(question, deps=deps)

        output = result.output
        seen: set[str] = set()
        citations: list[Citation] = []
        for sr in sandbox._search_results:
            if sr.chunk_id and sr.chunk_id not in seen:
                seen.add(sr.chunk_id)
                citations.append(
                    Citation(
                        index=len(seen),
                        document_id=sr.document_id or "",
                        chunk_id=sr.chunk_id,
                        document_uri=sr.document_uri or "",
                        document_title=sr.document_title,
                        page_numbers=sr.page_numbers,
                        headings=sr.headings,
                        content=sr.content,
                    )
                )
        return AnalysisResult(
            answer=output.answer,
            program=output.program,
            citations=citations,
        )

    async def visualize_chunk(self, chunk: Chunk) -> list:
        """Render page images with bounding box highlights for a chunk.

        Expands the chunk's context to find the full section, then resolves
        bounding boxes from all items in the expanded range. This ensures
        visualization covers all pages the expanded content spans.

        Args:
            chunk: The chunk to visualize.

        Returns:
            List of PIL Image objects, one per page with bounding boxes.
            Empty list if no bounding boxes or page images available.
        """
        from copy import deepcopy

        from PIL import ImageDraw

        from haiku.rag.store.models.chunk import ChunkMetadata

        # Get the document structure (from cache if available)
        if not chunk.document_id:
            return []

        doc = await self.document_repository.get_docling_data(chunk.document_id)
        if not doc:
            return []

        docling_doc = doc.get_docling_document()
        if not docling_doc:
            return []

        # Expand context to get all doc_item_refs in the section
        chunk_meta = chunk.get_chunk_metadata()
        if chunk_meta.doc_item_refs:
            search_result = SearchResult(
                content=chunk.content,
                score=1.0,
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                doc_item_refs=chunk_meta.doc_item_refs,
                page_numbers=chunk_meta.page_numbers,
            )
            expanded = await self.expand_context([search_result])
            refs = expanded[0].doc_item_refs if expanded else chunk_meta.doc_item_refs
            meta = ChunkMetadata(doc_item_refs=refs)
        else:
            meta = chunk_meta
        bounding_boxes = meta.resolve_bounding_boxes(docling_doc)
        if not bounding_boxes:
            return []

        # Group bounding boxes by page
        boxes_by_page: dict[int, list] = {}
        for bbox in bounding_boxes:
            if bbox.page_no not in boxes_by_page:
                boxes_by_page[bbox.page_no] = []
            boxes_by_page[bbox.page_no].append(bbox)

        # Load only the needed page images
        pages_doc = await self.document_repository.get_pages_data(chunk.document_id)
        if not pages_doc:
            return []
        page_images = pages_doc.get_page_images(list(boxes_by_page.keys()))

        # Render each page with its bounding boxes
        images = []
        for page_no in sorted(boxes_by_page.keys()):
            if page_no not in page_images:
                continue

            page = page_images[page_no]
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
                fill_color = (255, 255, 0, 40)  # Yellow with transparency
                outline_color = (255, 165, 0, 100)  # Orange outline

                draw.rectangle([(x0, y0), (x1, y1)], fill=fill_color, outline=None)
                draw.rectangle([(x0, y0), (x1, y1)], outline=outline_color, width=1)

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
                - TITLE_ONLY: Only generate titles for untitled documents

        Yields:
            The ID of the document currently being processed.
        """
        # Wait for any background vacuum before destructive table operations
        await self._await_vacuum_tasks()

        # Update settings to current config
        settings_repo = SettingsRepository(self.store)
        await settings_repo.save_current_settings()

        documents = await self.list_documents(include_content=True)

        if mode == RebuildMode.TITLE_ONLY:
            async for doc_id in self._rebuild_title_only(documents):
                yield doc_id
        elif mode == RebuildMode.EMBED_ONLY:
            async for doc_id in self._rebuild_embed_only(documents):
                yield doc_id
        elif mode == RebuildMode.RECHUNK:
            await self.chunk_repository.delete_all()
            await self.store.recreate_embeddings_table()
            async for doc_id in self._rebuild_rechunk(documents):
                yield doc_id
        else:  # FULL
            await self.chunk_repository.delete_all()
            await self.store.recreate_embeddings_table()
            async for doc_id in self._rebuild_full(documents):
                yield doc_id

        # Final maintenance if auto_vacuum enabled
        if self._config.storage.auto_vacuum:
            try:
                await self.store.vacuum()
            except Exception:
                pass

    async def _rebuild_title_only(
        self, documents: list[Document]
    ) -> AsyncGenerator[str, None]:
        """Generate titles for documents that don't have one."""
        for doc in documents:
            if doc.title is not None:
                continue
            assert doc.id is not None
            try:
                title = await self.generate_title(doc)
            except Exception:
                logger.warning(
                    "Failed to generate title for document %s", doc.id, exc_info=True
                )
                continue
            if title is not None:
                doc.title = title
                await self.document_repository.update(doc)
                yield doc.id

    async def _rebuild_embed_only(
        self, documents: list[Document]
    ) -> AsyncGenerator[str, None]:
        """Re-embed all chunks without changing chunk boundaries."""
        from haiku.rag.embeddings import contextualize

        # Collect all chunks with new embeddings
        all_chunk_data: list[tuple[str, dict]] = []

        for doc in documents:
            assert doc.id is not None
            chunks = await self.chunk_repository.get_by_document_id(doc.id)
            if not chunks:
                continue

            texts = contextualize(chunks)
            embeddings = await self.chunk_repository.embedder.embed_documents(texts)

            for chunk, content_fts, embedding in zip(chunks, texts, embeddings):
                all_chunk_data.append(
                    (
                        doc.id,
                        {
                            "id": chunk.id,
                            "document_id": chunk.document_id,
                            "content": chunk.content,
                            "content_fts": content_fts,
                            "metadata": json.dumps(chunk.metadata),
                            "order": chunk.order,
                            "vector": embedding,
                        },
                    )
                )

        # Recreate chunks table (handles dimension changes)
        await self.store.recreate_embeddings_table()

        # Insert all chunks
        if all_chunk_data:
            records = [self.store.ChunkRecord(**data) for _, data in all_chunk_data]
            await self.store.chunks_table.add(records)

        # Yield all processed doc IDs
        yielded_docs: set[str] = set()
        for doc_id, _ in all_chunk_data:
            if doc_id not in yielded_docs:
                yielded_docs.add(doc_id)
                yield doc_id

        # Yield docs with no chunks
        for doc in documents:
            if doc.id and doc.id not in yielded_docs:
                yield doc.id

    async def _flush_rebuild_batch(
        self, documents: list[Document], chunks: list[Chunk]
    ) -> None:
        """Batch write documents and chunks during rebuild.

        This performs two writes: one for all document updates, one for all chunks.
        Also repopulates document items from the stored docling document.
        Used by RECHUNK and FULL modes after the chunks table has been cleared.
        """
        from haiku.rag.store.engine import DocumentRecord

        if not documents:
            return

        now = datetime.now().isoformat()

        # Batch update documents using merge_insert (single LanceDB version)
        doc_records = []
        for doc in documents:
            assert doc.id is not None
            doc_records.append(
                DocumentRecord(
                    id=doc.id,
                    content=doc.content,
                    uri=doc.uri,
                    title=doc.title,
                    metadata=json.dumps(doc.metadata),
                    docling_document=doc.docling_document,
                    docling_pages=doc.docling_pages,
                    docling_version=doc.docling_version,
                    created_at=doc.created_at.isoformat() if doc.created_at else now,
                    updated_at=now,
                )
            )

        await (
            self.store.documents_table.merge_insert("id")
            .when_matched_update_all()
            .execute(doc_records)
        )

        # Batch create all chunks (single LanceDB version)
        if chunks:
            await self.chunk_repository.create(chunks)

        # Repopulate document items from stored docling data
        for doc in documents:
            assert doc.id is not None
            docling_doc = doc.get_docling_document()
            if docling_doc is not None:
                await self.document_item_repository.delete_by_document_id(doc.id)
                items = extract_items(doc.id, docling_doc)
                await self.document_item_repository.create_items(doc.id, items)

    async def _rebuild_rechunk(
        self, documents: list[Document]
    ) -> AsyncGenerator[str, None]:
        """Re-chunk and re-embed from existing document content."""
        from haiku.rag.embeddings import embed_chunks

        batch_size = 50
        pending_chunks: list[Chunk] = []
        pending_docs: list[Document] = []
        pending_doc_ids: list[str] = []

        converter = get_converter(self._config)

        for doc in documents:
            assert doc.id is not None

            # Convert stored markdown to DoclingDocument
            docling_document = await converter.convert_text(doc.content, format="md")

            # Chunk and embed
            chunks = await self.chunk(docling_document)
            embedded_chunks = await embed_chunks(chunks, self._config)

            # Update document fields
            doc.set_docling(docling_document)

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
        converter = get_converter(self._config)

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

            docling_document = await converter.convert_text(doc.content, format="md")
            chunks = await self.chunk(docling_document)
            embedded_chunks = await embed_chunks(chunks, self._config)

            doc.set_docling(docling_document)

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
        - Docling models
        - HuggingFace tokenizer
        - Sentence-transformers embedder (if configured)
        - HuggingFace reranker models (mxbai, jina-local)
        - Ollama models
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

        # Sentence-transformers embedder
        if (
            self._config.embeddings.model.provider == "sentence-transformers"
        ):  # pragma: no cover
            try:
                from sentence_transformers import (  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
                    SentenceTransformer,
                )

                model_name = self._config.embeddings.model.name
                yield DownloadProgress(model=model_name, status="start")
                await asyncio.to_thread(SentenceTransformer, model_name)
                yield DownloadProgress(model=model_name, status="done")
            except ImportError:
                pass

        # HuggingFace reranker models
        if self._config.reranking.model:  # pragma: no cover
            provider = self._config.reranking.model.provider
            model_name = self._config.reranking.model.name

            if provider == "mxbai":
                try:
                    from mxbai_rerank import MxbaiRerankV2

                    yield DownloadProgress(model=model_name, status="start")
                    await asyncio.to_thread(
                        MxbaiRerankV2, model_name, disable_transformers_warnings=True
                    )
                    yield DownloadProgress(model=model_name, status="done")
                except ImportError:
                    pass

            elif provider == "jina-local":
                try:
                    from transformers import AutoModel

                    yield DownloadProgress(model=model_name, status="start")
                    await asyncio.to_thread(
                        AutoModel.from_pretrained,
                        model_name,
                        trust_remote_code=True,
                    )
                    yield DownloadProgress(model=model_name, status="done")
                except ImportError:
                    pass

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
        pic_desc = self._config.processing.conversion_options.picture_description
        if pic_desc.enabled and pic_desc.model.provider == "ollama":
            required_models.add(pic_desc.model.name)
        if (
            self._config.processing.auto_title
            and self._config.processing.title_model.provider == "ollama"
        ):
            required_models.add(self._config.processing.title_model.name)

        if not required_models:
            return

        base_url = self._config.providers.ollama.base_url

        try:
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
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {base_url}. "
                "Is Ollama running? Start it with 'ollama serve'."
            )

    def close(self):
        """Close the underlying store connection."""
        self.store.close()
