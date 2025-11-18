import hashlib
import logging
import mimetypes
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from urllib.parse import urlparse

import httpx

from haiku.rag.config import AppConfig, Config
from haiku.rag.converters import get_converter
from haiku.rag.reranking import get_reranker
from haiku.rag.store.engine import Store
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.repositories.settings import SettingsRepository

logger = logging.getLogger(__name__)


class HaikuRAG:
    """High-level haiku-rag client."""

    def __init__(
        self,
        db_path: Path | None = None,
        config: AppConfig = Config,
        skip_validation: bool = False,
        allow_create: bool = True,
    ):
        """Initialize the RAG client with a database path.

        Args:
            db_path: Path to the database file. If None, uses config.storage.data_dir.
            config: Configuration to use. Defaults to global Config.
            skip_validation: Whether to skip configuration validation on database load.
            allow_create: Whether to allow database creation. If False, will raise error
                         if database doesn't exist (for read operations).
        """
        self._config = config
        if db_path is None:
            db_path = self._config.storage.data_dir / "haiku.rag.lancedb"
        self.store = Store(
            db_path,
            config=self._config,
            skip_validation=skip_validation,
            allow_create=allow_create,
        )
        self.document_repository = DocumentRepository(self.store)
        self.chunk_repository = ChunkRepository(self.store)

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

    async def _create_document_with_docling(
        self,
        docling_document,
        uri: str | None = None,
        title: str | None = None,
        metadata: dict | None = None,
        chunks: list[Chunk] | None = None,
    ) -> Document:
        """Create a new document from DoclingDocument."""
        content = docling_document.export_to_markdown()
        document = Document(
            content=content,
            uri=uri,
            title=title,
            metadata=metadata or {},
        )
        return await self.document_repository._create_and_chunk(
            document, docling_document, chunks
        )

    async def create_document(
        self,
        content: str,
        uri: str | None = None,
        title: str | None = None,
        metadata: dict | None = None,
        chunks: list[Chunk] | None = None,
    ) -> Document:
        """Create a new document with optional URI and metadata.

        Args:
            content: The text content of the document.
            uri: Optional URI identifier for the document.
            metadata: Optional metadata dictionary.
            chunks: Optional list of pre-created chunks to use instead of generating new ones.

        Returns:
            The created Document instance.
        """
        document = Document(
            content=content,
            uri=uri,
            title=title,
            metadata=metadata or {},
        )

        # Only create docling_document if we need to generate chunks
        if chunks is None:
            # Use converter to convert text
            converter = get_converter(self._config)
            docling_document = converter.convert_text(content)
        else:
            # Chunks already provided, no conversion needed
            docling_document = None

        return await self.document_repository._create_and_chunk(
            document, docling_document, chunks
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

        # Parse file only when content changed or new document
        converter = get_converter(self._config)
        docling_document = converter.convert_file(source_path)

        if existing_doc:
            # Update existing document
            existing_doc.content = docling_document.export_to_markdown()
            existing_doc.metadata = metadata
            if title is not None:
                existing_doc.title = title
            return await self.document_repository._update_and_rechunk(
                existing_doc, docling_document
            )
        else:
            # Create new document using DoclingDocument
            return await self._create_document_with_docling(
                docling_document=docling_document,
                uri=uri,
                title=title,
                metadata=metadata,
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
                mode="wb", suffix=file_extension
            ) as temp_file:
                temp_file.write(response.content)
                temp_file.flush()  # Ensure content is written to disk
                temp_path = Path(temp_file.name)

                # Parse the content using converter
                docling_document = converter.convert_file(temp_path)

            # Merge metadata with contentType and md5
            metadata.update({"contentType": content_type, "md5": md5_hash})

            if existing_doc:
                existing_doc.content = docling_document.export_to_markdown()
                existing_doc.metadata = metadata
                if title is not None:
                    existing_doc.title = title
                return await self.document_repository._update_and_rechunk(
                    existing_doc, docling_document
                )
            else:
                return await self._create_document_with_docling(
                    docling_document=docling_document,
                    uri=url,
                    title=title,
                    metadata=metadata,
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

    async def get_document_by_uri(self, uri: str) -> Document | None:
        """Get a document by its URI.

        Args:
            uri: The URI identifier of the document.

        Returns:
            The Document instance if found, None otherwise.
        """
        return await self.document_repository.get_by_uri(uri)

    async def update_document(self, document: Document) -> Document:
        """Update an existing document."""
        # Convert content to DoclingDocument
        converter = get_converter(self._config)
        docling_document = converter.convert_text(document.content)

        return await self.document_repository._update_and_rechunk(
            document, docling_document
        )

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
        limit: int = 5,
        search_type: str = "hybrid",
        filter: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search for relevant chunks using the specified search method with optional reranking.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.
            search_type: Type of search - "vector", "fts", or "hybrid" (default).
            filter: Optional SQL WHERE clause to filter documents before searching chunks.

        Returns:
            List of (chunk, score) tuples ordered by relevance.
        """
        # Get reranker if available
        reranker = get_reranker(config=self._config)

        if reranker is None:
            # No reranking - return direct search results
            return await self.chunk_repository.search(query, limit, search_type, filter)

        # Get more initial results (10X) for reranking
        search_limit = limit * 10
        search_results = await self.chunk_repository.search(
            query, search_limit, search_type, filter
        )

        # Apply reranking
        chunks = [chunk for chunk, _ in search_results]
        reranked_results = await reranker.rerank(query, chunks, top_n=limit)

        # Return reranked results with scores from reranker
        return reranked_results

    async def expand_context(
        self,
        search_results: list[tuple[Chunk, float]],
        radius: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Expand search results with adjacent chunks, merging overlapping chunks.

        Args:
            search_results: List of (chunk, score) tuples from search.
            radius: Number of adjacent chunks to include before/after each chunk.
                   If None, uses config.processing.context_chunk_radius.

        Returns:
            List of (chunk, score) tuples with expanded and merged context chunks.
        """
        if radius is None:
            radius = self._config.processing.context_chunk_radius
        if radius == 0:
            return search_results

        # Group chunks by document_id to handle merging within documents
        document_groups = {}
        for chunk, score in search_results:
            doc_id = chunk.document_id
            if doc_id not in document_groups:
                document_groups[doc_id] = []
            document_groups[doc_id].append((chunk, score))

        results = []

        for doc_id, doc_chunks in document_groups.items():
            # Get all expanded ranges for this document
            expanded_ranges = []
            for chunk, score in doc_chunks:
                adjacent_chunks = await self.chunk_repository.get_adjacent_chunks(
                    chunk, radius
                )

                all_chunks = adjacent_chunks + [chunk]

                # Get the range of orders for this expanded chunk
                orders = [c.order for c in all_chunks]
                min_order = min(orders)
                max_order = max(orders)

                expanded_ranges.append(
                    {
                        "original_chunk": chunk,
                        "score": score,
                        "min_order": min_order,
                        "max_order": max_order,
                        "all_chunks": sorted(all_chunks, key=lambda c: c.order),
                    }
                )

            # Merge overlapping/adjacent ranges
            merged_ranges = self._merge_overlapping_ranges(expanded_ranges)

            # Create merged chunks
            for merged_range in merged_ranges:
                combined_content_parts = [c.content for c in merged_range["all_chunks"]]

                # Use the first original chunk for metadata
                original_chunk = merged_range["original_chunks"][0]

                merged_chunk = Chunk(
                    id=original_chunk.id,
                    document_id=original_chunk.document_id,
                    content="".join(combined_content_parts),
                    metadata=original_chunk.metadata,
                    document_uri=original_chunk.document_uri,
                    document_title=original_chunk.document_title,
                    document_meta=original_chunk.document_meta,
                )

                # Use the highest score from merged chunks
                best_score = max(merged_range["scores"])
                results.append((merged_chunk, best_score))

        return results

    def _merge_overlapping_ranges(self, expanded_ranges):
        """Merge overlapping or adjacent expanded ranges."""
        if not expanded_ranges:
            return []

        # Sort by min_order
        sorted_ranges = sorted(expanded_ranges, key=lambda x: x["min_order"])
        merged = []

        current = {
            "min_order": sorted_ranges[0]["min_order"],
            "max_order": sorted_ranges[0]["max_order"],
            "original_chunks": [sorted_ranges[0]["original_chunk"]],
            "scores": [sorted_ranges[0]["score"]],
            "all_chunks": sorted_ranges[0]["all_chunks"],
        }

        for range_info in sorted_ranges[1:]:
            # Check if ranges overlap or are adjacent (max_order + 1 >= min_order)
            if current["max_order"] >= range_info["min_order"] - 1:
                # Merge ranges
                current["max_order"] = max(
                    current["max_order"], range_info["max_order"]
                )
                current["original_chunks"].append(range_info["original_chunk"])
                current["scores"].append(range_info["score"])

                # Merge all_chunks and deduplicate by order
                all_chunks_dict = {}
                for chunk in current["all_chunks"] + range_info["all_chunks"]:
                    order = chunk.order
                    all_chunks_dict[order] = chunk
                current["all_chunks"] = [
                    all_chunks_dict[order] for order in sorted(all_chunks_dict.keys())
                ]
            else:
                # No overlap, add current to merged and start new
                merged.append(current)
                current = {
                    "min_order": range_info["min_order"],
                    "max_order": range_info["max_order"],
                    "original_chunks": [range_info["original_chunk"]],
                    "scores": [range_info["score"]],
                    "all_chunks": range_info["all_chunks"],
                }

        # Add the last range
        merged.append(current)
        return merged

    async def ask(
        self, question: str, cite: bool = False, system_prompt: str | None = None
    ) -> str:
        """Ask a question using the configured QA agent.

        Args:
            question: The question to ask.
            cite: Whether to include citations in the response.
            system_prompt: Optional custom system prompt for the QA agent.

        Returns:
            The generated answer as a string.
        """
        from haiku.rag.qa import get_qa_agent

        qa_agent = get_qa_agent(
            self, config=self._config, use_citations=cite, system_prompt=system_prompt
        )
        return await qa_agent.answer(question)

    async def rebuild_database(self) -> AsyncGenerator[str, None]:
        """Rebuild the database by deleting all chunks and re-indexing all documents.

        For documents with URIs:
        - Re-adds from source if source exists
        - Re-embeds from existing content if source is missing

        For documents without URIs:
        - Re-creates chunks from existing content

        Yields:
            int: The ID of the document currently being processed
        """
        await self.chunk_repository.delete_all()
        self.store.recreate_embeddings_table()

        converter = get_converter(self._config)

        # Update settings to current config
        settings_repo = SettingsRepository(self.store)
        settings_repo.save_current_settings()

        documents = await self.list_documents()

        for doc in documents:
            assert doc.id is not None, "Document ID should not be None"
            if doc.uri:
                # Document has a URI - check if source is accessible
                source_accessible = False
                parsed_url = urlparse(doc.uri)

                try:
                    if parsed_url.scheme == "file":
                        # Check if file exists
                        source_path = Path(parsed_url.path)
                        source_accessible = source_path.exists()
                    elif parsed_url.scheme in ("http", "https"):
                        # For URLs, we'll try to create and catch errors
                        source_accessible = True
                    else:
                        source_accessible = False
                except Exception:
                    source_accessible = False

                if source_accessible:
                    # Source exists - delete and recreate from source
                    try:
                        await self.delete_document(doc.id)
                        new_doc = await self.create_document_from_source(
                            source=doc.uri, metadata=doc.metadata or {}
                        )
                        # URIs always point to single files/URLs, never directories
                        assert isinstance(new_doc, Document)
                        assert new_doc.id is not None, (
                            "New document ID should not be None"
                        )
                        yield new_doc.id
                    except Exception as e:
                        logger.error(
                            "Error recreating document from source %s: %s",
                            doc.uri,
                            e,
                        )
                        continue
                else:
                    # Source missing - re-embed from existing content
                    logger.warning(
                        "Source missing for %s, re-embedding from content", doc.uri
                    )
                    docling_document = converter.convert_text(doc.content)
                    await self.chunk_repository.create_chunks_for_document(
                        doc.id, docling_document
                    )
                    yield doc.id
            else:
                # Document without URI - re-create chunks from existing content
                docling_document = converter.convert_text(doc.content)
                await self.chunk_repository.create_chunks_for_document(
                    doc.id, docling_document
                )
                yield doc.id

        # Final maintenance: centralized vacuum to curb disk usage
        try:
            await self.store.vacuum()
        except Exception:
            pass

    async def vacuum(self) -> None:
        """Optimize and clean up old versions across all tables."""
        await self.store.vacuum()

    def close(self):
        """Close the underlying store connection."""
        self.store.close()
