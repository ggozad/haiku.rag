import hashlib
import logging
import mimetypes
import tempfile
from collections.abc import AsyncGenerator
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
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
    from haiku.rag.graph.common.models import Citation

logger = logging.getLogger(__name__)


class RebuildMode(Enum):
    """Mode for rebuilding the database."""

    FULL = "full"  # Re-convert from source, re-chunk, re-embed
    RECHUNK = "rechunk"  # Re-chunk from existing content, re-embed
    EMBED_ONLY = "embed_only"  # Keep chunks, only regenerate embeddings


class HaikuRAG:
    """High-level haiku-rag client."""

    def __init__(
        self,
        db_path: Path | None = None,
        config: AppConfig = Config,
        skip_validation: bool = False,
        create: bool = False,
    ):
        """Initialize the RAG client with a database path.

        Args:
            db_path: Path to the database file. If None, uses config.storage.data_dir.
            config: Configuration to use. Defaults to global Config.
            skip_validation: Whether to skip configuration validation on database load.
            create: Whether to create the database if it doesn't exist.
        """
        self._config = config
        if db_path is None:
            db_path = self._config.storage.data_dir / "haiku.rag.lancedb"
        self.store = Store(
            db_path,
            config=self._config,
            skip_validation=skip_validation,
            create=create,
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
            docling_document_json=docling_document.model_dump_json(),
            docling_version=docling_document.version,
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
    ) -> Document:
        """Create a new document from text content.

        Converts the content, chunks it, and generates embeddings.

        Args:
            content: The text content of the document.
            uri: Optional URI identifier for the document.
            title: Optional title for the document.
            metadata: Optional metadata dictionary.

        Returns:
            The created Document instance.
        """
        converter = get_converter(self._config)
        docling_document = await converter.convert_text(content)

        document = Document(
            content=content,
            uri=uri,
            title=title,
            metadata=metadata or {},
            docling_document_json=docling_document.model_dump_json(),
            docling_version=docling_document.version,
        )

        return await self.document_repository._create_and_chunk(
            document, docling_document, None
        )

    async def import_document(
        self,
        content: str,
        chunks: list[Chunk],
        uri: str | None = None,
        title: str | None = None,
        metadata: dict | None = None,
        docling_document_json: str | None = None,
        docling_version: str | None = None,
    ) -> Document:
        """Import a pre-processed document with chunks.

        Use this when document conversion, chunking, and embedding were done
        externally and you want to store the results in haiku.rag.

        Args:
            content: The document content.
            chunks: Pre-created chunks (must include embeddings).
            uri: Optional URI identifier for the document.
            title: Optional title for the document.
            metadata: Optional metadata dictionary.
            docling_document_json: Optional serialized DoclingDocument JSON.
            docling_version: Optional DoclingDocument schema version.

        Returns:
            The created Document instance.

        Raises:
            ValueError: If docling_document_json is provided without docling_version
                or vice versa, or if the JSON is invalid.
        """
        # Validate docling parameters
        if (docling_document_json is None) != (docling_version is None):
            raise ValueError(
                "docling_document_json and docling_version must both be provided or both be None"
            )

        # Validate docling JSON parses if provided
        if docling_document_json is not None:
            try:
                from docling_core.types.doc.document import DoclingDocument

                DoclingDocument.model_validate_json(docling_document_json)
            except Exception as e:
                raise ValueError(f"Invalid docling_document_json: {e}") from e

        document = Document(
            content=content,
            uri=uri,
            title=title,
            metadata=metadata or {},
            docling_document_json=docling_document_json,
            docling_version=docling_version,
        )

        return await self.document_repository._create_and_chunk(document, None, chunks)

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
        docling_document = await converter.convert_file(source_path)

        if existing_doc:
            # Update existing document
            existing_doc.content = docling_document.export_to_markdown()
            existing_doc.metadata = metadata
            existing_doc.docling_document_json = docling_document.model_dump_json()
            existing_doc.docling_version = docling_document.version
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
                docling_document = await converter.convert_file(temp_path)

            # Merge metadata with contentType and md5
            metadata.update({"contentType": content_type, "md5": md5_hash})

            if existing_doc:
                existing_doc.content = docling_document.export_to_markdown()
                existing_doc.metadata = metadata
                existing_doc.docling_document_json = docling_document.model_dump_json()
                existing_doc.docling_version = docling_document.version
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
        docling_document = await converter.convert_text(document.content)

        # Store DoclingDocument JSON
        document.docling_document_json = docling_document.model_dump_json()
        document.docling_version = docling_document.version

        return await self.document_repository._update_and_rechunk(
            document, docling_document
        )

    async def update_document_fields(
        self,
        document_id: str,
        content: str | None = None,
        metadata: dict | None = None,
        chunks: list[Chunk] | None = None,
        title: str | None = None,
        docling_document_json: str | None = None,
        docling_version: str | None = None,
    ) -> Document:
        """Update specific fields of a document by ID.

        Args:
            document_id: The ID of the document to update
            content: New content for the document (mutually exclusive with docling_document_json)
            metadata: New metadata for the document
            chunks: Custom chunks to use instead of auto-generating
            title: New title for the document
            docling_document_json: Serialized DoclingDocument JSON (mutually exclusive with content)
            docling_version: DoclingDocument schema version (required with docling_document_json)

        Returns:
            The updated Document instance.

        Raises:
            ValueError: If both content and docling_document_json are provided,
                if docling_document_json is provided without docling_version,
                or if the JSON is invalid.
        """
        from docling_core.types.doc.document import DoclingDocument

        # Validate: content and docling_document_json are mutually exclusive
        if content is not None and docling_document_json is not None:
            raise ValueError(
                "content and docling_document_json are mutually exclusive. "
                "Provide one or the other, not both."
            )

        # Validate docling parameters must be provided together
        if (docling_document_json is None) != (docling_version is None):
            raise ValueError(
                "docling_document_json and docling_version must both be provided or both be None"
            )

        # Parse and validate docling JSON if provided
        docling_document: DoclingDocument | None = None
        if docling_document_json is not None:
            try:
                docling_document = DoclingDocument.model_validate_json(
                    docling_document_json
                )
            except Exception as e:
                raise ValueError(f"Invalid docling_document_json: {e}") from e

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
            # Update content field if provided
            if content is not None:
                existing_doc.content = content

            # Store docling data if provided
            if docling_document is not None:
                existing_doc.docling_document_json = docling_document_json
                existing_doc.docling_version = docling_version
                # Extract content from docling if not explicitly provided
                if content is None:
                    existing_doc.content = docling_document.export_to_markdown()

            # Delete existing chunks and use custom ones
            await self.chunk_repository.delete_by_document_id(document_id)
            await self.document_repository.update(existing_doc)

            for order, chunk in enumerate(chunks):
                chunk.document_id = document_id
                chunk.order = order
            await self.chunk_repository.create(chunks)

            return existing_doc

        # DoclingDocument provided without chunks - extract content and rechunk
        if docling_document is not None:
            existing_doc.content = docling_document.export_to_markdown()
            existing_doc.docling_document_json = docling_document_json
            existing_doc.docling_version = docling_version

            return await self.document_repository._update_and_rechunk(
                existing_doc, docling_document
            )

        # Content provided without chunks - convert and rechunk
        existing_doc.content = content  # type: ignore[assignment]
        converter = get_converter(self._config)
        converted_docling = await converter.convert_text(existing_doc.content)
        existing_doc.docling_document_json = converted_docling.model_dump_json()
        existing_doc.docling_version = converted_docling.version

        return await self.document_repository._update_and_rechunk(
            existing_doc, converted_docling
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
        resolve_bounding_boxes: bool = False,
    ) -> list[SearchResult]:
        """Search for relevant chunks using the specified search method with optional reranking.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.
            search_type: Type of search - "vector", "fts", or "hybrid" (default).
            filter: Optional SQL WHERE clause to filter documents before searching chunks.
            resolve_bounding_boxes: Whether to resolve bounding boxes from DoclingDocument.

        Returns:
            List of SearchResult objects ordered by relevance.
        """
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

        bounding_boxes_map: dict[str, list] | None = None
        if resolve_bounding_boxes:
            bounding_boxes_map = {}
            doc_cache: dict[str, Document | None] = {}

            for chunk, _ in chunk_results:
                if chunk.document_id and chunk.id:
                    if chunk.document_id not in doc_cache:
                        doc_cache[chunk.document_id] = await self.get_document_by_id(
                            chunk.document_id
                        )

                    doc = doc_cache[chunk.document_id]
                    if doc:
                        docling_doc = doc.get_docling_document()
                        if docling_doc:
                            meta = chunk.get_chunk_metadata()
                            bounding_boxes_map[chunk.id] = meta.resolve_bounding_boxes(
                                docling_doc
                            )

        results = []
        for chunk, score in chunk_results:
            bboxes = None
            if bounding_boxes_map and chunk.id:
                bboxes = bounding_boxes_map.get(chunk.id)
            results.append(SearchResult.from_chunk(chunk, score, bboxes))
        return results

    async def expand_context(
        self,
        search_results: list[SearchResult],
        radius: int | None = None,
    ) -> list[SearchResult]:
        """Expand search results with adjacent content from the source document.

        When DoclingDocument is available and results have doc_item_refs, expands
        by finding adjacent DocItems with accurate bounding boxes and metadata.
        Otherwise, falls back to chunk-based expansion using adjacent chunks.

        Args:
            search_results: List of SearchResult objects from search.
            radius: Number of adjacent items to include before/after.
                   If None, uses config.processing.context_chunk_radius.

        Returns:
            List of SearchResult objects with expanded content and resolved provenance.
        """
        if radius is None:
            radius = self._config.processing.context_chunk_radius
        if radius == 0:
            return search_results

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
                    doc_results, docling_doc, radius
                )
                expanded_results.extend(expanded)
            else:
                # Fall back to chunk-based expansion
                expanded = await self._expand_with_chunks(doc_id, doc_results, radius)
                expanded_results.extend(expanded)

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

    async def _expand_with_docling(
        self,
        results: list[SearchResult],
        docling_doc,
        radius: int,
    ) -> list[SearchResult]:
        """Expand results using DoclingDocument structure."""
        from haiku.rag.store.models.chunk import BoundingBox

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
            min_idx = max(0, min(indices) - radius)
            max_idx = min(len(all_items) - 1, max(indices) + radius)
            ranges.append((min_idx, max_idx, result))

        # Merge overlapping ranges
        merged = self._merge_ranges(ranges)

        final_results: list[SearchResult] = []
        for min_idx, max_idx, original_results in merged:
            content_parts, refs, pages, labels, bboxes = [], [], set(), set(), []

            for i in range(min_idx, max_idx + 1):
                item, _ = all_items[i]
                if text := getattr(item, "text", None):
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
                        if bbox := getattr(p, "bbox", None):
                            bboxes.append(
                                BoundingBox(
                                    page_no=page_no or 0,
                                    left=bbox.l,
                                    top=bbox.t,
                                    right=bbox.r,
                                    bottom=bbox.b,
                                )
                            )

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
                    bounding_boxes=bboxes or None,
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
                    bounding_boxes=first.bounding_boxes,
                )
            )

        return final_results + passthrough

    async def ask(
        self, question: str, system_prompt: str | None = None
    ) -> "tuple[str, list[Citation]]":
        """Ask a question using the configured QA agent.

        Args:
            question: The question to ask.
            system_prompt: Optional custom system prompt for the QA agent.

        Returns:
            Tuple of (answer text, list of resolved citations).
        """
        from haiku.rag.qa import get_qa_agent

        qa_agent = get_qa_agent(self, config=self._config, system_prompt=system_prompt)
        return await qa_agent.answer(question)

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

        # Final maintenance
        try:
            await self.store.vacuum()
        except Exception:
            pass

    async def _rebuild_embed_only(
        self, documents: list[Document]
    ) -> AsyncGenerator[str, None]:
        """Re-embed all chunks without changing chunk boundaries."""
        for doc in documents:
            assert doc.id is not None

            # Get raw chunk records directly from LanceDB
            chunk_records = list(
                self.store.chunks_table.search()
                .where(f"document_id = '{doc.id}'")
                .to_pydantic(self.store.ChunkRecord)
            )
            if not chunk_records:
                continue

            # Batch embed all chunk contents
            contents = [rec.content for rec in chunk_records]
            embeddings = await self.chunk_repository.embedder.embed(contents)

            # Build updated records only for chunks with changed embeddings
            updated_records = [
                self.store.ChunkRecord(
                    id=rec.id,
                    document_id=rec.document_id,
                    content=rec.content,
                    metadata=rec.metadata,
                    order=rec.order,
                    vector=embedding,
                )
                for rec, embedding in zip(chunk_records, embeddings)
                if rec.vector != embedding
            ]

            # Batch update chunks with changed embeddings
            if updated_records:
                self.store.chunks_table.merge_insert(
                    "id"
                ).when_matched_update_all().execute(updated_records)

            yield doc.id

    async def _rebuild_rechunk(
        self, documents: list[Document]
    ) -> AsyncGenerator[str, None]:
        """Re-chunk and re-embed from existing document content."""
        converter = get_converter(self._config)

        for doc in documents:
            assert doc.id is not None
            docling_document = await converter.convert_text(doc.content)

            # Update document with docling JSON
            doc.docling_document_json = docling_document.model_dump_json()
            doc.docling_version = docling_document.version
            await self.document_repository.update(doc)

            await self.chunk_repository.create_chunks_for_document(
                doc.id, docling_document
            )
            yield doc.id

    async def _rebuild_full(
        self, documents: list[Document]
    ) -> AsyncGenerator[str, None]:
        """Full rebuild: re-convert from source, re-chunk, re-embed."""
        converter = get_converter(self._config)

        for doc in documents:
            assert doc.id is not None
            if doc.uri:
                source_accessible = self._check_source_accessible(doc.uri)

                if source_accessible:
                    try:
                        await self.delete_document(doc.id)
                        new_doc = await self.create_document_from_source(
                            source=doc.uri, metadata=doc.metadata or {}
                        )
                        assert isinstance(new_doc, Document)
                        assert new_doc.id is not None
                        yield new_doc.id
                    except Exception as e:
                        logger.error(
                            "Error recreating document from source %s: %s",
                            doc.uri,
                            e,
                        )
                        continue
                else:
                    logger.warning(
                        "Source missing for %s, re-embedding from content", doc.uri
                    )
                    docling_document = await converter.convert_text(doc.content)

                    # Update document with docling JSON
                    doc.docling_document_json = docling_document.model_dump_json()
                    doc.docling_version = docling_document.version
                    await self.document_repository.update(doc)

                    await self.chunk_repository.create_chunks_for_document(
                        doc.id, docling_document
                    )
                    yield doc.id
            else:
                docling_document = await converter.convert_text(doc.content)

                # Update document with docling JSON
                doc.docling_document_json = docling_document.model_dump_json()
                doc.docling_version = docling_document.version
                await self.document_repository.update(doc)

                await self.chunk_repository.create_chunks_for_document(
                    doc.id, docling_document
                )
                yield doc.id

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
