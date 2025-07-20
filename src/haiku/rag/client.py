import hashlib
import mimetypes
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx

from haiku.rag.config import Config
from haiku.rag.reader import FileReader
from haiku.rag.reranking import get_reranker
from haiku.rag.store.engine import Store
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository


class HaikuRAG:
    """High-level haiku-rag client."""

    def __init__(
        self,
        db_path: Path | Literal[":memory:"] = Config.DEFAULT_DATA_DIR
        / "haiku.rag.sqlite",
        skip_validation: bool = False,
    ):
        """Initialize the RAG client with a database path.

        Args:
            db_path: Path to the SQLite database file or ":memory:" for in-memory database.
            skip_validation: Whether to skip configuration validation on database load.
        """
        if isinstance(db_path, Path):
            if not db_path.parent.exists():
                Path.mkdir(db_path.parent, parents=True)
        self.store = Store(db_path, skip_validation=skip_validation)
        self.document_repository = DocumentRepository(self.store)
        self.chunk_repository = ChunkRepository(self.store)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ARG002
        """Async context manager exit."""
        self.close()
        return False

    async def create_document(
        self, content: str, uri: str | None = None, metadata: dict | None = None
    ) -> Document:
        """Create a new document with optional URI and metadata.

        Args:
            content: The text content of the document.
            uri: Optional URI identifier for the document.
            metadata: Optional metadata dictionary.

        Returns:
            The created Document instance.
        """
        document = Document(
            content=content,
            uri=uri,
            metadata=metadata or {},
        )
        return await self.document_repository.create(document)

    async def create_document_from_source(
        self, source: str | Path, metadata: dict = {}
    ) -> Document:
        """Create or update a document from a file path or URL.

        Checks if a document with the same URI already exists:
        - If MD5 is unchanged, returns existing document
        - If MD5 changed, updates the document
        - If no document exists, creates a new one

        Args:
            source: File path (as string or Path) or URL to parse
            metadata: Optional metadata dictionary

        Returns:
            Document instance (created, updated, or existing)

        Raises:
            ValueError: If the file/URL cannot be parsed or doesn't exist
            httpx.RequestError: If URL request fails
        """

        # Check if it's a URL
        source_str = str(source)
        parsed_url = urlparse(source_str)
        if parsed_url.scheme in ("http", "https"):
            return await self._create_or_update_document_from_url(source_str, metadata)

        # Handle as file path
        source_path = Path(source) if isinstance(source, str) else source
        if source_path.suffix.lower() not in FileReader.extensions:
            raise ValueError(f"Unsupported file extension: {source_path.suffix}")

        if not source_path.exists():
            raise ValueError(f"File does not exist: {source_path}")

        uri = source_path.as_uri()
        md5_hash = hashlib.md5(source_path.read_bytes()).hexdigest()

        # Check if document already exists
        existing_doc = await self.get_document_by_uri(uri)
        if existing_doc and existing_doc.metadata.get("md5") == md5_hash:
            # MD5 unchanged, return existing document
            return existing_doc

        content = FileReader.parse_file(source_path)

        # Get content type from file extension
        content_type, _ = mimetypes.guess_type(str(source_path))
        if not content_type:
            content_type = "application/octet-stream"

        # Merge metadata with contentType and md5
        metadata.update({"contentType": content_type, "md5": md5_hash})

        if existing_doc:
            # Update existing document
            existing_doc.content = content
            existing_doc.metadata = metadata
            return await self.update_document(existing_doc)
        else:
            # Create new document
            return await self.create_document(
                content=content, uri=uri, metadata=metadata
            )

    async def _create_or_update_document_from_url(
        self, url: str, metadata: dict = {}
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
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()

            md5_hash = hashlib.md5(response.content).hexdigest()

            # Check if document already exists
            existing_doc = await self.get_document_by_uri(url)
            if existing_doc and existing_doc.metadata.get("md5") == md5_hash:
                # MD5 unchanged, return existing document
                return existing_doc

            # Get content type to determine file extension
            content_type = response.headers.get("content-type", "").lower()
            file_extension = self._get_extension_from_content_type_or_url(
                url, content_type
            )

            if file_extension not in FileReader.extensions:
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

                # Parse the content using FileReader
                content = FileReader.parse_file(temp_path)

            # Merge metadata with contentType and md5
            metadata.update({"contentType": content_type, "md5": md5_hash})

            if existing_doc:
                existing_doc.content = content
                existing_doc.metadata = metadata
                return await self.update_document(existing_doc)
            else:
                return await self.create_document(
                    content=content, uri=url, metadata=metadata
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

    async def get_document_by_id(self, document_id: int) -> Document | None:
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
        return await self.document_repository.update(document)

    async def delete_document(self, document_id: int) -> bool:
        """Delete a document by its ID."""
        return await self.document_repository.delete(document_id)

    async def list_documents(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[Document]:
        """List all documents with optional pagination.

        Args:
            limit: Maximum number of documents to return.
            offset: Number of documents to skip.

        Returns:
            List of Document instances.
        """
        return await self.document_repository.list_all(limit=limit, offset=offset)

    async def search(
        self, query: str, limit: int = 5, k: int = 60, rerank=Config.RERANK
    ) -> list[tuple[Chunk, float]]:
        """Search for relevant chunks using hybrid search (vector similarity + full-text search) with reranking.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.
            k: Parameter for Reciprocal Rank Fusion (default: 60).

        Returns:
            List of (chunk, score) tuples ordered by relevance.
        """

        if not rerank:
            return await self.chunk_repository.search_chunks_hybrid(query, limit, k)

        # Get more initial results (3X) for reranking
        search_results = await self.chunk_repository.search_chunks_hybrid(
            query, limit * 3, k
        )
        # Apply reranking
        reranker = get_reranker()
        chunks = [chunk for chunk, _ in search_results]
        reranked_results = await reranker.rerank(query, chunks, top_n=limit)

        # Return reranked results with scores from reranker
        return reranked_results

    async def ask(self, question: str) -> str:
        """Ask a question using the configured QA agent.

        Args:
            question: The question to ask.

        Returns:
            The generated answer as a string.
        """
        from haiku.rag.qa import get_qa_agent

        qa_agent = get_qa_agent(self)
        return await qa_agent.answer(question)

    async def rebuild_database(self) -> AsyncGenerator[int, None]:
        """Rebuild the database by deleting all chunks and re-indexing all documents.

        Yields:
            int: The ID of the document currently being processed
        """
        await self.chunk_repository.delete_all()
        self.store.recreate_embeddings_table()

        # Update settings to current config
        from haiku.rag.store.repositories.settings import SettingsRepository

        settings_repo = SettingsRepository(self.store)
        settings_repo.save()

        documents = await self.list_documents()

        for doc in documents:
            if doc.id is not None:
                await self.chunk_repository.create_chunks_for_document(
                    doc.id, doc.content, commit=False
                )
                yield doc.id

        if self.store._connection:
            self.store._connection.commit()

    def close(self):
        """Close the underlying store connection."""
        self.store.close()
