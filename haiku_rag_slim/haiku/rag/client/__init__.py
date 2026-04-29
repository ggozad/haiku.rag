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
        # If _initialize fails mid-way (e.g. migration check raises after
        # connect), close the store so we don't leak the LanceDB connection —
        # __aexit__ won't run because the `async with` never entered.
        try:
            await self.store._initialize()
        except BaseException:
            self.store.close()
            raise
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

    # =========================================================================
    # Title Generation
    # =========================================================================

    async def generate_title(self, document: Document) -> str | None:
        from haiku.rag.client.titles import generate_title

        return await generate_title(self._config, document)

    async def create_document(
        self,
        content: str,
        uri: str | None = None,
        title: str | None = None,
        metadata: dict | None = None,
        format: str = "md",
    ) -> Document:
        from haiku.rag.client.documents import create_document

        return await create_document(self, content, uri, title, metadata, format)

    async def import_document(
        self,
        docling_document: "DoclingDocument",
        chunks: list[Chunk],
        uri: str | None = None,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> Document:
        from haiku.rag.client.documents import import_document

        return await import_document(
            self, docling_document, chunks, uri, title, metadata
        )

    async def create_document_from_source(
        self,
        source: str | Path,
        title: str | None = None,
        metadata: dict | None = None,
        storage_options: dict[str, str] | None = None,
    ) -> Document | list[Document]:
        from haiku.rag.client.documents import create_document_from_source

        return await create_document_from_source(
            self, source, title, metadata, storage_options=storage_options
        )

    async def update_document(
        self,
        document_id: str,
        content: str | None = None,
        metadata: dict | None = None,
        chunks: list[Chunk] | None = None,
        title: str | None = None,
        docling_document: "DoclingDocument | None" = None,
    ) -> Document:
        from haiku.rag.client.documents import update_document

        return await update_document(
            self,
            document_id,
            content,
            metadata,
            chunks,
            title,
            docling_document,
        )

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

    async def vacuum(self) -> None:
        """Optimize and clean up old versions across all tables."""
        await self.store.vacuum()

    def close(self):
        """Close the underlying store connection."""
        self.store.close()
