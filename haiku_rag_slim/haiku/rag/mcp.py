from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, Config
from haiku.rag.store.models import Document, SearchResult
from haiku.rag.tools.document import DocumentInfo
from haiku.rag.utils import format_citations


def create_mcp_server(
    db_path: Path, config: AppConfig = Config, read_only: bool = False
) -> FastMCP:
    """Create an MCP server with the specified database path.

    Args:
        db_path: Path to the database file.
        config: Configuration to use.
        read_only: If True, write tools (add_document_*, delete_document) are not registered.
    """
    mcp = FastMCP("haiku-rag")

    # Write tools - only registered when not in read-only mode
    if not read_only:

        @mcp.tool()
        async def add_document_from_file(
            file_path: str,
            metadata: dict[str, Any] | None = None,
            title: str | None = None,
        ) -> str | None:
            """Add a document to the RAG system from a file path."""
            try:
                async with HaikuRAG(db_path, config=config) as rag:
                    result = await rag.create_document_from_source(
                        Path(file_path), title=title, metadata=metadata or {}
                    )
                    # Handle both single document and list of documents (directories)
                    if isinstance(result, list):
                        return result[0].id if result else None
                    return result.id
            except Exception:
                return None

        @mcp.tool()
        async def add_document_from_url(
            url: str, metadata: dict[str, Any] | None = None, title: str | None = None
        ) -> str | None:
            """Add a document to the RAG system from a URL."""
            try:
                async with HaikuRAG(db_path, config=config) as rag:
                    result = await rag.create_document_from_source(
                        url, title=title, metadata=metadata or {}
                    )
                    # Handle both single document and list of documents
                    if isinstance(result, list):
                        return result[0].id if result else None
                    return result.id
            except Exception:
                return None

        @mcp.tool()
        async def add_document_from_text(
            content: str,
            uri: str | None = None,
            metadata: dict[str, Any] | None = None,
            title: str | None = None,
        ) -> str | None:
            """Add a document to the RAG system from text content."""
            try:
                async with HaikuRAG(db_path, config=config) as rag:
                    document = await rag.create_document(
                        content, uri, title=title, metadata=metadata or {}
                    )
                    return document.id
            except Exception:
                return None

        @mcp.tool()
        async def delete_document(document_id: str) -> bool:
            """Delete a document by its ID."""
            try:
                async with HaikuRAG(db_path, config=config) as rag:
                    return await rag.delete_document(document_id)
            except Exception:
                return False

    # Read tools - always registered
    @mcp.tool()
    async def search_documents(
        query: str, limit: int | None = None, include_images: bool = True
    ) -> list[SearchResult]:
        """Search the RAG system for documents using hybrid search (vector similarity + full-text search).

        When include_images is True (default) and a picture-labeled chunk is
        in the result set, ``SearchResult.image_data`` carries base64-encoded
        PNG bytes keyed by self_ref. Set to False to omit the bytes from the
        response (smaller JSON payload for plain-text consumers).
        """
        try:
            async with HaikuRAG(db_path, config=config, read_only=read_only) as rag:
                return await rag.search(
                    query, limit=limit, include_images=include_images
                )
        except Exception:
            return []

    # Image-as-query tool, only registered when the configured embedder
    # supports image embeddings. Probed at server-build time when no Store is
    # open, so there is no cached embedder to read; this is the one place
    # outside Store that builds one.
    from haiku.rag.embeddings import get_embedder

    if get_embedder(config).supports_images:

        @mcp.tool()
        async def search_documents_by_image(
            image_base64: str,
            limit: int | None = None,
            include_images: bool = True,
        ) -> list[SearchResult]:
            """Search the RAG system using an image as the query.

            ``image_base64`` is a base64-encoded image (PNG/JPEG bytes). The
            image is embedded via the configured multimodal embedder and the
            chunks table is searched vector-only. ``include_images`` controls
            whether picture bytes are attached to picture-labeled results.
            """
            import base64

            try:
                raw = base64.b64decode(image_base64)
            except Exception:
                return []
            try:
                async with HaikuRAG(db_path, config=config, read_only=read_only) as rag:
                    return await rag.search(
                        raw, limit=limit, include_images=include_images
                    )
            except Exception:
                return []

    @mcp.tool()
    async def get_document(document_id: str) -> Document | None:
        """Get a document by its ID."""
        try:
            async with HaikuRAG(db_path, config=config, read_only=read_only) as rag:
                return await rag.get_document_by_id(document_id)
        except Exception:
            return None

    @mcp.tool()
    async def list_documents(
        limit: int | None = None,
        offset: int | None = None,
        filter: str | None = None,
    ) -> list[DocumentInfo]:
        """List all documents with optional pagination and filtering.

        Args:
            limit: Maximum number of documents to return.
            offset: Number of documents to skip.
            filter: Optional SQL WHERE clause to filter documents.
        """
        try:
            async with HaikuRAG(db_path, config=config, read_only=read_only) as rag:
                documents = await rag.list_documents(limit, offset, filter)

                return [
                    DocumentInfo(
                        id=doc.id,
                        title=doc.title or "Untitled",
                        uri=doc.uri or "",
                        created=doc.created_at.strftime("%Y-%m-%d"),
                    )
                    for doc in documents
                ]
        except Exception:
            return []

    @mcp.tool()
    async def ask_question(
        question: str,
        cite: bool = False,
    ) -> str:
        """Ask a question using the QA agent.

        Args:
            question: The question to ask.
            cite: Whether to include citations in the response.

        Returns:
            The answer as a string.
        """
        try:
            async with HaikuRAG(db_path, config=config, read_only=read_only) as rag:
                answer, citations = await rag.ask(question)
                if cite and citations:
                    answer += "\n\n" + format_citations(citations)
                return answer
        except Exception as e:
            return f"Error answering question: {e!s}"

    @mcp.tool()
    async def analyze(
        question: str,
        filter: str | None = None,
    ) -> str:
        """Answer complex questions using the rag-analysis skill.

        Use this for questions requiring computation, aggregation, or
        structural traversal across documents. The skill can write and
        execute Python code in a sandboxed interpreter.

        Args:
            question: The question to answer.
            filter: Optional SQL WHERE clause to filter documents.

        Returns:
            The answer as a string.
        """
        try:
            async with HaikuRAG(db_path, config=config, read_only=read_only) as rag:
                result = await rag.analyze(question, filter=filter)
                return result.answer
        except Exception as e:
            return f"Error running analysis skill: {e!s}"

    return mcp
