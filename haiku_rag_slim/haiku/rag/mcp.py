from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, Config
from haiku.rag.graph.research.models import ResearchReport
from haiku.rag.store.models import SearchResult
from haiku.rag.utils import format_citations


class DocumentResult(BaseModel):
    id: str | None
    content: str
    uri: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = {}
    created_at: str
    updated_at: str


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
        query: str, limit: int | None = None
    ) -> list[SearchResult]:
        """Search the RAG system for documents using hybrid search (vector similarity + full-text search)."""
        try:
            async with HaikuRAG(db_path, config=config, read_only=read_only) as rag:
                return await rag.search(query, limit=limit)
        except Exception:
            return []

    @mcp.tool()
    async def get_document(document_id: str) -> DocumentResult | None:
        """Get a document by its ID."""
        try:
            async with HaikuRAG(db_path, config=config, read_only=read_only) as rag:
                document = await rag.get_document_by_id(document_id)

                if document is None:
                    return None

                return DocumentResult(
                    id=document.id,
                    content=document.content,
                    uri=document.uri,
                    title=document.title,
                    metadata=document.metadata,
                    created_at=str(document.created_at),
                    updated_at=str(document.updated_at),
                )
        except Exception:
            return None

    @mcp.tool()
    async def list_documents(
        limit: int | None = None,
        offset: int | None = None,
        filter: str | None = None,
    ) -> list[DocumentResult]:
        """List all documents with optional pagination and filtering.

        Args:
            limit: Maximum number of documents to return.
            offset: Number of documents to skip.
            filter: Optional SQL WHERE clause to filter documents.

        Returns:
            List of DocumentResult instances matching the criteria.
        """
        try:
            async with HaikuRAG(db_path, config=config, read_only=read_only) as rag:
                documents = await rag.list_documents(limit, offset, filter)

                return [
                    DocumentResult(
                        id=doc.id,
                        content=doc.content,
                        uri=doc.uri,
                        title=doc.title,
                        metadata=doc.metadata,
                        created_at=str(doc.created_at),
                        updated_at=str(doc.updated_at),
                    )
                    for doc in documents
                ]
        except Exception:
            return []

    @mcp.tool()
    async def ask_question(
        question: str,
        cite: bool = False,
        deep: bool = False,
    ) -> str:
        """Ask a question using the QA agent.

        Args:
            question: The question to ask.
            cite: Whether to include citations in the response.
            deep: Use deep multi-agent QA for complex questions that require decomposition.

        Returns:
            The answer as a string.
        """
        try:
            async with HaikuRAG(db_path, config=config, read_only=read_only) as rag:
                if deep:
                    from haiku.rag.graph.research.dependencies import ResearchContext
                    from haiku.rag.graph.research.graph import build_research_graph
                    from haiku.rag.graph.research.state import (
                        ResearchDeps,
                        ResearchState,
                    )

                    graph = build_research_graph(config=config)
                    context = ResearchContext(original_question=question)
                    state = ResearchState.from_config(
                        context=context,
                        config=config,
                        max_iterations=2,
                        confidence_threshold=0.0,
                    )
                    deps = ResearchDeps(client=rag)

                    result = await graph.run(state=state, deps=deps)
                    answer = result.executive_summary
                    citations = []
                else:
                    answer, citations = await rag.ask(question)
                if cite and citations:
                    answer += "\n\n" + format_citations(citations)
                return answer
        except Exception as e:
            return f"Error answering question: {e!s}"

    @mcp.tool()
    async def research_question(
        question: str,
    ) -> ResearchReport | None:
        """Run multi-agent research to investigate a complex question.

        The research process uses multiple agents to plan, search, evaluate, and synthesize
        information iteratively until confidence threshold is met or max iterations reached.

        Args:
            question: The research question to investigate.

        Returns:
            A research report with findings, or None if an error occurred.
        """
        try:
            from haiku.rag.graph.research.dependencies import ResearchContext
            from haiku.rag.graph.research.graph import build_research_graph
            from haiku.rag.graph.research.state import ResearchDeps, ResearchState

            async with HaikuRAG(db_path, config=config, read_only=read_only) as rag:
                graph = build_research_graph(config=config)
                context = ResearchContext(original_question=question)
                state = ResearchState.from_config(context=context, config=config)
                deps = ResearchDeps(client=rag)

                result = await graph.run(state=state, deps=deps)

                return result
        except Exception:
            return None

    return mcp
