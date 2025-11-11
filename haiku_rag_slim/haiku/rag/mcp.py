from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, Config
from haiku.rag.graph.research.models import ResearchReport


class SearchResult(BaseModel):
    document_id: str
    content: str
    score: float


class DocumentResult(BaseModel):
    id: str | None
    content: str
    uri: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = {}
    created_at: str
    updated_at: str


def create_mcp_server(db_path: Path, config: AppConfig = Config) -> FastMCP:
    """Create an MCP server with the specified database path."""
    mcp = FastMCP("haiku-rag")

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
    async def search_documents(query: str, limit: int = 5) -> list[SearchResult]:
        """Search the RAG system for documents using hybrid search (vector similarity + full-text search)."""
        try:
            async with HaikuRAG(db_path, config=config) as rag:
                results = await rag.search(query, limit)

                search_results = []
                for chunk, score in results:
                    assert chunk.document_id is not None, (
                        "Chunk document_id should not be None in search results"
                    )
                    search_results.append(
                        SearchResult(
                            document_id=chunk.document_id,
                            content=chunk.content,
                            score=score,
                        )
                    )

                return search_results
        except Exception:
            return []

    @mcp.tool()
    async def get_document(document_id: str) -> DocumentResult | None:
        """Get a document by its ID."""
        try:
            async with HaikuRAG(db_path, config=config) as rag:
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
            async with HaikuRAG(db_path, config=config) as rag:
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
    async def delete_document(document_id: str) -> bool:
        """Delete a document by its ID."""
        try:
            async with HaikuRAG(db_path, config=config) as rag:
                return await rag.delete_document(document_id)
        except Exception:
            return False

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
            async with HaikuRAG(db_path, config=config) as rag:
                if deep:
                    from haiku.rag.graph.deep_qa.dependencies import DeepQAContext
                    from haiku.rag.graph.deep_qa.graph import build_deep_qa_graph
                    from haiku.rag.graph.deep_qa.state import DeepQADeps, DeepQAState

                    graph = build_deep_qa_graph(config=config)
                    context = DeepQAContext(
                        original_question=question, use_citations=cite
                    )
                    state = DeepQAState.from_config(context=context, config=config)
                    deps = DeepQADeps(client=rag)

                    result = await graph.run(state=state, deps=deps)
                    answer = result.answer
                else:
                    answer = await rag.ask(question, cite=cite)
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

            async with HaikuRAG(db_path, config=config) as rag:
                graph = build_research_graph(config=config)
                context = ResearchContext(original_question=question)
                state = ResearchState.from_config(context=context, config=config)
                deps = ResearchDeps(client=rag)

                result = await graph.run(state=state, deps=deps)

                return result
        except Exception:
            return None

    return mcp
