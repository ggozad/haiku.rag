from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models import SearchResult
from haiku.rag.utils import get_model

if TYPE_CHECKING:
    from haiku.rag.graph.agui.emitter import AGUIEmitter


class CitationInfo(BaseModel):
    """Citation info for frontend display."""

    index: int
    document_id: str
    chunk_id: str
    document_uri: str
    document_title: str | None = None
    page_numbers: list[int] = []
    headings: list[str] | None = None
    content: str


class ChatSessionState(BaseModel):
    """State shared between frontend and agent via AG-UI."""

    session_id: str = ""
    citations: list[CitationInfo] = []


@dataclass
class ChatDeps:
    """Dependencies for chat agent."""

    client: HaikuRAG
    config: AppConfig
    agui_emitter: "AGUIEmitter | None" = None
    search_results: list[SearchResult] | None = None


CHAT_SYSTEM_PROMPT = """You are a helpful research assistant powered by haiku.rag, a knowledge base system.

You have access to a knowledge base of documents. Use your tools to search and answer questions.

CRITICAL RULES:
1. For greetings or casual chat: respond directly WITHOUT using any tools
2. For questions: ALWAYS use the "ask" tool - it provides answers with proper citations
3. NEVER make up information - always use tools to get facts from the knowledge base

How to decide which tool to use:
- "ask" - DEFAULT CHOICE for any question. Use this for questions like "What is X?", "How does Y work?", "Explain Z", etc. Returns answers with citations.
- "search" - ONLY use when explicitly exploring/browsing the knowledge base, or when the user asks to "search for" or "find" something without needing an answer.

Be friendly and conversational. When you use tools, summarize the key findings for the user."""


def create_chat_agent(config: AppConfig) -> Agent[ChatDeps, str]:
    """Create the chat agent with search and ask tools."""
    model = get_model(config.qa.model, config)

    agent: Agent[ChatDeps, str] = Agent(
        model,
        deps_type=ChatDeps,
        output_type=str,
        instructions=CHAT_SYSTEM_PROMPT,
    )

    @agent.tool
    async def search(
        ctx: RunContext[ChatDeps],
        query: str,
        limit: int = 5,
        document_filter: str | None = None,
    ) -> str:
        """Search the knowledge base for relevant documents.

        Use this when you need to find documents or explore the knowledge base.
        Returns relevant chunks with metadata.

        Args:
            query: The search query
            limit: Maximum number of results (default 5)
            document_filter: Optional SQL WHERE clause to filter documents (e.g. "id IN ('doc1', 'doc2')")
        """
        if ctx.deps.agui_emitter:
            ctx.deps.agui_emitter.log(f"Searching: {query}")

        results = await ctx.deps.client.search(
            query, limit=limit, filter=document_filter
        )
        results = await ctx.deps.client.expand_context(results)

        # Store for potential citation resolution
        ctx.deps.search_results = results

        if not results:
            return "No results found for your query."

        # Format results for the agent
        parts = [r.format_for_agent() for r in results]
        return "\n\n".join(parts)

    @agent.tool
    async def ask(
        ctx: RunContext[ChatDeps],
        question: str,
        document_filter: str | None = None,
    ) -> str:
        """Answer a specific question using the knowledge base.

        Use this for direct questions that need a focused answer with citations.

        Args:
            question: The question to answer
            document_filter: Optional SQL WHERE clause to filter documents (e.g. "id IN ('doc1', 'doc2')")
        """
        if ctx.deps.agui_emitter:
            ctx.deps.agui_emitter.log(f"Answering: {question}")

        answer, citations = await ctx.deps.client.ask(question, filter=document_filter)

        # Emit citations via AG-UI state for frontend rendering
        if citations and ctx.deps.agui_emitter:
            citation_infos = [
                CitationInfo(
                    index=i + 1,
                    document_id=c.document_id,
                    chunk_id=c.chunk_id,
                    document_uri=c.document_uri,
                    document_title=c.document_title,
                    page_numbers=c.page_numbers,
                    headings=c.headings,
                    content=c.content,
                )
                for i, c in enumerate(citations)
            ]
            ctx.deps.agui_emitter.update_state(
                ChatSessionState(citations=citation_infos)
            )

        # Format answer with citation references
        if citations:
            citation_refs = " ".join(f"[{i + 1}]" for i in range(len(citations)))
            return f"{answer}\n\nSources: {citation_refs}"

        return answer

    return agent
