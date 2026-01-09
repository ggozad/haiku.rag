from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, format_as_xml

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


class QAResponse(BaseModel):
    """A Q&A pair from conversation history."""

    question: str
    answer: str
    sources: list[str] = []


class ChatSessionState(BaseModel):
    """State shared between frontend and agent via AG-UI."""

    session_id: str = ""
    citations: list[CitationInfo] = []
    qa_history: list[QAResponse] = []


def format_conversation_context(qa_history: list[QAResponse]) -> str:
    """Format conversation history as XML for inclusion in prompts."""
    if not qa_history:
        return ""

    context_data = {
        "previous_qa": [
            {
                "question": qa.question,
                "answer": qa.answer,
                "sources": qa.sources,
            }
            for qa in qa_history
        ],
    }
    return format_as_xml(context_data, root_tag="conversation_context")


@dataclass
class ChatDeps:
    """Dependencies for chat agent."""

    client: HaikuRAG
    config: AppConfig
    agui_emitter: "AGUIEmitter | None" = None
    search_results: list[SearchResult] | None = None
    session_state: ChatSessionState | None = None


CHAT_SYSTEM_PROMPT = """You are a helpful research assistant powered by haiku.rag, a knowledge base system.

You have access to a knowledge base of documents. Use your tools to search and answer questions.

CRITICAL RULES:
1. For greetings or casual chat: respond directly WITHOUT using any tools
2. For questions: Use the "ask" tool EXACTLY ONCE - it handles query expansion internally
3. For searches: Use the "search" tool EXACTLY ONCE - it handles multi-query expansion internally
4. NEVER call the same tool multiple times for a single user message
5. NEVER make up information - always use tools to get facts from the knowledge base

How to decide which tool to use:
- "get_document" - Use when the user references a SPECIFIC document by name, title, or URI (e.g., "summarize document X", "get the paper about Y", "fetch 2412.00566"). Retrieves the full document content.
- "ask" - Use for general questions about topics in the knowledge base when no specific document is named. It searches across all documents and returns answers with citations.
- "search" - Use when the user explicitly asks to search/find/explore documents. Call it ONCE. After calling search, just output the list of results returned by the tool verbatim. Do NOT summarize or add commentary.

Be friendly and conversational. When you use the "ask" tool, summarize the key findings for the user."""


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
    ) -> str:
        """Search the knowledge base for relevant documents.

        Use this when you need to find documents or explore the knowledge base.
        Results are displayed to the user - just list the titles found.

        Args:
            query: The search query
        """
        from search_agent import SearchAgent

        if ctx.deps.agui_emitter:
            ctx.deps.agui_emitter.log(f"Searching: {query}")

        # Build context from conversation history
        context = None
        if ctx.deps.session_state and ctx.deps.session_state.qa_history:
            context = format_conversation_context(ctx.deps.session_state.qa_history)

        # Use search agent for query expansion and deduplication
        search_agent = SearchAgent(ctx.deps.client, ctx.deps.config)
        results = await search_agent.search(query, context=context)

        # Store for potential citation resolution
        ctx.deps.search_results = results

        if not results:
            return "No results found."

        # Build citation infos for frontend display
        citation_infos = [
            CitationInfo(
                index=i + 1,
                document_id=r.document_id or "",
                chunk_id=r.chunk_id or "",
                document_uri=r.document_uri or "",
                document_title=r.document_title,
                page_numbers=r.page_numbers or [],
                headings=r.headings,
                content=r.content,
            )
            for i, r in enumerate(results)
        ]

        # Emit search results as citations
        if ctx.deps.agui_emitter:
            ctx.deps.agui_emitter.update_state(
                ChatSessionState(
                    session_id=(
                        ctx.deps.session_state.session_id
                        if ctx.deps.session_state
                        else ""
                    ),
                    citations=citation_infos,
                    qa_history=(
                        ctx.deps.session_state.qa_history
                        if ctx.deps.session_state
                        else []
                    ),
                )
            )

        # Return simple list of titles for the agent to present
        titles = []
        for i, r in enumerate(results):
            title = r.document_title or r.document_uri or "Unknown"
            titles.append(f"[{i + 1}] {title}")
        return f"Found {len(results)} results:\n" + "\n".join(titles)

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

        # Build context-aware system prompt if we have history
        system_prompt = None
        if ctx.deps.session_state and ctx.deps.session_state.qa_history:
            from haiku.rag.qa.prompts import QA_SYSTEM_PROMPT

            context_xml = format_conversation_context(ctx.deps.session_state.qa_history)
            system_prompt = (
                f"{QA_SYSTEM_PROMPT}\n\n"
                f"{context_xml}\n\n"
                "Use this conversation context to provide informed answers. "
                "Reference previous answers when relevant."
            )

        answer, citations = await ctx.deps.client.ask(
            question, system_prompt=system_prompt, filter=document_filter
        )

        # Accumulate Q&A in session state
        if ctx.deps.session_state is not None:
            sources = (
                [c.document_title or c.document_uri for c in citations]
                if citations
                else []
            )
            qa_response = QAResponse(
                question=question,
                answer=answer,
                sources=list(dict.fromkeys(sources)),  # dedupe preserving order
            )
            ctx.deps.session_state.qa_history.append(qa_response)

        # Build citation infos for frontend
        citation_infos = []
        if citations:
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

        # Emit updated state with citations AND accumulated qa_history
        if ctx.deps.agui_emitter:
            ctx.deps.agui_emitter.update_state(
                ChatSessionState(
                    session_id=(
                        ctx.deps.session_state.session_id
                        if ctx.deps.session_state
                        else ""
                    ),
                    citations=citation_infos,
                    qa_history=(
                        ctx.deps.session_state.qa_history
                        if ctx.deps.session_state
                        else []
                    ),
                )
            )

        # Format answer with citation references
        if citations:
            citation_refs = " ".join(f"[{i + 1}]" for i in range(len(citations)))
            return f"{answer}\n\nSources: {citation_refs}"

        return answer

    @agent.tool
    async def get_document(
        ctx: RunContext[ChatDeps],
        query: str,
    ) -> str:
        """Retrieve a specific document by title or URI.

        Use this when the user wants to fetch/get/retrieve a specific document.

        Args:
            query: The document title or URI to look up
        """
        if ctx.deps.agui_emitter:
            ctx.deps.agui_emitter.log(f"Fetching document: {query}")

        # Try exact URI match first
        doc = await ctx.deps.client.get_document_by_uri(query)

        escaped_query = query.replace("'", "''")

        # If not found, try partial URI match
        if doc is None:
            docs = await ctx.deps.client.list_documents(
                limit=1, filter=f"LOWER(uri) LIKE LOWER('%{escaped_query}%')"
            )
            if docs:
                doc = docs[0]

        # If still not found, try partial title match
        if doc is None:
            docs = await ctx.deps.client.list_documents(
                limit=1, filter=f"LOWER(title) LIKE LOWER('%{escaped_query}%')"
            )
            if docs:
                doc = docs[0]

        if doc is None:
            return f"Document not found: {query}"

        return (
            f"**{doc.title or 'Untitled'}**\n\n"
            f"- ID: {doc.id}\n"
            f"- URI: {doc.uri or 'N/A'}\n"
            f"- Created: {doc.created_at.strftime('%Y-%m-%d %H:%M')}\n\n"
            f"**Content:**\n{doc.content}"
        )

    return agent
