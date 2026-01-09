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
    """A Q&A pair from conversation history with citations."""

    question: str
    answer: str
    confidence: float = 0.9
    citations: list[CitationInfo] = []

    @property
    def sources(self) -> list[str]:
        """Source names for display."""
        return list(
            dict.fromkeys(c.document_title or c.document_uri for c in self.citations)
        )


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


def build_document_filter(document_name: str) -> str:
    """Build SQL filter for document name matching."""
    escaped = document_name.replace("'", "''")
    no_spaces = escaped.replace(" ", "")
    return (
        f"LOWER(uri) LIKE LOWER('%{escaped}%') OR LOWER(title) LIKE LOWER('%{escaped}%') "
        f"OR LOWER(uri) LIKE LOWER('%{no_spaces}%') OR LOWER(title) LIKE LOWER('%{no_spaces}%')"
    )


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
- "search" - Use when the user explicitly asks to search/find/explore documents. Call it ONCE. After calling search, copy the ENTIRE tool response to your output INCLUDING the content snippets. Do NOT shorten, summarize, or omit any part of the results.

IMPORTANT - When user mentions a document in search/ask:
- If user says "search in <doc>", "find in <doc>", "answer from <doc>", or "<topic> in <doc>":
  - Extract the TOPIC as `query`/`question`
  - Extract the DOCUMENT NAME as `document_name`
- Examples for search:
  - "search for latrines in TB MED 593" → query="latrines", document_name="TB MED 593"
  - "find waste disposal in the army manual" → query="waste disposal", document_name="army manual"
- Examples for ask:
  - "what does TB MED 593 say about latrines?" → question="what are the guidelines for latrines?", document_name="TB MED 593"
  - "answer from the army manual about sanitation" → question="what are the sanitation guidelines?", document_name="army manual"

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
        document_name: str | None = None,
    ) -> str:
        """Search the knowledge base for relevant documents.

        Use this when you need to find documents or explore the knowledge base.
        Results are displayed to the user - just list the titles found.

        Args:
            query: The search query (what to search for)
            document_name: Optional document name/title to search within (e.g., "tbmed593", "army manual")
        """
        from search_agent import SearchAgent

        if ctx.deps.agui_emitter:
            msg = f"Searching: {query}"
            if document_name:
                msg += f" (in {document_name})"
            ctx.deps.agui_emitter.log(msg)

        # Build context from conversation history
        context = None
        if ctx.deps.session_state and ctx.deps.session_state.qa_history:
            context = format_conversation_context(ctx.deps.session_state.qa_history)

        # Build filter from document_name
        doc_filter = build_document_filter(document_name) if document_name else None

        # Use search agent for query expansion and deduplication
        search_agent = SearchAgent(ctx.deps.client, ctx.deps.config)
        results = await search_agent.search(query, context=context, filter=doc_filter)

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

        # Return detailed results for the agent to present
        result_lines = []
        for i, r in enumerate(results):
            title = r.document_title or r.document_uri or "Unknown"
            # Truncate content for display
            snippet = r.content[:300].replace("\n", " ").strip()
            if len(r.content) > 300:
                snippet += "..."

            line = f"[{i + 1}] **{title}**"
            if r.page_numbers:
                line += f" (pages {', '.join(map(str, r.page_numbers))})"
            line += f"\n    {snippet}"
            result_lines.append(line)

        return f"Found {len(results)} results:\n\n" + "\n\n".join(result_lines)

    @agent.tool
    async def ask(
        ctx: RunContext[ChatDeps],
        question: str,
        document_name: str | None = None,
    ) -> str:
        """Answer a specific question using the knowledge base.

        Use this for direct questions that need a focused answer with citations.
        Uses a research graph for planning, searching, and synthesis.

        Args:
            question: The question to answer
            document_name: Optional document name/title to search within (e.g., "tbmed593", "army manual")
        """
        from haiku.rag.graph.research.dependencies import ResearchContext
        from haiku.rag.graph.research.graph import build_conversational_graph
        from haiku.rag.graph.research.models import Citation, SearchAnswer
        from haiku.rag.graph.research.state import ResearchDeps, ResearchState

        if ctx.deps.agui_emitter:
            msg = f"Answering: {question}"
            if document_name:
                msg += f" (in {document_name})"
            ctx.deps.agui_emitter.log(msg)

        # Build filter from document_name
        doc_filter = build_document_filter(document_name) if document_name else None

        # Convert existing qa_history to SearchAnswers for context seeding
        existing_qa: list[SearchAnswer] = []
        if ctx.deps.session_state and ctx.deps.session_state.qa_history:
            for qa in ctx.deps.session_state.qa_history:
                citations = [
                    Citation(
                        document_id=c.document_id,
                        chunk_id=c.chunk_id,
                        document_uri=c.document_uri,
                        document_title=c.document_title,
                        page_numbers=c.page_numbers,
                        headings=c.headings,
                        content=c.content,
                    )
                    for c in qa.citations
                ]
                existing_qa.append(
                    SearchAnswer(
                        query=qa.question,
                        answer=qa.answer,
                        confidence=qa.confidence,
                        cited_chunks=[c.chunk_id for c in qa.citations],
                        citations=citations,
                    )
                )

        # Build and run the conversational research graph
        graph = build_conversational_graph(config=ctx.deps.config)

        context = ResearchContext(
            original_question=question,
            qa_responses=existing_qa,
        )
        state = ResearchState(
            context=context,
            max_iterations=1,
            confidence_threshold=0.0,
            search_filter=doc_filter,
            max_concurrency=ctx.deps.config.research.max_concurrency,
        )
        # Don't pass agui_emitter to research graph - its state model differs from ChatSessionState
        # The ask tool handles final state emission with citations
        deps = ResearchDeps(
            client=ctx.deps.client,
        )

        result = await graph.run(state=state, deps=deps)

        # Build citation infos for frontend and history
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
            for i, c in enumerate(result.citations)
        ]

        # Accumulate Q&A in session state with full citation metadata
        if ctx.deps.session_state is not None:
            qa_response = QAResponse(
                question=question,
                answer=result.answer,
                confidence=result.confidence,
                citations=citation_infos,
            )
            ctx.deps.session_state.qa_history.append(qa_response)

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

        # Format answer with citation references and confidence
        answer_text = result.answer
        if citation_infos:
            citation_refs = " ".join(f"[{i + 1}]" for i in range(len(citation_infos)))
            answer_text = f"{answer_text}\n\nSources: {citation_refs}"

        return answer_text

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
        # Also try without spaces for matching "TB MED 593" to "tbmed593"
        no_spaces = escaped_query.replace(" ", "")

        # If not found, try partial URI match (with and without spaces)
        if doc is None:
            docs = await ctx.deps.client.list_documents(
                limit=1,
                filter=f"LOWER(uri) LIKE LOWER('%{escaped_query}%') OR LOWER(uri) LIKE LOWER('%{no_spaces}%')",
            )
            if docs:
                doc = docs[0]

        # If still not found, try partial title match (with and without spaces)
        if doc is None:
            docs = await ctx.deps.client.list_documents(
                limit=1,
                filter=f"LOWER(title) LIKE LOWER('%{escaped_query}%') OR LOWER(title) LIKE LOWER('%{no_spaces}%')",
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
