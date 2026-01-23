import asyncio

from ag_ui.core import EventType, StateSnapshotEvent
from pydantic_ai import Agent, RunContext, ToolReturn

from haiku.rag.agents.chat.context import (
    get_cached_session_context,
    update_session_context,
)
from haiku.rag.agents.chat.prompts import CHAT_SYSTEM_PROMPT
from haiku.rag.agents.chat.search import SearchAgent
from haiku.rag.agents.chat.state import (
    MAX_QA_HISTORY,
    ChatDeps,
    ChatSessionState,
    QAResponse,
    build_document_filter,
)
from haiku.rag.agents.research.dependencies import ResearchContext
from haiku.rag.agents.research.graph import build_conversational_graph
from haiku.rag.agents.research.models import Citation
from haiku.rag.agents.research.state import ResearchDeps, ResearchState
from haiku.rag.config.models import AppConfig
from haiku.rag.utils import get_model

# Track summarization tasks per session to allow cancellation
_summarization_tasks: dict[str, asyncio.Task[None]] = {}


async def _update_context_background(
    qa_history: list[QAResponse],
    config: AppConfig,
    session_state: ChatSessionState,
) -> None:
    """Background task to update session context after an ask."""
    try:
        await update_session_context(
            qa_history=qa_history,
            config=config,
            session_state=session_state,
        )
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


def create_chat_agent(config: AppConfig) -> Agent[ChatDeps, str]:
    """Create the chat agent with search and ask tools."""
    model = get_model(config.qa.model, config)

    agent: Agent[ChatDeps, str] = Agent(
        model,
        deps_type=ChatDeps,
        output_type=str,
        instructions=CHAT_SYSTEM_PROMPT,
        retries=3,
    )

    @agent.system_prompt
    async def add_background_context(ctx: RunContext[ChatDeps]) -> str:
        """Add background_context to system prompt when available."""
        if ctx.deps.session_state and ctx.deps.session_state.background_context:
            return f"\nBACKGROUND CONTEXT:\n{ctx.deps.session_state.background_context}"
        return ""

    @agent.tool
    async def search(
        ctx: RunContext[ChatDeps],
        query: str,
        document_name: str | None = None,
        limit: int | None = None,
    ) -> ToolReturn:
        """Search the knowledge base for relevant documents.

        Use this when you need to find documents or explore the knowledge base.
        Results are displayed to the user - just list the titles found.

        Args:
            query: The search query (what to search for)
            document_name: Optional document name/title to search within
            limit: Number of results to return (default: 5)
        """
        # Build filter from document_name
        doc_filter = build_document_filter(document_name) if document_name else None

        # Use search agent for query expansion and deduplication
        search_agent = SearchAgent(ctx.deps.client, ctx.deps.config)
        results = await search_agent.search(query, filter=doc_filter, limit=limit)

        # Store for potential citation resolution
        ctx.deps.search_results = results

        if not results:
            return ToolReturn(return_value="No results found.")

        # Build citation infos for frontend display
        citation_infos = [
            Citation(
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

        # Build new state with citations
        session_id = ctx.deps.session_state.session_id if ctx.deps.session_state else ""
        new_state = ChatSessionState(
            session_id=session_id,
            citations=citation_infos,
            qa_history=(
                ctx.deps.session_state.qa_history if ctx.deps.session_state else []
            ),
            background_context=(
                ctx.deps.session_state.background_context
                if ctx.deps.session_state
                else None
            ),
            session_context=get_cached_session_context(session_id)
            if session_id
            else None,
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

        snapshot = new_state.model_dump()
        if ctx.deps.state_key:
            snapshot = {ctx.deps.state_key: snapshot}

        return ToolReturn(
            return_value=f"Found {len(results)} results:\n\n"
            + "\n\n".join(result_lines),
            metadata=[
                StateSnapshotEvent(
                    type=EventType.STATE_SNAPSHOT,
                    snapshot=snapshot,
                )
            ],
        )

    @agent.tool
    async def ask(
        ctx: RunContext[ChatDeps],
        question: str,
        document_name: str | None = None,
    ) -> ToolReturn:
        """Answer a specific question using the knowledge base.

        Use this for direct questions that need a focused answer with citations.
        Uses a research graph for planning, searching, and synthesis.

        Args:
            question: The question to answer
            document_name: Optional document name/title to search within (e.g., "tbmed593", "army manual")
        """
        # Build filter from document_name
        doc_filter = build_document_filter(document_name) if document_name else None

        # Build and run the conversational research graph
        graph = build_conversational_graph(config=ctx.deps.config)

        # Determine context strategy:
        # 1. Read from server cache (ignoring client state)
        # 2. Fall back to explicit background_context (first request)
        background_context: str | None = None
        session_id = ctx.deps.session_state.session_id if ctx.deps.session_state else ""
        if ctx.deps.session_state:
            cached_context = (
                get_cached_session_context(session_id) if session_id else None
            )

            if cached_context and cached_context.summary:
                # Use cached SessionContext from previous summarization
                background_context = cached_context.render_markdown()
            elif ctx.deps.session_state.background_context:
                # Fall back to explicit background_context (first request)
                background_context = ctx.deps.session_state.background_context

        context = ResearchContext(
            original_question=question,
            background_context=background_context,
        )
        state = ResearchState(
            context=context,
            max_iterations=1,
            confidence_threshold=0.0,
            search_filter=doc_filter,
            max_concurrency=ctx.deps.config.research.max_concurrency,
        )
        deps = ResearchDeps(
            client=ctx.deps.client,
        )

        result = await graph.run(state=state, deps=deps)

        # Build citation infos for frontend and history
        citation_infos = [
            Citation(
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
            # Enforce FIFO limit
            if len(ctx.deps.session_state.qa_history) > MAX_QA_HISTORY:
                ctx.deps.session_state.qa_history = ctx.deps.session_state.qa_history[
                    -MAX_QA_HISTORY:
                ]

            # Spawn background task to update session context
            # Cancel any previous summarization for this session
            if session_id in _summarization_tasks:
                _summarization_tasks[session_id].cancel()

            task = asyncio.create_task(
                _update_context_background(
                    qa_history=list(ctx.deps.session_state.qa_history),
                    config=ctx.deps.config,
                    session_state=ctx.deps.session_state,
                )
            )
            _summarization_tasks[session_id] = task
            task.add_done_callback(lambda t: _summarization_tasks.pop(session_id, None))

        # Build new state with citations AND accumulated qa_history
        new_state = ChatSessionState(
            session_id=session_id,
            citations=citation_infos,
            qa_history=(
                ctx.deps.session_state.qa_history if ctx.deps.session_state else []
            ),
            background_context=(
                ctx.deps.session_state.background_context
                if ctx.deps.session_state
                else None
            ),
            session_context=get_cached_session_context(session_id)
            if session_id
            else None,
        )

        # Format answer with citation references and confidence
        answer_text = result.answer
        if citation_infos:
            citation_refs = " ".join(f"[{i + 1}]" for i in range(len(citation_infos)))
            answer_text = f"{answer_text}\n\nSources: {citation_refs}"

        snapshot = new_state.model_dump()
        if ctx.deps.state_key:
            snapshot = {ctx.deps.state_key: snapshot}

        return ToolReturn(
            return_value=answer_text,
            metadata=[
                StateSnapshotEvent(
                    type=EventType.STATE_SNAPSHOT,
                    snapshot=snapshot,
                )
            ],
        )

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
