import asyncio
import math

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
    build_multi_document_filter,
    combine_filters,
)
from haiku.rag.agents.research.dependencies import ResearchContext
from haiku.rag.agents.research.graph import build_conversational_graph
from haiku.rag.agents.research.models import Citation
from haiku.rag.agents.research.state import ResearchDeps, ResearchState
from haiku.rag.config.models import AppConfig
from haiku.rag.embeddings import get_embedder
from haiku.rag.utils import get_model

# Similarity threshold for recall matching
RECALL_SIMILARITY_THRESHOLD = 0.8


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


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
        # Build session filter from document_filter
        session_filter = None
        if ctx.deps.session_state and ctx.deps.session_state.document_filter:
            session_filter = build_multi_document_filter(
                ctx.deps.session_state.document_filter
            )

        # Build tool filter from document_name parameter
        tool_filter = build_document_filter(document_name) if document_name else None

        # Combine filters: session AND tool
        doc_filter = combine_filters(session_filter, tool_filter)

        # Use search agent for query expansion and deduplication
        search_agent = SearchAgent(ctx.deps.client, ctx.deps.config)
        results = await search_agent.search(query, filter=doc_filter, limit=limit)

        # Store for potential citation resolution
        ctx.deps.search_results = results

        if not results:
            return ToolReturn(return_value="No results found.")

        # Build citation infos using stable registry indices
        citation_infos = []
        for r in results:
            chunk_id = r.chunk_id or ""
            if ctx.deps.session_state is not None and chunk_id:
                index = ctx.deps.session_state.get_or_assign_index(chunk_id)
            else:
                index = len(citation_infos) + 1
            citation_infos.append(
                Citation(
                    index=index,
                    document_id=r.document_id or "",
                    chunk_id=chunk_id,
                    document_uri=r.document_uri or "",
                    document_title=r.document_title,
                    page_numbers=r.page_numbers or [],
                    headings=r.headings,
                    content=r.content,
                )
            )

        # Build new state with citations and registry
        session_id = ctx.deps.session_state.session_id if ctx.deps.session_state else ""
        new_state = ChatSessionState(
            session_id=session_id,
            citations=citation_infos,
            qa_history=(
                ctx.deps.session_state.qa_history if ctx.deps.session_state else []
            ),
            session_context=get_cached_session_context(session_id)
            if session_id
            else None,
            document_filter=(
                ctx.deps.session_state.document_filter if ctx.deps.session_state else []
            ),
            citation_registry=(
                ctx.deps.session_state.citation_registry
                if ctx.deps.session_state
                else {}
            ),
        )

        # Return detailed results for the agent to present
        result_lines = []
        for c in citation_infos:
            title = c.document_title or c.document_uri or "Unknown"
            # Truncate content for display
            snippet = c.content[:300].replace("\n", " ").strip()
            if len(c.content) > 300:
                snippet += "..."

            line = f"[{c.index}] **{title}**"
            if c.page_numbers:
                line += f" (pages {', '.join(map(str, c.page_numbers))})"
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
        # Build session filter from document_filter
        session_filter = None
        if ctx.deps.session_state and ctx.deps.session_state.document_filter:
            session_filter = build_multi_document_filter(
                ctx.deps.session_state.document_filter
            )

        # Build tool filter from document_name parameter
        tool_filter = build_document_filter(document_name) if document_name else None

        # Combine filters: session AND tool
        doc_filter = combine_filters(session_filter, tool_filter)

        # Build and run the conversational research graph
        graph = build_conversational_graph(config=ctx.deps.config)
        session_id = ctx.deps.session_state.session_id if ctx.deps.session_state else ""

        # Get session context from server cache for planning
        cached_context = get_cached_session_context(session_id) if session_id else None
        session_context = (
            cached_context.render_markdown()
            if cached_context and cached_context.summary
            else None
        )

        context = ResearchContext(
            original_question=question,
            session_context=session_context,
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

        # Build citation infos using stable registry indices
        citation_infos = []
        for c in result.citations:
            # Use registry for stable indices across calls
            if ctx.deps.session_state is not None:
                index = ctx.deps.session_state.get_or_assign_index(c.chunk_id)
            else:
                index = len(citation_infos) + 1
            citation_infos.append(
                Citation(
                    index=index,
                    document_id=c.document_id,
                    chunk_id=c.chunk_id,
                    document_uri=c.document_uri,
                    document_title=c.document_title,
                    page_numbers=c.page_numbers,
                    headings=c.headings,
                    content=c.content,
                )
            )

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

        # Build new state with citations, qa_history, and registry
        new_state = ChatSessionState(
            session_id=session_id,
            citations=citation_infos,
            qa_history=(
                ctx.deps.session_state.qa_history if ctx.deps.session_state else []
            ),
            session_context=get_cached_session_context(session_id)
            if session_id
            else None,
            document_filter=(
                ctx.deps.session_state.document_filter if ctx.deps.session_state else []
            ),
            citation_registry=(
                ctx.deps.session_state.citation_registry
                if ctx.deps.session_state
                else {}
            ),
        )

        # Format answer with citation references using stable indices
        answer_text = result.answer
        if citation_infos:
            citation_refs = " ".join(f"[{c.index}]" for c in citation_infos)
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

    @agent.tool
    async def recall(
        ctx: RunContext[ChatDeps],
        topic: str,
    ) -> ToolReturn:
        """Search conversation history for a previous answer on this topic.

        Use this FIRST when the user asks about something that may have been
        discussed before. Returns the previous answer with citations if found,
        or indicates no match exists.

        Args:
            topic: The topic or question to search for in conversation history
        """
        if ctx.deps.session_state is None:
            return ToolReturn(return_value="No conversation history available.")

        qa_history = ctx.deps.session_state.qa_history
        if not qa_history:
            return ToolReturn(return_value="No previous answers found.")

        # Get embedder and embed the topic
        embedder = get_embedder(ctx.deps.config)
        topic_embedding = await embedder.embed_query(topic)

        # Embed all previous questions
        questions = [qa.question for qa in qa_history]
        question_embeddings = await embedder.embed_documents(questions)

        # Find best match by cosine similarity
        best_match_idx = -1
        best_similarity = 0.0
        for i, q_embedding in enumerate(question_embeddings):
            similarity = _cosine_similarity(topic_embedding, q_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = i

        # Check if similarity exceeds threshold
        if best_similarity < RECALL_SIMILARITY_THRESHOLD:
            return ToolReturn(return_value="No previous answer found on this topic.")

        # Build result with the matching answer
        matched_qa = qa_history[best_match_idx]
        result = f"**Previous answer found** (similarity: {best_similarity:.2f}):\n\n"
        result += f"**Question:** {matched_qa.question}\n\n"
        result += f"**Answer:** {matched_qa.answer}\n\n"

        if matched_qa.citations:
            citation_refs = " ".join(f"[{c.index}]" for c in matched_qa.citations)
            result += f"Sources: {citation_refs}"

        # Emit state with citations so frontend can display them
        session_id = ctx.deps.session_state.session_id
        new_state = ChatSessionState(
            session_id=session_id,
            citations=matched_qa.citations,
            qa_history=ctx.deps.session_state.qa_history,
            session_context=get_cached_session_context(session_id)
            if session_id
            else None,
            document_filter=ctx.deps.session_state.document_filter,
            citation_registry=ctx.deps.session_state.citation_registry,
        )

        snapshot = new_state.model_dump()
        if ctx.deps.state_key:
            snapshot = {ctx.deps.state_key: snapshot}

        return ToolReturn(
            return_value=result,
            metadata=[
                StateSnapshotEvent(
                    type=EventType.STATE_SNAPSHOT,
                    snapshot=snapshot,
                )
            ],
        )

    return agent
