import asyncio
import math
import uuid

from pydantic_ai import Agent, RunContext, ToolReturn

from haiku.rag.agents.chat.context import (
    get_cached_session_context,
    update_session_context,
)
from haiku.rag.agents.chat.prompts import CHAT_SYSTEM_PROMPT, DOCUMENT_SUMMARY_PROMPT
from haiku.rag.agents.chat.search import SearchAgent
from haiku.rag.agents.chat.state import (
    MAX_QA_HISTORY,
    ChatDeps,
    ChatSessionState,
    DocumentInfo,
    DocumentListResponse,
    QAResponse,
    build_document_filter,
    build_multi_document_filter,
    combine_filters,
    emit_state_event,
)
from haiku.rag.agents.research.dependencies import ResearchContext
from haiku.rag.agents.research.graph import build_research_graph
from haiku.rag.agents.research.models import Citation
from haiku.rag.agents.research.state import ResearchDeps, ResearchState
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.embeddings import get_embedder
from haiku.rag.utils import get_model

# Similarity threshold for finding relevant prior answers
PRIOR_ANSWER_RELEVANCE_THRESHOLD = 0.7


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

        new_state = ctx.deps.session_state.model_copy(deep=True)
        if not new_state.session_id:
            new_state.session_id = str(uuid.uuid4())

        # Build citation infos using the copy's registry
        citation_infos = []
        for r in results:
            chunk_id = r.chunk_id or ""
            if chunk_id:
                index = new_state.get_or_assign_index(chunk_id)
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

        # Update new_state with citations and fresh session_context
        new_state.citations = citation_infos
        if new_state.session_id:
            new_state.session_context = get_cached_session_context(new_state.session_id)

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

        state_event = emit_state_event(
            ctx.deps.session_state, new_state, ctx.deps.state_key
        )

        return ToolReturn(
            return_value=f"Found {len(results)} results:\n\n"
            + "\n\n".join(result_lines),
            metadata=[state_event] if state_event else None,
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
        session_filter = build_multi_document_filter(
            ctx.deps.session_state.document_filter
        )

        # Build tool filter from document_name parameter
        tool_filter = build_document_filter(document_name) if document_name else None

        # Combine filters: session AND tool
        doc_filter = combine_filters(session_filter, tool_filter)

        # Build and run the conversational research graph
        graph = build_research_graph(
            config=ctx.deps.config, output_mode="conversational"
        )
        session_id = ctx.deps.session_state.session_id

        # Get session context from server cache for planning, fallback to initial_context
        cached_context = get_cached_session_context(session_id)
        session_context = (
            cached_context.render_markdown()
            if cached_context and cached_context.summary
            else ctx.deps.session_state.initial_context
        )

        # Find relevant prior answers from qa_history
        prior_answers = []
        if ctx.deps.session_state.qa_history:
            embedder = get_embedder(ctx.deps.config)
            question_embedding = await embedder.embed_query(question)

            # Collect questions that need embedding (not cached)
            to_embed = []
            to_embed_indices = []
            for i, qa in enumerate(ctx.deps.session_state.qa_history):
                if qa.question_embedding is None:
                    to_embed.append(qa.question)
                    to_embed_indices.append(i)

            # Batch embed uncached questions
            if to_embed:
                new_embeddings = await embedder.embed_documents(to_embed)
                for i, idx in enumerate(to_embed_indices):
                    ctx.deps.session_state.qa_history[
                        idx
                    ].question_embedding = new_embeddings[i]

            # Compare against all questions and collect relevant prior answers
            for qa in ctx.deps.session_state.qa_history:
                if qa.question_embedding is not None:
                    similarity = _cosine_similarity(
                        question_embedding, qa.question_embedding
                    )
                    if similarity >= PRIOR_ANSWER_RELEVANCE_THRESHOLD:
                        prior_answers.append(qa.to_search_answer())

        context = ResearchContext(
            original_question=question,
            session_context=session_context,
            qa_responses=prior_answers,
        )
        state = ResearchState(
            context=context,
            max_iterations=1,
            search_filter=doc_filter,
            max_concurrency=ctx.deps.config.research.max_concurrency,
        )
        deps = ResearchDeps(
            client=ctx.deps.client,
        )

        result = await graph.run(state=state, deps=deps)

        new_state = ctx.deps.session_state.model_copy(deep=True)
        if not new_state.session_id:
            new_state.session_id = str(uuid.uuid4())

        # Build citation infos using the copy's registry
        citation_infos = []
        for c in result.citations:
            index = new_state.get_or_assign_index(c.chunk_id)
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

        # Add Q&A to the copy's history
        qa_response = QAResponse(
            question=question,
            answer=result.answer,
            confidence=result.confidence,
            citations=citation_infos,
        )
        new_state.qa_history.append(qa_response)
        # Enforce FIFO limit
        if len(new_state.qa_history) > MAX_QA_HISTORY:
            new_state.qa_history = new_state.qa_history[-MAX_QA_HISTORY:]

        # Update citations and session_context
        new_state.citations = citation_infos
        if new_state.session_id:
            new_state.session_context = get_cached_session_context(new_state.session_id)

        # Spawn background task to update session context
        if new_state.session_id in _summarization_tasks:
            _summarization_tasks[new_state.session_id].cancel()

        task = asyncio.create_task(
            _update_context_background(
                qa_history=list(new_state.qa_history),
                config=ctx.deps.config,
                session_state=new_state,
            )
        )
        _summarization_tasks[new_state.session_id] = task
        task.add_done_callback(
            lambda t, sid=new_state.session_id: _summarization_tasks.pop(sid, None)
        )

        # Format answer with citation references using stable indices
        answer_text = result.answer
        if citation_infos:
            citation_refs = " ".join(f"[{c.index}]" for c in citation_infos)
            answer_text = f"{answer_text}\n\nSources: {citation_refs}"

        state_event = emit_state_event(
            ctx.deps.session_state, new_state, ctx.deps.state_key
        )

        return ToolReturn(
            return_value=answer_text,
            metadata=[state_event] if state_event else None,
        )

    @agent.tool
    async def list_documents(
        ctx: RunContext[ChatDeps],
        page: int = 1,
    ) -> DocumentListResponse:
        """List available documents in the knowledge base.

        Use this when the user wants to browse or see what documents are available.

        Args:
            page: Page number (default: 1, 50 documents per page)
        """
        page_size = 50
        offset = (page - 1) * page_size

        doc_filter = build_multi_document_filter(ctx.deps.session_state.document_filter)

        docs = await ctx.deps.client.list_documents(
            limit=page_size, offset=offset, filter=doc_filter
        )
        total = await ctx.deps.client.count_documents(filter=doc_filter)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1

        return DocumentListResponse(
            documents=[
                DocumentInfo(
                    title=doc.title or "Untitled",
                    uri=doc.uri or "",
                    created=doc.created_at.strftime("%Y-%m-%d"),
                )
                for doc in docs
            ],
            page=page,
            total_pages=total_pages,
            total_documents=total,
        )

    async def _find_document(client: HaikuRAG, query: str):
        """Find a document by exact URI, partial URI, or partial title match."""
        # Try exact URI match first
        doc = await client.get_document_by_uri(query)
        if doc is not None:
            return doc

        escaped_query = query.replace("'", "''")
        # Also try without spaces for matching "TB MED 593" to "tbmed593"
        no_spaces = escaped_query.replace(" ", "")

        # Try partial URI match (with and without spaces)
        docs = await client.list_documents(
            limit=1,
            filter=f"LOWER(uri) LIKE LOWER('%{escaped_query}%') OR LOWER(uri) LIKE LOWER('%{no_spaces}%')",
        )
        if docs:
            return docs[0]

        # Try partial title match (with and without spaces)
        docs = await client.list_documents(
            limit=1,
            filter=f"LOWER(title) LIKE LOWER('%{escaped_query}%') OR LOWER(title) LIKE LOWER('%{no_spaces}%')",
        )
        if docs:
            return docs[0]

        return None

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
        doc = await _find_document(ctx.deps.client, query)

        if doc is None:
            return f"Document not found: {query}"

        return (
            f"**{doc.title or 'Untitled'}**\n\n"
            f"- ID: {doc.id}\n"
            f"- URI: {doc.uri}\n"
            f"- Created: {doc.created_at.strftime('%Y-%m-%d %H:%M')}\n\n"
            f"**Content:**\n{doc.content}"
        )

    @agent.tool
    async def summarize_document(
        ctx: RunContext[ChatDeps],
        query: str,
    ) -> str:
        """Generate a summary of a specific document.

        Use this when the user wants an overview or summary of a document's content.

        Args:
            query: The document title or URI to summarize
        """
        doc = await _find_document(ctx.deps.client, query)

        if doc is None:
            return f"Document not found: {query}"

        # Use LLM to generate summary
        summary_model = get_model(ctx.deps.config.qa.model, ctx.deps.config)
        summary_agent: Agent[None, str] = Agent(
            summary_model,
            output_type=str,
        )
        result = await summary_agent.run(
            DOCUMENT_SUMMARY_PROMPT.format(content=doc.content or "")
        )

        return f"**Summary of {doc.title or doc.uri}:**\n\n{result.output}"

    return agent
