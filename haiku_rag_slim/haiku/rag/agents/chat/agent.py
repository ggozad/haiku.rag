from ag_ui.core import EventType, StateSnapshotEvent
from pydantic_ai import Agent, RunContext, ToolReturn

from haiku.rag.agents.chat.prompts import CHAT_SYSTEM_PROMPT
from haiku.rag.agents.chat.search import SearchAgent
from haiku.rag.agents.chat.state import (
    MAX_QA_HISTORY,
    ChatDeps,
    ChatSessionState,
    CitationInfo,
    QAResponse,
    build_document_filter,
    rank_qa_history_by_similarity,
)
from haiku.rag.agents.research.dependencies import ResearchContext
from haiku.rag.agents.research.graph import build_conversational_graph
from haiku.rag.agents.research.models import Citation, SearchAnswer
from haiku.rag.agents.research.state import ResearchDeps, ResearchState
from haiku.rag.config.models import AppConfig
from haiku.rag.utils import get_model


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

        # Build new state with citations
        new_state = ChatSessionState(
            session_id=(
                ctx.deps.session_state.session_id if ctx.deps.session_state else ""
            ),
            citations=citation_infos,
            qa_history=(
                ctx.deps.session_state.qa_history if ctx.deps.session_state else []
            ),
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

        return ToolReturn(
            return_value=f"Found {len(results)} results:\n\n"
            + "\n\n".join(result_lines),
            metadata=[
                StateSnapshotEvent(
                    type=EventType.STATE_SNAPSHOT,
                    snapshot=new_state.model_dump(),
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

        # Filter and rank qa_history
        ranked_history: list[QAResponse] = []
        if ctx.deps.session_state and ctx.deps.session_state.qa_history:
            # Step 1: Filter out low-confidence responses
            filtered_history = [
                qa for qa in ctx.deps.session_state.qa_history if qa.confidence >= 0.3
            ]

            # Step 2: Rank filtered history by similarity to current question
            embedder = ctx.deps.client.chunk_repository.embedder
            ranked_history = await rank_qa_history_by_similarity(
                current_question=question,
                qa_history=filtered_history,
                embedder=embedder,
                top_k=5,
            )

        # Convert ranked qa_history to SearchAnswers for context seeding
        existing_qa: list[SearchAnswer] = []
        if ranked_history:
            for qa in ranked_history:
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
            # Enforce FIFO limit
            if len(ctx.deps.session_state.qa_history) > MAX_QA_HISTORY:
                ctx.deps.session_state.qa_history = ctx.deps.session_state.qa_history[
                    -MAX_QA_HISTORY:
                ]

        # Build new state with citations AND accumulated qa_history
        new_state = ChatSessionState(
            session_id=(
                ctx.deps.session_state.session_id if ctx.deps.session_state else ""
            ),
            citations=citation_infos,
            qa_history=(
                ctx.deps.session_state.qa_history if ctx.deps.session_state else []
            ),
        )

        # Format answer with citation references and confidence
        answer_text = result.answer
        if citation_infos:
            citation_refs = " ".join(f"[{i + 1}]" for i in range(len(citation_infos)))
            answer_text = f"{answer_text}\n\nSources: {citation_refs}"

        return ToolReturn(
            return_value=answer_text,
            metadata=[
                StateSnapshotEvent(
                    type=EventType.STATE_SNAPSHOT,
                    snapshot=new_state.model_dump(),
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
