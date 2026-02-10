import math

from pydantic import BaseModel, Field
from pydantic_ai import FunctionToolset, ToolReturn

from haiku.rag.agents.chat.context import (
    cache_question_embedding,
    get_cached_embedding,
    trigger_background_summarization,
)
from haiku.rag.agents.chat.state import SessionContext
from haiku.rag.agents.research.dependencies import ResearchContext
from haiku.rag.agents.research.graph import build_research_graph
from haiku.rag.agents.research.models import Citation, SearchAnswer
from haiku.rag.agents.research.state import ResearchDeps, ResearchState
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.embeddings import get_embedder
from haiku.rag.tools.context import ToolContext
from haiku.rag.tools.filters import (
    build_document_filter,
    combine_filters,
    get_session_filter,
)
from haiku.rag.tools.models import QAResult
from haiku.rag.tools.session import (
    SESSION_NAMESPACE,
    SessionState,
    compute_combined_state_delta,
)

PRIOR_ANSWER_RELEVANCE_THRESHOLD = 0.7


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


class QAHistoryEntry(BaseModel):
    """A Q&A pair with optional cached embedding for similarity matching."""

    question: str
    answer: str
    confidence: float = 0.9
    citations: list[Citation] = []
    question_embedding: list[float] | None = Field(default=None, exclude=True)

    @property
    def sources(self) -> list[str]:
        """Source names for display."""
        return list(
            dict.fromkeys(c.document_title or c.document_uri for c in self.citations)
        )

    def to_search_answer(self) -> SearchAnswer:
        """Convert to SearchAnswer for research graph context."""
        return SearchAnswer(
            query=self.question,
            answer=self.answer,
            confidence=self.confidence,
            cited_chunks=[c.chunk_id for c in self.citations],
            citations=self.citations,
        )


# Resolve ChatSessionState forward reference to QAHistoryEntry
from haiku.rag.agents.chat.state import _rebuild_models  # noqa: E402

_rebuild_models(QAHistoryEntry)


class QASessionState(BaseModel):
    """Extended session state for QA with embedding cache."""

    qa_history: list[QAHistoryEntry] = []
    session_context: str | None = None
    incoming_session_context: SessionContext | None = Field(
        default=None, exclude=True
    )  # Track what client sent


QA_SESSION_NAMESPACE = "haiku.rag.qa_session"
MAX_QA_HISTORY = 50


def create_qa_toolset(
    client: HaikuRAG,
    config: AppConfig,
    context: ToolContext | None = None,
    base_filter: str | None = None,
    tool_name: str = "ask",
    session_context: str | None = None,
    prior_answers: list[SearchAnswer] | None = None,
) -> FunctionToolset:
    """Create a toolset with Q&A capabilities using research graph.

    Args:
        client: HaikuRAG client for search operations.
        config: Application configuration.
        context: Optional ToolContext for state accumulation.
            If SessionState is registered, it will be used for dynamic
            document filtering and citation indexing.
        base_filter: Optional base SQL WHERE clause applied to searches.
        tool_name: Name for the ask tool. Defaults to "ask".
        session_context: Optional session context for the research graph.
            Overridden by QASessionState.session_context if available.
        prior_answers: Optional list of prior answers for context.
            Overridden by similarity-matched answers from QASessionState if available.

    Returns:
        FunctionToolset with an ask tool.
    """

    async def ask(
        question: str,
        document_name: str | None = None,
    ) -> ToolReturn | QAResult:
        """Answer a question using the knowledge base.

        Uses a research graph for searching and synthesizing answers.

        Args:
            question: The question to answer.
            document_name: Optional document name/title to search within.

        Returns:
            QAResult with answer, confidence, and citations.
        """
        # Get session states
        session_state: SessionState | None = None
        qa_session_state: QASessionState | None = None
        old_state_snapshot: dict | None = None

        if context is not None:
            session_state = context.get(SESSION_NAMESPACE, SessionState)
            qa_session_state = context.get(QA_SESSION_NAMESPACE, QASessionState)

            # Capture combined state snapshot before changes
            # Use incoming values (what client sent) so delta shows server-side updates
            if session_state is not None:
                old_state_snapshot = {
                    "session_id": session_state.incoming_session_id,
                    "document_filter": session_state.document_filter.copy(),
                    "citation_registry": session_state.citation_registry.copy(),
                    "citations": [c.model_dump() for c in session_state.citations],
                }
                if qa_session_state is not None:
                    old_state_snapshot["qa_history"] = [
                        qa.model_dump() for qa in qa_session_state.qa_history
                    ]
                    # Use incoming_session_context so delta shows what client sent
                    if qa_session_state.incoming_session_context is not None:
                        old_state_snapshot["session_context"] = (
                            qa_session_state.incoming_session_context.model_dump(
                                mode="json"
                            )
                        )
                    else:
                        old_state_snapshot["session_context"] = None

        # Build filter from session state, base_filter, and document_name
        doc_filter = build_document_filter(document_name) if document_name else None
        effective_filter = combine_filters(
            get_session_filter(context, base_filter), doc_filter
        )

        # Determine session context
        effective_session_context = session_context
        if qa_session_state is not None and qa_session_state.session_context:
            effective_session_context = qa_session_state.session_context

        # Find relevant prior answers via similarity matching
        effective_prior_answers = prior_answers or []
        session_id = session_state.session_id if session_state is not None else ""
        if qa_session_state is not None and qa_session_state.qa_history:
            embedder = get_embedder(config)
            question_embedding = await embedder.embed_query(question)

            # Collect questions that need embedding
            to_embed = []
            to_embed_indices = []
            for i, qa in enumerate(qa_session_state.qa_history):
                if qa.question_embedding is None:
                    # Check per-session cache first
                    if session_id:
                        cached = get_cached_embedding(session_id, qa.question)
                        if cached:
                            qa.question_embedding = cached
                            continue
                    to_embed.append(qa.question)
                    to_embed_indices.append(i)

            # Batch embed uncached questions
            if to_embed:
                new_embeddings = await embedder.embed_documents(to_embed)
                for i, idx in enumerate(to_embed_indices):
                    embedding = new_embeddings[i]
                    qa_session_state.qa_history[idx].question_embedding = embedding
                    # Cache per-session for next request
                    if session_id:
                        cache_question_embedding(
                            session_id,
                            qa_session_state.qa_history[idx].question,
                            embedding,
                        )

            # Find similar prior answers
            matched_answers = []
            for qa in qa_session_state.qa_history:
                if qa.question_embedding is not None:
                    similarity = _cosine_similarity(
                        question_embedding, qa.question_embedding
                    )
                    if similarity >= PRIOR_ANSWER_RELEVANCE_THRESHOLD:
                        matched_answers.append(qa.to_search_answer())

            if matched_answers:
                effective_prior_answers = matched_answers

        # Build and run the research graph
        graph = build_research_graph(config=config, output_mode="conversational")

        research_context = ResearchContext(
            original_question=question,
            session_context=effective_session_context,
            qa_responses=effective_prior_answers,
        )
        research_state = ResearchState(
            context=research_context,
            max_iterations=1,
            search_filter=effective_filter,
            max_concurrency=config.research.max_concurrency,
        )
        deps = ResearchDeps(client=client)

        result = await graph.run(state=research_state, deps=deps)

        # Build citations with stable indices from session state
        citations = []
        for i, c in enumerate(result.citations):
            if session_state is not None:
                index = session_state.get_or_assign_index(c.chunk_id)
            else:
                index = i + 1

            citations.append(
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

        qa_result = QAResult(
            question=question,
            answer=result.answer,
            confidence=result.confidence,
            citations=citations,
        )

        # Update session state with citations
        if session_state is not None:
            session_state.citations = citations

        # Update QA session state with history entry
        if qa_session_state is not None:
            qa_session_state.qa_history.append(
                QAHistoryEntry(
                    question=question,
                    answer=result.answer,
                    confidence=result.confidence,
                    citations=citations,
                )
            )
            # Enforce FIFO limit
            if len(qa_session_state.qa_history) > MAX_QA_HISTORY:
                qa_session_state.qa_history = qa_session_state.qa_history[
                    -MAX_QA_HISTORY:
                ]
            # Trigger background summarization
            trigger_background_summarization(
                qa_session_state=qa_session_state,
                config=config,
                session_id=session_id,
            )

        # Compute and return state delta if session state changed
        if session_state is not None and old_state_snapshot is not None:
            # Build new combined state snapshot
            new_state_snapshot = {
                "session_id": session_state.session_id,
                "document_filter": session_state.document_filter,
                "citation_registry": session_state.citation_registry,
                "citations": [c.model_dump() for c in session_state.citations],
            }
            if qa_session_state is not None:
                new_state_snapshot["qa_history"] = [
                    qa.model_dump() for qa in qa_session_state.qa_history
                ]
                if qa_session_state.session_context:
                    new_state_snapshot["session_context"] = SessionContext(
                        summary=qa_session_state.session_context
                    ).model_dump(mode="json")
                else:
                    new_state_snapshot["session_context"] = None

            state_event = compute_combined_state_delta(
                old_state_snapshot,
                new_state_snapshot,
                state_key=session_state.state_key,
            )

            # Format answer with citation references
            answer_text = result.answer
            if citations:
                citation_refs = " ".join(f"[{c.index}]" for c in citations)
                answer_text = f"{answer_text}\n\nSources: {citation_refs}"

            metadata = [state_event] if state_event is not None else None
            return ToolReturn(return_value=answer_text, metadata=metadata)

        return qa_result

    toolset = FunctionToolset()
    toolset.add_function(ask, name=tool_name)
    return toolset
