import math
from collections.abc import Callable

from ag_ui.core import EventType, StateSnapshotEvent
from pydantic import BaseModel, Field
from pydantic_ai import FunctionToolset, RunContext, ToolReturn

from haiku.rag.agents.research.dependencies import ResearchContext
from haiku.rag.agents.research.graph import build_research_graph
from haiku.rag.agents.research.models import Citation, SearchAnswer
from haiku.rag.agents.research.state import ResearchDeps, ResearchState
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.embeddings import get_embedder
from haiku.rag.tools.context import RAGDeps, ToolContext
from haiku.rag.tools.filters import (
    build_document_filter,
    combine_filters,
    get_session_filter,
)
from haiku.rag.tools.models import QAResult
from haiku.rag.tools.session import (
    SESSION_NAMESPACE,
    SessionContext,
    SessionState,
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


class QASessionState(BaseModel):
    """Extended session state for QA with embedding cache."""

    qa_history: list[QAHistoryEntry] = []
    session_context: SessionContext | None = None


QA_SESSION_NAMESPACE = "haiku.rag.qa_session"
MAX_QA_HISTORY = 50


async def run_qa_core(
    client: HaikuRAG,
    config: AppConfig,
    question: str,
    document_name: str | None = None,
    *,
    context: ToolContext | None = None,
    base_filter: str | None = None,
    session_context: str | None = None,
    prior_answers: list[SearchAnswer] | None = None,
    on_qa_complete: Callable[[QASessionState, AppConfig], None] | None = None,
) -> QAResult:
    """Run the QA flow and return a QAResult.

    This is the core QA implementation shared by toolsets and client APIs.
    It updates session state and QA history when context is provided.
    """
    session_state: SessionState | None = None
    qa_session_state: QASessionState | None = None

    if context is not None:
        session_state = context.get(SESSION_NAMESPACE, SessionState)
        qa_session_state = context.get(QA_SESSION_NAMESPACE, QASessionState)

    doc_filter = build_document_filter(document_name) if document_name else None
    effective_filter = combine_filters(
        get_session_filter(context, base_filter), doc_filter
    )

    effective_session_context = session_context
    if qa_session_state is not None and qa_session_state.session_context is not None:
        effective_session_context = qa_session_state.session_context.summary

    effective_prior_answers = prior_answers or []
    if qa_session_state is not None and qa_session_state.qa_history:
        embedder = get_embedder(config)
        question_embedding = await embedder.embed_query(question)

        to_embed = []
        to_embed_indices = []
        for i, qa in enumerate(qa_session_state.qa_history):
            if qa.question_embedding is None:
                to_embed.append(qa.question)
                to_embed_indices.append(i)

        if to_embed:
            new_embeddings = await embedder.embed_documents(to_embed)
            for i, idx in enumerate(to_embed_indices):
                qa_session_state.qa_history[idx].question_embedding = new_embeddings[i]

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

    if session_state is not None:
        session_state.citations = citations
        session_state.citations_history.append(citations)

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
            qa_session_state.qa_history = qa_session_state.qa_history[-MAX_QA_HISTORY:]
        if on_qa_complete is not None:
            on_qa_complete(qa_session_state, config)

    return qa_result


def create_qa_toolset(
    config: AppConfig,
    base_filter: str | None = None,
    tool_name: str = "ask",
    on_ask_complete: Callable[[QASessionState, AppConfig], None] | None = None,
) -> FunctionToolset:
    """Create a toolset with Q&A capabilities using research graph.

    Args:
        config: Application configuration.
        base_filter: Optional base SQL WHERE clause applied to searches.
        tool_name: Name for the ask tool. Defaults to "ask".
        on_ask_complete: Optional callback invoked after each QA cycle with
            the updated QASessionState and config. Use this to trigger
            background summarization or other post-processing.

    Returns:
        FunctionToolset with an ask tool.
    """

    async def ask(
        ctx: RunContext[RAGDeps],
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
        client = ctx.deps.client
        tool_context = ctx.deps.tool_context

        state_key: str | None = None

        if tool_context is not None:
            state_key = tool_context.state_key

        qa_result = await run_qa_core(
            client=client,
            config=config,
            question=question,
            document_name=document_name,
            context=tool_context,
            base_filter=base_filter,
            on_qa_complete=on_ask_complete,
        )

        if tool_context is not None and tool_context.namespaces:
            snapshot = tool_context.build_state_snapshot()
            if state_key:
                snapshot = {state_key: snapshot}

            answer_text = qa_result.answer
            if qa_result.citations:
                citation_refs = " ".join(f"[{c.index}]" for c in qa_result.citations)
                answer_text = f"{answer_text}\n\nSources: {citation_refs}"

            state_event = StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=snapshot,
            )
            return ToolReturn(return_value=answer_text, metadata=[state_event])

        return qa_result

    toolset = FunctionToolset()
    toolset.add_function(ask, name=tool_name)
    return toolset
