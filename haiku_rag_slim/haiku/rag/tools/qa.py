from pydantic import BaseModel
from pydantic_ai import FunctionToolset

from haiku.rag.agents.research.dependencies import ResearchContext
from haiku.rag.agents.research.graph import build_research_graph
from haiku.rag.agents.research.models import Citation, SearchAnswer
from haiku.rag.agents.research.state import ResearchDeps, ResearchState
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.tools.context import ToolContext
from haiku.rag.tools.filters import build_document_filter, combine_filters
from haiku.rag.tools.models import QAResult

QA_NAMESPACE = "haiku.rag.qa"


class QAState(BaseModel):
    """State for QA toolset.

    Tracks Q&A history across tool invocations.
    """

    history: list[QAResult] = []


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
            If provided, Q&A results are accumulated in QAState.
        base_filter: Optional base SQL WHERE clause applied to searches.
        tool_name: Name for the ask tool. Defaults to "ask".
        session_context: Optional session context for the research graph.
        prior_answers: Optional list of prior answers for context.

    Returns:
        FunctionToolset with an ask tool.
    """
    # Get or create QA state if context provided
    state: QAState | None = None
    if context is not None:
        state = context.get_or_create(QA_NAMESPACE, QAState)

    async def ask(
        question: str,
        document_name: str | None = None,
    ) -> QAResult:
        """Answer a question using the knowledge base.

        Uses a research graph for searching and synthesizing answers.

        Args:
            question: The question to answer.
            document_name: Optional document name/title to search within.

        Returns:
            QAResult with answer, confidence, and citations.
        """
        # Build filter from base_filter and document_name
        doc_filter = build_document_filter(document_name) if document_name else None
        effective_filter = combine_filters(base_filter, doc_filter)

        # Build and run the research graph
        graph = build_research_graph(config=config, output_mode="conversational")

        research_context = ResearchContext(
            original_question=question,
            session_context=session_context,
            qa_responses=prior_answers or [],
        )
        research_state = ResearchState(
            context=research_context,
            max_iterations=1,
            search_filter=effective_filter,
            max_concurrency=config.research.max_concurrency,
        )
        deps = ResearchDeps(client=client)

        result = await graph.run(state=research_state, deps=deps)

        # Convert to QAResult
        citations = [
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

        qa_result = QAResult(
            question=question,
            answer=result.answer,
            confidence=result.confidence,
            citations=citations,
        )

        # Accumulate in state if context provided
        if state is not None:
            state.history.append(qa_result)

        return qa_result

    toolset = FunctionToolset()
    toolset.add_function(ask, name=tool_name)
    return toolset
