from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from haiku.rag.agents.research.models import Citation, ResearchReport
    from haiku.rag.client import HaikuRAG
    from haiku.rag.sandbox import AnalysisResult


async def ask(
    client: "HaikuRAG",
    question: str,
    filter: str | None = None,
) -> "tuple[str, list[Citation]]":
    """Ask a question against the knowledge base via the rag skill.

    Args:
        client: The HaikuRAG client.
        question: The question to ask.
        filter: SQL WHERE clause to filter documents.

    Returns:
        Tuple of (answer text, list of resolved citations).
    """
    from haiku.rag.skills.rag import RAGState, create_skill
    from haiku.rag.utils import get_model
    from haiku.skills import run_skill

    skill = create_skill(db_path=client.store.db_path, config=client._config)
    state = RAGState(document_filter=filter)
    model = get_model(client._config.qa.model, client._config)
    answer, _, _ = await run_skill(model, skill, question, state=state)
    citations = [
        state.citation_index[cid]
        for cid in state.citations
        if cid in state.citation_index
    ]
    return answer, citations


async def research(
    client: "HaikuRAG",
    question: str,
    *,
    filter: str | None = None,
    max_iterations: int | None = None,
) -> "ResearchReport":
    """Run multi-agent research to investigate a question.

    Args:
        client: The HaikuRAG client.
        question: The research question to investigate.
        filter: SQL WHERE clause to filter documents.
        max_iterations: Override max iterations (None uses config default).

    Returns:
        ResearchReport with structured findings.
    """
    from haiku.rag.agents.research.dependencies import ResearchContext
    from haiku.rag.agents.research.graph import build_research_graph
    from haiku.rag.agents.research.state import ResearchDeps, ResearchState

    graph = build_research_graph(config=client._config)
    context = ResearchContext(original_question=question)
    state = ResearchState.from_config(
        context=context, config=client._config, max_iterations=max_iterations
    )
    state.search_filter = filter
    deps = ResearchDeps(client=client)

    return await graph.run(state=state, deps=deps)


async def analyze(
    client: "HaikuRAG",
    question: str,
    filter: str | None = None,
) -> "AnalysisResult":
    """Answer a question against the knowledge base via the rag-analysis skill.

    The analysis skill exposes ``search``, ``execute_code``, and ``cite`` tools.
    The driving model decides when to reach for code (structural traversal,
    computation, aggregation) versus a direct ``search → cite → answer``.

    Args:
        client: The HaikuRAG client.
        question: The question to answer.
        filter: SQL WHERE clause to filter documents during searches.

    Returns:
        AnalysisResult with the answer and resolved citations.
    """
    from haiku.rag.sandbox import AnalysisResult
    from haiku.rag.skills.analysis import AnalysisState, create_skill
    from haiku.rag.utils import get_model
    from haiku.skills import run_skill

    skill = create_skill(db_path=client.store.db_path, config=client._config)
    state = AnalysisState(document_filter=filter)
    model = get_model(
        client._config.analysis.model or client._config.qa.model, client._config
    )
    answer, _, _ = await run_skill(model, skill, question, state=state)
    citations = [
        state.citation_index[cid]
        for cid in state.citations
        if cid in state.citation_index
    ]
    return AnalysisResult(answer=answer, citations=citations)
