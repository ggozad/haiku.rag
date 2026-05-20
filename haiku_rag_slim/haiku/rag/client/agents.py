from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG
    from haiku.rag.sandbox import AnalysisResult
    from haiku.rag.store.models.citation import Citation


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
