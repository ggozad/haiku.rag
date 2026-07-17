from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG
    from haiku.rag.sandbox import AnalysisResult
    from haiku.rag.store.models.citation import Citation


@dataclass
class _AgentDeps:
    state: dict[str, Any] = field(default_factory=dict)


async def ask(
    client: "HaikuRAG",
    question: str,
    filter: str | None = None,
) -> "tuple[str, list[Citation]]":
    """Ask a question against the knowledge base via the RAG capability.

    Args:
        client: The HaikuRAG client.
        question: The question to ask.
        filter: SQL WHERE clause to filter documents.

    Returns:
        Tuple of (answer text, list of resolved citations).
    """
    from haiku.rag.capabilities.rag import (
        AGENT_PREAMBLE,
        RAGState,
        create_capability,
    )
    from haiku.rag.utils import get_model

    capability = create_capability(
        db_path=client.store.db_path,
        config=client._config,
        defer_loading=False,
    )
    deps = _AgentDeps(
        state={"rag": RAGState(document_filter=filter).model_dump(mode="json")}
    )
    model = get_model(client._config.qa.model, client._config)
    agent = Agent(
        model,
        deps_type=_AgentDeps,
        instructions=AGENT_PREAMBLE,
        capabilities=[capability],
    )
    result = await agent.run(question, deps=deps)
    state = RAGState.model_validate(deps.state["rag"])
    citations = [
        state.citation_index[cid]
        for cid in state.citations
        if cid in state.citation_index
    ]
    return result.output, citations


async def analyze(
    client: "HaikuRAG",
    question: str,
    filter: str | None = None,
) -> "AnalysisResult":
    """Answer a question using the analysis capability.

    The capability exposes search, code execution, and citation tools.
    The driving model decides when to reach for code (structural traversal,
    computation, aggregation) versus a direct ``search → cite → answer``.

    Args:
        client: The HaikuRAG client.
        question: The question to answer.
        filter: SQL WHERE clause to filter documents during searches.

    Returns:
        AnalysisResult with the answer and resolved citations.
    """
    from haiku.rag.capabilities.analysis import AnalysisState, create_capability
    from haiku.rag.sandbox import AnalysisResult
    from haiku.rag.utils import get_model

    capability = create_capability(
        db_path=client.store.db_path,
        config=client._config,
        defer_loading=False,
    )
    deps = _AgentDeps(
        state={
            "analysis": AnalysisState(document_filter=filter).model_dump(mode="json")
        }
    )
    model = get_model(
        client._config.analysis.model or client._config.qa.model, client._config
    )
    agent = Agent(
        model,
        deps_type=_AgentDeps,
        capabilities=[capability],
    )
    result = await agent.run(
        question,
        deps=deps,
        usage_limits=UsageLimits(request_limit=capability.default_request_limit),
    )
    state = AnalysisState.model_validate(deps.state["analysis"])
    citations = [
        state.citation_index[cid]
        for cid in state.citations
        if cid in state.citation_index
    ]
    return AnalysisResult(answer=result.output, citations=citations)
