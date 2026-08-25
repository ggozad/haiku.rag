from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent

if TYPE_CHECKING:
    from pydantic_ai.messages import BinaryContent

    from haiku.rag.client import HaikuRAG
    from haiku.rag.config.models import ModelConfig
    from haiku.rag.sandbox import AnalysisResult
    from haiku.rag.store.models.citation import Citation


@dataclass
class _AgentDeps:
    state: dict[str, Any] = field(default_factory=dict)


def _build_user_prompt(
    question: str,
    images: Sequence[bytes] | None,
    model_config: "ModelConfig",
) -> "str | list[str | BinaryContent]":
    if not images:
        return question
    if not model_config.vision:
        raise ValueError(
            f"Model {model_config.provider}:{model_config.name} is not configured "
            "for vision (set `vision: true` on the model config to pass images)."
        )
    from haiku.rag.utils import image_binary_content

    return [question, *(image_binary_content(data) for data in images)]


async def ask(
    client: "HaikuRAG",
    question: str,
    filter: str | None = None,
    images: Sequence[bytes] | None = None,
    sources: list[str] | None = None,
) -> "tuple[str, list[Citation]]":
    """Ask a question against the knowledge base via the RAG capability.

    Args:
        client: The HaikuRAG client.
        question: The question to ask.
        filter: SQL WHERE clause to filter documents.
        images: Raw image bytes attached to the question (requires a
            vision-capable QA model).
        sources: Names of the databases to ask across. None asks across every
            configured database.

    Returns:
        Tuple of (answer text, list of resolved citations).
    """
    from haiku.rag.capabilities.rag import (
        AGENT_PREAMBLE,
        RAGState,
        create_capability,
    )
    from haiku.rag.utils import get_model

    # No `db_path`: the lent client is what the capability reads through, and it
    # already knows which databases that is.
    capability = create_capability(
        config=client._config,
        rag=client,
        defer_loading=False,
    )
    deps = _AgentDeps(
        state={
            "rag": RAGState(document_filter=filter, sources=sources).model_dump(
                mode="json"
            )
        }
    )
    user_prompt = _build_user_prompt(question, images, client._config.qa.model)
    model = get_model(client._config.qa.model, client._config)
    agent = Agent(
        model,
        deps_type=_AgentDeps,
        instructions=AGENT_PREAMBLE,
        capabilities=[capability],
    )
    result = await agent.run(user_prompt, deps=deps)
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
    images: Sequence[bytes] | None = None,
    sources: list[str] | None = None,
) -> "AnalysisResult":
    """Answer a question using the analysis capability.

    The capability exposes search, code execution, and citation tools.
    The driving model decides when to reach for code (structural traversal,
    computation, aggregation) versus a direct ``search → cite → answer``.

    Args:
        client: The HaikuRAG client.
        question: The question to answer.
        filter: SQL WHERE clause to filter documents during searches.
        images: Raw image bytes attached to the question (requires a
            vision-capable analysis model).
        sources: Names of the databases to analyze across. None covers every
            configured database.

    Returns:
        AnalysisResult with the answer and resolved citations.
    """
    from haiku.rag.capabilities.analysis import AnalysisState, create_capability
    from haiku.rag.sandbox import AnalysisResult
    from haiku.rag.utils import get_model

    # No `db_path`: the lent client is what the capability reads through, and it
    # already knows which databases that is.
    capability = create_capability(
        config=client._config,
        rag=client,
        defer_loading=False,
    )
    deps = _AgentDeps(
        state={
            "analysis": AnalysisState(
                document_filter=filter, sources=sources
            ).model_dump(mode="json")
        }
    )
    model_config = client._config.analysis.model or client._config.qa.model
    user_prompt = _build_user_prompt(question, images, model_config)
    model = get_model(model_config, client._config)
    agent = Agent(
        model,
        deps_type=_AgentDeps,
        capabilities=[capability],
    )
    result = await agent.run(user_prompt, deps=deps)
    state = AnalysisState.model_validate(deps.state["analysis"])
    citations = [
        state.citation_index[cid]
        for cid in state.citations
        if cid in state.citation_index
    ]
    return AnalysisResult(answer=result.output, citations=citations)
