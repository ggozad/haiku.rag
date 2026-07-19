from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic_ai import Agent
from pydantic_ai.models import Model

from haiku.rag.capabilities import RAGCapabilityBase
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models.chunk import SearchResult
from haiku.rag.store.models.citation import Citation

CapabilityFactory = Callable[..., RAGCapabilityBase[Any]]


class _RagLikeState(Protocol):
    document_filter: str | None
    citation_index: dict[str, Citation]
    citations: list[str]
    searches: dict[str, list[SearchResult]]


@dataclass
class CapabilityRunResult:
    answer: str
    cited_uris: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)
    searched_uris: list[str] = field(default_factory=list)
    n_searches: int = 0
    n_executions: int = 0


@dataclass
class _EvalDeps:
    state: dict[str, Any] = field(default_factory=dict)


async def run_capability_question(
    capability_factory: CapabilityFactory,
    db_path: Path,
    config: AppConfig,
    question: str,
    capability_model: str | Model,
    document_filter: str | None = None,
    request_limit: int | None = None,
) -> CapabilityRunResult:
    """Run a single question through a capability and return answer + retrieval data.

    Builds a native capability via ``capability_factory(db_path=..., config=...)``.
    After the run, citations and searched documents
    are extracted from the state for downstream eval scoring.

    The capability must produce a state with RAG-capability-shaped fields (citation
    index, searches, optional document filter) — i.e. ``RAGState`` or
    ``AnalysisState`` from ``haiku.rag.capabilities``.
    """
    capability = capability_factory(
        db_path=db_path,
        config=config,
        defer_loading=False,
    )
    if request_limit is not None:
        capability.request_limit = request_limit
    state = capability.state_type()
    typed = cast(_RagLikeState, state)
    if document_filter is not None:
        typed.document_filter = document_filter

    deps = _EvalDeps(state={capability.state_namespace: state.model_dump(mode="json")})
    agent = Agent(
        capability_model,
        deps_type=_EvalDeps,
        capabilities=[capability],
    )
    agent_result = await agent.run(question, deps=deps)
    state = capability.state_type.model_validate(deps.state[capability.state_namespace])
    typed = cast(_RagLikeState, state)

    cited_chunk_ids: list[str] = list(typed.citations)
    seen_cited: set[str] = set()
    cited_uris: list[str] = []
    for chunk_id in cited_chunk_ids:
        citation = typed.citation_index.get(chunk_id)
        if citation is None:
            continue
        if citation.document_uri not in seen_cited:
            seen_cited.add(citation.document_uri)
            cited_uris.append(citation.document_uri)

    seen_searched: set[str] = set()
    searched_uris: list[str] = []
    for results in typed.searches.values():
        for search_result in results:
            uri = search_result.document_uri
            if uri and uri not in seen_searched:
                seen_searched.add(uri)
                searched_uris.append(uri)

    executions = getattr(state, "executions", None)
    n_executions = len(executions) if executions is not None else 0

    return CapabilityRunResult(
        answer=agent_result.output,
        cited_uris=cited_uris,
        cited_chunk_ids=cited_chunk_ids,
        searched_uris=searched_uris,
        n_searches=len(typed.searches),
        n_executions=n_executions,
    )
