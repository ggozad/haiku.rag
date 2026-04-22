from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from haiku.rag.config.models import AppConfig
from haiku.skills.state import SkillRunDeps

if TYPE_CHECKING:
    from haiku.rag.agents.analysis.sandbox import Sandbox
    from haiku.rag.client import HaikuRAG


@dataclass
class RAGRunDeps(SkillRunDeps):
    rag: "HaikuRAG | None" = None
    search_count: int = 0


@dataclass
class AnalysisRunDeps(RAGRunDeps):
    sandbox: "Sandbox | None" = None


def _reset_invocation_state(state: Any) -> None:
    """Clear state fields scoped to a single invocation.

    Keeps ``citation_index`` (accumulates resolved citations across the session
    for lookup) and ``document_filter`` (session-level). Clears ``citations``,
    ``searches``, and (for analysis) ``executions``.
    """
    if state is None:
        return
    citations = getattr(state, "citations", None)
    if citations is not None:
        citations.clear()
    searches = getattr(state, "searches", None)
    if searches is not None:
        searches.clear()
    executions = getattr(state, "executions", None)
    if executions is not None:
        executions.clear()


def make_rag_lifespan(db_path: Path, config: AppConfig):
    @asynccontextmanager
    async def lifespan(deps: RAGRunDeps) -> AsyncIterator[None]:
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(db_path, config=config, read_only=True) as rag:
            deps.rag = rag
            deps.search_count = 0
            _reset_invocation_state(deps.state)
            yield

    return lifespan


def make_analysis_lifespan(db_path: Path, config: AppConfig):
    @asynccontextmanager
    async def lifespan(deps: AnalysisRunDeps) -> AsyncIterator[None]:
        from haiku.rag.agents.analysis.dependencies import AnalysisContext
        from haiku.rag.agents.analysis.sandbox import Sandbox
        from haiku.rag.client import HaikuRAG

        doc_filter = getattr(deps.state, "document_filter", None)
        async with HaikuRAG(db_path, config=config, read_only=True) as rag:
            deps.rag = rag
            deps.search_count = 0
            deps.sandbox = Sandbox(
                db_path=db_path,
                config=config,
                context=AnalysisContext(filter=doc_filter),
            )
            _reset_invocation_state(deps.state)
            yield

    return lifespan
