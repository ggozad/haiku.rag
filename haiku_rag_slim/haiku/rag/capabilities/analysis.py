from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import RunContext, ToolFailed
from pydantic_ai.messages import ToolReturn
from pydantic_ai.toolsets import FunctionToolset

from haiku.rag.capabilities._base import (
    CodeExecutionEntry,
    RAGCapabilityBase,
    resolve_db_path,
)
from haiku.rag.config.models import AppConfig
from haiku.rag.sandbox import AnalysisContext, Sandbox
from haiku.rag.store.models.chunk import SearchResult
from haiku.rag.store.models.citation import Citation

STATE_NAMESPACE = "analysis"
_CAPABILITY_ID = "haiku-rag-analysis"
_TOOL_NAMES = frozenset({"analysis_search", "analysis_execute_code", "analysis_cite"})
_instructions_path = Path(__file__).parent / "instructions" / "analysis.md"


class AnalysisState(BaseModel):
    document_filter: str | None = None
    executions: list[CodeExecutionEntry] = Field(default_factory=list)
    citation_index: dict[str, Citation] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
    searches: dict[str, list[SearchResult]] = Field(default_factory=dict)


@cache
def instructions() -> str:
    return _instructions_path.read_text().strip()


@dataclass
class AnalysisCapability(RAGCapabilityBase[AnalysisState]):
    """Deferred capability for sandboxed computation over a RAG corpus."""

    sandbox: Sandbox | None = field(default=None, repr=False)
    execute_count: int = field(default=0, repr=False)

    async def for_run(self, ctx: RunContext[Any]) -> "AnalysisCapability":
        capability = await super().for_run(ctx)
        assert isinstance(capability, AnalysisCapability)
        capability.sandbox = None
        capability.execute_count = 0
        return capability

    async def _ensure_sandbox(self) -> Sandbox:
        if self.sandbox is None:
            rag = await self._ensure_rag()
            assert self.state is not None
            self.sandbox = Sandbox(
                db_path=self.db_path,
                config=self.config,
                context=AnalysisContext(filter=self.state.document_filter),
                rag=rag,
                lock=self.rag_lock,
            )
        return self.sandbox

    async def _close(self) -> None:
        if self.sandbox is not None:
            await self.sandbox.close()
            self.sandbox = None
        await super()._close()

    def _spent_tool_names(self) -> set[str]:
        spent = super()._spent_tool_names()
        if self.execute_count >= self.config.analysis.max_executions:
            spent.add("analysis_execute_code")
        return spent

    async def _execute_code(self, code: str) -> str:
        assert self.state is not None
        self.execute_count += 1
        if self.execute_count > self.config.analysis.max_executions:
            raise ToolFailed(
                "Code-execution limit reached. Give your final answer now from what "
                "you already have; do not call analysis_execute_code again."
            )
        sandbox = await self._ensure_sandbox()
        result = await sandbox.execute(code)
        if sandbox._search_results:
            existing = self.state.searches.get("_sandbox", [])
            seen = {item.chunk_id for item in existing}
            for item in sandbox._search_results:
                if item.chunk_id not in seen:
                    existing.append(item)
                    seen.add(item.chunk_id)
            self.state.searches["_sandbox"] = existing
        self.state.executions.append(
            CodeExecutionEntry(
                code=code,
                stdout=result.stdout,
                stderr=result.stderr,
                success=result.success,
            )
        )
        if not result.success:
            raise ToolFailed(f"{result.stderr}\n\nOutput: {result.stdout}")
        return result.stdout or "No output."

    def get_toolset(self) -> FunctionToolset[Any]:
        async def analysis_search(
            ctx: RunContext[Any], query: str, limit: int | None = None
        ) -> str | ToolReturn:
            """Search the knowledge base for evidence to analyze."""
            return await self._with_state(self._search(query, limit))

        async def analysis_execute_code(ctx: RunContext[Any], code: str) -> Any:
            """Execute Python against the sandboxed document filesystem."""
            return await self._with_state(self._execute_code(code))

        async def analysis_cite(ctx: RunContext[Any], chunk_ids: list[str]) -> Any:
            """Register exact retrieved chunk IDs as citations for the answer."""
            return await self._with_state(self._cite(chunk_ids))

        return FunctionToolset(
            [analysis_search, analysis_execute_code, analysis_cite],
            id=_CAPABILITY_ID,
            max_retries=3,
            sequential=True,
        )


def create_capability(
    db_path: Path | None = None,
    config: AppConfig | None = None,
    *,
    defer_loading: bool = True,
    request_limit: int | None = 30,
    vision: bool | None = None,
) -> AnalysisCapability:
    """Create a native Pydantic AI analysis capability.

    ``vision`` gates whether picture chunks are attached to search results as
    images, and should reflect the model the hosting agent actually runs.
    Defaults to ``config.analysis.model.vision`` (falling back to
    ``config.qa.model.vision``).
    """
    if config is None:
        from haiku.rag.config import get_config

        config = get_config()
    analysis_model = config.analysis.model or config.qa.model
    return AnalysisCapability(
        db_path=resolve_db_path(db_path, config),
        config=config,
        state_type=AnalysisState,
        state_namespace=STATE_NAMESPACE,
        instruction_text=instructions(),
        vision=analysis_model.vision if vision is None else vision,
        tool_names=_TOOL_NAMES,
        request_limit=request_limit,
        id=_CAPABILITY_ID,
        description=(
            "Analyze the haiku.rag corpus with search and sandboxed Python code. "
            "Use for counting, aggregation, statistics, data traversal, comparison "
            "across documents, and other tasks best solved by writing Python code."
        ),
        defer_loading=defer_loading,
    )


__all__ = [
    "AnalysisCapability",
    "AnalysisState",
    "STATE_NAMESPACE",
    "create_capability",
    "instructions",
]
