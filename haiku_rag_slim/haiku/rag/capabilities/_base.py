import asyncio
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AgentToolset

from haiku.rag.capabilities._tools import CodeExecutionEntry, search_corpus
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig, ModelConfig
from haiku.rag.store.models.chunk import SearchResult
from haiku.rag.store.models.citation import Citation, resolve_citations
from haiku.rag.tools.search import build_binary_parts_from_results


def resolve_db_path(db_path: Path | None, config: AppConfig) -> Path:
    if db_path is not None:
        return db_path
    if env_db := os.environ.get("HAIKU_RAG_DB"):
        return Path(env_db).expanduser()
    return config.storage.data_dir / "haiku.rag.lancedb"


def _clear_invocation_state(state: BaseModel) -> None:
    for field_name in ("citations", "searches", "executions"):
        value = getattr(state, field_name, None)
        if hasattr(value, "clear"):
            value.clear()


def _compact_old_tool_returns(
    messages: list[ModelMessage], tool_names: frozenset[str]
) -> list[ModelMessage]:
    """Remove bulky prior-turn evidence while retaining the current turn.

    Tool call and return parts remain paired; only the old return payload is
    replaced. This keeps provider histories valid and preserves all evidence
    gathered since the most recent user prompt.
    """
    latest_user_message = -1
    for index, message in enumerate(messages):
        if isinstance(message, ModelRequest) and any(
            isinstance(part, UserPromptPart) for part in message.parts
        ):
            latest_user_message = index

    if latest_user_message < 0:
        return messages

    compacted = list(messages)
    for index, message in enumerate(messages[:latest_user_message]):
        if not isinstance(message, ModelRequest):
            continue
        parts = [
            replace(
                part,
                content="[Prior-turn RAG tool output removed; citations remain in state.]",
            )
            if isinstance(part, ToolReturnPart) and part.tool_name in tool_names
            else part
            for part in message.parts
        ]
        if parts != message.parts:
            compacted[index] = replace(message, parts=parts)
    return compacted


@dataclass
class RAGCapabilityBase[StateT: BaseModel](AbstractCapability[Any]):
    db_path: Path
    config: AppConfig
    state_type: type[StateT]
    state_namespace: str
    instruction_text: str
    model: ModelConfig
    tool_names: frozenset[str]
    request_limit: int | None = None
    state: StateT | None = field(default=None, repr=False)
    outer_state: dict[str, Any] | None = field(default=None, repr=False)
    rag: HaikuRAG | None = field(default=None, repr=False)
    rag_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    resource_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    search_count: int = field(default=0, repr=False)
    request_count: int = field(default=0, repr=False)

    async def for_run(self, ctx: RunContext[Any]) -> "RAGCapabilityBase[StateT]":
        outer = getattr(ctx.deps, "state", None)
        outer_state = outer if isinstance(outer, dict) else None
        raw_state = outer_state.get(self.state_namespace) if outer_state else None
        state = self.state_type.model_validate(raw_state or {})
        _clear_invocation_state(state)
        run_capability = replace(
            self,
            state=state,
            outer_state=outer_state,
            rag=None,
            rag_lock=asyncio.Lock(),
            resource_lock=asyncio.Lock(),
            search_count=0,
            request_count=0,
        )
        run_capability._sync_state()
        return run_capability

    def get_instructions(self) -> str:
        if self.config.prompts.domain_preamble:
            return f"{self.config.prompts.domain_preamble}\n\n{self.instruction_text}"
        return self.instruction_text

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        request_context.messages = _compact_old_tool_returns(
            request_context.messages, self.tool_names
        )
        if self._request_limit_reached:
            current_request = request_context.messages[-1]
            if isinstance(current_request, ModelRequest):
                instruction = (
                    f"The {self.state_namespace} capability has reached its request "
                    "limit. Its tools are no longer available. Give the best answer "
                    "possible using the evidence already gathered."
                )
                current_request.instructions = "\n\n".join(
                    part for part in (current_request.instructions, instruction) if part
                )
                parameters = request_context.model_request_parameters
                request_context.model_request_parameters = replace(
                    parameters,
                    instruction_parts=[
                        *(parameters.instruction_parts or []),
                        InstructionPart(content=instruction, dynamic=True),
                    ],
                )
        else:
            self.request_count += 1
        return request_context

    async def prepare_tools(
        self,
        ctx: RunContext[Any],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Remove only this capability's tools after its per-question limit."""
        if not self._request_limit_reached:
            return tool_defs
        return [tool for tool in tool_defs if tool.capability_id != self.id]

    @property
    def _request_limit_reached(self) -> bool:
        return (
            self.request_limit is not None and self.request_count >= self.request_limit
        )

    async def after_run(
        self, ctx: RunContext[Any], *, result: AgentRunResult[Any]
    ) -> AgentRunResult[Any]:
        await self._close()
        return result

    async def on_run_error(
        self, ctx: RunContext[Any], *, error: BaseException
    ) -> AgentRunResult[Any]:
        await self._close()
        raise error

    async def _ensure_rag(self) -> HaikuRAG:
        if self.rag is None:
            async with self.resource_lock:
                if self.rag is None:
                    rag = HaikuRAG(self.db_path, config=self.config, read_only=True)
                    await rag.__aenter__()
                    self.rag = rag
        return self.rag

    async def _close(self) -> None:
        if self.rag is not None:
            await self.rag.__aexit__(None, None, None)
            self.rag = None

    def _sync_state(self) -> None:
        if self.outer_state is not None and self.state is not None:
            self.outer_state[self.state_namespace] = self.state.model_dump(mode="json")

    async def _with_state(self, operation: Any) -> Any:
        """Execute an operation and copy its state back to the host dependencies."""
        result = await operation
        self._sync_state()
        return result

    async def _search(self, query: str, limit: int | None) -> str | ToolReturn:
        assert self.state is not None
        self.search_count += 1
        if self.search_count > self.config.qa.max_searches:
            return (
                "Search limit reached. Answer the question using "
                "the results you already have."
            )
        async with self.rag_lock:
            formatted, results = await search_corpus(
                await self._ensure_rag(),
                query,
                limit=limit,
                document_filter=getattr(self.state, "document_filter", None),
            )
        state = cast(Any, self.state)
        state.searches[query] = results
        if self.model.vision and (parts := build_binary_parts_from_results(results)):
            return ToolReturn(return_value=formatted, content=parts)
        return formatted

    async def _cite(self, chunk_ids: list[str]) -> str:
        assert self.state is not None
        if not chunk_ids:
            return "Registered 0 citations (empty chunk_ids)."

        all_results: list[SearchResult] = []
        state = cast(Any, self.state)
        for results in state.searches.values():
            all_results.extend(results)
        citations = resolve_citations(chunk_ids, all_results)
        resolved = {citation.chunk_id for citation in citations}
        missing = [
            cid.strip("[]") for cid in chunk_ids if cid.strip("[]") not in resolved
        ]

        if missing:
            async with self.rag_lock:
                rag = await self._ensure_rag()
                synthetic: list[SearchResult] = []
                documents: dict[str, Any] = {}
                for chunk_id in missing:
                    chunk = await rag.get_chunk_by_id(chunk_id)
                    if chunk is None or not chunk.document_id:
                        continue
                    document = documents.get(chunk.document_id)
                    if chunk.document_id not in documents:
                        document = await rag.get_document_by_id(chunk.document_id)
                        documents[chunk.document_id] = document
                    chunk.document_uri = document.uri if document else None
                    chunk.document_title = document.title if document else None
                    chunk.document_meta = document.metadata if document else {}
                    synthetic.append(SearchResult.from_chunk(chunk, score=1.0))
                citations.extend(resolve_citations(missing, synthetic))

        if not citations:
            raise ModelRetry(
                f"None of the supplied chunk_ids {list(chunk_ids)} could be resolved. "
                "Copy chunk_ids verbatim from search results."
            )
        self._register_citations(citations)
        return f"Registered {len(citations)} citation(s)."

    def _register_citations(self, citations: list[Citation]) -> None:
        assert self.state is not None
        state = cast(Any, self.state)
        next_index = len(state.citation_index) + 1
        for citation in citations:
            if citation.chunk_id not in state.citation_index:
                citation.index = next_index
                next_index += 1
                state.citation_index[citation.chunk_id] = citation
            if citation.chunk_id not in state.citations:
                state.citations.append(citation.chunk_id)

    def get_toolset(self) -> AgentToolset[Any] | None:
        raise NotImplementedError


__all__ = [
    "CodeExecutionEntry",
    "RAGCapabilityBase",
    "resolve_db_path",
]
