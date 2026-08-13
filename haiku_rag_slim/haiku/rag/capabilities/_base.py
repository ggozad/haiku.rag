import asyncio
import os
from dataclasses import dataclass, field, replace
from difflib import get_close_matches
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel
from pydantic_ai import ModelRetry, RunContext, ToolFailed
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturn,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AgentToolset

from haiku.rag.capabilities._tools import CodeExecutionEntry, search_corpus
from haiku.rag.capabilities.ledger import CapabilityEvidenceRecord, EvidenceRef
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models.chunk import SearchResult
from haiku.rag.store.models.citation import Citation, resolve_citations
from haiku.rag.tools.search import build_image_content_from_results

CITATION_GRACE_REQUESTS = 2
"""Requests calling this capability's tools that its cite tool outlives the rest by.

A loop guard, not a budget: cite consumes no retry budget and raises nothing, so
left available forever a stuck model calls it until the agent's own request limit
raises ``UsageLimitExceeded`` and the question returns no answer at all. Only
engagement can loop, which is why other capabilities' turns do not spend it.
"""

CHUNK_ID_MATCH_CUTOFF = 0.75
"""Similarity a cited chunk id needs to be treated as a corrupted known id.

Calibration knob. Two unrelated UUID4s reach about 0.5, while dropping or
duplicating a character or a whole group stays above 0.75, so the gap is wide.
"""


def _nearest_known_id(chunk_id: str, known_ids: list[str]) -> str:
    """Recover a chunk id the model damaged while transcribing it.

    Models copying opaque UUIDs drop and duplicate characters and whole
    hyphen-separated groups. Candidates are limited to ids the run actually
    retrieved, so a wrong match needs both a near miss and a same-run neighbour.
    Ids that match nothing are returned unchanged for the caller to report.
    """
    if not known_ids or chunk_id in known_ids:
        return chunk_id
    match = get_close_matches(chunk_id, known_ids, n=1, cutoff=CHUNK_ID_MATCH_CUTOFF)
    return match[0] if match else chunk_id


def resolve_db_path(db_path: Path | None, config: AppConfig) -> Path:
    if db_path is not None:
        return db_path
    if env_db := os.environ.get("HAIKU_RAG_DB"):
        return Path(env_db).expanduser()
    return config.storage.data_dir / "haiku.rag.lancedb"


def _clear_invocation_state(state: BaseModel) -> None:
    """Drop the working evidence of the previous question.

    Only ever called when a new question starts. A resumption keeps it: the
    results belong to the question still being answered, and dropping them leaves
    a later citation unable to resolve against the expanded result the model saw,
    recording no provenance for it.
    """
    for field_name in ("citations", "searches", "executions"):
        value = getattr(state, field_name, None)
        if hasattr(value, "clear"):
            value.clear()


def _is_resumption(prompt: Any, messages: list[ModelMessage]) -> bool:
    """Whether this run continues a question rather than asking a new one.

    Two signals, either of which is enough, because getting this wrong hands the
    model a notice where its own evidence should be:

    - no prompt: how pydantic-ai resumes for interruptions and suspensions.
    - an unfinished tail: the history ends with a request the model has not
      answered, or with a response whose tool calls have no returns yet. Deferred
      tool results may arrive *with* a prompt, so the prompt alone is not enough.

    A settled history ends with the previous answer, so a genuinely new question
    is not mistaken for a continuation. The framework's own first-new-message
    index would be better than either signal, but it is not public here.
    """
    if prompt is None:
        return True
    if not messages:
        return False
    last = messages[-1]
    if isinstance(last, ModelRequest):
        return True
    return any(isinstance(part, ToolCallPart) for part in last.parts)


def _called_own_tool(messages: list[ModelMessage], tool_names: frozenset[str]) -> bool:
    """Whether the model's most recent response called one of these tools."""
    for message in reversed(messages):
        if isinstance(message, ModelResponse):
            return any(
                isinstance(part, ToolCallPart) and part.tool_name in tool_names
                for part in message.parts
            )
    return False


@dataclass
class RAGCapabilityBase[StateT: BaseModel](AbstractCapability[Any]):
    db_path: Path
    config: AppConfig
    state_type: type[StateT]
    state_namespace: str
    instruction_text: str
    vision: bool
    tool_names: frozenset[str]
    request_limit: int | None = None
    state: StateT | None = field(default=None, repr=False)
    outer_state: dict[str, Any] | None = field(default=None, repr=False)
    rag: HaikuRAG | None = field(default=None, repr=False)
    rag_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    resource_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    search_count: int = field(default=0, repr=False)
    request_count: int = field(default=0, repr=False)
    grace_requests_used: int = field(default=0, repr=False)
    epoch: int = field(default=0, repr=False)

    async def for_run(self, ctx: RunContext[Any]) -> "RAGCapabilityBase[StateT]":
        """Start a run's own copy, and settle which question it is answering.

        A new question takes the message count as its identity, which every
        participant derives identically from the same history. A resumption keeps
        the identity already recorded: the question is the one in progress, and
        adopting the current count would relabel it as a new one and judge its
        declarations against the wrong question. A resumption with no recorded
        identity is a state this design does not produce, so it is reported rather
        than guessed at. With no history at all there is nothing in progress: an
        absent prompt is then an instructions-only first question, which takes an
        identity like any other.
        """
        outer = getattr(ctx.deps, "state", None)
        outer_state = outer if isinstance(outer, dict) else None
        raw_state = outer_state.get(self.state_namespace) if outer_state else None
        resuming = _is_resumption(ctx.prompt, ctx.messages)
        continuing = resuming and bool(ctx.messages)
        state = self.state_type.model_validate(raw_state or {})
        record = cast(CapabilityEvidenceRecord, cast(Any, state).evidence)
        if continuing and record.question is None:
            raise RuntimeError(
                f"The {self.state_namespace} capability is resuming a question with "
                "no stored question identity. Capabilities cannot be added, removed "
                "or migrated while a question is unfinished, and the run's state "
                "must be carried between its runs."
            )
        if not continuing:
            _clear_invocation_state(state)
            record.begin_question(len(ctx.messages))
        run_capability = replace(
            self,
            state=state,
            outer_state=outer_state,
            rag=None,
            rag_lock=asyncio.Lock(),
            resource_lock=asyncio.Lock(),
            search_count=0,
            request_count=0,
            grace_requests_used=0,
            epoch=0,
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
        self.epoch = len(ctx.messages)
        if instruction := self._budget_notice():
            current_request = request_context.messages[-1]
            if isinstance(current_request, ModelRequest):
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
        if self._request_limit_reached and _called_own_tool(
            request_context.messages, self.tool_names
        ):
            self.grace_requests_used += 1
        self.request_count += 1
        return request_context

    def _budget_notice(self) -> str | None:
        """Tell the model which of this capability's budgets just ran out.

        Never names a tool ``prepare_tools`` has already withdrawn: pointing the
        model at a tool that is gone costs it the agent's unknown-tool retry
        budget and can abort the run.
        """
        if self._citation_grace_expired:
            return (
                f"The {self.state_namespace} capability's tools are no longer "
                "available. Give the best answer possible using the evidence "
                "already gathered."
            )
        if self._request_limit_reached:
            return (
                f"The {self.state_namespace} capability has reached its request "
                f"limit. Only {self._cite_tool_name} remains among its tools: "
                "register the chunk_ids supporting your answer, then answer from "
                "the evidence already gathered."
            )
        if spent := self._spent_tool_names():
            names = ", ".join(sorted(spent))
            if remaining := sorted(self.evidence_tool_names() - spent):
                return (
                    f"The {self.state_namespace} capability has spent its budget "
                    f"for {names}; further calls to them fail. Gather any further "
                    f"evidence with {', '.join(remaining)}, or call "
                    f"{self._cite_tool_name} with the chunk_ids you have and "
                    "answer."
                )
            return (
                f"The {self.state_namespace} capability has spent its budget for "
                f"{names}; further calls to them fail. Answer from the evidence "
                f"already gathered and call {self._cite_tool_name} with the "
                "chunk_ids supporting it."
            )
        return None

    async def prepare_tools(
        self,
        ctx: RunContext[Any],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Remove this capability's tools past its limit, cite tool last.

        Tools whose own budget is spent stay declared on purpose. Removing one
        makes a model that calls it anyway hit ``Unknown tool name``, charged
        against the agent's unknown-tool retry budget, which kills the run after
        two attempts. A spent tool that keeps failing only wastes requests.
        """
        if self._citation_grace_expired:
            return [tool for tool in tool_defs if tool.capability_id != self.id]
        if not self._request_limit_reached:
            return tool_defs
        return [
            tool
            for tool in tool_defs
            if tool.capability_id != self.id or tool.name == self._cite_tool_name
        ]

    @property
    def cite_available(self) -> bool:
        """Whether this capability's cite tool is still declared to the model.

        Public because the citation policy has to know whether asking for a
        citation is even possible: past the grace window the tool is gone, and
        pointing the model at it would cost the agent's unknown-tool retries.
        """
        return not self._citation_grace_expired

    def evidence_tool_names(self) -> set[str]:
        """Tools that can bring new evidence into the run.

        Public because compaction needs to know whose output on the wire is
        evidence: a cite acknowledgement is a receipt of the model's own action and
        must survive, while a code execution that reached the corpus is evidence.
        """
        return {f"{self.state_namespace}_search"}

    def _spent_tool_names(self) -> set[str]:
        """This capability's tools whose own budget is exhausted."""
        if self.search_count >= self._max_searches:
            return {f"{self.state_namespace}_search"}
        return set()

    @property
    def _cite_tool_name(self) -> str:
        return f"{self.state_namespace}_cite"

    @property
    def _max_searches(self) -> int:
        return self.config.qa.max_searches

    @property
    def _request_limit_reached(self) -> bool:
        return (
            self.request_limit is not None and self.request_count >= self.request_limit
        )

    @property
    def _citation_grace_expired(self) -> bool:
        # No `request_limit is None` guard: the counter only advances under
        # `_request_limit_reached`, which already requires a limit.
        return self.grace_requests_used >= CITATION_GRACE_REQUESTS

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

    async def get_picture_bytes(self, document_id: str, self_ref: str) -> bytes | None:
        """Fetch a picture of this capability's evidence, for whoever re-attaches it.

        Public because compaction rehydrates cited pictures and this capability
        already holds the connection they came from; bytes are never kept in state.
        """
        async with self.rag_lock:
            rag = await self._ensure_rag()
            return await rag.document_item_repository.get_picture_bytes(
                document_id, self_ref
            )

    async def _close(self) -> None:
        if self.rag is not None:
            await self.rag.__aexit__(None, None, None)
            self.rag = None

    def _sync_state(self) -> None:
        if self.outer_state is not None and self.state is not None:
            self.outer_state[self.state_namespace] = self.state.model_dump(mode="json")

    async def _with_state(self, operation: Any) -> Any:
        """Execute an operation and copy its state back to the host dependencies.

        A failing tool still syncs, so evidence it gathered before the failure
        reaches the host.
        """
        try:
            return await operation
        finally:
            self._sync_state()

    def _evidence_record(self) -> CapabilityEvidenceRecord:
        assert self.state is not None
        return cast(CapabilityEvidenceRecord, cast(Any, self.state).evidence)

    def _note_evidence(self) -> None:
        """Record an outcome the model can ground an answer on.

        Includes an empty search result and a failed execution that still printed
        output: negative evidence grounds a refusal. Excludes a spent budget, which
        yields nothing to ground anything on.
        """
        self._evidence_record().note_evidence(self.epoch)

    def _declare(self, citations: list[Citation]) -> None:
        """Record what the model cited, once the ids have resolved.

        Declaring earlier would let a call naming only unresolvable ids read as a
        grounded answer.
        """
        state = cast(Any, self.state)
        retrieved = {
            result.chunk_id
            for results in state.searches.values()
            for result in results
            if result.chunk_id
        }
        self._evidence_record().declare(
            [
                EvidenceRef(capability=self.state_namespace, chunk_id=c.chunk_id)
                for c in citations
            ],
            epoch=self.epoch,
            retrieved_now=retrieved,
        )

    async def _search(self, query: str, limit: int | None) -> str | ToolReturn:
        assert self.state is not None
        self.search_count += 1
        if self.search_count > self._max_searches:
            raise ToolFailed(
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
        self._note_evidence()
        if self.vision and (parts := build_image_content_from_results(results)):
            return ToolReturn(return_value=formatted, content=parts)
        return formatted

    async def _cite(self, chunk_ids: list[str]) -> str:
        """Register the evidence behind this answer, or declare there is none.

        An empty list is a valid answer to "what grounds this?", and the only way
        the model can say "nothing" other than staying silent — which is
        indistinguishable from forgetting to cite at all. It declares the question
        ungrounded, which is not the same as leaving it undeclared.
        """
        assert self.state is not None
        if not chunk_ids:
            self._declare([])
            return "Recorded: this answer cites no knowledge-base evidence."

        all_results: list[SearchResult] = []
        state = cast(Any, self.state)
        for results in state.searches.values():
            all_results.extend(results)
        known_ids = [result.chunk_id for result in all_results if result.chunk_id]
        requested = [_nearest_known_id(cid.strip("[]"), known_ids) for cid in chunk_ids]
        citations = resolve_citations(requested, all_results)
        resolved = {citation.chunk_id for citation in citations}
        missing = [cid for cid in requested if cid not in resolved]

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
        self._declare(citations)
        resolved = {citation.chunk_id for citation in citations}
        unresolved = [cid for cid in missing if cid not in resolved]
        if unresolved:
            return (
                f"Registered {len(citations)} citation(s); "
                f"ignored {len(unresolved)} unresolvable id(s): "
                f"{unresolved}. Copy chunk_ids verbatim from search "
                "results and cite again."
            )
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
