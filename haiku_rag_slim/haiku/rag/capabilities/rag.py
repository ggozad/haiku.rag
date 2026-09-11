import asyncio
from dataclasses import dataclass, field, replace
from difflib import get_close_matches
from functools import cache
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field
from pydantic_ai import (
    DeferredToolRequests,
    ModelRetry,
    RunContext,
    ToolFailed,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    BinaryContent,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturn,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset

from haiku.rag.capabilities._tools import (
    CodeExecutionEntry,
    EvidenceKey,
    merge_results,
    search_corpus,
)
from haiku.rag.capabilities.ledger import CapabilityEvidenceRecord, EvidenceRef
from haiku.rag.client import HaikuRAG, all_found
from haiku.rag.client.scope import DatabaseScope
from haiku.rag.config.models import AppConfig
from haiku.rag.sandbox import AnalysisContext, Sandbox, recovery_hint
from haiku.rag.store.exceptions import AmbiguousCitationError, AmbiguousDatabaseError
from haiku.rag.store.models.chunk import SearchResult
from haiku.rag.store.models.citation import (
    Citation,
    ambiguous_citation,
    resolve_citations,
)
from haiku.rag.tools.search import PictureKey, build_image_content_from_results

STATE_NAMESPACE = "rag"
CAPABILITY_ID = "haiku-rag"
SEARCH_TOOL = "search"
EXECUTE_CODE_TOOL = "execute_code"
CITE_TOOL = "cite"
TOOL_NAMES = frozenset({SEARCH_TOOL, EXECUTE_CODE_TOOL, CITE_TOOL})

CITATION_GRACE_REQUESTS = 2
"""Requests calling this capability's tools that its cite tool outlives the rest by.

A loop guard, not a budget: cite consumes no retry budget and raises nothing, so
left available forever a stuck model calls it until the agent's own request limit
raises ``UsageLimitExceeded`` and the question returns no answer at all. Only
engagement can loop, which is why other tools' turns do not spend it.
"""

CHUNK_ID_MATCH_CUTOFF = 0.75
"""Similarity a cited chunk id needs to be treated as a corrupted known id.

Calibration knob. Two unrelated UUID4s reach about 0.5, while dropping or
duplicating a character or a whole group stays above 0.75, so the gap is wide.
"""

FREE_SIBLINGS_PER_ROUND = 3
"""Searches one budget unit covers when emitted in the same model response.

Calibration knob, sized to the measured modal burst. ``qa.max_searches``
counts units, so a model rephrasing its query a few times in one response
spends one unit, while every search of a sequential searcher is a unit of its
own.
"""

_instructions_path = Path(__file__).parent / "instructions" / "rag.md"
_multiple_collections_path = (
    Path(__file__).parent / "instructions" / "rag_multiple_collections.md"
)


class RAGState(BaseModel):
    """What the capability accumulates while answering, carried between its runs.

    Hosts dump this to JSON and hand it back on the next turn, including over
    AG-UI, so the field names and their nesting are a compatibility surface: keep
    it flat and don't rename. Key order is not part of it — every carry point
    re-validates by key.
    """

    citation_index: dict[str, Citation] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
    evidence: CapabilityEvidenceRecord = Field(default_factory=CapabilityEvidenceRecord)
    document_filter: str | None = None
    sources: list[str] | None = None
    searches: dict[str, list[SearchResult]] = Field(default_factory=dict)
    executions: list[CodeExecutionEntry] = Field(default_factory=list)

    def begin_invocation(self) -> None:
        """Drop the working evidence of the previous question.

        Only ever called when a new question starts. A resumption keeps it: the
        results belong to the question still being answered, and dropping them
        leaves a later citation unable to resolve against the expanded result the
        model saw, recording no provenance for it.

        `document_filter` and `sources` scope the conversation, and `evidence`
        carries question identity, so none of them is working evidence.
        """
        self.citations.clear()
        self.searches.clear()
        self.executions.clear()


@cache
def instructions() -> str:
    return _instructions_path.read_text().strip()


@cache
def multiple_collections_instructions() -> str:
    """Appended for a run that spans more than one collection."""
    return _multiple_collections_path.read_text().rstrip()


def _ambiguous_retry(error: AmbiguousCitationError) -> ModelRetry:
    """The only way out of an id that names a chunk in two databases.

    The model cannot say which it meant, so the retry asks for other evidence.
    """
    return ModelRetry(
        f"{error}. Cite a chunk id that appears once across the databases "
        "searched, or cite nothing."
    )


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


def resolve_scope(
    db_path: Path | str | None,
    config: AppConfig,
    sources: list[str] | None = None,
    *,
    rag: HaikuRAG | None = None,
) -> DatabaseScope:
    """The databases the capability covers, resolved once at its entry point."""
    if sources is not None:
        if db_path is not None:
            raise AmbiguousDatabaseError(
                "a path and `sources` both say which databases the capability "
                f"covers: db_path={Path(db_path)} and sources "
                f"{', '.join(sources)}; pass one of them"
            )
        if rag is not None:
            raise AmbiguousDatabaseError(
                "`sources` and a lent client both say which databases the "
                "capability covers, and the client is what it reads; narrow "
                "the client instead"
            )
    scope = DatabaseScope.resolve(config, database_path=db_path)
    return scope if sources is None else scope.select(sources)


def _awaits_the_model(messages: list[ModelMessage]) -> bool:
    """Whether the history unmistakably leaves the model something to answer.

    Used to validate what the record already says, never to decide it. Only two
    shapes are unambiguous: a response whose tool calls have no returns, and a
    retry the model has not answered. A trailing tool return is not one of them,
    being both how a settled structured answer ends and how results reach a
    question still in progress.
    """
    if not messages:
        return False
    last = messages[-1]
    if isinstance(last, ModelResponse):
        return any(isinstance(part, ToolCallPart) for part in last.parts)
    return any(isinstance(part, RetryPromptPart) for part in last.parts)


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
class RAGCapability(AbstractCapability[Any]):
    """Deferred, native Pydantic AI capability for grounded answers over a corpus.

    Search, sandboxed Python over the documents, and citations, in one place.
    """

    scope: DatabaseScope
    config: AppConfig
    vision: bool
    request_limit: int | None = None
    state: RAGState | None = field(default=None, repr=False)
    outer_state: dict[str, Any] | None = field(default=None, repr=False)
    rag: HaikuRAG | None = field(default=None, repr=False)
    """A connection this capability opened, and must close."""
    borrowed_rag: HaikuRAG | None = field(default=None, repr=False)
    """A caller's connection, reused and never closed here."""
    rag_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    resource_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    sandbox: Sandbox | None = field(default=None, repr=False)
    search_count: int = field(default=0, repr=False)
    search_step: int = field(default=0, repr=False)
    """The run_step whose searches are being priced and deduplicated."""
    step_searches: int = field(default=0, repr=False)
    step_rejected: bool = field(default=False, repr=False)
    step_shown: set[EvidenceKey] = field(default_factory=set, repr=False)
    step_pictures: set[PictureKey] = field(default_factory=set, repr=False)
    request_count: int = field(default=0, repr=False)
    grace_requests_used: int = field(default=0, repr=False)
    execute_count: int = field(default=0, repr=False)
    epoch: int = field(default=0, repr=False)
    state_carried: bool = field(default=False, repr=False)
    """Whether the host handed back a record a previous question had stamped.

    False on a first question, and equally on every question of a host that does
    not carry state between runs. Capabilities that need the record to mean
    anything across questions refuse when this is False.
    """

    state_namespace: ClassVar[str] = STATE_NAMESPACE
    state_type: ClassVar[type[RAGState]] = RAGState
    tool_names: ClassVar[frozenset[str]] = TOOL_NAMES

    @classmethod
    def from_spec(
        cls,
        db_path: Path | None = None,
        config: AppConfig | None = None,
        *,
        defer_loading: bool = True,
        request_limit: int | None = 30,
        sources: list[str] | None = None,
        vision: bool | None = None,
    ) -> "RAGCapability":
        """Build from an agent spec, mirroring the factory's serializable arguments.

        A live ``HaikuRAG`` client cannot be written in a spec, so ``rag`` is
        absent here. ``config`` arrives as a mapping and is validated.
        """
        return create_capability(
            db_path,
            AppConfig.model_validate(config) if config is not None else None,
            defer_loading=defer_loading,
            request_limit=request_limit,
            sources=sources,
            vision=vision,
        )

    async def for_run(self, ctx: RunContext[Any]) -> "RAGCapability":
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
        state = RAGState.model_validate(raw_state or {})
        record = state.evidence
        continuing = record.in_progress
        state_carried = record.question is not None
        if not continuing and _awaits_the_model(ctx.messages):
            raise RuntimeError(
                f"The {self.state_namespace} capability is resuming a question with "
                "no stored question identity. Capabilities cannot be added, removed "
                "or migrated while a question is unfinished, and the run's state "
                "must be carried between its runs."
            )
        if not continuing:
            state.begin_invocation()
            record.begin_question(len(ctx.messages))
        run_capability = replace(
            self,
            state=state,
            outer_state=outer_state,
            rag=None,
            rag_lock=asyncio.Lock(),
            resource_lock=asyncio.Lock(),
            sandbox=None,
            search_count=0,
            search_step=0,
            step_searches=0,
            step_rejected=False,
            step_shown=set(),
            step_pictures=set(),
            request_count=0,
            grace_requests_used=0,
            execute_count=0,
            epoch=0,
            state_carried=state_carried,
        )
        run_capability._sync_state()
        return run_capability

    @property
    def spans_collections(self) -> bool:
        """Whether this run reads more than one collection.

        A question narrows the conversation through `sources`, so a capability
        built over a set can still run against one collection, and telling it
        how to attribute across collections it cannot reach is noise.
        """
        if self.state is not None and self.state.sources is not None:
            return len(set(self.state.sources)) > 1
        if self.borrowed_rag is not None:
            return self.borrowed_rag.covers_multiple
        return self.scope.covers_multiple

    def get_instructions(self) -> str:
        parts = [instructions()]
        if self.config.prompts.domain_preamble:
            parts.insert(0, self.config.prompts.domain_preamble)
        if self.spans_collections:
            parts.append(multiple_collections_instructions())
        return "\n\n".join(parts)

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
                f"limit. Only {CITE_TOOL} remains among its tools: "
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
                    f"{CITE_TOOL} with the chunk_ids you have and "
                    "answer."
                )
            return (
                f"The {self.state_namespace} capability has spent its budget for "
                f"{names}; further calls to them fail. Answer from the evidence "
                f"already gathered and call {CITE_TOOL} with the "
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
            if tool.capability_id != self.id or tool.name == CITE_TOOL
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
        Searching from inside the sandbox is not priced against
        ``qa.max_searches``, so code execution outlives a spent search budget as a
        way to reach new evidence.
        """
        return {SEARCH_TOOL, EXECUTE_CODE_TOOL}

    def _spent_tool_names(self) -> set[str]:
        """This capability's tools whose own budget is exhausted."""
        spent = set()
        if self.search_count >= self._max_searches:
            spent.add(SEARCH_TOOL)
        if self.execute_count >= self.config.qa.max_executions:
            spent.add(EXECUTE_CODE_TOOL)
        return spent

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
        """Close the question, unless the run is only pausing for deferred results.

        A run that raised never arrives here, which is what leaves an interrupted
        question in progress for the resumption to claim.
        """
        if self.state is not None and not isinstance(
            result.output, DeferredToolRequests
        ):
            self.evidence_record().end_question()
            self._sync_state()
        await self._close()
        return result

    async def on_run_error(
        self, ctx: RunContext[Any], *, error: BaseException
    ) -> AgentRunResult[Any]:
        await self._close()
        raise error

    async def _ensure_rag(self) -> HaikuRAG:
        if self.borrowed_rag is not None:
            return self.borrowed_rag
        if self.rag is None:
            async with self.resource_lock:
                if self.rag is None:
                    rag = HaikuRAG._covering(self.scope, self.config, read_only=True)
                    await rag.__aenter__()
                    self.rag = rag
        return self.rag

    async def _ensure_sandbox(self) -> Sandbox:
        if self.sandbox is None:
            rag = await self._ensure_rag()
            assert self.state is not None
            self.sandbox = Sandbox._covering(
                scope=self.scope,
                config=self.config,
                context=AnalysisContext(
                    filter=self.state.document_filter,
                    sources=self.state.sources,
                ),
                rag=rag,
                lock=self.rag_lock,
                executions=self.config.qa.max_executions,
            )
        return self.sandbox

    async def get_picture_bytes(
        self, document_id: str, self_ref: str, source: str | None = None
    ) -> bytes | None:
        """Fetch a picture of this capability's evidence, for whoever re-attaches it.

        Public because compaction rehydrates cited pictures and this capability
        already holds the connection they came from; bytes are never kept in state.
        """
        async with self.rag_lock:
            rag = await self._ensure_rag()
            return await rag.get_picture_bytes(document_id, self_ref, source)

    async def _close(self) -> None:
        if self.sandbox is not None:
            await self.sandbox.close()
            self.sandbox = None
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

    def evidence_record(self) -> CapabilityEvidenceRecord:
        """What this capability has retrieved and cited, per question."""
        assert self.state is not None
        return self.state.evidence

    def citation_index(self) -> dict[str, Citation]:
        """Citations registered so far, by chunk id."""
        assert self.state is not None
        return self.state.citation_index

    def _note_evidence(self) -> None:
        """Record an outcome the model can ground an answer on.

        Includes an empty search result and a failed execution that still printed
        output: negative evidence grounds a refusal. Excludes a spent budget, which
        yields nothing to ground anything on.
        """
        self.evidence_record().note_evidence(self.epoch)

    def _declare(self, citations: list[Citation]) -> None:
        """Record what the model cited, once the ids have resolved.

        Declaring earlier would let a call naming only unresolvable ids read as a
        grounded answer.
        """
        assert self.state is not None
        state = self.state
        retrieved = {
            result.chunk_id
            for results in state.searches.values()
            for result in results
            if result.chunk_id
        }
        self.evidence_record().declare(
            [
                EvidenceRef(capability=self.state_namespace, chunk_id=c.chunk_id)
                for c in citations
            ],
            epoch=self.epoch,
            retrieved_now=retrieved,
        )

    async def _search(
        self, query: str, limit: int | None, run_step: int
    ) -> str | ToolReturn:
        assert self.state is not None
        if run_step != self.search_step:
            self.search_step = run_step
            self.step_searches = 0
            self.step_rejected = False
            self.step_shown = set()
            self.step_pictures = set()
        self.step_searches += 1
        if (self.step_searches - 1) % FREE_SIBLINGS_PER_ROUND == 0:
            self.search_count += 1
        if self.step_rejected or self.search_count > self._max_searches:
            self.step_rejected = True
            raise ToolFailed(
                "Search limit reached. Answer the question using "
                "the results you already have."
            )
        async with self.rag_lock:
            formatted, results, rendered, include_collection = await search_corpus(
                await self._ensure_rag(),
                query,
                limit=limit,
                document_filter=self.state.document_filter,
                sources=self.state.sources,
                shown=self.step_shown,
            )
        parts: list[str | BinaryContent] = []
        emitted: set[PictureKey] = set()
        if self.vision:
            parts, emitted = build_image_content_from_results(
                results,
                include_collection=include_collection,
                exclude=self.step_pictures,
            )
        # Everything the search produced commits together, after formatting and
        # image construction have both succeeded: a search that raises must not
        # leave results citable, note evidence the model never received, or
        # suppress a later sibling's results.
        state = self.state
        # A model can search the same query twice with different limits, and the
        # narrower return must not drop what the wider one already showed it.
        merge_results(state.searches.setdefault(query, []), results)
        self._note_evidence()
        self.step_shown |= rendered
        self.step_pictures |= emitted
        if parts:
            return ToolReturn(return_value=formatted, content=parts)
        return formatted

    async def _execute_code(self, code: str) -> str:
        assert self.state is not None
        self.execute_count += 1
        if self.execute_count > self.config.qa.max_executions:
            raise ToolFailed(
                "Code-execution limit reached. Give your final answer now from what "
                f"you already have; do not call {EXECUTE_CODE_TOOL} again."
            )
        sandbox = await self._ensure_sandbox()
        result = await sandbox.execute(code)
        if result.success or result.stdout:
            self._note_evidence()
        if sandbox._search_results:
            merge_results(
                self.state.searches.setdefault("_sandbox", []),
                sandbox._search_results,
            )
        self.state.executions.append(
            CodeExecutionEntry(
                code=code,
                stdout=result.stdout,
                stderr=result.stderr,
                success=result.success,
            )
        )
        if not result.success:
            raise ToolFailed(
                f"{result.stderr}{recovery_hint(result.stderr)}"
                f"\n\nOutput: {result.stdout}"
            )
        return result.stdout or "No output."

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
        state = self.state
        for results in state.searches.values():
            all_results.extend(results)
        known_ids = [result.chunk_id for result in all_results if result.chunk_id]
        requested = [_nearest_known_id(cid.strip("[]"), known_ids) for cid in chunk_ids]
        try:
            citations = resolve_citations(requested, all_results)
        except AmbiguousCitationError as error:
            raise _ambiguous_retry(error) from error
        resolved = {citation.chunk_id for citation in citations}
        missing = [cid for cid in requested if cid not in resolved]

        if missing:
            async with self.rag_lock:
                rag = await self._ensure_rag()
                # A chunk id names no database, so the fallback covers exactly
                # what the question covers, and nothing it does not.
                lookups = await rag.clients_covering(self.state.sources)
                synthetic: list[SearchResult] = []
                documents: dict[tuple[str | None, str], Any] = {}
                for chunk_id in missing:
                    # All holders are asked: a collision shows only across
                    # every database's answer.
                    holders = await all_found(
                        lookups, lambda owner: owner.get_chunk_by_id(chunk_id)
                    )
                    if len(holders) > 1:
                        raise _ambiguous_retry(
                            ambiguous_citation(
                                chunk_id, [owner.source for owner, _ in holders]
                            )
                        )
                    if not holders:
                        continue
                    [(owner, chunk)] = holders
                    if not chunk.document_id:
                        continue
                    key = (owner.source, chunk.document_id)
                    if key not in documents:
                        documents[key] = await owner.get_document_by_id(
                            chunk.document_id
                        )
                    document = documents[key]
                    chunk.document_uri = document.uri if document else None
                    chunk.document_title = document.title if document else None
                    chunk.document_meta = document.metadata if document else {}
                    result = SearchResult.from_chunk(chunk, score=1.0)
                    result.source = owner.source
                    synthetic.append(result)
                citations.extend(resolve_citations(missing, synthetic))

        if not citations:
            raise ModelRetry(
                f"None of the supplied chunk_ids {list(chunk_ids)} could be resolved. "
                "Copy chunk_ids verbatim from search results."
            )
        try:
            self._register_citations(citations)
        except AmbiguousCitationError as error:
            raise _ambiguous_retry(error) from error
        self._declare(citations)
        resolved = {citation.chunk_id for citation in citations}
        unresolved = [cid for cid in missing if cid not in resolved]
        if unresolved:
            # States the outcome without asking for another call. Reaching here
            # means something registered, so the answer already has grounding: a
            # model that keeps mangling ids would obey an invitation to retry
            # until the run dies on output retries.
            return (
                f"Registered {len(citations)} citation(s); "
                f"ignored {len(unresolved)} unresolvable id(s): "
                f"{unresolved}, which were not verbatim from search results."
            )
        return f"Registered {len(citations)} citation(s)."

    def _register_citations(self, citations: list[Citation]) -> None:
        """Index the citations, numbering the ones not already registered.

        The index is keyed by chunk id and outlives the question, so an id
        already registered from another database is refused here.
        """
        assert self.state is not None
        state = self.state
        for citation in citations:
            held = state.citation_index.get(citation.chunk_id)
            if held is not None and held.source != citation.source:
                raise AmbiguousCitationError(
                    f"chunk id {citation.chunk_id} was already cited from "
                    "another database in this conversation; a citation records "
                    "the id alone and cannot say which"
                )
        next_index = len(state.citation_index) + 1
        for citation in citations:
            if citation.chunk_id not in state.citation_index:
                citation.index = next_index
                next_index += 1
                state.citation_index[citation.chunk_id] = citation
            if citation.chunk_id not in state.citations:
                state.citations.append(citation.chunk_id)

    def get_toolset(self) -> FunctionToolset[Any]:
        async def search(
            ctx: RunContext[Any], query: str, limit: int | None = None
        ) -> str | ToolReturn:
            """Search the knowledge base for evidence to analyze."""
            return await self._with_state(self._search(query, limit, ctx.run_step))

        async def execute_code(ctx: RunContext[Any], code: str) -> Any:
            """Execute Python against the sandboxed document filesystem."""
            return await self._with_state(self._execute_code(code))

        async def cite(ctx: RunContext[Any], chunk_ids: list[str]) -> Any:
            """Register exact retrieved chunk IDs as citations for the answer."""
            return await self._with_state(self._cite(chunk_ids))

        return FunctionToolset(
            [search, execute_code, cite],
            id=CAPABILITY_ID,
            max_retries=3,
            sequential=True,
        )


def create_capability(
    db_path: Path | str | None = None,
    config: AppConfig | None = None,
    *,
    defer_loading: bool = True,
    rag: HaikuRAG | None = None,
    request_limit: int | None = 30,
    sources: list[str] | None = None,
    vision: bool | None = None,
) -> RAGCapability:
    """Create the native Pydantic AI RAG capability.

    ``sources`` names the configured databases the capability covers, all of
    them when omitted. ``vision`` gates whether picture chunks are attached to
    search results as images, and should reflect the model the hosting agent
    actually runs. Defaults to ``config.qa.model.vision``.
    """
    if config is None:
        from haiku.rag.config import get_config

        config = get_config()
    scope = resolve_scope(db_path, config, sources, rag=rag)
    return RAGCapability(
        scope=scope,
        config=config,
        borrowed_rag=rag,
        vision=config.qa.model.vision if vision is None else vision,
        request_limit=request_limit,
        id=CAPABILITY_ID,
        description=(
            "Search the haiku.rag knowledge base, run Python over its documents, "
            "and cite evidence for grounded answers."
        ),
        defer_loading=defer_loading,
    )


__all__ = [
    "CAPABILITY_ID",
    "CITATION_GRACE_REQUESTS",
    "CITE_TOOL",
    "EXECUTE_CODE_TOOL",
    "SEARCH_TOOL",
    "STATE_NAMESPACE",
    "TOOL_NAMES",
    "CodeExecutionEntry",
    "RAGCapability",
    "RAGState",
    "create_capability",
    "instructions",
    "resolve_scope",
]
