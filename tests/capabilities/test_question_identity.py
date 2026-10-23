import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
from ag_ui.core import Message
from pydantic import TypeAdapter
from pydantic_ai import (
    Agent,
    CallDeferred,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.ui.ag_ui import AGUIAdapter

from haiku.rag.capabilities.compaction import create_capability as create_compaction
from haiku.rag.capabilities.policy import REDIRECT_HINT
from haiku.rag.capabilities.policy import create_capability as create_policy
from haiku.rag.capabilities.rag import RAGCapability
from haiku.rag.capabilities.rag import create_capability as create_rag
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models.chunk import SearchResult

STORED = json.loads(
    (Path(__file__).parents[1] / "data" / "agui_stored_conversations.json").read_text(
        encoding="utf-8"
    )
)


@dataclass
class Deps:
    state: dict[str, Any] = field(default_factory=dict)


async def stub_search(
    self, query: str, _limit: int | None, _run_step: int, _sources=None
) -> str:
    cast(Any, self.state).searches[query] = [
        SearchResult(content=f"evidence for {query}", score=1.0, chunk_id=query)
    ]
    self._note_evidence()
    return f"EVIDENCE for {query}"


def request(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(text)])


def response(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(text)])


SHAPES: dict[str, tuple[str | None, list[ModelMessage]]] = {
    "prompt": ("q", []),
    "prompt after a response": ("q", [request("hi"), response("hello")]),
    "prompt after a request": (
        "q",
        [request("hi"), response("hello"), request("earlier")],
    ),
    "prompt after consecutive requests": (
        "q",
        [request("a"), request("b"), response("x")],
    ),
    "question in history": (None, [request("hi"), response("hello"), request("q")]),
    "question alone in history": (None, [request("q")]),
    "consecutive requests": (None, [request("a"), request("q")]),
    "consecutive requests after a response": (
        None,
        [request("hi"), response("hello"), request("a"), request("q")],
    ),
}


def redirects(messages: list[ModelMessage]) -> int:
    return sum(
        REDIRECT_HINT in str(part.content)
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart)
    )


@dataclass
class OpeningRequest(AbstractCapability[Any]):
    """Records the stored identity and the history at each model request."""

    seen: list[tuple[int, list[ModelMessage]]] = field(default_factory=list)

    async def before_model_request(self, ctx: RunContext[Any], request_context):
        self.seen.append(
            (ctx.deps.state["rag"]["evidence"]["question"], list(ctx.messages))
        )
        return request_context


def scripted(responses: list[list[Any]], sent: list[list[ModelMessage]]):
    turns = iter(responses)

    async def model(messages, _info):
        sent.append(list(messages))
        return ModelResponse(parts=next(turns))

    return FunctionModel(model)


def streamed(responses: list[list[Any]], sent: list[list[ModelMessage]]):
    turns = iter(responses)

    async def model(messages, _info):
        sent.append(list(messages))
        for index, part in enumerate(next(turns)):
            if isinstance(part, TextPart):
                yield part.content
            else:
                yield {
                    index: DeltaToolCall(
                        name=part.tool_name,
                        json_args=json.dumps(part.args),
                        tool_call_id=part.tool_call_id,
                    )
                }

    return FunctionModel(stream_function=model)


async def run_shape(db_path, shape: str, responses, *, extra=(), deferred=False):
    prompt, history = SHAPES[shape]
    sent: list[list[ModelMessage]] = []
    agent = Agent(
        scripted(responses, sent),
        deps_type=Deps,
        capabilities=[
            create_rag(db_path=db_path, config=AppConfig(), defer_loading=deferred),
            create_policy(),
            *extra,
        ],
    )
    deps = Deps()
    with patch.object(RAGCapability, "_search", stub_search):
        await agent.run(prompt, deps=deps, message_history=copy.deepcopy(history))
    return deps, sent


UNCITED_AFTER_A_SEARCH = [
    [ToolCallPart("search", {"query": "supervisor"}, "call-1")],
    [TextPart("uncited")],
    [TextPart("still uncited")],
]


@pytest.mark.parametrize("shape", SHAPES)
async def test_a_single_round_answer_is_enforced(temp_db_path, shape):
    deps, sent = await run_shape(temp_db_path, shape, UNCITED_AFTER_A_SEARCH)

    question = deps.state["rag"]["evidence"]["question"]
    assert redirects(sent[-1]) == 1
    assert deps.state["citation_policy"]["violations"] == [question]


@pytest.mark.parametrize("shape", SHAPES)
async def test_a_single_execution_is_enforced(rag_db, shape):
    deps, sent = await run_shape(
        rag_db,
        shape,
        [
            [ToolCallPart("execute_code", {"code": "print(6 * 7)"}, "call-1")],
            [TextPart("uncited")],
            [TextPart("still uncited")],
        ],
    )

    question = deps.state["rag"]["evidence"]["question"]
    assert redirects(sent[-1]) == 1
    assert deps.state["citation_policy"]["violations"] == [question]


@pytest.mark.parametrize(
    "shape", [s for s, (prompt, _) in SHAPES.items() if not prompt]
)
async def test_a_greeting_in_the_history_is_left_alone(temp_db_path, shape):
    deps, sent = await run_shape(temp_db_path, shape, [[TextPart("hello back")]])

    assert redirects(sent[-1]) == 0
    assert deps.state["citation_policy"]["violations"] == []


@pytest.mark.parametrize("shape", SHAPES)
async def test_the_identity_is_the_index_of_the_opening_request(temp_db_path, shape):
    probe = OpeningRequest()
    deps, _ = await run_shape(
        temp_db_path,
        shape,
        [
            [ToolCallPart("search", {"query": "supervisor"}, "call-1")],
            [ToolCallPart("cite", {"chunk_ids": ["supervisor"]}, "call-2")],
            [TextPart("a grounded answer")],
        ],
        extra=[probe],
    )

    identity, messages = probe.seen[0]
    assert messages[identity] is messages[-1]
    assert isinstance(messages[identity], ModelRequest)
    evidence = deps.state["rag"]["evidence"]
    assert evidence["question"] == identity
    assert evidence["latest_evidence_epoch"] == identity + 1


@pytest.mark.parametrize("compacted", [False, True])
@pytest.mark.parametrize("shape", ["prompt", "question in history"])
async def test_a_capability_loaded_mid_question_takes_the_request_after_the_load(
    temp_db_path, shape, compacted
):
    deps, sent = await run_shape(
        temp_db_path,
        shape,
        [
            [ToolCallPart("load_capability", {"id": "haiku-rag"}, "call-0")],
            *UNCITED_AFTER_A_SEARCH,
        ],
        extra=[create_compaction()] if compacted else (),
        deferred=True,
    )

    loaded = next(
        index
        for index, message in enumerate(sent[-1])
        if any(
            isinstance(part, ToolReturnPart) and part.tool_name == "load_capability"
            for part in message.parts
        )
    )
    question = deps.state["rag"]["evidence"]["question"]
    assert question == loaded
    assert redirects(sent[-1]) == 1
    assert deps.state["citation_policy"]["violations"] == [question]


async def test_a_capability_never_loaded_leaves_no_question_for_the_next_run(
    temp_db_path,
):
    """With the compactor, a question the deferred capability never saw is not
    carried over, and the run that loads it opens at the request after the load."""
    sent: list[list[ModelMessage]] = []
    agent = Agent(
        scripted(
            [
                [TextPart("hello back")],
                [ToolCallPart("load_capability", {"id": "haiku-rag"}, "call-0")],
                [ToolCallPart("search", {"query": "supervisor"}, "call-1")],
                [ToolCallPart("cite", {"chunk_ids": ["supervisor"]}, "call-2")],
                [TextPart("a grounded answer")],
            ],
            sent,
        ),
        deps_type=Deps,
        capabilities=[
            create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=True),
            create_compaction(),
        ],
    )
    deps = Deps()
    with patch.object(RAGCapability, "_search", stub_search):
        first = await agent.run("hi", deps=deps)
        assert deps.state.get("rag", {}).get("evidence", {}).get("in_progress") in (
            None,
            False,
        )
        await agent.run("a question", deps=deps, message_history=first.all_messages())

    loaded = next(
        index
        for index, message in enumerate(sent[-1])
        if any(
            isinstance(part, ToolReturnPart) and part.tool_name == "load_capability"
            for part in message.parts
        )
    )
    assert deps.state["rag"]["evidence"]["question"] == loaded


@pytest.mark.parametrize(
    ("shape", "identity"), [("prompt", 0), ("prompt after a response", 2)]
)
async def test_a_prompt_over_an_unmerged_history_keeps_its_identity(
    temp_db_path, shape, identity
):
    deps, _ = await run_shape(temp_db_path, shape, [[TextPart("hello back")]])

    assert deps.state["rag"]["evidence"]["question"] == identity


async def test_a_resumed_question_keeps_its_identity_and_its_evidence(temp_db_path):
    """The first question is compacted, the resumed one is not, and the redirect
    the resumed question already received is not sent again."""
    sent: list[list[ModelMessage]] = []
    # ty cannot resolve the deps type parameter through a multi-member
    # `output_type`.
    agent = Agent(  # ty: ignore[no-matching-overload]
        scripted(
            [
                [ToolCallPart("search", {"query": "first"}, "call-1")],
                [ToolCallPart("cite", {"chunk_ids": ["first"]}, "call-2")],
                [TextPart("first answer")],
                [ToolCallPart("search", {"query": "second"}, "call-3")],
                [TextPart("uncited")],
                [ToolCallPart("confirm", {}, "call-4")],
                [TextPart("still uncited")],
            ],
            sent,
        ),
        deps_type=Deps,
        output_type=[str, DeferredToolRequests],
        capabilities=[
            create_rag(db_path=temp_db_path, config=AppConfig()),
            create_compaction(),
            create_policy(),
        ],
    )

    @agent.tool_plain
    def confirm() -> str:
        raise CallDeferred

    deps = Deps()

    with patch.object(RAGCapability, "_search", stub_search):
        first = await agent.run(
            None,
            deps=deps,  # ty: ignore[invalid-argument-type]
            message_history=[request("first")],
        )
        paused = await agent.run(
            None,
            deps=deps,  # ty: ignore[invalid-argument-type]
            message_history=[*first.all_messages(), request("second")],
        )
        assert isinstance(paused.output, DeferredToolRequests)
        question = deps.state["rag"]["evidence"]["question"]
        await agent.run(
            None,
            deps=deps,  # ty: ignore[invalid-argument-type]
            message_history=paused.all_messages(),
            deferred_tool_results=DeferredToolResults(calls={"call-4": "confirmed"}),
        )

    assert deps.state["rag"]["evidence"]["question"] == question
    assert redirects(sent[-1]) == 1
    assert deps.state["citation_policy"]["violations"] == [question]
    returns = {
        part.tool_call_id: str(part.content)
        for message in sent[-1]
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "search"
    }
    assert returns["call-3"] == "EVIDENCE for second"
    assert returns["call-1"] != "EVIDENCE for first"


@dataclass
class FailsBeforeTheModel(AbstractCapability[Any]):
    async def before_run(self, ctx: RunContext[Any]) -> None:
        raise RuntimeError("refused")


async def test_a_run_failing_before_its_first_request_leaves_the_record(
    temp_db_path,
):
    deps = Deps(state=copy.deepcopy(STORED["completed"]["state"]))
    agent = Agent(
        scripted([], []),
        deps_type=Deps,
        capabilities=[
            create_rag(db_path=temp_db_path, config=AppConfig()),
            FailsBeforeTheModel(),
        ],
    )

    history = AGUIAdapter.load_messages(
        TypeAdapter(list[Message]).validate_python(STORED["completed"]["messages"])
    )

    with pytest.raises(RuntimeError, match="refused"):
        await agent.run("and who supervises them?", deps=deps, message_history=history)

    assert deps.state["rag"] == STORED["completed"]["state"]["rag"]


async def run_agui(db_path, messages, state, responses, tools=()):
    sent: list[list[ModelMessage]] = []
    agent = Agent(
        streamed(responses, sent),
        deps_type=Deps,
        capabilities=[
            create_rag(db_path=db_path, config=AppConfig()),
            create_compaction(),
            create_policy(),
        ],
    )
    body = {
        "threadId": "thread",
        "runId": "run",
        "state": copy.deepcopy(state),
        "tools": list(tools),
        "context": [],
        "forwardedProps": {},
        "messages": messages,
    }
    adapter = AGUIAdapter(
        agent=agent, run_input=AGUIAdapter.build_run_input(json.dumps(body).encode())
    )
    deps = Deps()
    with patch.object(RAGCapability, "_search", stub_search):
        events = [event async for event in adapter.run_stream(deps=deps)]
    assert not [e for e in events if "RUN_ERROR" in str(e.type)]
    return deps, sent


async def test_agui_enforces_a_single_round_answer(temp_db_path):
    deps, sent = await run_agui(
        temp_db_path,
        [{"id": "u1", "role": "user", "content": "what does the supervisor do?"}],
        {},
        UNCITED_AFTER_A_SEARCH,
    )

    assert redirects(sent[-1]) == 1
    assert deps.state["citation_policy"]["violations"] == [0]


async def test_stored_agui_state_continues_with_a_new_question(temp_db_path):
    stored = STORED["completed"]

    deps, sent = await run_agui(
        temp_db_path,
        [
            *stored["messages"],
            {"id": "u2", "role": "user", "content": "and who supervises them?"},
        ],
        stored["state"],
        UNCITED_AFTER_A_SEARCH,
    )

    question = deps.state["rag"]["evidence"]["question"]
    assert question > stored["state"]["rag"]["evidence"]["question"]
    assert redirects(sent[-1]) == 1
    assert deps.state["citation_policy"]["violations"] == [question]


async def test_a_streamed_new_question_compacts_the_stored_one(temp_db_path):
    stored = STORED["completed"]

    _, sent = await run_agui(
        temp_db_path,
        [
            *stored["messages"],
            {"id": "u2", "role": "user", "content": "and who supervises them?"},
        ],
        stored["state"],
        [[TextPart("an answer")], [TextPart("still an answer")]],
    )

    returns = [
        part.content
        for message in sent[0]
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert "EVIDENCE" not in returns
    assert any("[chunk-1]" in content for content in returns)


async def test_stored_agui_state_resumes_an_interrupted_question(temp_db_path):
    stored = STORED["interrupted"]

    deps, _ = await run_agui(
        temp_db_path,
        [
            *stored["messages"],
            {"id": "t2", "role": "tool", "toolCallId": "call-2", "content": "ok"},
        ],
        stored["state"],
        [[TextPart("an answer")]],
        tools=[
            {
                "name": "confirm",
                "description": "Ask the user to confirm.",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )

    evidence = deps.state["rag"]["evidence"]
    assert evidence["question"] == stored["state"]["rag"]["evidence"]["question"]
    assert evidence["in_progress"] is False
