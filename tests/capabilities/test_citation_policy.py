import json
from dataclasses import dataclass, field
from typing import Any, cast
from unittest.mock import patch

import pytest
from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    DeferredToolResults,
    NativeOutput,
    PromptedOutput,
    RunContext,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelResponse, RetryPromptPart, TextPart, ToolCallPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.output import OutputContext
from pydantic_ai.usage import RunUsage

from haiku.rag.capabilities.policy import (
    CITATION_REDIRECT_TAG,
    REDIRECT_HINT,
    CitationPolicyState,
)
from haiku.rag.capabilities.policy import (
    create_capability as create_policy,
)
from haiku.rag.capabilities.rag import RAGCapability
from haiku.rag.capabilities.rag import create_capability as create_rag
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models.chunk import SearchResult


@dataclass
class Deps:
    state: dict[str, Any] = field(default_factory=dict)


async def stub_search(
    self, query: str, _limit: int | None, _run_step: int, _sources=None
) -> str:
    cast(Any, self.state).searches[query] = [
        SearchResult(content="evidence", score=1.0, chunk_id="chunk-1")
    ]
    self._note_evidence()
    return "EVIDENCE"


def prompts_of(messages) -> list[str]:
    return [
        str(part.content)
        for message in messages
        for part in message.parts
        if type(part).__name__ == "UserPromptPart"
    ]


async def run_with_policy(temp_db_path, responses, *, policy=True, config=None):
    """Answer one question with the given model responses, policy optional."""
    rag = create_rag(db_path=temp_db_path, config=config or AppConfig())
    capabilities: list[Any] = [rag]
    if policy:
        capabilities.append(create_policy())
    turns = iter(responses)
    sent: list[list[Any]] = []

    async def model(messages, _info):
        sent.append(list(messages))
        return ModelResponse(parts=next(turns))

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=capabilities)
    deps = Deps()
    with patch.object(RAGCapability, "_search", stub_search):
        result = await agent.run("what does the supervisor do?", deps=deps)
    return result, deps, sent


async def test_an_answer_without_a_citation_is_sent_back_once(temp_db_path):
    """The last response of a question is the last moment to notice."""
    result, deps, sent = await run_with_policy(
        temp_db_path,
        [
            [ToolCallPart("search", {"query": "supervisor"}, "call-1")],
            [TextPart("an answer with no citation")],
            [ToolCallPart("cite", {"chunk_ids": ["chunk-1"]}, "call-2")],
            [TextPart("an answer with no citation")],
        ],
    )

    redirects = [prompt for prompt in prompts_of(sent[-1]) if REDIRECT_HINT in prompt]
    assert len(redirects) == 1
    assert deps.state["rag"]["citations"] == ["chunk-1"]
    assert result.output == "an answer with no citation"


async def test_a_grounded_answer_is_left_alone(temp_db_path):
    _, _, sent = await run_with_policy(
        temp_db_path,
        [
            [ToolCallPart("search", {"query": "supervisor"}, "call-1")],
            [ToolCallPart("cite", {"chunk_ids": ["chunk-1"]}, "call-2")],
            [TextPart("a grounded answer")],
        ],
    )

    assert not [p for p in prompts_of(sent[-1]) if REDIRECT_HINT in p]


async def test_an_explicitly_ungrounded_answer_is_left_alone(temp_db_path):
    """Citing nothing is a declaration, not an omission."""
    _, deps, sent = await run_with_policy(
        temp_db_path,
        [
            [ToolCallPart("search", {"query": "supervisor"}, "call-1")],
            [ToolCallPart("cite", {"chunk_ids": []}, "call-2")],
            [TextPart("I cannot find this in the knowledge base")],
        ],
    )

    assert not [p for p in prompts_of(sent[-1]) if REDIRECT_HINT in p]
    assert deps.state["citation_policy"]["violations"] == []


async def test_a_question_that_gathered_no_evidence_is_left_alone(temp_db_path):
    """Nothing was retrieved, so there is no grounding to declare."""
    _, _, sent = await run_with_policy(temp_db_path, [[TextPart("hello back")]])

    assert not [p for p in prompts_of(sent[-1]) if REDIRECT_HINT in p]


async def test_a_violation_is_recorded_when_the_cite_tool_is_gone(temp_db_path):
    """Asking for a withdrawn tool costs the agent's unknown-tool retries."""
    with patch(
        "haiku.rag.capabilities.rag.RAGCapability.cite_available",
        new_callable=lambda: property(lambda self: False),
    ):
        _, deps, sent = await run_with_policy(
            temp_db_path,
            [
                [ToolCallPart("search", {"query": "supervisor"}, "call-1")],
                [TextPart("an answer with no citation")],
            ],
        )

    assert not [p for p in prompts_of(sent[-1]) if REDIRECT_HINT in p]
    assert deps.state["citation_policy"]["violations"] == [0]


async def test_without_the_policy_capability_nothing_is_enforced(temp_db_path):
    """Omission is the switch, so there is no flag to test."""
    _, deps, sent = await run_with_policy(
        temp_db_path,
        [
            [ToolCallPart("search", {"query": "supervisor"}, "call-1")],
            [TextPart("an answer with no citation")],
        ],
        policy=False,
    )

    assert not [p for p in prompts_of(sent[-1]) if REDIRECT_HINT in p]
    assert "citation_policy" not in deps.state


def test_two_policy_capabilities_fail_fast(temp_db_path):
    rag = create_rag(db_path=temp_db_path, config=AppConfig())

    async def model(_messages, _info):  # pragma: no cover - never reached
        return ModelResponse(parts=[TextPart("answer")])

    with pytest.raises(UserError, match="haiku-rag-citation-policy"):
        Agent(
            FunctionModel(model),
            deps_type=Deps,
            capabilities=[rag, create_policy(), create_policy()],
        )


def test_the_policy_state_round_trips():
    state = CitationPolicyState(violations=[4, 12])

    restored = CitationPolicyState.model_validate(state.model_dump(mode="json"))

    assert restored.violations == [4, 12]


async def test_a_second_question_can_be_redirected_again(temp_db_path):
    """The redirect fires once per question, not once per conversation."""
    rag = create_rag(db_path=temp_db_path, config=AppConfig())
    turns = iter(
        [
            [ToolCallPart("search", {"query": "supervisor"}, "call-1")],
            [TextPart("first, uncited")],
            [TextPart("first, uncited")],
            [ToolCallPart("search", {"query": "again"}, "call-2")],
            [TextPart("second, uncited")],
            [TextPart("second, uncited")],
        ]
    )
    sent: list[list[Any]] = []

    async def model(messages, _info):
        sent.append(list(messages))
        return ModelResponse(parts=next(turns))

    agent = Agent(
        FunctionModel(model), deps_type=Deps, capabilities=[rag, create_policy()]
    )
    deps = Deps()
    with patch.object(RAGCapability, "_search", stub_search):
        first = await agent.run("what does the supervisor do?", deps=deps)
        await agent.run(
            "and who supervises them?", deps=deps, message_history=first.all_messages()
        )

    assert len([p for p in prompts_of(sent[-1]) if REDIRECT_HINT in p]) == 2


@dataclass
class StatelessDeps:
    """A host that keeps no capability state, which is allowed."""


async def test_a_violation_with_nowhere_to_record_it_does_not_fail_the_run(
    temp_db_path,
):
    rag = create_rag(db_path=temp_db_path, config=AppConfig())
    turns = iter(
        [
            [ToolCallPart("search", {"query": "supervisor"}, "call-1")],
            [TextPart("an answer with no citation")],
        ]
    )

    async def model(_messages, _info):
        return ModelResponse(parts=next(turns))

    agent = Agent(
        FunctionModel(model),
        deps_type=StatelessDeps,
        capabilities=[rag, create_policy()],
    )

    with (
        patch(
            "haiku.rag.capabilities.rag.RAGCapability.cite_available",
            new_callable=lambda: property(lambda self: False),
        ),
        patch.object(RAGCapability, "_search", stub_search),
    ):
        result = await agent.run("what does the supervisor do?", deps=StatelessDeps())

    assert result.output == "an answer with no citation"


async def test_a_follow_up_answered_from_retained_evidence_is_enforced(temp_db_path):
    """The multi-turn case is the one enforcement exists for.

    A follow-up about something already cited needs no new search — the evidence is
    still on the wire, whether in a capsule or in full — so requiring a fresh
    evidence outcome let exactly those answers through undeclared.
    """
    rag = create_rag(db_path=temp_db_path, config=AppConfig())
    turns = iter(
        [
            [ToolCallPart("search", {"query": "supervisor"}, "call-1")],
            [ToolCallPart("cite", {"chunk_ids": ["chunk-1"]}, "call-2")],
            [TextPart("first answer")],
            [TextPart("a follow-up answered from what is already here")],
            [TextPart("a follow-up answered from what is already here")],
        ]
    )
    sent: list[list[Any]] = []

    async def model(messages, _info):
        sent.append(list(messages))
        return ModelResponse(parts=next(turns))

    agent = Agent(
        FunctionModel(model), deps_type=Deps, capabilities=[rag, create_policy()]
    )
    deps = Deps()

    with patch.object(RAGCapability, "_search", stub_search):
        first = await agent.run("what does the supervisor do?", deps=deps)
        await agent.run(
            "and what colour is the box in it?",
            deps=deps,
            message_history=first.all_messages(),
        )

    assert [p for p in prompts_of(sent[-1]) if REDIRECT_HINT in p]


async def test_a_conversation_that_never_cited_anything_is_still_left_alone(
    temp_db_path,
):
    """A greeting has nothing to declare, and no evidence exists to declare from."""
    _, _, sent = await run_with_policy(temp_db_path, [[TextPart("hello back")]])

    assert not [p for p in prompts_of(sent[-1]) if REDIRECT_HINT in p]


async def test_a_resumed_question_is_not_redirected_twice(temp_db_path):
    """Once per question has to mean once, across every run of that question.

    Tracking it on the run instance forgot it at the next `for_run`, so resuming an
    interrupted question asked for the citation again.
    """
    rag = create_rag(db_path=temp_db_path, config=AppConfig())
    turns = iter(
        [
            [ToolCallPart("search", {"query": "supervisor"}, "call-1")],
            [TextPart("uncited")],
            [TextPart("uncited again")],
            [TextPart("uncited a third time")],
            [TextPart("uncited a fourth time")],
            [TextPart("uncited a fifth time")],
        ]
    )
    sent: list[list[Any]] = []

    async def model(messages, _info):
        sent.append(list(messages))
        return ModelResponse(parts=next(turns))

    agent = Agent(
        FunctionModel(model), deps_type=Deps, capabilities=[rag, create_policy()]
    )
    deps = Deps()

    with patch.object(RAGCapability, "_search", stub_search):
        first = await agent.run("what does the supervisor do?", deps=deps)
        # The same question again, continued rather than asked anew: a run that
        # ends awaiting external work leaves the question in progress.
        deps.state["rag"]["evidence"]["in_progress"] = True
        await agent.run(
            deps=deps,
            message_history=[
                *first.all_messages(),
                ModelResponse(parts=[ToolCallPart("external_tool", {}, "call-9")]),
            ],
            deferred_tool_results=DeferredToolResults(
                calls={"call-9": "external result"}
            ),
        )

    redirects = [p for p in prompts_of(sent[-1]) if REDIRECT_HINT in p]
    assert len(redirects) == 1


async def test_a_user_quoting_the_redirect_does_not_suppress_enforcement(temp_db_path):
    """Prose is not proof that we asked: a user can write any phrase.

    Matching the wording let a question that merely mentioned it pass as already
    asked, which silently switches enforcement off.
    """
    rag = create_rag(db_path=temp_db_path, config=AppConfig())
    turns = iter(
        [
            [ToolCallPart("search", {"query": "supervisor"}, "call-1")],
            [TextPart("uncited")],
            [TextPart("uncited again")],
        ]
    )
    sent: list[list[Any]] = []

    async def model(messages, _info):
        sent.append(list(messages))
        return ModelResponse(parts=next(turns))

    agent = Agent(
        FunctionModel(model), deps_type=Deps, capabilities=[rag, create_policy()]
    )

    with patch.object(RAGCapability, "_search", stub_search):
        await agent.run(
            "Please record what grounded the answer you already gave, in your notes.",
            deps=Deps(),
        )

    assert [p for p in prompts_of(sent[-1]) if CITATION_REDIRECT_TAG in p]


class Answer(BaseModel):
    """A structured output, which the model returns through an output tool."""

    text: str


@pytest.mark.parametrize("output_mode", ["tool", "native", "prompted"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_invalid_structured_output_does_not_queue_redirects(
    temp_db_path, output_mode, streaming
):
    rag = create_rag(db_path=temp_db_path, config=AppConfig())
    turns = iter(
        [
            [ToolCallPart("search", {"query": "supervisor"}, "search-1")],
            [ToolCallPart("final_result", {}, "invalid-output")],
            [ToolCallPart("final_result", {"text": "answer"}, "valid-output")],
            [ToolCallPart("cite", {"chunk_ids": ["chunk-1"]}, "cite-1")],
            [ToolCallPart("final_result", {"text": "answer"}, "final-output")],
        ]
    )
    sent: list[list[Any]] = []

    async def model(messages, _info):
        sent.append(list(messages))
        return ModelResponse(
            parts=[
                TextPart(json.dumps(part.args))
                if output_mode != "tool" and part.tool_name == "final_result"
                else part
                for part in next(turns)
            ]
        )

    async def stream_model(messages, info):
        response = await model(messages, info)
        for index, part in enumerate(response.parts):
            if isinstance(part, TextPart):
                yield part.content
            elif isinstance(part, ToolCallPart):
                yield {
                    index: DeltaToolCall(
                        name=part.tool_name,
                        json_args=json.dumps(part.args),
                        tool_call_id=part.tool_call_id,
                    )
                }

    async def consume_events(_ctx, events):
        async for _event in events:
            pass

    agent = Agent(
        FunctionModel(model, stream_function=stream_model),
        deps_type=Deps,
        output_type={
            "tool": Answer,
            "native": NativeOutput(Answer),
            "prompted": PromptedOutput(Answer),
        }[output_mode],
        retries={"output": 2},
        capabilities=[rag, create_policy()],
    )
    deps = Deps()

    with patch.object(RAGCapability, "_search", stub_search):
        result = await agent.run(
            "what does the supervisor do?",
            deps=deps,
            event_stream_handler=consume_events if streaming else None,
        )

    redirects = [
        prompt for prompt in prompts_of(sent[-1]) if CITATION_REDIRECT_TAG in prompt
    ]
    assert not redirects, f"Expected no queued redirects; received {len(redirects)}"
    assert result.output == Answer(text="answer")
    assert deps.state["rag"]["citations"] == ["chunk-1"]
    assert deps.state["citation_policy"]["violations"] == []
    retries = [
        part
        for message in sent[-1]
        for part in message.parts
        if isinstance(part, RetryPromptPart)
    ]
    assert len(retries) == 2
    assert isinstance(retries[0].content, list)
    assert CITATION_REDIRECT_TAG in retries[1].content


@pytest.mark.parametrize("declaration", [None, [], ["chunk-1"]])
async def test_structured_answers_with_nothing_missing_are_left_alone(
    temp_db_path, declaration
):
    responses = []
    if declaration is not None:
        responses.extend(
            [
                [ToolCallPart("search", {"query": "supervisor"}, "search-1")],
                [ToolCallPart("cite", {"chunk_ids": declaration}, "cite-1")],
            ]
        )
    responses.append([ToolCallPart("final_result", {"text": "answer"}, "output-1")])
    turns = iter(responses)

    async def model(_messages, _info):
        return ModelResponse(parts=next(turns))

    agent = Agent(
        FunctionModel(model),
        deps_type=Deps,
        output_type=Answer,
        capabilities=[
            create_rag(db_path=temp_db_path, config=AppConfig()),
            create_policy(),
        ],
    )
    deps = Deps()
    with patch.object(RAGCapability, "_search", stub_search):
        result = await agent.run("what does the supervisor do?", deps=deps)

    assert result.output == Answer(text="answer")
    assert result.usage.requests == len(responses)
    assert deps.state["citation_policy"]["violations"] == []


async def test_partial_structured_output_does_not_enforce_citations():
    policy = create_policy()
    output = Answer(text="partial")
    ctx = RunContext(
        deps=Deps(),
        model=FunctionModel(lambda _messages, _info: ModelResponse(parts=[])),
        usage=RunUsage(),
        partial_output=True,
    )
    output_context = OutputContext(
        mode="tool", output_type=Answer, object_def=None, has_function=False
    )
    with patch("haiku.rag.capabilities.policy.discover_evidence") as discover:
        result = await policy.after_output_validate(
            ctx, output_context=output_context, output=output
        )

    assert result is output
    discover.assert_not_called()


@pytest.mark.parametrize("deps_type", [Deps, StatelessDeps])
async def test_structured_answer_with_unavailable_cite_is_not_retried(
    temp_db_path, deps_type
):
    turns = iter(
        [
            [ToolCallPart("search", {"query": "supervisor"}, "search-1")],
            [ToolCallPart("final_result", {"text": "answer"}, "output-1")],
        ]
    )

    async def model(_messages, _info):
        return ModelResponse(parts=next(turns))

    agent = Agent(
        FunctionModel(model),
        deps_type=deps_type,
        output_type=Answer,
        capabilities=[
            create_rag(db_path=temp_db_path, config=AppConfig()),
            create_policy(),
        ],
    )
    deps = deps_type()
    with (
        patch.object(RAGCapability, "_search", stub_search),
        patch.object(RAGCapability, "cite_available", property(lambda _self: False)),
    ):
        result = await agent.run("what does the supervisor do?", deps=deps)

    assert result.output == Answer(text="answer")
    assert result.usage.requests == 2
    if isinstance(deps, Deps):
        assert deps.state["citation_policy"]["violations"] == [0]


async def test_a_structured_output_answer_does_not_escape_enforcement(temp_db_path):
    """A structured answer receives one retry before a violation is recorded."""
    rag = create_rag(db_path=temp_db_path, config=AppConfig())
    turns = iter(
        [
            [ToolCallPart("search", {"query": "supervisor"}, "call-1")],
            [
                ToolCallPart(
                    "final_result", {"text": "uncited structured answer"}, "out"
                )
            ],
            [
                ToolCallPart(
                    "final_result", {"text": "uncited structured answer"}, "out"
                )
            ],
        ]
    )
    sent: list[list[Any]] = []

    async def model(messages, _info):
        sent.append(list(messages))
        return ModelResponse(parts=next(turns))

    agent = Agent(
        FunctionModel(model),
        deps_type=Deps,
        output_type=Answer,
        capabilities=[rag, create_policy()],
    )
    deps = Deps()

    with patch.object(RAGCapability, "_search", stub_search):
        await agent.run("what does the supervisor do?", deps=deps)

    retries = [
        part
        for message in sent[-1]
        for part in message.parts
        if isinstance(part, RetryPromptPart)
        and isinstance(part.content, str)
        and CITATION_REDIRECT_TAG in part.content
    ]
    assert len(retries) == 1
    assert retries[0].tool_name == "final_result"
    assert not [p for p in prompts_of(sent[-1]) if CITATION_REDIRECT_TAG in p]
    assert deps.state["citation_policy"]["violations"] == [0]


async def test_a_question_asked_once_and_still_undeclared_is_recorded(temp_db_path):
    """Being asked is not an outcome; the question still ended undeclared.

    Returning early on the redirect marker meant a question that was asked, ignored,
    and then finished — with the cite tool possibly gone by that point — was neither
    redirected again nor recorded anywhere.
    """
    rag = create_rag(db_path=temp_db_path, config=AppConfig())
    turns = iter(
        [
            [ToolCallPart("search", {"query": "supervisor"}, "call-1")],
            [TextPart("uncited")],
            [TextPart("still uncited after being asked")],
        ]
    )

    async def model(_messages, _info):
        return ModelResponse(parts=next(turns))

    agent = Agent(
        FunctionModel(model), deps_type=Deps, capabilities=[rag, create_policy()]
    )
    deps = Deps()

    with patch.object(RAGCapability, "_search", stub_search):
        await agent.run("what does the supervisor do?", deps=deps)

    assert deps.state["citation_policy"]["violations"] == [0]
