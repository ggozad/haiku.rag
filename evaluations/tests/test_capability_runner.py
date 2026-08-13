from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel

from evaluations.capability_runner import (
    CapabilityRunResult,
    _count_tool_traffic,
    run_capability_question,
)
from haiku.rag.capabilities.analysis import create_capability as create_analysis
from haiku.rag.capabilities.rag import create_capability as create_rag
from haiku.rag.config.models import AppConfig

ANALYSIS_TOOLS = frozenset(
    {"analysis_search", "analysis_execute_code", "analysis_cite"}
)


def test_count_tool_traffic_sees_a_rejected_cite_call():
    """`_cite` rejects with ModelRetry, which is not a failed ToolReturnPart."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content="q")]),
        ModelResponse(parts=[ToolCallPart("analysis_cite", {"chunk_ids": []})]),
        ModelRequest(
            parts=[
                RetryPromptPart(
                    tool_name="analysis_cite",
                    content="No citations registered: chunk_ids was empty.",
                    tool_call_id="1",
                )
            ]
        ),
        ModelResponse(parts=[TextPart("done")]),
    ]

    traffic = _count_tool_traffic(messages, "analysis", ANALYSIS_TOOLS)

    assert traffic.n_failed_tools == 1
    assert traffic.n_rejected_searches == 0


def test_count_tool_traffic_separates_search_rejections_from_code_errors():
    """A crash in model-written Python must not read as budget exhaustion."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content="q")]),
        ModelResponse(parts=[ToolCallPart("analysis_execute_code", {"code": "1/0"})]),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="analysis_execute_code",
                    content="ZeroDivisionError",
                    tool_call_id="1",
                    outcome="failed",
                )
            ]
        ),
        ModelResponse(parts=[TextPart("done")]),
    ]

    traffic = _count_tool_traffic(messages, "analysis", ANALYSIS_TOOLS)

    assert traffic.n_search_calls == 0
    assert traffic.n_rejected_searches == 0
    assert traffic.n_failed_tools == 1
    assert traffic.n_requests == 2


def test_count_tool_traffic_counts_attempts_not_distinct_queries():
    """Rejected and repeated calls both count; `state.searches` hides them."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content="q")]),
        ModelResponse(
            parts=[
                ToolCallPart("analysis_search", {"query": "same"}),
                ToolCallPart("analysis_search", {"query": "same"}),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="analysis_search", content="results", tool_call_id="1"
                ),
                ToolReturnPart(
                    tool_name="analysis_search",
                    content="Search limit reached.",
                    tool_call_id="2",
                    outcome="failed",
                ),
            ]
        ),
        ModelResponse(parts=[TextPart("done")]),
    ]

    traffic = _count_tool_traffic(messages, "analysis", ANALYSIS_TOOLS)

    assert traffic.n_search_calls == 2
    assert traffic.n_rejected_searches == 1
    assert traffic.n_requests == 2


async def test_runs_rag_capability_without_legacy_capability_layer(tmp_path):
    result = await run_capability_question(
        create_rag,
        tmp_path / "rag.lancedb",
        AppConfig(),
        "hello",
        TestModel(call_tools=[]),
        document_filter="uri = 'manual.pdf'",
    )

    assert result.answer == "success (no tool calls)"
    assert result.cited_uris == []
    assert result.n_searches == 0
    assert result.citation_status == "missing"


async def test_runs_analysis_capability_without_legacy_capability_layer(tmp_path):
    result = await run_capability_question(
        create_analysis,
        tmp_path / "rag.lancedb",
        AppConfig(),
        "hello",
        TestModel(call_tools=[]),
        request_limit=5,
    )

    assert result.answer == "success (no tool calls)"
    assert result.n_executions == 0


@pytest.mark.parametrize(("override", "expected"), [(None, 30), (5, 5)])
async def test_analysis_capability_applies_request_limit(tmp_path, override, expected):
    capability = create_analysis(
        db_path=tmp_path / "rag.lancedb",
        config=AppConfig(),
        defer_loading=False,
    )
    with patch(
        "evaluations.capability_runner.Agent.run", new_callable=AsyncMock
    ) as run:
        run.return_value = SimpleNamespace(
            output="done", all_messages=lambda: [], new_messages=lambda: []
        )

        await run_capability_question(
            lambda **_kwargs: capability,
            tmp_path / "rag.lancedb",
            AppConfig(),
            "hello",
            TestModel(call_tools=[]),
            request_limit=override,
        )

    assert capability.request_limit == expected
    assert "usage_limits" not in run.call_args.kwargs


class TestCitationStatusDerivation:
    """`citation_status` distinguishes an answer that declared nothing
    (`missing`) from one that declared ungrounded (`ungrounded`) — refusals
    now cite an empty list."""

    def _result(self, record) -> CapabilityRunResult:
        from evaluations.capability_runner import ToolTraffic, _result_from_run
        from haiku.rag.capabilities.rag import RAGState

        state = RAGState(evidence=record)
        return _result_from_run("answer", state, ToolTraffic(0, 0, 0, 1))

    def test_grounded(self) -> None:
        from haiku.rag.capabilities.ledger import (
            CapabilityEvidenceRecord,
            CitationDeclaration,
            EvidenceRef,
        )

        record = CapabilityEvidenceRecord(
            question=2,
            latest_evidence_epoch=3,
            declaration=CitationDeclaration(
                question=2,
                epoch=5,
                refs=[EvidenceRef(capability="rag", chunk_id="c1")],
            ),
        )
        assert self._result(record).citation_status == "grounded"

    def test_ungrounded(self) -> None:
        from haiku.rag.capabilities.ledger import (
            CapabilityEvidenceRecord,
            CitationDeclaration,
        )

        record = CapabilityEvidenceRecord(
            question=2,
            latest_evidence_epoch=3,
            declaration=CitationDeclaration(question=2, epoch=5, refs=[]),
        )
        assert self._result(record).citation_status == "ungrounded"

    def test_missing(self) -> None:
        from haiku.rag.capabilities.ledger import CapabilityEvidenceRecord

        record = CapabilityEvidenceRecord(question=2, latest_evidence_epoch=3)
        assert self._result(record).citation_status == "missing"

    def test_none_without_a_question(self) -> None:
        """A record no run ever stamped (mocked runs) derives no status."""
        from haiku.rag.capabilities.ledger import CapabilityEvidenceRecord

        assert self._result(CapabilityEvidenceRecord()).citation_status is None


class TestPrefixToMessages:
    def test_maps_turns_to_model_messages(self) -> None:
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            TextPart,
            UserPromptPart,
        )

        from evaluations.capability_runner import prefix_to_messages
        from evaluations.config import Turn

        messages = prefix_to_messages(
            [
                Turn(speaker="user", text="who takes photos of planes?"),
                Turn(speaker="agent", text="Ground-to-air photographers."),
            ]
        )

        assert len(messages) == 2
        assert isinstance(messages[0], ModelRequest)
        assert isinstance(messages[0].parts[0], UserPromptPart)
        assert messages[0].parts[0].content == "who takes photos of planes?"
        assert isinstance(messages[1], ModelResponse)
        assert isinstance(messages[1].parts[0], TextPart)
        assert messages[1].parts[0].content == "Ground-to-air photographers."

    def test_empty_prefix(self) -> None:
        from evaluations.capability_runner import prefix_to_messages

        assert prefix_to_messages([]) == []


async def test_message_history_passed_to_agent_run(tmp_path):
    from evaluations.capability_runner import prefix_to_messages
    from evaluations.config import Turn

    history = prefix_to_messages([Turn(speaker="user", text="earlier question")])
    capability = create_rag(
        db_path=tmp_path / "rag.lancedb",
        config=AppConfig(),
        defer_loading=False,
    )
    with patch(
        "evaluations.capability_runner.Agent.run", new_callable=AsyncMock
    ) as run:
        run.return_value = SimpleNamespace(output="done", new_messages=lambda: [])

        await run_capability_question(
            lambda **_kwargs: capability,
            tmp_path / "rag.lancedb",
            AppConfig(),
            "follow-up question",
            TestModel(call_tools=[]),
            message_history=history,
        )

    assert run.call_args.kwargs["message_history"] is history


async def test_conversation_threads_own_messages_across_turns(tmp_path):
    """Each turn runs with the previous turn's full message history (including
    tool traffic), so prior-turn compaction operates on real history."""
    from evaluations.capability_runner import run_capability_conversation

    capability = create_rag(
        db_path=tmp_path / "rag.lancedb",
        config=AppConfig(),
        defer_loading=False,
    )
    histories: list[object] = []

    async def _run(question, deps=None, message_history=None):
        histories.append(message_history)
        return SimpleNamespace(
            output=f"answer to {question}",
            all_messages=lambda: [f"history after {question}"],
            new_messages=lambda: [],
        )

    with patch("evaluations.capability_runner.Agent.run", side_effect=_run):
        result = await run_capability_conversation(
            lambda **_kwargs: capability,
            tmp_path / "rag.lancedb",
            AppConfig(),
            ["q1", "q2", "q3"],
            TestModel(call_tools=[]),
        )

    assert [t.answer for t in result] == [
        "answer to q1",
        "answer to q2",
        "answer to q3",
    ]
    assert histories == [None, ["history after q1"], ["history after q2"]]


async def test_conversation_carries_one_state_dict_across_turns(tmp_path):
    """Capabilities read and write state through the deps dict; carrying the
    same dict across turns is what lets compaction see earlier questions'
    records instead of refusing."""
    from evaluations.capability_runner import run_capability_conversation

    deps_seen: list[object] = []

    async def _run(question, deps=None, message_history=None):
        deps_seen.append(deps)
        return SimpleNamespace(
            output="a", all_messages=lambda: [], new_messages=lambda: []
        )

    with patch("evaluations.capability_runner.Agent.run", side_effect=_run):
        await run_capability_conversation(
            create_rag,
            tmp_path / "rag.lancedb",
            AppConfig(),
            ["q1", "q2", "q3"],
            TestModel(call_tools=[]),
        )

    assert deps_seen[0] is deps_seen[1] is deps_seen[2]


@pytest.mark.parametrize(("compaction", "expected"), [(False, 0), (True, 1)])
async def test_conversation_compaction_registration(tmp_path, compaction, expected):
    from haiku.rag.capabilities.compaction import EvidenceCompactionCapability

    from evaluations.capability_runner import run_capability_conversation

    with patch("evaluations.capability_runner.Agent") as agent_cls:
        agent_cls.return_value.run = AsyncMock(
            return_value=SimpleNamespace(
                output="a", all_messages=lambda: [], new_messages=lambda: []
            )
        )
        await run_capability_conversation(
            create_rag,
            tmp_path / "rag.lancedb",
            AppConfig(),
            ["q1"],
            TestModel(call_tools=[]),
            compaction=compaction,
        )
        capabilities = agent_cls.call_args.kwargs["capabilities"]

    compactors = [
        c for c in capabilities if isinstance(c, EvidenceCompactionCapability)
    ]
    assert len(compactors) == expected
    assert len(capabilities) == 1 + expected


async def test_conversation_end_to_end_with_compaction(tmp_path):
    from evaluations.capability_runner import run_capability_conversation

    result = await run_capability_conversation(
        create_rag,
        tmp_path / "rag.lancedb",
        AppConfig(),
        ["first question", "follow-up"],
        TestModel(call_tools=[]),
        compaction=True,
    )

    assert [turn.answer for turn in result] == ["success (no tool calls)"] * 2


async def test_conversation_end_to_end_with_test_model(tmp_path):
    from evaluations.capability_runner import run_capability_conversation

    result = await run_capability_conversation(
        create_rag,
        tmp_path / "rag.lancedb",
        AppConfig(),
        ["first question", "follow-up"],
        TestModel(call_tools=[]),
    )

    assert len(result) == 2
    assert all(turn.answer == "success (no tool calls)" for turn in result)
    assert all(turn.cited_uris == [] for turn in result)


async def test_gold_prefix_run_answers_with_history(tmp_path):
    """End-to-end through a real Agent: the prefix rides along as history."""
    from evaluations.capability_runner import prefix_to_messages
    from evaluations.config import Turn

    history = prefix_to_messages(
        [
            Turn(speaker="user", text="who takes photos of planes?"),
            Turn(speaker="agent", text="Ground-to-air photographers."),
        ]
    )
    result = await run_capability_question(
        create_rag,
        tmp_path / "rag.lancedb",
        AppConfig(),
        "No, I meant photos in the air.",
        TestModel(call_tools=[]),
        message_history=history,
    )

    assert result.answer == "success (no tool calls)"
