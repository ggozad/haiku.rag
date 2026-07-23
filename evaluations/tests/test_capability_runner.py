from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai.models.test import TestModel

from evaluations.capability_runner import run_capability_question
from haiku.rag.capabilities.analysis import create_capability as create_analysis
from haiku.rag.capabilities.rag import create_capability as create_rag
from haiku.rag.config.models import AppConfig


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
        run.return_value = SimpleNamespace(output="done")

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
        run.return_value = SimpleNamespace(output="done")

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
