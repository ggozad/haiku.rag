from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel

from evaluations.capability_runner import _count_tool_traffic, run_capability_question
from haiku.rag.capabilities.analysis import create_capability as create_analysis
from haiku.rag.capabilities.rag import create_capability as create_rag
from haiku.rag.config.models import AppConfig


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

    search_calls, rejected_searches, failed_tools, requests = _count_tool_traffic(
        messages, "analysis"
    )

    assert search_calls == 0
    assert rejected_searches == 0
    assert failed_tools == 1
    assert requests == 2


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

    search_calls, rejected, _failed, requests = _count_tool_traffic(
        messages, "analysis"
    )

    assert search_calls == 2
    assert rejected == 1
    assert requests == 2


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
        run.return_value = SimpleNamespace(output="done", all_messages=lambda: [])

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
