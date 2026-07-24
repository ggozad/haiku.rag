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
