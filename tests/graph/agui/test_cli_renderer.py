"""Tests for AGUIConsoleRenderer."""

import pytest
from pydantic import BaseModel
from rich.console import Console

from haiku.rag.graph.agui.cli_renderer import AGUIConsoleRenderer
from haiku.rag.graph.agui.events import (
    emit_activity,
    emit_run_error,
    emit_run_finished,
    emit_run_started,
    emit_state_snapshot,
    emit_step_finished,
    emit_step_started,
    emit_text_message,
)


class SimpleState(BaseModel):
    """Simple state for testing."""

    value: int


async def async_gen(items):
    """Helper to create async generator from list."""
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_renderer_basic_flow():
    """Test basic event rendering flow."""
    console = Console(file=None, force_terminal=False)  # Don't actually print
    renderer = AGUIConsoleRenderer(console)

    events = [
        emit_run_started("t1", "r1"),
        emit_state_snapshot(SimpleState(value=1)),
        emit_step_started("plan"),
        emit_activity("m1", "planning", "Planning research"),
        emit_run_finished("t1", "r1", {"status": "complete"}),
    ]

    result = await renderer.render(async_gen(events))

    assert result == {"status": "complete"}


@pytest.mark.asyncio
async def test_renderer_state_diff():
    """Test that renderer computes state diffs correctly."""
    renderer = AGUIConsoleRenderer()

    # First state
    old_state = {"value": 1, "text": "old", "nested": {"a": 1, "b": 2}}
    # Second state with changes
    new_state = {"value": 2, "text": "old", "nested": {"a": 1, "b": 3, "c": 4}}

    diff = renderer._compute_diff(old_state, new_state)

    assert diff == {
        "value": 2,
        "nested": {"b": 3, "c": 4},  # Only changed/new fields in nested
    }


@pytest.mark.asyncio
async def test_renderer_state_diff_no_changes():
    """Test that no diff is computed when states are equal."""
    renderer = AGUIConsoleRenderer()

    state = {"value": 1, "text": "test"}
    diff = renderer._compute_diff(state, state)

    assert diff == {}


@pytest.mark.asyncio
async def test_renderer_handles_all_event_types():
    """Test that renderer handles all event types without errors."""
    renderer = AGUIConsoleRenderer()

    events = [
        emit_run_started("t1", "r1"),
        emit_state_snapshot(SimpleState(value=1)),
        emit_step_started("step1"),
        emit_step_finished("step1"),
        emit_text_message("message"),
        emit_activity("m1", "work", "Working"),
        emit_run_error("error occurred"),
        emit_run_finished("t1", "r1", {"result": "done"}),
    ]

    # Should not raise any exceptions
    result = await renderer.render(async_gen(events))
    assert result == {"result": "done"}


@pytest.mark.asyncio
async def test_renderer_initial_state():
    """Test that initial state is rendered."""
    renderer = AGUIConsoleRenderer()

    events = [
        emit_state_snapshot(SimpleState(value=1)),
        emit_state_snapshot(SimpleState(value=2)),
    ]

    await renderer.render(async_gen(events))

    # After processing, internal state should be the last state
    assert renderer._state == {"value": 2}


@pytest.mark.asyncio
async def test_renderer_no_result():
    """Test renderer when no RUN_FINISHED event."""
    renderer = AGUIConsoleRenderer()

    events = [
        emit_run_started("t1", "r1"),
        emit_step_started("step1"),
    ]

    result = await renderer.render(async_gen(events))
    assert result is None


@pytest.mark.asyncio
async def test_renderer_nested_state_diff():
    """Test nested state diff computation."""
    renderer = AGUIConsoleRenderer()

    old = {"level1": {"level2": {"value": 1, "text": "old"}, "other": "same"}}

    new = {"level1": {"level2": {"value": 2, "text": "old"}, "other": "same"}}

    diff = renderer._compute_diff(old, new)

    # Should only show the changed nested value
    assert diff == {"level1": {"level2": {"value": 2}}}


@pytest.mark.asyncio
async def test_renderer_with_empty_state():
    """Test renderer handles empty state gracefully."""
    renderer = AGUIConsoleRenderer()

    # Create a state with no changes
    events = [
        emit_step_started("step1"),  # No state snapshot
        emit_run_finished("t1", "r1", {"result": "ok"}),
    ]

    result = await renderer.render(async_gen(events))
    assert result == {"result": "ok"}
