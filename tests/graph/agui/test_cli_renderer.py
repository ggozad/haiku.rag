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
    emit_state_delta,
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
async def test_renderer_multiple_snapshots():
    """Test that renderer handles multiple snapshots without errors."""
    renderer = AGUIConsoleRenderer()

    events = [
        emit_state_snapshot(SimpleState(value=1)),
        emit_state_snapshot(SimpleState(value=2)),
    ]

    # Should render both snapshots without error
    result = await renderer.render(async_gen(events))
    assert result is None  # No run finished event


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
async def test_renderer_state_snapshots():
    """Test that state snapshots are rendered."""
    renderer = AGUIConsoleRenderer()

    events = [
        emit_state_snapshot(SimpleState(value=1)),
        emit_state_snapshot(SimpleState(value=2)),
        emit_run_finished("t1", "r1", {"done": True}),
    ]

    result = await renderer.render(async_gen(events))
    assert result == {"done": True}


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
async def test_renderer_snapshot_then_delta():
    """Test that renderer handles snapshot followed by deltas."""
    renderer = AGUIConsoleRenderer()

    state1 = SimpleState(value=1)
    state2 = SimpleState(value=2)
    state3 = SimpleState(value=3)

    events = [
        emit_state_snapshot(state1),  # Initial snapshot
        emit_state_delta(state1, state2),  # Delta to value=2
        emit_state_delta(state2, state3),  # Delta to value=3
        emit_run_finished("t1", "r1", {"complete": True}),
    ]

    result = await renderer.render(async_gen(events))
    assert result == {"complete": True}


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


@pytest.mark.asyncio
async def test_renderer_state_delta():
    """Test that renderer renders state deltas."""
    renderer = AGUIConsoleRenderer()

    state1 = SimpleState(value=1)
    state2 = SimpleState(value=2)

    events = [
        emit_state_snapshot(state1),  # Initial state
        emit_state_delta(state1, state2),  # Delta update
        emit_run_finished("t1", "r1", None),
    ]

    result = await renderer.render(async_gen(events))
    assert result is None


@pytest.mark.asyncio
async def test_renderer_state_delta_without_initial():
    """Test that renderer handles delta without initial state gracefully."""
    renderer = AGUIConsoleRenderer()

    state1 = SimpleState(value=1)
    state2 = SimpleState(value=2)

    # Send delta without initial snapshot - should still render it
    events = [
        emit_state_delta(state1, state2),
        emit_run_finished("t1", "r1", {"ok": True}),
    ]

    result = await renderer.render(async_gen(events))
    assert result == {"ok": True}
