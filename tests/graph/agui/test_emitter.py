"""Tests for AGUIEmitter."""

import asyncio

import pytest
from pydantic import BaseModel

from haiku.rag.graph.agui.emitter import AGUIEmitter


class TestState(BaseModel):
    """Test state model."""

    value: int
    text: str


class TestResult(BaseModel):
    """Test result model."""

    status: str


@pytest.mark.asyncio
async def test_emitter_lifecycle():
    """Test emitter lifecycle events."""
    emitter: AGUIEmitter[TestState, TestResult] = AGUIEmitter()

    initial_state = TestState(value=1, text="initial")
    emitter.start_run(initial_state)

    events = []
    async for event in emitter:
        events.append(event)
        if event["type"] == "RUN_STARTED":
            # Close after getting run started
            result = TestResult(status="complete")
            emitter.finish_run(result)
            await emitter.close()

    assert len(events) >= 3  # RUN_STARTED, STATE_SNAPSHOT, RUN_FINISHED
    assert events[0]["type"] == "RUN_STARTED"
    assert events[-1]["type"] == "RUN_FINISHED"
    assert events[-1]["result"] == {"status": "complete"}


@pytest.mark.asyncio
async def test_emitter_step_events():
    """Test step lifecycle events."""
    emitter: AGUIEmitter[TestState, TestResult] = AGUIEmitter()

    emitter.start_step("test_step")
    emitter.finish_step("test_step")
    await emitter.close()

    events = []
    async for event in emitter:
        events.append(event)

    step_events = [e for e in events if e["type"] in ("STEP_STARTED", "STEP_FINISHED")]
    assert len(step_events) == 2
    assert step_events[0]["type"] == "STEP_STARTED"
    assert step_events[0]["stepName"] == "test_step"
    assert step_events[1]["type"] == "STEP_FINISHED"
    assert step_events[1]["stepName"] == "test_step"


@pytest.mark.asyncio
async def test_emitter_state_updates():
    """Test state update events with snapshots."""
    emitter: AGUIEmitter[TestState, TestResult] = AGUIEmitter(use_deltas=False)

    state1 = TestState(value=1, text="first")
    state2 = TestState(value=2, text="second")

    emitter.update_state(state1)
    emitter.update_state(state2)
    await emitter.close()

    events = []
    async for event in emitter:
        events.append(event)

    state_events = [e for e in events if e["type"] == "STATE_SNAPSHOT"]
    assert len(state_events) == 2
    assert state_events[0]["snapshot"] == {"value": 1, "text": "first"}
    assert state_events[1]["snapshot"] == {"value": 2, "text": "second"}


@pytest.mark.asyncio
async def test_emitter_state_deltas():
    """Test state update events with deltas."""
    emitter: AGUIEmitter[TestState, TestResult] = AGUIEmitter(use_deltas=True)

    state1 = TestState(value=1, text="first")
    state2 = TestState(value=2, text="second")

    emitter.update_state(state1)
    emitter.update_state(state2)
    await emitter.close()

    events = []
    async for event in emitter:
        events.append(event)

    # First update should be a snapshot (no previous state)
    snapshot_events = [e for e in events if e["type"] == "STATE_SNAPSHOT"]
    assert len(snapshot_events) == 1
    assert snapshot_events[0]["snapshot"] == {"value": 1, "text": "first"}

    # Second update should be a delta
    delta_events = [e for e in events if e["type"] == "STATE_DELTA"]
    assert len(delta_events) == 1
    # Delta should contain replace operations for changed fields
    delta = delta_events[0]["delta"]
    assert isinstance(delta, list)
    assert len(delta) == 2  # Two fields changed
    assert any(op["path"] == "/value" and op["value"] == 2 for op in delta)
    assert any(op["path"] == "/text" and op["value"] == "second" for op in delta)


@pytest.mark.asyncio
async def test_emitter_activity_events():
    """Test activity events."""
    emitter: AGUIEmitter[TestState, TestResult] = AGUIEmitter()

    # Activity without a step
    emitter.update_activity("processing", {"message": "Processing data"})

    # Activity within a step (stepName explicitly included in content)
    emitter.start_step("analyze")
    emitter.update_activity(
        "done", {"stepName": "analyze", "message": "Completed"}, message_id="msg-1"
    )
    emitter.finish_step("analyze")

    await emitter.close()

    events = []
    async for event in emitter:
        events.append(event)

    activity_events = [e for e in events if e["type"] == "ACTIVITY_SNAPSHOT"]
    assert len(activity_events) == 2
    assert activity_events[0]["activityType"] == "processing"
    assert activity_events[0]["content"]["message"] == "Processing data"
    assert "stepName" not in activity_events[0]["content"]  # No step context

    assert activity_events[1]["messageId"] == "msg-1"
    assert activity_events[1]["activityType"] == "done"
    assert activity_events[1]["content"]["message"] == "Completed"
    assert activity_events[1]["content"]["stepName"] == "analyze"  # Has step context


@pytest.mark.asyncio
async def test_emitter_text_messages():
    """Test text message events."""
    emitter: AGUIEmitter[TestState, TestResult] = AGUIEmitter()

    emitter.log("Test message", role="assistant")
    emitter.log("Another message", role="user")
    await emitter.close()

    events = []
    async for event in emitter:
        events.append(event)

    text_events = [e for e in events if e["type"] == "TEXT_MESSAGE_CHUNK"]
    assert len(text_events) == 2
    assert text_events[0]["delta"] == "Test message"
    assert text_events[0]["role"] == "assistant"
    assert text_events[1]["delta"] == "Another message"
    assert text_events[1]["role"] == "user"


@pytest.mark.asyncio
async def test_emitter_error():
    """Test error event emission."""
    emitter: AGUIEmitter[TestState, TestResult] = AGUIEmitter()

    error = ValueError("Test error")
    emitter.error(error, code="TEST_ERROR")
    await emitter.close()

    events = []
    async for event in emitter:
        events.append(event)

    error_events = [e for e in events if e["type"] == "RUN_ERROR"]
    assert len(error_events) == 1
    assert error_events[0]["message"] == "Test error"
    assert error_events[0]["code"] == "TEST_ERROR"


@pytest.mark.asyncio
async def test_emitter_thread_and_run_ids():
    """Test thread and run ID management."""
    emitter: AGUIEmitter[TestState, TestResult] = AGUIEmitter(
        thread_id="thread-1", run_id="run-1"
    )

    assert emitter.thread_id == "thread-1"
    assert emitter.run_id == "run-1"

    initial_state = TestState(value=1, text="test")
    emitter.start_run(initial_state)
    await emitter.close()

    events = []
    async for event in emitter:
        events.append(event)

    run_started = [e for e in events if e["type"] == "RUN_STARTED"][0]
    assert run_started["threadId"] == "thread-1"
    assert run_started["runId"] == "run-1"


@pytest.mark.asyncio
async def test_emitter_generates_thread_id():
    """Test that thread ID is generated from state hash when not provided."""
    emitter: AGUIEmitter[TestState, TestResult] = AGUIEmitter()

    initial_state = TestState(value=42, text="test")
    emitter.start_run(initial_state)

    # Thread ID should be generated deterministically from state
    assert emitter.thread_id is not None
    assert len(emitter.thread_id) > 0

    await emitter.close()
    async for _ in emitter:
        pass


@pytest.mark.asyncio
async def test_emitter_closes_properly():
    """Test that emitter closes and stops iteration."""
    emitter: AGUIEmitter[TestState, TestResult] = AGUIEmitter()

    emitter.log("Message 1")
    await emitter.close()

    # Attempting to iterate after close should work and stop
    events = []
    async for event in emitter:
        events.append(event)

    # Should have received the message and then stopped
    assert len(events) == 1


@pytest.mark.asyncio
async def test_emitter_concurrent_emission():
    """Test that multiple events can be emitted concurrently."""
    emitter: AGUIEmitter[TestState, TestResult] = AGUIEmitter()

    async def emit_many():
        for i in range(10):
            emitter.log(f"Message {i}")
            await asyncio.sleep(0.001)  # Simulate some work
        await emitter.close()

    # Start emission in background
    emit_task = asyncio.create_task(emit_many())

    # Collect events
    events = []
    async for event in emitter:
        events.append(event)

    await emit_task

    # Should have all 10 messages
    text_events = [e for e in events if e["type"] == "TEXT_MESSAGE_CHUNK"]
    assert len(text_events) == 10


@pytest.mark.asyncio
async def test_emitter_concurrent_steps():
    """Test that multiple steps can run concurrently and finish independently."""
    emitter: AGUIEmitter[TestState, TestResult] = AGUIEmitter()

    async def run_step(step_name: str, delay: float):
        emitter.start_step(step_name)
        await asyncio.sleep(delay)
        emitter.finish_step(step_name)

    # Start collecting events in background
    events: list[dict] = []

    async def collect():
        async for event in emitter:
            events.append(event)

    collector = asyncio.create_task(collect())

    # Run three steps concurrently with different durations
    # Step A finishes last, Step B first, Step C middle
    await asyncio.gather(
        run_step("step_a", 0.03),
        run_step("step_b", 0.01),
        run_step("step_c", 0.02),
    )

    await emitter.close()
    await collector

    started = [e for e in events if e["type"] == "STEP_STARTED"]
    finished = [e for e in events if e["type"] == "STEP_FINISHED"]

    # All three steps should have started
    started_names = {e["stepName"] for e in started}
    assert started_names == {"step_a", "step_b", "step_c"}

    # All three steps should have finished
    finished_names = {e["stepName"] for e in finished}
    assert finished_names == {"step_a", "step_b", "step_c"}
