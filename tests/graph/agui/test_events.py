"""Tests for AG-UI event creation utilities."""

from pydantic import BaseModel

from haiku.rag.graph.agui.events import (
    emit_activity,
    emit_run_error,
    emit_run_finished,
    emit_run_started,
    emit_state_snapshot,
    emit_step_finished,
    emit_step_started,
    emit_text_message,
    emit_tool_call_args,
    emit_tool_call_end,
    emit_tool_call_start,
)


class TestState(BaseModel):
    """Test state model."""

    value: int


class TestResult(BaseModel):
    """Test result model."""

    status: str


def test_emit_run_started():
    """Test RUN_STARTED event creation."""
    event = emit_run_started("thread-1", "run-1")

    assert event["type"] == "RUN_STARTED"
    assert event["threadId"] == "thread-1"
    assert event["runId"] == "run-1"
    assert "input" not in event


def test_emit_run_started_with_input():
    """Test RUN_STARTED event with input data."""
    event = emit_run_started("thread-1", "run-1", input_data="test input")

    assert event["type"] == "RUN_STARTED"
    assert event["input"] == "test input"


def test_emit_run_finished():
    """Test RUN_FINISHED event creation."""
    result = TestResult(status="complete")
    event = emit_run_finished("thread-1", "run-1", result)

    assert event["type"] == "RUN_FINISHED"
    assert event["threadId"] == "thread-1"
    assert event["runId"] == "run-1"
    assert event["result"] == {"status": "complete"}


def test_emit_run_finished_with_dict():
    """Test RUN_FINISHED event with dict result."""
    result = {"status": "complete", "count": 42}
    event = emit_run_finished("thread-1", "run-1", result)

    assert event["type"] == "RUN_FINISHED"
    assert event["result"] == result


def test_emit_run_error():
    """Test RUN_ERROR event creation."""
    event = emit_run_error("Something went wrong")

    assert event["type"] == "RUN_ERROR"
    assert event["message"] == "Something went wrong"
    assert "code" not in event


def test_emit_run_error_with_code():
    """Test RUN_ERROR event with error code."""
    event = emit_run_error("Something went wrong", code="ERR_001")

    assert event["type"] == "RUN_ERROR"
    assert event["message"] == "Something went wrong"
    assert event["code"] == "ERR_001"


def test_emit_step_started():
    """Test STEP_STARTED event creation."""
    event = emit_step_started("plan")

    assert event["type"] == "STEP_STARTED"
    assert event["stepName"] == "plan"


def test_emit_step_finished():
    """Test STEP_FINISHED event creation."""
    event = emit_step_finished("plan")

    assert event["type"] == "STEP_FINISHED"
    assert event["stepName"] == "plan"


def test_emit_text_message():
    """Test TEXT_MESSAGE_CHUNK event creation."""
    event = emit_text_message("Hello world")

    assert event["type"] == "TEXT_MESSAGE_CHUNK"
    assert event["delta"] == "Hello world"
    assert event["role"] == "assistant"
    assert "messageId" in event


def test_emit_text_message_with_role():
    """Test TEXT_MESSAGE_CHUNK event with custom role."""
    event = emit_text_message("Hello", role="user")

    assert event["type"] == "TEXT_MESSAGE_CHUNK"
    assert event["role"] == "user"


def test_emit_state_snapshot():
    """Test STATE_SNAPSHOT event creation."""
    state = TestState(value=42)
    event = emit_state_snapshot(state)

    assert event["type"] == "STATE_SNAPSHOT"
    assert event["snapshot"] == {"value": 42}


def test_emit_activity():
    """Test ACTIVITY_SNAPSHOT event creation."""
    event = emit_activity("msg-1", "processing", {"message": "Working on task"})

    assert event["type"] == "ACTIVITY_SNAPSHOT"
    assert event["messageId"] == "msg-1"
    assert event["activityType"] == "processing"
    assert event["content"] == {"message": "Working on task"}


def test_emit_tool_call_start():
    """Test TOOL_CALL_START event creation."""
    event = emit_tool_call_start("call-1", "search_documents")

    assert event["type"] == "TOOL_CALL_START"
    assert event["toolCallId"] == "call-1"
    assert event["toolCallName"] == "search_documents"
    assert "parentMessageId" not in event


def test_emit_tool_call_start_with_parent():
    """Test TOOL_CALL_START event with parent message ID."""
    event = emit_tool_call_start("call-1", "search", parent_message_id="msg-1")

    assert event["type"] == "TOOL_CALL_START"
    assert event["toolCallId"] == "call-1"
    assert event["toolCallName"] == "search"
    assert event["parentMessageId"] == "msg-1"


def test_emit_tool_call_args():
    """Test TOOL_CALL_ARGS event creation."""
    import json

    args = {"query": "test query", "limit": 10}
    event = emit_tool_call_args("call-1", args)

    assert event["type"] == "TOOL_CALL_ARGS"
    assert event["toolCallId"] == "call-1"
    assert event["delta"] == json.dumps(args)


def test_emit_tool_call_end():
    """Test TOOL_CALL_END event creation."""
    event = emit_tool_call_end("call-1")

    assert event["type"] == "TOOL_CALL_END"
    assert event["toolCallId"] == "call-1"


def test_event_structure_consistency():
    """Test that all events have consistent structure."""
    events = [
        emit_run_started("t1", "r1"),
        emit_run_finished("t1", "r1", {"result": "done"}),
        emit_run_error("error"),
        emit_step_started("step1"),
        emit_step_finished("step1"),
        emit_text_message("text"),
        emit_state_snapshot(TestState(value=1)),
        emit_activity("m1", "type", {"content": "value"}),
        emit_tool_call_start("c1", "tool"),
        emit_tool_call_args("c1", {"arg": "value"}),
        emit_tool_call_end("c1"),
    ]

    for event in events:
        assert isinstance(event, dict)
        assert "type" in event
        assert isinstance(event["type"], str)
        assert event["type"].isupper()  # Event types are uppercase
