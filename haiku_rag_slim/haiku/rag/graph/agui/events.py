"""Generic AG-UI event creation utilities for any graph."""

from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from haiku.rag.graph.agui.state import compute_state_delta

# Type aliases for AG-UI events (actual types from ag_ui.core will be used at runtime)
AGUIEvent = dict[str, Any]


def emit_run_started(
    thread_id: str, run_id: str, input_data: str | None = None
) -> dict[str, Any]:
    """Create a RunStarted event.

    Args:
        thread_id: Unique identifier for the conversation thread
        run_id: Unique identifier for this run
        input_data: Optional input that started the run

    Returns:
        RunStarted event dict
    """
    event: dict[str, Any] = {
        "type": "RUN_STARTED",
        "threadId": thread_id,
        "runId": run_id,
    }
    if input_data:
        event["input"] = input_data
    return event


def emit_run_finished(thread_id: str, run_id: str, result: Any) -> dict[str, Any]:
    """Create a RunFinished event.

    Args:
        thread_id: Unique identifier for the conversation thread
        run_id: Unique identifier for this run
        result: The final result of the run

    Returns:
        RunFinished event dict
    """
    # Convert result to dict if it's a Pydantic model
    if hasattr(result, "model_dump"):
        result = result.model_dump()

    return {
        "type": "RUN_FINISHED",
        "threadId": thread_id,
        "runId": run_id,
        "result": result,
    }


def emit_run_error(message: str, code: str | None = None) -> dict[str, Any]:
    """Create a RunError event.

    Args:
        message: Error message
        code: Optional error code

    Returns:
        RunError event dict
    """
    event: dict[str, Any] = {
        "type": "RUN_ERROR",
        "message": message,
    }
    if code:
        event["code"] = code
    return event


def emit_step_started(step_name: str) -> dict[str, Any]:
    """Create a StepStarted event.

    Args:
        step_name: Name of the step being started

    Returns:
        StepStarted event dict
    """
    return {
        "type": "STEP_STARTED",
        "stepName": step_name,
    }


def emit_step_finished(step_name: str) -> dict[str, Any]:
    """Create a StepFinished event.

    Args:
        step_name: Name of the step that finished

    Returns:
        StepFinished event dict
    """
    return {
        "type": "STEP_FINISHED",
        "stepName": step_name,
    }


def emit_text_message(content: str, role: str = "assistant") -> dict[str, Any]:
    """Create a TextMessageChunk event (convenience wrapper).

    This creates a complete text message in one event.

    Args:
        content: The message content
        role: The role of the sender (default: assistant)

    Returns:
        TextMessageChunk event dict
    """
    message_id = str(uuid4())
    return {
        "type": "TEXT_MESSAGE_CHUNK",
        "messageId": message_id,
        "role": role,
        "delta": content,
    }


def emit_text_message_start(message_id: str, role: str = "assistant") -> dict[str, Any]:
    """Create a TextMessageStart event.

    Args:
        message_id: Unique identifier for this message
        role: The role of the sender

    Returns:
        TextMessageStart event dict
    """
    return {
        "type": "TEXT_MESSAGE_START",
        "messageId": message_id,
        "role": role,
    }


def emit_text_message_content(message_id: str, delta: str) -> dict[str, Any]:
    """Create a TextMessageContent event.

    Args:
        message_id: Identifier for the message being streamed
        delta: Content chunk to append

    Returns:
        TextMessageContent event dict
    """
    return {
        "type": "TEXT_MESSAGE_CONTENT",
        "messageId": message_id,
        "delta": delta,
    }


def emit_text_message_end(message_id: str) -> dict[str, Any]:
    """Create a TextMessageEnd event.

    Args:
        message_id: Identifier for the message being completed

    Returns:
        TextMessageEnd event dict
    """
    return {
        "type": "TEXT_MESSAGE_END",
        "messageId": message_id,
    }


def emit_state_snapshot(state: BaseModel) -> dict[str, Any]:
    """Create a StateSnapshot event.

    Args:
        state: The complete state to snapshot (any Pydantic BaseModel)

    Returns:
        StateSnapshot event dict
    """
    return {
        "type": "STATE_SNAPSHOT",
        "snapshot": state.model_dump(),
    }


def emit_state_delta(old_state: BaseModel, new_state: BaseModel) -> dict[str, Any]:
    """Create a StateDelta event with JSON Patch operations.

    Args:
        old_state: Previous state (any Pydantic BaseModel)
        new_state: Current state (same type as old_state)

    Returns:
        StateDelta event dict
    """
    delta = compute_state_delta(old_state, new_state)
    return {
        "type": "STATE_DELTA",
        "delta": delta,
    }


def emit_activity(
    message_id: str,
    activity_type: str,
    content: str,
) -> dict[str, Any]:
    """Create an ActivitySnapshot event.

    Args:
        message_id: Message ID to associate activity with (required)
        activity_type: Type of activity (e.g., "planning", "searching")
        content: Description of the activity

    Returns:
        ActivitySnapshot event dict
    """
    return {
        "type": "ACTIVITY_SNAPSHOT",
        "messageId": message_id,
        "activityType": activity_type,
        "content": content,
    }


def emit_activity_delta(
    message_id: str,
    activity_type: str,
    patch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create an ActivityDelta event with JSON Patch operations.

    Args:
        message_id: Message ID of the activity being updated
        activity_type: Type of activity being updated
        patch: JSON Patch operations to apply

    Returns:
        ActivityDelta event dict
    """
    return {
        "type": "ACTIVITY_DELTA",
        "messageId": message_id,
        "activityType": activity_type,
        "patch": patch,
    }
