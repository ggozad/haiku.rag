"""Generic AG-UI protocol support for haiku.rag graphs."""

from haiku.rag.agui.emitter import AGUIEmitter
from haiku.rag.agui.events import (
    AGUIEvent,
    emit_activity,
    emit_activity_delta,
    emit_run_error,
    emit_run_finished,
    emit_run_started,
    emit_state_delta,
    emit_state_snapshot,
    emit_step_finished,
    emit_step_started,
    emit_text_message,
    emit_text_message_content,
    emit_text_message_end,
    emit_text_message_start,
)
from haiku.rag.agui.state import compute_state_delta

__all__ = [
    "AGUIEmitter",
    "AGUIEvent",
    "compute_state_delta",
    "emit_activity",
    "emit_activity_delta",
    "emit_run_error",
    "emit_run_finished",
    "emit_run_started",
    "emit_state_delta",
    "emit_state_snapshot",
    "emit_step_finished",
    "emit_step_started",
    "emit_text_message",
    "emit_text_message_content",
    "emit_text_message_end",
    "emit_text_message_start",
]
