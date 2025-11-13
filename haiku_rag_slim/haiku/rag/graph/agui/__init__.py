"""Generic AG-UI protocol support for haiku.rag graphs."""

from haiku.rag.graph.agui.cli_renderer import AGUIConsoleRenderer
from haiku.rag.graph.agui.emitter import AGUIEmitter
from haiku.rag.graph.agui.events import (
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
from haiku.rag.graph.agui.server import (
    RunAgentInput,
    create_agui_app,
    create_agui_server,
    format_sse_event,
)
from haiku.rag.graph.agui.state import compute_state_delta
from haiku.rag.graph.agui.stream import stream_graph

__all__ = [
    "AGUIConsoleRenderer",
    "AGUIEmitter",
    "AGUIEvent",
    "RunAgentInput",
    "compute_state_delta",
    "create_agui_app",
    "create_agui_server",
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
    "format_sse_event",
    "stream_graph",
]
