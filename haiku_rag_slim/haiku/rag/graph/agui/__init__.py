"""Generic AG-UI protocol support for haiku.rag graphs."""

from haiku.rag.graph.agui.cli_renderer import AGUIConsoleRenderer
from haiku.rag.graph.agui.emitter import (
    AGUIEmitter,
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
    emit_tool_call_args,
    emit_tool_call_end,
    emit_tool_call_start,
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
    "emit_tool_call_args",
    "emit_tool_call_end",
    "emit_tool_call_start",
    "format_sse_event",
    "stream_graph",
]
