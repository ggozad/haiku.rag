"""Generic AG-UI event emitter for any graph execution."""

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from ag_ui.core import (
    ActivitySnapshotEvent,
    BaseEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateDeltaEvent,
    StateSnapshotEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageChunkEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from pydantic import BaseModel

from haiku.rag.graph.agui.state import compute_state_delta

AGUIEvent = dict[str, Any]


def _serialize_event(event: BaseEvent) -> AGUIEvent:
    """Serialize an ag_ui event to a dict with camelCase keys."""
    return event.model_dump(mode="json", by_alias=True, exclude_none=True)


class AGUIEmitter[StateT: BaseModel, ResultT]:
    """Generic queue-backed AG-UI event emitter for any graph.

    Manages the lifecycle of AG-UI events including:
    - Run lifecycle (start, finish, error)
    - Step lifecycle (start, finish)
    - Text messages
    - State synchronization (snapshots and deltas)
    - Activity updates

    Type parameters:
        StateT: The Pydantic BaseModel type for graph state
        ResultT: The result type returned by the graph
    """

    def __init__(
        self,
        thread_id: str | None = None,
        run_id: str | None = None,
        use_deltas: bool = True,
    ):
        """Initialize the emitter.

        Args:
            thread_id: Optional thread ID (generated from input hash if not provided)
            run_id: Optional run ID (random UUID if not provided)
            use_deltas: Whether to emit state deltas instead of full snapshots (default: True)
        """
        self._queue: asyncio.Queue[AGUIEvent | None] = asyncio.Queue()
        self._closed = False
        self._thread_id = thread_id or str(uuid4())
        self._run_id = run_id or str(uuid4())
        self._last_state: StateT | None = None
        self._current_step: str | None = None
        self._use_deltas = use_deltas

    @property
    def thread_id(self) -> str:
        """Get the thread ID for this emitter."""
        return self._thread_id

    @property
    def run_id(self) -> str:
        """Get the run ID for this emitter."""
        return self._run_id

    def start_run(self, initial_state: StateT) -> None:
        """Emit RunStarted and initial StateSnapshot.

        Args:
            initial_state: The initial state of the graph
        """
        # If thread_id wasn't provided, generate from state hash
        if not self._thread_id or self._thread_id == str(uuid4()):
            state_json = initial_state.model_dump_json()
            self._thread_id = self._generate_thread_id(state_json)

        # RunStarted (state snapshot follows immediately with full state)
        self.emit(
            _serialize_event(
                RunStartedEvent(thread_id=self._thread_id, run_id=self._run_id)
            )
        )
        self.emit(
            _serialize_event(StateSnapshotEvent(snapshot=initial_state.model_dump()))
        )
        # Store a deep copy to detect future changes
        self._last_state = initial_state.model_copy(deep=True)

    def start_step(self, step_name: str) -> None:
        """Emit StepStarted event.

        Args:
            step_name: Name of the step being started
        """
        self._current_step = step_name
        self.emit(_serialize_event(StepStartedEvent(step_name=step_name)))

    def finish_step(self) -> None:
        """Emit StepFinished event for the current step."""
        if self._current_step:
            self.emit(_serialize_event(StepFinishedEvent(step_name=self._current_step)))
            self._current_step = None

    def log(self, message: str, role: str = "assistant") -> None:
        """Emit a text message event.

        Args:
            message: The message content
            role: The role of the sender (default: assistant)
        """
        message_id = str(uuid4())
        self.emit(
            _serialize_event(
                TextMessageChunkEvent(
                    message_id=message_id,
                    role=role,  # type: ignore[arg-type]
                    delta=message,
                )
            )
        )

    def update_state(self, new_state: StateT) -> None:
        """Emit StateDelta or StateSnapshot for state change.

        Args:
            new_state: The updated state
        """
        if self._use_deltas and self._last_state is not None:
            # Emit delta for incremental updates
            delta = compute_state_delta(self._last_state, new_state)
            self.emit(_serialize_event(StateDeltaEvent(delta=delta)))
        else:
            # Emit full snapshot for initial state or when deltas disabled
            self.emit(
                _serialize_event(StateSnapshotEvent(snapshot=new_state.model_dump()))
            )
        # Store a deep copy to detect future changes
        self._last_state = new_state.model_copy(deep=True)

    def update_activity(
        self,
        activity_type: str,
        content: dict[str, Any],
        message_id: str | None = None,
    ) -> None:
        """Emit ActivitySnapshot event.

        Args:
            activity_type: Type of activity (e.g., "planning", "searching")
            content: Structured payload representing the activity state
            message_id: Optional message ID to associate activity with (auto-generated if None)
        """
        if message_id is None:
            message_id = str(uuid4())
        self.emit(
            _serialize_event(
                ActivitySnapshotEvent(
                    message_id=message_id,
                    activity_type=activity_type,
                    content=content,
                )
            )
        )

    def finish_run(self, result: ResultT) -> None:
        """Emit RunFinished event.

        Args:
            result: The final result from the graph
        """
        # Convert result to dict if it's a Pydantic model
        result_data: Any = result
        if hasattr(result, "model_dump"):
            result_data = result.model_dump()  # type: ignore[union-attr]

        self.emit(
            _serialize_event(
                RunFinishedEvent(
                    thread_id=self._thread_id, run_id=self._run_id, result=result_data
                )
            )
        )

    def error(self, error: Exception, code: str | None = None) -> None:
        """Emit RunError event.

        Args:
            error: The exception that occurred
            code: Optional error code
        """
        self.emit(_serialize_event(RunErrorEvent(message=str(error), code=code)))

    def emit(self, event: AGUIEvent) -> None:
        """Put event in queue.

        Args:
            event: The event to emit
        """
        if not self._closed:
            self._queue.put_nowait(event)

    async def close(self) -> None:
        """Close the emitter and stop event iteration."""
        if self._closed:
            return
        self._closed = True
        await self._queue.put(None)

    def __aiter__(self) -> AsyncIterator[AGUIEvent]:
        """Enable async iteration over events."""
        return self._iter_events()

    async def _iter_events(self) -> AsyncIterator[AGUIEvent]:
        """Iterate over events from the queue."""
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event

    @staticmethod
    def _generate_thread_id(input_data: str) -> str:
        """Generate a deterministic thread ID from input data.

        Args:
            input_data: The input data (e.g., question, prompt)

        Returns:
            A stable thread ID based on input hash
        """
        # Use hash of input for deterministic thread ID
        hash_obj = hashlib.sha256(input_data.encode("utf-8"))
        return hash_obj.hexdigest()[:16]


def emit_text_message_start(message_id: str, role: str = "assistant") -> AGUIEvent:
    """Create a TextMessageStart event."""
    return _serialize_event(
        TextMessageStartEvent(message_id=message_id, role=role)  # type: ignore[arg-type]
    )


def emit_text_message_content(message_id: str, delta: str) -> AGUIEvent:
    """Create a TextMessageContent event."""
    return _serialize_event(TextMessageContentEvent(message_id=message_id, delta=delta))


def emit_text_message_end(message_id: str) -> AGUIEvent:
    """Create a TextMessageEnd event."""
    return _serialize_event(TextMessageEndEvent(message_id=message_id))


def emit_tool_call_start(
    tool_call_id: str,
    tool_name: str,
    parent_message_id: str | None = None,
) -> AGUIEvent:
    """Create a ToolCallStart event."""
    return _serialize_event(
        ToolCallStartEvent(
            tool_call_id=tool_call_id,
            tool_call_name=tool_name,
            parent_message_id=parent_message_id,
        )
    )


def emit_tool_call_args(tool_call_id: str, args: dict[str, Any]) -> AGUIEvent:
    """Create a ToolCallArgs event."""
    return _serialize_event(
        ToolCallArgsEvent(tool_call_id=tool_call_id, delta=json.dumps(args))
    )


def emit_tool_call_end(tool_call_id: str) -> AGUIEvent:
    """Create a ToolCallEnd event."""
    return _serialize_event(ToolCallEndEvent(tool_call_id=tool_call_id))


def emit_run_started(thread_id: str, run_id: str) -> AGUIEvent:
    """Create a RunStarted event."""
    return _serialize_event(RunStartedEvent(thread_id=thread_id, run_id=run_id))


def emit_run_finished(thread_id: str, run_id: str, result: Any) -> AGUIEvent:
    """Create a RunFinished event."""
    # Convert result to dict if it's a Pydantic model
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    return _serialize_event(
        RunFinishedEvent(thread_id=thread_id, run_id=run_id, result=result)
    )


def emit_run_error(message: str, code: str | None = None) -> AGUIEvent:
    """Create a RunError event."""
    return _serialize_event(RunErrorEvent(message=message, code=code))


def emit_step_started(step_name: str) -> AGUIEvent:
    """Create a StepStarted event."""
    return _serialize_event(StepStartedEvent(step_name=step_name))


def emit_step_finished(step_name: str) -> AGUIEvent:
    """Create a StepFinished event."""
    return _serialize_event(StepFinishedEvent(step_name=step_name))


def emit_text_message(content: str, role: str = "assistant") -> AGUIEvent:
    """Create a TextMessageChunk event (convenience wrapper)."""
    message_id = str(uuid4())
    return _serialize_event(
        TextMessageChunkEvent(
            message_id=message_id,
            role=role,  # type: ignore[arg-type]
            delta=content,
        )
    )


def emit_state_snapshot(state: BaseModel) -> AGUIEvent:
    """Create a StateSnapshot event."""
    return _serialize_event(StateSnapshotEvent(snapshot=state.model_dump()))


def emit_state_delta(old_state: BaseModel, new_state: BaseModel) -> AGUIEvent:
    """Create a StateDelta event with JSON Patch operations."""
    delta = compute_state_delta(old_state, new_state)
    return _serialize_event(StateDeltaEvent(delta=delta))


def emit_activity(
    message_id: str,
    activity_type: str,
    content: dict[str, Any],
) -> AGUIEvent:
    """Create an ActivitySnapshot event."""
    return _serialize_event(
        ActivitySnapshotEvent(
            message_id=message_id,
            activity_type=activity_type,
            content=content,
        )
    )


def emit_activity_delta(
    message_id: str,
    activity_type: str,
    patch: list[dict[str, Any]],
) -> AGUIEvent:
    """Create an ActivityDelta event with JSON Patch operations."""
    from ag_ui.core import ActivityDeltaEvent

    return _serialize_event(
        ActivityDeltaEvent(
            message_id=message_id,
            activity_type=activity_type,
            patch=patch,
        )
    )
