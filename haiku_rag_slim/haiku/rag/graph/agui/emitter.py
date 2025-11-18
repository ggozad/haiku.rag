"""Generic AG-UI event emitter for any graph execution."""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from uuid import uuid4

from pydantic import BaseModel

from haiku.rag.graph.agui.events import (
    AGUIEvent,
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
        self._emit(emit_run_started(self._thread_id, self._run_id))
        self._emit(emit_state_snapshot(initial_state))
        # Store a deep copy to detect future changes
        self._last_state = initial_state.model_copy(deep=True)

    def start_step(self, step_name: str) -> None:
        """Emit StepStarted event.

        Args:
            step_name: Name of the step being started
        """
        self._current_step = step_name
        self._emit(emit_step_started(step_name))

    def finish_step(self) -> None:
        """Emit StepFinished event for the current step."""
        if self._current_step:
            self._emit(emit_step_finished(self._current_step))
            self._current_step = None

    def log(self, message: str, role: str = "assistant") -> None:
        """Emit a text message event.

        Args:
            message: The message content
            role: The role of the sender (default: assistant)
        """
        self._emit(emit_text_message(message, role))

    def update_state(self, new_state: StateT) -> None:
        """Emit StateDelta or StateSnapshot for state change.

        Args:
            new_state: The updated state
        """
        if self._use_deltas and self._last_state is not None:
            # Emit delta for incremental updates
            self._emit(emit_state_delta(self._last_state, new_state))
        else:
            # Emit full snapshot for initial state or when deltas disabled
            self._emit(emit_state_snapshot(new_state))
        # Store a deep copy to detect future changes
        self._last_state = new_state.model_copy(deep=True)

    def update_activity(
        self, activity_type: str, content: str, message_id: str | None = None
    ) -> None:
        """Emit ActivitySnapshot event.

        Args:
            activity_type: Type of activity (e.g., "planning", "searching")
            content: Description of the activity
            message_id: Optional message ID to associate activity with (auto-generated if None)
        """
        if message_id is None:
            message_id = str(uuid4())
        self._emit(emit_activity(message_id, activity_type, content))

    def finish_run(self, result: ResultT) -> None:
        """Emit RunFinished event.

        Args:
            result: The final result from the graph
        """
        self._emit(emit_run_finished(self._thread_id, self._run_id, result))

    def error(self, error: Exception, code: str | None = None) -> None:
        """Emit RunError event.

        Args:
            error: The exception that occurred
            code: Optional error code
        """
        self._emit(emit_run_error(str(error), code))

    def _emit(self, event: AGUIEvent) -> None:
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
