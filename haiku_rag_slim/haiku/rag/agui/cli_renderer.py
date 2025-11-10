"""Generic CLI renderer for AG-UI events with Rich console output."""

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel
from rich.console import Console

from haiku.rag.agui.events import AGUIEvent


class AGUIConsoleRenderer:
    """Renders AG-UI events to Rich console with formatted output.

    Generic renderer that processes AG-UI protocol events and renders them
    with Rich formatting. Works with any graph that emits AG-UI events.
    """

    def __init__(self, console: Console | None = None):
        """Initialize the renderer.

        Args:
            console: Optional Rich console instance (creates new one if not provided)
        """
        self.console = console or Console()
        self._state: BaseModel | None = None

    async def render(self, events: AsyncIterator[AGUIEvent]) -> Any | None:
        """Process events and render to console, return final result.

        Args:
            events: Async iterator of AG-UI events

        Returns:
            The final result from RunFinished event, or None
        """
        result = None

        async for event in events:
            event_type = event.get("type")

            if event_type == "RUN_STARTED":
                self._render_run_started(event)
            elif event_type == "RUN_FINISHED":
                result = event.get("result")
                self._render_run_finished()
            elif event_type == "RUN_ERROR":
                self._render_error(event)
            elif event_type == "STEP_STARTED":
                self._render_step_started(event)
            elif event_type == "STEP_FINISHED":
                self._render_step_finished(event)
            elif event_type == "TEXT_MESSAGE_CHUNK":
                self._render_text_message(event)
            elif event_type == "TEXT_MESSAGE_START":
                pass  # Start of streaming message, no output needed
            elif event_type == "TEXT_MESSAGE_CONTENT":
                self._render_text_content(event)
            elif event_type == "TEXT_MESSAGE_END":
                pass  # End of streaming message, no output needed
            elif event_type == "STATE_SNAPSHOT":
                self._state = event.get("snapshot")
            elif event_type == "STATE_DELTA":
                self._apply_state_delta(event)
            elif event_type == "ACTIVITY_SNAPSHOT":
                self._render_activity(event)
            elif event_type == "ACTIVITY_DELTA":
                pass  # Activity deltas don't need separate rendering

        return result

    def _render_run_started(self, event: AGUIEvent) -> None:
        """Render run start event.

        Args:
            event: RunStarted event
        """
        # Currently silent - could render run metadata later

    def _render_run_finished(self) -> None:
        """Render run completion."""
        # Currently silent - the result is rendered separately

    def _render_error(self, event: AGUIEvent) -> None:
        """Render error event.

        Args:
            event: RunError event
        """
        message = event.get("message", "Unknown error")
        self.console.print(f"[bold red]❌ Error:[/bold red] {message}")

    def _render_step_started(self, event: AGUIEvent) -> None:
        """Render step start event.

        Args:
            event: StepStarted event
        """
        step_name = event.get("stepName", "")
        if step_name:
            # Format step name for display
            display_name = step_name.replace("_", " ").title()
            self.console.print(f"\n[bold cyan]{display_name}[/bold cyan]")

    def _render_step_finished(self, event: AGUIEvent) -> None:
        """Render step finish event.

        Args:
            event: StepFinished event
        """
        # Step completion is implicit from the next step or activity

    def _render_text_message(self, event: AGUIEvent) -> None:
        """Render complete text message.

        Args:
            event: TextMessageChunk event
        """
        delta = event.get("delta", "")
        # The delta contains the text content to display
        self.console.print(delta)

    def _render_text_content(self, event: AGUIEvent) -> None:
        """Render streaming text content delta.

        Args:
            event: TextMessageContent event
        """
        delta = event.get("delta", "")
        # Print delta without newline for streaming effect
        self.console.print(delta, end="")

    def _render_activity(self, event: AGUIEvent) -> None:
        """Render activity update.

        Args:
            event: ActivitySnapshot event
        """
        content = event.get("content", "")

        # Render activity content with emphasis
        if content:
            self.console.print(f"[dim]{content}[/dim]")

    def _apply_state_delta(self, event: AGUIEvent) -> None:
        """Apply state delta to current state.

        Args:
            event: StateDelta event
        """
        # Currently not applying deltas - could implement state patching later
        # For now, the text messages contain all the information we need to display
