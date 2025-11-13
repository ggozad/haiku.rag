"""Generic CLI renderer for AG-UI events with Rich console output."""

from collections.abc import AsyncIterator
from typing import Any

from rich.console import Console

from haiku.rag.graph.agui.events import AGUIEvent


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
                self._render_state_snapshot(event)
            elif event_type == "STATE_DELTA":
                self._render_state_delta(event)
            elif event_type == "ACTIVITY_SNAPSHOT":
                self._render_activity(event)
            elif event_type == "ACTIVITY_DELTA":
                pass  # Activity deltas don't need separate rendering

        return result

    def _render_run_started(self, event: AGUIEvent) -> None:
        """Render run start event."""
        run_id = event.get("runId", "")
        if run_id:
            # Show shortened run ID (first 8 chars like our UUIDs)
            short_id = run_id[:8] if len(run_id) > 8 else run_id
            self.console.print(f"[bold green][RUN_STARTED][/bold green] Run {short_id}")

    def _render_run_finished(self) -> None:
        """Render run completion."""
        self.console.print("[bold green][RUN_FINISHED][/bold green] Completed")

    def _render_error(self, event: AGUIEvent) -> None:
        """Render error event."""
        message = event.get("message", "Unknown error")
        self.console.print(f"[bold red][RUN_ERROR][/bold red] {message}")

    def _render_step_started(self, event: AGUIEvent) -> None:
        """Render step start event."""
        step_name = event.get("stepName", "")
        if step_name:
            display_name = step_name.replace("_", " ").title()
            self.console.print(
                f"\n[bold cyan][STEP_STARTED][/bold cyan] {display_name}"
            )

    def _render_step_finished(self, event: AGUIEvent) -> None:
        """Render step finish event."""
        step_name = event.get("stepName", "")
        if step_name:
            display_name = step_name.replace("_", " ").title()
            self.console.print(f"[cyan][STEP_FINISHED][/cyan] {display_name}")

    def _render_text_message(self, event: AGUIEvent) -> None:
        """Render complete text message."""
        delta = event.get("delta", "")
        self.console.print(f"[magenta][TEXT_MESSAGE][/magenta] {delta}")

    def _render_text_content(self, event: AGUIEvent) -> None:
        """Render streaming text content delta."""
        delta = event.get("delta", "")
        self.console.print(delta, end="")

    def _render_activity(self, event: AGUIEvent) -> None:
        """Render activity update."""
        content = event.get("content", "")
        if content:
            self.console.print(f"[yellow][ACTIVITY][/yellow] {content}")

    def _render_state_snapshot(self, event: AGUIEvent) -> None:
        """Render full state snapshot."""
        snapshot = event.get("snapshot")
        if not snapshot:
            return

        self.console.print("[blue][STATE_SNAPSHOT][/blue]")
        self.console.print(snapshot, style="dim")

    def _render_state_delta(self, event: AGUIEvent) -> None:
        """Render state delta operations."""
        delta = event.get("delta", [])
        if not delta:
            return

        self.console.print("[blue][STATE_DELTA][/blue]")
        self.console.print(delta, style="dim")
