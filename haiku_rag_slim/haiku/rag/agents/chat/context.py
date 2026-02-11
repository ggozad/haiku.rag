import asyncio
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic_ai import Agent

from haiku.rag.agents.chat.prompts import SESSION_SUMMARY_PROMPT
from haiku.rag.agents.chat.state import SessionContext
from haiku.rag.config.models import AppConfig
from haiku.rag.utils import get_model

if TYPE_CHECKING:
    from haiku.rag.tools.qa import QAHistoryEntry, QASessionState


# Track summarization tasks to allow cancellation
_summarization_tasks: dict[int, asyncio.Task[None]] = {}


async def summarize_session(
    qa_history: list["QAHistoryEntry"],
    config: AppConfig,
    current_context: str | None = None,
) -> str:
    """Summarize qa_history into compact context.

    Args:
        qa_history: List of Q&A pairs from the conversation.
        config: AppConfig for model selection.
        current_context: Previous session_context.summary to incorporate.
            The summarizer will build upon this.

    Returns:
        Markdown summary of the conversation history.
    """
    if not qa_history:
        return ""

    model = get_model(config.qa.model, config)
    agent: Agent[None, str] = Agent(
        model,
        output_type=str,
        instructions=SESSION_SUMMARY_PROMPT,
        retries=2,
    )

    history_text = _format_qa_history(qa_history)
    if current_context:
        history_text = f"## Current Context\n{current_context}\n\n{history_text}"
    result = await agent.run(history_text)
    return result.output


async def update_session_context(
    qa_history: list["QAHistoryEntry"],
    config: AppConfig,
    current_context: str | None = None,
) -> SessionContext:
    """Summarize qa_history and return the resulting session context.

    Args:
        qa_history: List of Q&A pairs from the conversation.
        config: AppConfig for model selection.
        current_context: Previous summary to incorporate.

    Returns:
        The new SessionContext with summary and timestamp.
    """
    summary = await summarize_session(
        qa_history, config, current_context=current_context
    )
    return SessionContext(
        summary=summary,
        last_updated=datetime.now(),
    )


def _format_qa_history(qa_history: list["QAHistoryEntry"]) -> str:
    """Format qa_history for input to summarization."""
    lines: list[str] = []
    for i, qa in enumerate(qa_history, 1):
        lines.append(f"## Q{i}: {qa.question}")
        lines.append(f"**Answer** (confidence: {qa.confidence:.0%}):")
        lines.append(qa.answer)

        if qa.sources:
            lines.append(f"**Sources:** {', '.join(qa.sources)}")
        lines.append("")

    return "\n".join(lines)


async def _update_context_background(
    qa_session_state: "QASessionState",
    config: AppConfig,
) -> None:
    """Background task to update session context after an ask."""
    try:
        result = await update_session_context(
            qa_history=list(qa_session_state.qa_history),
            config=config,
            current_context=qa_session_state.session_context,
        )

        if result.summary:
            qa_session_state.session_context = result.summary

    except asyncio.CancelledError:
        pass
    except Exception as e:
        import logging

        logging.getLogger(__name__).exception(f"Background summarization failed: {e}")


def trigger_background_summarization(
    qa_session_state: "QASessionState",
    config: AppConfig,
) -> None:
    """Trigger background session summarization if qa_history has entries.

    Args:
        qa_session_state: QASessionState with qa_history to summarize.
        config: AppConfig for model selection.
    """
    if not qa_session_state.qa_history:
        return

    key = id(qa_session_state)

    # Cancel any existing summarization task for this state
    if key in _summarization_tasks:
        _summarization_tasks[key].cancel()

    # Spawn background task
    task = asyncio.create_task(
        _update_context_background(
            qa_session_state=qa_session_state,
            config=config,
        )
    )
    _summarization_tasks[key] = task
    task.add_done_callback(lambda _t, k=key: _summarization_tasks.pop(k, None))
