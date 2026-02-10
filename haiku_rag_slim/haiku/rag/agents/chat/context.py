import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from pydantic_ai import Agent

from haiku.rag.agents.chat.prompts import SESSION_SUMMARY_PROMPT
from haiku.rag.agents.chat.state import ChatSessionState, SessionContext
from haiku.rag.config.models import AppConfig
from haiku.rag.utils import get_model

if TYPE_CHECKING:
    from haiku.rag.tools.qa import QAHistoryEntry, QASessionState


@dataclass
class SessionCache:
    """Per-session cache for context and embeddings."""

    context: SessionContext | None = None
    embeddings: dict[str, list[float]] = field(default_factory=dict)


# Cache for session data (session_id -> SessionCache)
# Used to persist async summarization results and embeddings between requests
_session_cache: dict[str, SessionCache] = {}
_cache_timestamps: dict[str, datetime] = {}
_CACHE_TTL = timedelta(hours=1)

# Track summarization tasks per session to allow cancellation
_summarization_tasks: dict[str, asyncio.Task[None]] = {}


def _cleanup_stale_cache() -> None:
    """Remove cache entries older than TTL."""
    now = datetime.now()
    stale = [sid for sid, ts in _cache_timestamps.items() if now - ts > _CACHE_TTL]
    for sid in stale:
        _session_cache.pop(sid, None)
        _cache_timestamps.pop(sid, None)


def _get_or_create_session_cache(session_id: str) -> SessionCache:
    """Get or create session cache for a given session_id."""
    _cleanup_stale_cache()
    if session_id not in _session_cache:
        _session_cache[session_id] = SessionCache()
    _cache_timestamps[session_id] = datetime.now()
    return _session_cache[session_id]


def cache_session_context(session_id: str, context: SessionContext) -> None:
    """Store session context in cache."""
    cache = _get_or_create_session_cache(session_id)
    cache.context = context


def get_cached_session_context(session_id: str) -> SessionContext | None:
    """Get session context from server cache."""
    _cleanup_stale_cache()
    cache = _session_cache.get(session_id)
    return cache.context if cache else None


def cache_question_embedding(
    session_id: str, question: str, embedding: list[float]
) -> None:
    """Store question embedding in session cache."""
    cache = _get_or_create_session_cache(session_id)
    cache.embeddings[question] = embedding


def get_cached_embedding(session_id: str, question: str) -> list[float] | None:
    """Get cached embedding for a question in this session."""
    _cleanup_stale_cache()
    cache = _session_cache.get(session_id)
    return cache.embeddings.get(question) if cache else None


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
    session_state: ChatSessionState,
) -> None:
    """Update session context in the session state.

    Args:
        qa_history: List of Q&A pairs from the conversation.
        config: AppConfig for model selection.
        session_state: The session state to update.
    """
    # Use existing session_context summary if available, else initial_context
    current_context: str | None = None
    if session_state.session_context and session_state.session_context.summary:
        current_context = session_state.session_context.summary
    elif session_state.initial_context:
        current_context = session_state.initial_context

    summary = await summarize_session(
        qa_history, config, current_context=current_context
    )
    session_state.session_context = SessionContext(
        summary=summary,
        last_updated=datetime.now(),
    )
    # Also cache for next-run delivery in stateless contexts
    if session_state.session_id:
        cache_session_context(session_state.session_id, session_state.session_context)


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
    session_id: str,
) -> None:
    """Background task to update session context after an ask."""
    try:
        qa_history = list(qa_session_state.qa_history)

        session_state = ChatSessionState(
            session_id=session_id,
            qa_history=qa_history,
        )

        await update_session_context(
            qa_history=qa_history,
            config=config,
            session_state=session_state,
        )

        # Update the QASessionState with the new context
        cached = get_cached_session_context(session_id)
        if cached and cached.summary:
            qa_session_state.session_context = cached.summary

    except asyncio.CancelledError:
        pass
    except Exception as e:
        import logging

        logging.getLogger(__name__).exception(f"Background summarization failed: {e}")


def trigger_background_summarization(
    qa_session_state: "QASessionState",
    config: AppConfig,
    session_id: str,
) -> None:
    """Trigger background session summarization if qa_history has entries.

    Args:
        qa_session_state: QASessionState with qa_history to summarize.
        config: AppConfig for model selection.
        session_id: Session ID for caching results.
    """
    if not qa_session_state.qa_history or not session_id:
        return

    # Cancel any existing summarization task for this session
    if session_id in _summarization_tasks:
        _summarization_tasks[session_id].cancel()

    # Spawn background task
    task = asyncio.create_task(
        _update_context_background(
            qa_session_state=qa_session_state,
            config=config,
            session_id=session_id,
        )
    )
    _summarization_tasks[session_id] = task
    task.add_done_callback(
        lambda _t, sid=session_id: _summarization_tasks.pop(sid, None)
    )
