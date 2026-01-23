from datetime import datetime, timedelta

from pydantic_ai import Agent

from haiku.rag.agents.chat.prompts import SESSION_SUMMARY_PROMPT
from haiku.rag.agents.chat.state import ChatSessionState, QAResponse, SessionContext
from haiku.rag.config.models import AppConfig
from haiku.rag.utils import get_model

# Cache for session contexts (session_id -> SessionContext)
# Used to persist async summarization results between requests
_session_context_cache: dict[str, SessionContext] = {}
_cache_timestamps: dict[str, datetime] = {}
_CACHE_TTL = timedelta(hours=1)


def _cleanup_stale_cache() -> None:
    """Remove cache entries older than TTL."""
    now = datetime.now()
    stale = [sid for sid, ts in _cache_timestamps.items() if now - ts > _CACHE_TTL]
    for sid in stale:
        _session_context_cache.pop(sid, None)
        _cache_timestamps.pop(sid, None)


def cache_session_context(session_id: str, context: SessionContext) -> None:
    """Store session context in cache."""
    _cleanup_stale_cache()
    _session_context_cache[session_id] = context
    _cache_timestamps[session_id] = datetime.now()


def get_cached_session_context(session_id: str) -> SessionContext | None:
    """Get session context from server cache.

    Args:
        session_id: The session identifier.

    Returns:
        Cached SessionContext, or None if not cached.
    """
    _cleanup_stale_cache()
    return _session_context_cache.get(session_id)


async def summarize_session(
    qa_history: list[QAResponse],
    config: AppConfig,
    current_context: str | None = None,
) -> str:
    """Summarize qa_history into compact context.

    Args:
        qa_history: List of Q&A pairs from the conversation.
        config: AppConfig for model selection.
        current_context: Previous context to incorporate (background_context or
            previous session_context.summary). The summarizer will build upon this.

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
    qa_history: list[QAResponse],
    config: AppConfig,
    session_state: ChatSessionState,
) -> None:
    """Update session context in the session state.

    Args:
        qa_history: List of Q&A pairs from the conversation.
        config: AppConfig for model selection.
        session_state: The session state to update.
    """
    # Determine current context to incorporate:
    # 1. If session_context already exists, use its summary
    # 2. Otherwise, use background_context (if available)
    current_context: str | None = None
    if session_state.session_context and session_state.session_context.summary:
        current_context = session_state.session_context.summary
    elif session_state.background_context:
        current_context = session_state.background_context

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


def _format_qa_history(qa_history: list[QAResponse]) -> str:
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
