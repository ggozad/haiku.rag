from datetime import datetime
from pathlib import Path

import pytest

from haiku.rag.agents.chat.state import (
    QAResponse,
    SessionContext,
)
from haiku.rag.agents.research.models import Citation
from haiku.rag.config import Config


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent.parent / "cassettes" / "test_chat_context")


class TestSessionContext:
    """Tests for SessionContext model."""

    def test_session_context_creation_empty(self):
        """Test SessionContext can be created with defaults."""
        ctx = SessionContext()
        assert ctx.summary == ""
        assert ctx.last_updated is None

    def test_session_context_creation_with_values(self):
        """Test SessionContext can be created with provided values."""
        now = datetime.now()
        ctx = SessionContext(
            summary="User discussed authentication patterns.",
            last_updated=now,
        )
        assert ctx.summary == "User discussed authentication patterns."
        assert ctx.last_updated == now

    def test_render_markdown_empty(self):
        """Test render_markdown returns empty string when no summary."""
        ctx = SessionContext()
        assert ctx.render_markdown() == ""

    def test_render_markdown_with_summary(self):
        """Test render_markdown returns the summary directly."""
        summary = "## Key Facts\n- Authentication uses JWT\n- Rate limit is 100/min"
        ctx = SessionContext(summary=summary)
        assert ctx.render_markdown() == summary

    def test_session_context_serialization_roundtrip(self):
        """Test SessionContext serializes and deserializes correctly."""
        now = datetime.now()
        original = SessionContext(
            summary="Test summary with facts.",
            last_updated=now,
        )
        # Serialize to dict
        data = original.model_dump()
        # Deserialize back
        restored = SessionContext(**data)

        assert restored.summary == original.summary
        assert restored.last_updated == original.last_updated


class TestSummarizeSession:
    """Tests for summarize_session function."""

    @pytest.mark.asyncio
    async def test_summarize_session_empty_history(self):
        """Test summarize_session with empty qa_history returns empty string."""
        from haiku.rag.agents.chat.context import summarize_session

        result = await summarize_session(qa_history=[], config=Config)
        assert result == ""

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_summarize_session_single_entry(
        self, allow_model_requests, temp_db_path
    ):
        """Test summarize_session with a single qa entry."""
        from haiku.rag.agents.chat.context import summarize_session

        qa_history = [
            QAResponse(
                question="What is the authentication method?",
                answer="The API uses JWT tokens for authentication.",
                confidence=0.95,
                citations=[
                    Citation(
                        index=1,
                        document_id="doc-1",
                        chunk_id="chunk-1",
                        document_uri="auth-guide.md",
                        document_title="Auth Guide",
                        content="JWT token details...",
                    )
                ],
            )
        ]

        result = await summarize_session(qa_history=qa_history, config=Config)

        # Should produce a non-empty summary
        assert len(result) > 0
        # Summary should mention authentication or JWT
        assert "authentication" in result.lower() or "jwt" in result.lower()

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_summarize_session_multiple_entries(
        self, allow_model_requests, temp_db_path
    ):
        """Test summarize_session with multiple qa entries produces consolidated summary."""
        from haiku.rag.agents.chat.context import summarize_session

        qa_history = [
            QAResponse(
                question="What is the authentication method?",
                answer="The API uses JWT tokens for authentication.",
                confidence=0.95,
                citations=[
                    Citation(
                        index=1,
                        document_id="doc-1",
                        chunk_id="chunk-1",
                        document_uri="auth-guide.md",
                        document_title="Auth Guide",
                        content="JWT token details...",
                    )
                ],
            ),
            QAResponse(
                question="What is the rate limit?",
                answer="Rate limiting is set to 100 requests per minute.",
                confidence=0.9,
                citations=[
                    Citation(
                        index=1,
                        document_id="doc-2",
                        chunk_id="chunk-2",
                        document_uri="api-reference.md",
                        document_title="API Reference",
                        content="Rate limit config...",
                    )
                ],
            ),
            QAResponse(
                question="How do I refresh tokens?",
                answer="Use the /refresh endpoint with your refresh token.",
                confidence=0.85,
                citations=[
                    Citation(
                        index=1,
                        document_id="doc-1",
                        chunk_id="chunk-3",
                        document_uri="auth-guide.md",
                        document_title="Auth Guide",
                        content="Token refresh...",
                    )
                ],
            ),
        ]

        result = await summarize_session(qa_history=qa_history, config=Config)

        # Should produce a non-empty summary
        assert len(result) > 0
        # Summary should contain structured sections
        result_lower = result.lower()
        assert "key facts" in result_lower or "established" in result_lower
        assert "documents" in result_lower or "sources" in result_lower

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_summarize_session_with_current_context(self, allow_model_requests):
        """Test summarize_session incorporates current_context into the summary."""
        from haiku.rag.agents.chat.context import summarize_session

        qa_history = [
            QAResponse(
                question="What's the rate limit?",
                answer="100 requests per minute.",
                confidence=0.9,
            )
        ]

        # Provide current_context (e.g., previous summary)
        current_context = "Focus on Python APIs. User is building a web application."

        result = await summarize_session(
            qa_history=qa_history,
            config=Config,
            current_context=current_context,
        )

        # Summary should be non-empty and ideally incorporate context about Python/web
        assert len(result) > 0
        # The context about "Python" or "web application" should influence the summary
        result_lower = result.lower()
        assert (
            "rate" in result_lower or "limit" in result_lower or "100" in result_lower
        )


class TestUpdateSessionContext:
    """Tests for update_session_context function."""

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_update_session_context_updates_state(
        self, allow_model_requests, temp_db_path
    ):
        """Test update_session_context updates the session_state."""
        from haiku.rag.agents.chat.context import update_session_context
        from haiku.rag.agents.chat.state import ChatSessionState

        session_state = ChatSessionState(session_id="test-session")

        qa_history = [
            QAResponse(
                question="What is the authentication method?",
                answer="The API uses JWT tokens.",
                confidence=0.95,
            )
        ]

        await update_session_context(
            qa_history=qa_history,
            config=Config,
            session_state=session_state,
        )

        # session_context should now be populated
        assert session_state.session_context is not None
        assert session_state.session_context.summary != ""
        assert session_state.session_context.last_updated is not None

    @pytest.mark.asyncio
    async def test_update_session_context_with_empty_history(self):
        """Test update_session_context with empty history sets empty context."""
        from haiku.rag.agents.chat.context import update_session_context
        from haiku.rag.agents.chat.state import ChatSessionState

        session_state = ChatSessionState(session_id="test-session")

        await update_session_context(
            qa_history=[],
            config=Config,
            session_state=session_state,
        )

        # session_context should exist but have empty summary
        assert session_state.session_context is not None
        assert session_state.session_context.summary == ""


class TestSessionContextCache:
    """Tests for server-side session context caching."""

    def test_cache_and_retrieve_session_context(self):
        """Test caching and retrieving a session context."""
        from haiku.rag.agents.chat.context import (
            _session_context_cache,
            cache_session_context,
            get_cached_session_context,
        )

        # Clear cache
        _session_context_cache.clear()

        now = datetime.now()
        ctx = SessionContext(summary="Test summary", last_updated=now)

        cache_session_context("session-1", ctx)
        result = get_cached_session_context("session-1")

        assert result is not None
        assert result.summary == "Test summary"
        assert result.last_updated == now

    def test_get_cached_session_context_returns_none_when_not_cached(self):
        """Test get_cached_session_context returns None when nothing cached."""
        from haiku.rag.agents.chat.context import (
            _session_context_cache,
            get_cached_session_context,
        )

        _session_context_cache.clear()

        result = get_cached_session_context("nonexistent-session")
        assert result is None

    def test_cache_ttl_cleanup_removes_stale_entries(self):
        """Test that stale cache entries are cleaned up."""
        from datetime import timedelta

        from haiku.rag.agents.chat.context import (
            _CACHE_TTL,
            _cache_timestamps,
            _session_context_cache,
            cache_session_context,
            get_cached_session_context,
        )

        _session_context_cache.clear()
        _cache_timestamps.clear()

        # Add an entry
        ctx = SessionContext(summary="Old summary", last_updated=datetime.now())
        cache_session_context("stale-session", ctx)

        # Make the entry stale by backdating its timestamp
        _cache_timestamps["stale-session"] = (
            datetime.now() - _CACHE_TTL - timedelta(seconds=1)
        )

        # Getting session context should trigger cleanup
        result = get_cached_session_context("stale-session")

        # Should be None because the entry was cleaned up
        assert result is None
        assert "stale-session" not in _session_context_cache

    @pytest.mark.asyncio
    async def test_update_session_context_caches_result(self):
        """Test update_session_context stores result in cache."""
        from unittest.mock import AsyncMock, patch

        from haiku.rag.agents.chat.context import (
            _session_context_cache,
            get_cached_session_context,
            update_session_context,
        )
        from haiku.rag.agents.chat.state import ChatSessionState

        _session_context_cache.clear()

        session_state = ChatSessionState(session_id="cache-test-session")

        qa_history = [
            QAResponse(
                question="What is Python?",
                answer="A programming language.",
                confidence=0.95,
            )
        ]

        # Mock summarize_session to avoid LLM call (we're testing caching, not summarization)
        with patch(
            "haiku.rag.agents.chat.context.summarize_session",
            new=AsyncMock(return_value="Mocked summary"),
        ):
            await update_session_context(
                qa_history=qa_history,
                config=Config,
                session_state=session_state,
            )

        # Should be cached
        cached = get_cached_session_context("cache-test-session")
        assert cached is not None
        assert cached.summary == "Mocked summary"
        assert session_state.session_context is not None
        assert cached.summary == session_state.session_context.summary

    @pytest.mark.asyncio
    async def test_update_session_context_no_cache_without_session_id(self):
        """Test update_session_context doesn't cache without session_id."""
        from haiku.rag.agents.chat.context import (
            _session_context_cache,
            get_cached_session_context,
            update_session_context,
        )
        from haiku.rag.agents.chat.state import ChatSessionState

        _session_context_cache.clear()

        # No session_id
        session_state = ChatSessionState()

        await update_session_context(
            qa_history=[],
            config=Config,
            session_state=session_state,
        )

        # Nothing should be cached (empty session_id)
        cached = get_cached_session_context("")
        assert cached is None

    @pytest.mark.asyncio
    async def test_update_session_context_uses_initial_context_as_fallback(self):
        """Test update_session_context uses initial_context when no session_context exists."""
        from unittest.mock import patch

        from haiku.rag.agents.chat.context import (
            _session_context_cache,
            update_session_context,
        )
        from haiku.rag.agents.chat.state import ChatSessionState

        _session_context_cache.clear()

        # Create session_state with initial_context but no session_context
        session_state = ChatSessionState(
            session_id="initial-context-test",
            initial_context="User is working on a Python web application with FastAPI.",
        )

        qa_history = [
            QAResponse(
                question="What is JWT?",
                answer="JSON Web Token for authentication.",
                confidence=0.95,
            )
        ]

        # Mock summarize_session to capture what gets passed as current_context
        captured_current_context = []

        async def mock_summarize(qa_history, config, current_context=None):
            captured_current_context.append(current_context)
            return "Mocked summary"

        with patch(
            "haiku.rag.agents.chat.context.summarize_session",
            new=mock_summarize,
        ):
            await update_session_context(
                qa_history=qa_history,
                config=Config,
                session_state=session_state,
            )

        # initial_context should have been passed as current_context
        assert len(captured_current_context) == 1
        assert (
            captured_current_context[0]
            == "User is working on a Python web application with FastAPI."
        )

    @pytest.mark.asyncio
    async def test_update_session_context_session_context_takes_precedence(self):
        """Test session_context.summary takes precedence over initial_context."""
        from unittest.mock import patch

        from haiku.rag.agents.chat.context import (
            _session_context_cache,
            update_session_context,
        )
        from haiku.rag.agents.chat.state import ChatSessionState, SessionContext

        _session_context_cache.clear()

        # Create session_state with BOTH initial_context and session_context
        session_state = ChatSessionState(
            session_id="precedence-test",
            initial_context="Initial background info",
            session_context=SessionContext(summary="Evolved session summary"),
        )

        qa_history = [
            QAResponse(
                question="What is JWT?",
                answer="JSON Web Token.",
                confidence=0.95,
            )
        ]

        # Mock summarize_session to capture what gets passed as current_context
        captured_current_context = []

        async def mock_summarize(qa_history, config, current_context=None):
            captured_current_context.append(current_context)
            return "Mocked summary"

        with patch(
            "haiku.rag.agents.chat.context.summarize_session",
            new=mock_summarize,
        ):
            await update_session_context(
                qa_history=qa_history,
                config=Config,
                session_state=session_state,
            )

        # session_context.summary should take precedence over initial_context
        assert len(captured_current_context) == 1
        assert captured_current_context[0] == "Evolved session summary"
