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
    @pytest.mark.vcr()
    async def test_summarize_session_empty_history(self, allow_model_requests):
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

        # Provide current_context (e.g., background_context or previous summary)
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
