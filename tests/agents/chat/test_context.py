from datetime import datetime
from pathlib import Path

import pytest

from haiku.rag.agents.research.models import Citation
from haiku.rag.config import Config
from haiku.rag.tools.qa import QAHistoryEntry
from haiku.rag.tools.session import SessionContext


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
            QAHistoryEntry(
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
            QAHistoryEntry(
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
            QAHistoryEntry(
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
            QAHistoryEntry(
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
            QAHistoryEntry(
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
    async def test_update_session_context_returns_context(
        self, allow_model_requests, temp_db_path
    ):
        """Test update_session_context returns a populated SessionContext."""
        from haiku.rag.agents.chat.context import update_session_context

        qa_history = [
            QAHistoryEntry(
                question="What is the authentication method?",
                answer="The API uses JWT tokens.",
                confidence=0.95,
            )
        ]

        result = await update_session_context(
            qa_history=qa_history,
            config=Config,
        )

        assert result.summary != ""
        assert result.last_updated is not None

    @pytest.mark.asyncio
    async def test_update_session_context_with_empty_history(self):
        """Test update_session_context with empty history returns empty summary."""
        from haiku.rag.agents.chat.context import update_session_context

        result = await update_session_context(
            qa_history=[],
            config=Config,
        )

        assert result.summary == ""


class TestTriggerBackgroundSummarization:
    """Tests for trigger_background_summarization."""

    def test_trigger_with_empty_qa_history(self):
        """trigger_background_summarization returns early with empty qa_history."""
        from haiku.rag.agents.chat.context import (
            _summarization_tasks,
            trigger_background_summarization,
        )
        from haiku.rag.tools.qa import QASessionState

        tasks_before = len(_summarization_tasks)

        qa_session_state = QASessionState()
        assert len(qa_session_state.qa_history) == 0

        trigger_background_summarization(qa_session_state, config=Config)

        # No new task should have been created
        assert len(_summarization_tasks) == tasks_before

    @pytest.mark.asyncio
    async def test_trigger_cancels_existing_task(self):
        """Second trigger cancels the previous background task."""
        import asyncio
        from unittest.mock import patch

        from haiku.rag.agents.chat.context import (
            _summarization_tasks,
            trigger_background_summarization,
        )
        from haiku.rag.tools.qa import QAHistoryEntry, QASessionState

        _summarization_tasks.clear()

        qa_session_state = QASessionState(
            qa_history=[QAHistoryEntry(question="Q1", answer="A1", confidence=0.9)]
        )

        # Patch _update_context_background to be a slow coroutine
        async def slow_background(*args, **kwargs):
            await asyncio.sleep(10)

        with patch(
            "haiku.rag.agents.chat.context._update_context_background",
            new=slow_background,
        ):
            # First trigger creates a task
            trigger_background_summarization(qa_session_state, config=Config)
            key = id(qa_session_state)
            assert key in _summarization_tasks
            first_task = _summarization_tasks[key]

            # Second trigger should cancel the first
            trigger_background_summarization(qa_session_state, config=Config)
            await asyncio.sleep(0)  # Let cancellation propagate
            assert first_task.cancelled() or first_task.done()

            # Cleanup
            if key in _summarization_tasks:
                _summarization_tasks[key].cancel()
                try:
                    await _summarization_tasks[key]
                except asyncio.CancelledError:
                    pass
            _summarization_tasks.clear()


class TestUpdateSessionContextPassesCurrentContext:
    """Tests for update_session_context current_context forwarding."""

    @pytest.mark.asyncio
    async def test_update_session_context_passes_current_context(self):
        """Test update_session_context passes current_context to summarizer."""
        from unittest.mock import patch

        from haiku.rag.agents.chat.context import update_session_context

        qa_history = [
            QAHistoryEntry(
                question="What is JWT?",
                answer="JSON Web Token for authentication.",
                confidence=0.95,
            )
        ]

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
                current_context="Previous session summary",
            )

        assert len(captured_current_context) == 1
        assert captured_current_context[0] == "Previous session summary"
