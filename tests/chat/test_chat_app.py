from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from haiku.rag.capabilities.rag import RAGState, create_capability
from haiku.rag.cli import _cli as cli

runner = CliRunner()


def test_chat_command():
    """Test chat command launches chat TUI."""
    with patch("haiku.rag.chat.run_chat") as mock_chat:
        mock_chat.return_value = None

        result = runner.invoke(cli, ["chat"])

        assert result.exit_code == 0
        mock_chat.assert_called_once()


def test_run_chat_creates_app_and_runs(temp_db_path: Path):
    """Test run_chat() eagerly attaches one capability and runs the app."""
    with patch("haiku.rag.chat.app.ChatApp") as mock_app:
        from haiku.rag.chat import run_chat

        run_chat(db_path=temp_db_path)

    mock_app.return_value.run.assert_called_once()
    attached = mock_app.call_args.kwargs["capabilities"]
    assert len(attached) == 1
    assert attached[0].defer_loading is False


def test_run_chat_covers_a_configured_set(tmp_path, monkeypatch):
    """A configured set has no single path to fall back to: the app is handed
    None so the client opens the set."""
    import haiku.rag.config as config_module
    from haiku.rag.config import AppConfig, set_config
    from haiku.rag.config.models import LanceDBConfig

    monkeypatch.setattr(config_module, "_config", None)
    set_config(
        AppConfig(
            lancedb=LanceDBConfig(
                databases={
                    "a": str(tmp_path / "a.lancedb"),
                    "b": str(tmp_path / "b.lancedb"),
                }
            )
        )
    )
    with patch("haiku.rag.chat.app.ChatApp") as app:
        from haiku.rag.chat import run_chat

        run_chat(db_path=None)

    assert app.call_args.args[0] is None


def test_run_chat_defers_multiple_capabilities(temp_db_path: Path):
    """Test chat only defers capabilities when routing between multiple choices."""
    with patch("haiku.rag.chat.app.ChatApp") as mock_app:
        from haiku.rag.chat import run_chat

        run_chat(db_path=temp_db_path, capabilities=["rag", "analysis"])

    attached = mock_app.call_args.kwargs["capabilities"]
    assert len(attached) == 2
    assert all(capability.defer_loading for capability in attached)


@pytest.mark.parametrize(
    ("enabled", "expected_model", "expected_vision"),
    [
        (["analysis"], "analysis-model", False),
        (["rag"], "qa-model", True),
        (["rag", "analysis"], "qa-model", True),
    ],
)
def test_run_chat_gates_capability_vision_on_driving_model(
    temp_db_path: Path, enabled, expected_model, expected_vision
):
    """Analysis-only chat runs on analysis.model; otherwise on qa.model. Every
    attached capability's vision gate tracks that one driving model."""
    from haiku.rag.config.models import AppConfig, ModelConfig

    config = AppConfig()
    config.qa.model = ModelConfig(provider="openai", name="qa-model", vision=True)
    config.analysis.model = ModelConfig(
        provider="openai", name="analysis-model", vision=False
    )
    captured: dict[str, str] = {}

    def fake_get_model(model_config, _config):
        captured["name"] = model_config.name
        return "resolved-model"

    with (
        patch("haiku.rag.chat.app.ChatApp") as mock_app,
        patch("haiku.rag.config.get_config", return_value=config),
        patch("haiku.rag.utils.get_model", side_effect=fake_get_model),
    ):
        from haiku.rag.chat import run_chat

        run_chat(db_path=temp_db_path, capabilities=enabled)

    assert captured["name"] == expected_model
    attached = mock_app.call_args.kwargs["capabilities"]
    assert {capability.vision for capability in attached} == {expected_vision}


def _make_mock_client():
    """Create a mock HaikuRAG client."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    # Covers one database: a bare AsyncMock answers `_federated` with a truthy
    # Mock, which would send every read down the covering-a-set branch.
    mock_client._federated = {}
    mock_client._source = None
    return mock_client


def _make_app(db_path: Path, mock_client: AsyncMock | None = None):
    """Create a ChatApp with mocked HaikuRAG."""
    from haiku.rag.chat.app import ChatApp

    if mock_client is None:
        mock_client = _make_mock_client()

    return ChatApp(
        db_path=db_path,
        capabilities=[create_capability(db_path=db_path)],
        read_only=True,
    ), mock_client


def _make_app_with_state(db_path: Path, mock_client: AsyncMock | None = None):
    """Create a ChatApp with a RAG capability and state."""
    from haiku.rag.chat.app import ChatApp

    if mock_client is None:
        mock_client = _make_mock_client()

    return ChatApp(
        db_path=db_path,
        capabilities=[create_capability(db_path=db_path)],
        read_only=True,
    ), mock_client


@pytest.mark.asyncio
async def test_chat_app_has_required_widgets(temp_db_path: Path):
    """Test that ChatApp has the required widgets: ChatHistory, FlexibleInput."""
    from haiku.rag.chat.widgets.chat_history import ChatHistory

    app, mock_client = _make_app(temp_db_path)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test():
            chat_history = app.query_one(ChatHistory)
            assert chat_history is not None

            from haiku.rag.chat.widgets.prompt import FlexibleInput

            chat_input = app.query_one(FlexibleInput)
            assert chat_input is not None


@pytest.mark.asyncio
async def test_chat_app_quit_binding(temp_db_path: Path):
    """Test that pressing ctrl+q quits the app."""
    app, mock_client = _make_app(temp_db_path)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test() as pilot:
            assert app.is_running
            await pilot.press("ctrl+q")
            assert not app.is_running


@pytest.mark.asyncio
async def test_chat_history_can_add_message(temp_db_path: Path):
    """Test that ChatHistory can display messages."""
    from haiku.rag.chat.widgets.chat_history import ChatHistory

    app, mock_client = _make_app(temp_db_path)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test():
            chat_history = app.query_one(ChatHistory)

            await chat_history.add_message("user", "Hello, how are you?")
            assert len(chat_history.messages) == 1
            assert chat_history.messages[0] == ("user", "Hello, how are you?")

            await chat_history.add_message("assistant", "I'm doing well, thank you!")
            assert len(chat_history.messages) == 2


@pytest.mark.asyncio
async def test_chat_history_can_add_tool_calls(temp_db_path: Path):
    """Test that ChatHistory can display inline tool calls."""
    from haiku.rag.chat.widgets.chat_history import ChatHistory, ToolCallWidget

    app, mock_client = _make_app(temp_db_path)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test():
            chat_history = app.query_one(ChatHistory)

            tool_widget = await chat_history.add_tool_call(
                "tool-1", "search", {"query": "test"}
            )
            assert isinstance(tool_widget, ToolCallWidget)
            assert tool_widget._completed is False

            chat_history.mark_tool_complete("tool-1")
            assert tool_widget._completed is True


@pytest.mark.asyncio
async def test_chat_history_can_add_citations(temp_db_path: Path):
    """Test that ChatHistory can display inline citations."""
    from haiku.rag.chat.widgets.chat_history import ChatHistory, CitationWidget
    from haiku.rag.store.models.citation import Citation

    app, mock_client = _make_app(temp_db_path)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test():
            chat_history = app.query_one(ChatHistory)

            test_citations = [
                Citation(
                    index=1,
                    document_id="doc1",
                    chunk_id="chunk1",
                    document_uri="file:///test/doc1.pdf",
                    document_title="Test Document 1",
                    page_numbers=[1, 2],
                    headings=["Section 1"],
                    content="This is some test content from doc 1",
                ),
                Citation(
                    index=2,
                    document_id="doc2",
                    chunk_id="chunk2",
                    document_uri="file:///test/doc2.pdf",
                    document_title="Test Document 2",
                    page_numbers=[5],
                    headings=["Section 2", "Subsection"],
                    content="This is test content from doc 2",
                ),
            ]

            await chat_history.add_citations(test_citations)

            citation_widgets = chat_history.query(CitationWidget)
            assert len(list(citation_widgets)) == 2


@pytest.mark.asyncio
async def test_chat_history_thinking_indicator(temp_db_path: Path):
    """Test that ChatHistory can show and hide thinking indicator."""
    from haiku.rag.chat.widgets.chat_history import ChatHistory, ThinkingWidget

    app, mock_client = _make_app(temp_db_path)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test() as pilot:
            chat_history = app.query_one(ChatHistory)

            await chat_history.show_thinking()
            thinking = chat_history.query(ThinkingWidget)
            assert len(list(thinking)) == 1

            chat_history.hide_thinking()
            await pilot.pause()
            thinking = chat_history.query(ThinkingWidget)
            assert len(list(thinking)) == 0


@pytest.mark.asyncio
async def test_clear_chat_resets_state(temp_db_path: Path):
    """Test that clearing chat resets state, messages, and conversation id."""
    from haiku.rag.chat.widgets.chat_history import ChatHistory

    app, mock_client = _make_app(temp_db_path)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test() as pilot:
            chat_history = app.query_one(ChatHistory)

            await chat_history.add_message("user", "Hello")
            await chat_history.add_message("assistant", "Hi there")
            assert len(chat_history.messages) == 2

            previous_conversation_id = app._conversation_id
            await app.action_clear_chat()
            await pilot.pause()

            assert len(chat_history.messages) == 0
            assert app._conversation_id != previous_conversation_id


@pytest.mark.asyncio
async def test_citation_expand_collapse_with_enter(temp_db_path: Path):
    """Test that pressing Enter on a focused citation toggles expand/collapse."""
    from haiku.rag.chat.widgets.chat_history import ChatHistory, CitationWidget
    from haiku.rag.store.models.citation import Citation

    app, mock_client = _make_app(temp_db_path)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test() as pilot:
            chat_history = app.query_one(ChatHistory)

            test_citation = Citation(
                index=1,
                document_id="doc1",
                chunk_id="chunk1",
                document_uri="file:///test/doc1.pdf",
                document_title="Test Document",
                page_numbers=[1],
                content="Test content",
            )
            await chat_history.add_citations([test_citation])

            citation_widget = chat_history.query_one(CitationWidget)
            assert citation_widget.collapsed is True

            citation_widget.focus()
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()
            assert citation_widget.collapsed is False

            await pilot.press("enter")
            await pilot.pause()
            assert citation_widget.collapsed is True


@pytest.mark.asyncio
async def test_show_citations_renders_from_flat_state(temp_db_path: Path):
    """Citations in state (flat list[str]) render into the chat history."""
    from haiku.rag.chat.app import RAG_STATE_NAMESPACE
    from haiku.rag.chat.widgets.chat_history import ChatHistory, CitationWidget
    from haiku.rag.store.models.citation import Citation

    app, mock_client = _make_app_with_state(temp_db_path)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test() as pilot:
            rag_state = RAGState.model_validate(app._state[RAG_STATE_NAMESPACE])

            citation = Citation(
                index=1,
                document_id="doc1",
                chunk_id="chunk1",
                document_uri="file:///test/doc1.pdf",
                document_title="Test Document",
                page_numbers=[1],
                content="Cited content",
            )
            rag_state.citation_index["chunk1"] = citation
            rag_state.citations.append("chunk1")
            app._state[RAG_STATE_NAMESPACE] = rag_state.model_dump(mode="json")

            chat_history = app.query_one(ChatHistory)
            await app._show_citations_and_programs(chat_history)
            await pilot.pause()

            widgets = list(chat_history.query(CitationWidget))
            assert len(widgets) == 1
            assert widgets[0].citation.chunk_id == "chunk1"


@pytest.mark.asyncio
async def test_document_filter_updates_rag_state(temp_db_path: Path):
    """Test that selecting document filters updates RAGState.document_filter."""
    from haiku.rag.chat.app import RAG_STATE_NAMESPACE
    from haiku.rag.chat.widgets.document_filter_modal import DocumentFilterModal
    from haiku.rag.tools.filters import build_multi_document_filter

    app, mock_client = _make_app_with_state(temp_db_path)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test():
            # Simulate the FilterChanged message
            selected = ["AI Overview", "ML Basics"]
            app.on_document_filter_modal_filter_changed(
                DocumentFilterModal.FilterChanged(selected)
            )

            # RAGState.document_filter should be set
            rag_state = RAGState.model_validate(app._state[RAG_STATE_NAMESPACE])
            expected_filter = build_multi_document_filter(selected)
            assert rag_state.document_filter == expected_filter

            # The state snapshot should also reflect the change
            assert app._state["rag"]["document_filter"] == expected_filter


@pytest.mark.asyncio
async def test_document_filter_cleared_when_empty(temp_db_path: Path):
    """Test that clearing all document filters sets document_filter to None."""
    from haiku.rag.chat.app import RAG_STATE_NAMESPACE
    from haiku.rag.chat.widgets.document_filter_modal import DocumentFilterModal

    app, mock_client = _make_app_with_state(temp_db_path)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test():
            # First set a filter
            app.on_document_filter_modal_filter_changed(
                DocumentFilterModal.FilterChanged(["AI Overview"])
            )
            rag_state = RAGState.model_validate(app._state[RAG_STATE_NAMESPACE])
            assert rag_state.document_filter is not None

            # Then clear it
            app.on_document_filter_modal_filter_changed(
                DocumentFilterModal.FilterChanged([])
            )
            rag_state = RAGState.model_validate(app._state[RAG_STATE_NAMESPACE])
            assert rag_state.document_filter is None
            assert app._state["rag"]["document_filter"] is None


@pytest.mark.asyncio
async def test_chat_app_open_failure_surfaces_real_error(tmp_path: Path):
    """A failed database open must surface its own error, not an
    AttributeError from tearing down a client that never opened."""
    from haiku.rag.chat.app import ChatApp

    app = ChatApp(db_path=tmp_path / "missing.lancedb", capabilities=[])
    with pytest.raises(FileNotFoundError):
        async with app.run_test():
            pass


@pytest.mark.asyncio
async def test_a_cancelled_run_does_not_advance_persisted_state(temp_db_path: Path):
    """State and message history have to move together, or the thread bricks.

    A cancelled run keeps whatever the tools wrote but discards the run's messages.
    If the state advanced, the next question derives its identity from the shorter
    history, lands behind the recorded evidence epoch, and is refused as
    non-append-only — leaving the conversation unusable until cleared.
    """
    import asyncio

    app, mock_client = _make_app(temp_db_path)

    class CancellingRun:
        """A run that writes evidence through the tools, then is cancelled."""

        def __init__(self, deps):
            self._deps = deps

        async def __aenter__(self):
            self._deps.state["rag"] = {
                "evidence": {"question": 0, "latest_evidence_epoch": 7}
            }
            return self

        async def __aexit__(self, *_):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise asyncio.CancelledError

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test():
            app._state = {
                "rag": {"evidence": {"question": 0, "latest_evidence_epoch": 0}}
            }
            before = deepcopy(app._state)

            class Agent:
                def run_stream_events(self, *_, deps, **__):
                    return CancellingRun(deps)

            app._agent = Agent()  # type: ignore[assignment]
            await app._run_agent("a question")

            assert app._state == before
            assert app._messages == []


async def test_visual_grounding_uses_the_database_holding_the_citation(tmp_path):
    """Chunks, pages and boxes come from one database. Covering a set, the
    citation's source says which, and a covering client has no repositories."""
    from haiku.rag.chat.widgets.chat_history import ChatHistory, CitationWidget
    from haiku.rag.store.models import Chunk
    from haiku.rag.store.models.citation import Citation

    owner = _make_mock_client()
    owner._source = "beta"
    owner.get_chunk_by_id.return_value = Chunk(
        id="c1", document_id="d1", content="cited body"
    )

    covering = _make_mock_client()
    covering._federated = {"alpha": "/a.lancedb", "beta": "/b.lancedb"}
    covering.clients_for = AsyncMock(return_value=[owner])

    app, _ = _make_app(tmp_path / "unused.lancedb", covering)
    with patch("haiku.rag.chat.app.HaikuRAG", return_value=covering):
        async with app.run_test():
            history = app.query_one(ChatHistory)
            await history.add_citations(
                [
                    Citation(
                        chunk_id="c1",
                        document_id="d1",
                        document_uri="test://beta/one",
                        content="cited body",
                        source="beta",
                    )
                ]
            )
            widget = next(iter(app.query(CitationWidget)))
            widget.add_class("selected")

            with patch.object(app, "push_screen", new=AsyncMock()) as push:
                await app.action_show_visual()

    covering.clients_for.assert_awaited_once_with(["beta"])
    owner.get_chunk_by_id.assert_awaited_once_with("c1")
    assert push.await_args is not None
    assert push.await_args.args[0].client is owner


class TestDocumentSearchFilter:
    """The filter modal shows one page and asks the database for the rest, so the
    typed term reaches SQL."""

    def test_no_term_means_no_filter(self):
        from haiku.rag.chat.widgets.document_filter_modal import search_filter

        assert search_filter("   ") is None

    def test_a_term_matches_titles_and_uris(self):
        from haiku.rag.chat.widgets.document_filter_modal import search_filter

        built = search_filter("Nobel")

        assert built == (
            "LOWER(title) LIKE '%nobel%' ESCAPE '\\' "
            "OR LOWER(uri) LIKE '%nobel%' ESCAPE '\\'"
        )

    def test_the_search_is_case_insensitive(self):
        """LIKE is case-sensitive, so both sides are lowered."""
        from haiku.rag.chat.widgets.document_filter_modal import search_filter

        built = search_filter("NoBeL")

        assert built is not None
        assert "LOWER(title) LIKE '%nobel%'" in built
        assert "LOWER(uri) LIKE '%nobel%'" in built

    def test_like_wildcards_in_the_term_are_literal(self):
        """A term is text to find, not a pattern: `_` matches any character."""
        from haiku.rag.chat.widgets.document_filter_modal import search_filter

        built = search_filter("100%_raw")

        assert built is not None
        assert "100\\%\\_raw" in built
        assert "ESCAPE '\\'" in built

    def test_a_quote_in_the_term_is_escaped(self):
        """Whatever was typed is data, not syntax."""
        from haiku.rag.chat.widgets.document_filter_modal import search_filter

        built = search_filter("O'Brien")

        assert built is not None
        assert "o''brien" in built
