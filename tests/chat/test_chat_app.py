from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from haiku.rag.capabilities.rag import RAGState, create_capability
from haiku.rag.cli import _cli as cli
from tests.conftest import _covering_returns, for_path

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

    assert app.call_args.kwargs["scope"].names == ("a", "b")


def test_chat_capabilities_read_the_named_database(tmp_path, monkeypatch):
    """`--db PATH` and `--db-name NAME` reach the capabilities too."""
    import haiku.rag.config as config_module
    from haiku.rag.client.scope import DatabaseScope
    from haiku.rag.config import set_config
    from haiku.rag.config.models import AppConfig, LanceDBConfig

    monkeypatch.setattr(config_module, "_config", None)
    config = AppConfig(
        lancedb=LanceDBConfig(
            databases={
                "a": str(tmp_path / "a.lancedb"),
                "b": str(tmp_path / "b.lancedb"),
            }
        )
    )
    set_config(config)

    with patch("haiku.rag.chat.app.ChatApp") as chat_app:
        from haiku.rag.chat import run_chat

        run_chat(scope=DatabaseScope.resolve(config, database_name="b"))
        named_scope = chat_app.call_args.kwargs["scope"]
        [named] = chat_app.call_args.kwargs["capabilities"]

        run_chat(scope=DatabaseScope.resolve(config))
        covering_scope = chat_app.call_args.kwargs["scope"]
        [covering] = chat_app.call_args.kwargs["capabilities"]

    # The app opens the scope it is handed and lends that client to the
    # capabilities, which keep the configuration as the caller named it.
    assert named_scope.names == ("b",)
    assert covering_scope.names == ("a", "b")
    assert set(named.config.lancedb.databases) == {"a", "b"}
    assert set(covering.config.lancedb.databases) == {"a", "b"}


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
    # Covers one database; a bare AsyncMock answers `covers_multiple` with a
    # truthy Mock.
    mock_client.covers_multiple = False
    mock_client.source_names = ()
    mock_client.source = None
    return mock_client


def _make_app(db_path: Path, mock_client: AsyncMock | None = None):
    """Create a ChatApp with mocked HaikuRAG."""
    from haiku.rag.chat.app import ChatApp

    if mock_client is None:
        mock_client = _make_mock_client()

    return ChatApp(
        scope=for_path(db_path),
        capabilities=[create_capability(db_path=db_path)],
        read_only=True,
    ), mock_client


def _make_app_with_state(db_path: Path, mock_client: AsyncMock | None = None):
    """Create a ChatApp with a RAG capability and state."""
    from haiku.rag.chat.app import ChatApp

    if mock_client is None:
        mock_client = _make_mock_client()

    return ChatApp(
        scope=for_path(db_path),
        capabilities=[create_capability(db_path=db_path)],
        read_only=True,
    ), mock_client


@pytest.mark.asyncio
async def test_chat_app_has_required_widgets(temp_db_path: Path):
    """Test that ChatApp has the required widgets: ChatHistory, FlexibleInput."""
    from haiku.rag.chat.widgets.chat_history import ChatHistory

    app, mock_client = _make_app(temp_db_path)

    with (
        patch("haiku.rag.chat.app.HaikuRAG") as _stub_rag,
        _covering_returns(_stub_rag, mock_client),
    ):
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

    with (
        patch("haiku.rag.chat.app.HaikuRAG") as _stub_rag,
        _covering_returns(_stub_rag, mock_client),
    ):
        async with app.run_test() as pilot:
            assert app.is_running
            await pilot.press("ctrl+q")
            assert not app.is_running


@pytest.mark.asyncio
async def test_chat_history_can_add_message(temp_db_path: Path):
    """Test that ChatHistory can display messages."""
    from haiku.rag.chat.widgets.chat_history import ChatHistory

    app, mock_client = _make_app(temp_db_path)

    with (
        patch("haiku.rag.chat.app.HaikuRAG") as _stub_rag,
        _covering_returns(_stub_rag, mock_client),
    ):
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

    with (
        patch("haiku.rag.chat.app.HaikuRAG") as _stub_rag,
        _covering_returns(_stub_rag, mock_client),
    ):
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

    with (
        patch("haiku.rag.chat.app.HaikuRAG") as _stub_rag,
        _covering_returns(_stub_rag, mock_client),
    ):
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

    with (
        patch("haiku.rag.chat.app.HaikuRAG") as _stub_rag,
        _covering_returns(_stub_rag, mock_client),
    ):
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

    with (
        patch("haiku.rag.chat.app.HaikuRAG") as _stub_rag,
        _covering_returns(_stub_rag, mock_client),
    ):
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

    with (
        patch("haiku.rag.chat.app.HaikuRAG") as _stub_rag,
        _covering_returns(_stub_rag, mock_client),
    ):
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

    with (
        patch("haiku.rag.chat.app.HaikuRAG") as _stub_rag,
        _covering_returns(_stub_rag, mock_client),
    ):
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
    from haiku.rag.tools.filters import build_document_id_filter

    app, mock_client = _make_app_with_state(temp_db_path)

    with (
        patch("haiku.rag.chat.app.HaikuRAG") as _stub_rag,
        _covering_returns(_stub_rag, mock_client),
    ):
        async with app.run_test():
            # The selection is document ids, so a repeated title cannot widen it.
            selected = [
                (None, "6f1c2d4e-0000-4000-8000-000000000001"),
                (None, "6f1c2d4e-0000-4000-8000-000000000002"),
            ]
            app.on_document_filter_modal_filter_changed(
                DocumentFilterModal.FilterChanged(selected)
            )

            # RAGState.document_filter should be set
            rag_state = RAGState.model_validate(app._state[RAG_STATE_NAMESPACE])
            expected_filter = build_document_id_filter(
                [doc_id for _, doc_id in selected]
            )
            assert rag_state.document_filter == expected_filter
            assert rag_state.document_filter is not None
            assert "LIKE" not in rag_state.document_filter
            # An unnamed database leaves the question unscoped by source.
            assert rag_state.sources is None

            # The state snapshot should also reflect the change
            assert app._state["rag"]["document_filter"] == expected_filter


@pytest.mark.asyncio
async def test_document_filter_narrows_sources_to_the_selection(temp_db_path: Path):
    """The filter carries ids, and `sources` restricts the question to the
    databases the selection names."""
    from haiku.rag.chat.app import RAG_STATE_NAMESPACE
    from haiku.rag.chat.widgets.document_filter_modal import DocumentFilterModal

    app, mock_client = _make_app_with_state(temp_db_path)

    with (
        patch("haiku.rag.chat.app.HaikuRAG") as _stub_rag,
        _covering_returns(_stub_rag, mock_client),
    ):
        async with app.run_test():
            app.on_document_filter_modal_filter_changed(
                DocumentFilterModal.FilterChanged(
                    [("alpha", "id-one"), ("alpha", "id-two")]
                )
            )
            rag_state = RAGState.model_validate(app._state[RAG_STATE_NAMESPACE])
            assert rag_state.sources == ["alpha"]

            app.on_document_filter_modal_filter_changed(
                DocumentFilterModal.FilterChanged(
                    [("alpha", "id-one"), ("beta", "id-three")]
                )
            )
            rag_state = RAGState.model_validate(app._state[RAG_STATE_NAMESPACE])
            assert rag_state.sources == ["alpha", "beta"]

            app.on_document_filter_modal_filter_changed(
                DocumentFilterModal.FilterChanged([])
            )
            rag_state = RAGState.model_validate(app._state[RAG_STATE_NAMESPACE])
            assert rag_state.sources is None
            assert rag_state.document_filter is None


@pytest.mark.asyncio
async def test_document_filter_cleared_when_empty(temp_db_path: Path):
    """Test that clearing all document filters sets document_filter to None."""
    from haiku.rag.chat.app import RAG_STATE_NAMESPACE
    from haiku.rag.chat.widgets.document_filter_modal import DocumentFilterModal

    app, mock_client = _make_app_with_state(temp_db_path)

    with (
        patch("haiku.rag.chat.app.HaikuRAG") as _stub_rag,
        _covering_returns(_stub_rag, mock_client),
    ):
        async with app.run_test():
            # First set a filter
            app.on_document_filter_modal_filter_changed(
                DocumentFilterModal.FilterChanged([(None, "AI Overview")])
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

    app = ChatApp(scope=for_path(tmp_path / "missing.lancedb"), capabilities=[])
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

    with (
        patch("haiku.rag.chat.app.HaikuRAG") as _stub_rag,
        _covering_returns(_stub_rag, mock_client),
    ):
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
    owner.source = "beta"
    owner.get_chunk_by_id.return_value = Chunk(
        id="c1", document_id="d1", content="cited body"
    )

    covering = _make_mock_client()
    covering.covers_multiple = True
    covering.source_names = ("alpha", "beta")
    covering.reader_for = AsyncMock(return_value=owner)

    app, _ = _make_app(tmp_path / "unused.lancedb", covering)
    with (
        patch("haiku.rag.chat.app.HaikuRAG") as _stub_rag,
        _covering_returns(_stub_rag, covering),
    ):
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

    covering.reader_for.assert_awaited_once_with("beta")
    owner.get_chunk_by_id.assert_awaited_once_with("c1")
    assert push.await_args is not None
    assert push.await_args.args[0].client is owner


class TestLendingTheClient:
    @pytest.mark.asyncio
    async def test_mounting_lends_its_client_to_every_capability(self, temp_db_path):
        """Capabilities are built before the client exists, and each reads
        through the one the app opened."""
        client = _make_mock_client()
        app, _ = _make_app(temp_db_path, client)

        with (
            patch("haiku.rag.chat.app.HaikuRAG") as stub_rag,
            _covering_returns(stub_rag, client),
        ):
            async with app.run_test():
                borrowed = [c.borrowed_rag for c in app._capabilities]

        assert borrowed == [client] * len(app._capabilities)
        assert borrowed

    @pytest.mark.asyncio
    async def test_mounting_gives_every_capability_the_apps_scope(self, tmp_path):
        """A capability built over the configured set covers what the chat
        selected once mounted: the analysis sandbox is built over that scope."""
        from haiku.rag.chat.app import ChatApp
        from haiku.rag.client.scope import DatabaseScope
        from haiku.rag.config.models import AppConfig, LanceDBConfig

        config = AppConfig(
            lancedb=LanceDBConfig(
                databases={
                    "a": str(tmp_path / "a.lancedb"),
                    "b": str(tmp_path / "b.lancedb"),
                }
            )
        )
        selected = DatabaseScope.resolve(config, database_name="b")
        capability = create_capability(config=config)
        assert capability.scope.covers_multiple

        client = _make_mock_client()
        app = ChatApp(scope=selected, capabilities=[capability], read_only=True)
        with (
            patch("haiku.rag.chat.app.HaikuRAG") as stub_rag,
            _covering_returns(stub_rag, client),
        ):
            async with app.run_test():
                pass

        assert capability.scope == selected


class TestDocumentSelectionIdentity:
    """Two documents can share a title, within a corpus and across databases, so
    the selection is by id and the label says which database."""

    @pytest.mark.asyncio
    async def test_a_repeated_title_selects_one_document(self, temp_db_path: Path):
        from haiku.rag.chat.widgets.document_filter_modal import (
            DocumentCheckbox,
            DocumentFilterModal,
        )
        from haiku.rag.store.models.document import Document

        client = AsyncMock()
        client.covers_multiple = True
        client.source_names = ("arxiv", "wiki")
        client.list_documents.return_value = [
            Document(id="id-one", content="", title="Capital region", source="arxiv"),
            Document(id="id-two", content="", title="Capital region", source="wiki"),
        ]
        client.count_documents.return_value = 2

        modal = DocumentFilterModal(client=client)
        app, _ = _make_app(temp_db_path, client)
        with (
            patch("haiku.rag.chat.app.HaikuRAG") as _stub_rag,
            _covering_returns(_stub_rag, client),
        ):
            async with app.run_test() as pilot:
                await app.push_screen(modal)
                await pilot.pause()

                boxes = list(modal.query(DocumentCheckbox))
                selected_ids = [box.doc_id for box in boxes]
                assert selected_ids == ["id-one", "id-two"]
                labels = [str(b.label) for b in boxes]
                assert "Capital region  (arxiv)" in labels
                assert "Capital region  (wiki)" in labels

                boxes[0].value = True
                await pilot.pause()
                assert modal._selected == {("arxiv", "id-one")}

    @pytest.mark.asyncio
    async def test_a_shared_id_selects_only_the_named_database_copy(
        self, temp_db_path: Path
    ):
        """Copies of a database share document ids, so a selection carries the
        database name and checking one copy leaves the other unselected."""
        from haiku.rag.chat.widgets.document_filter_modal import (
            DocumentCheckbox,
            DocumentFilterModal,
        )
        from haiku.rag.store.models.document import Document

        client = AsyncMock()
        client.covers_multiple = True
        client.source_names = ("alpha", "beta")
        client.list_documents.return_value = [
            Document(id="id-x", content="", title="Report", source="alpha"),
            Document(id="id-x", content="", title="Report", source="beta"),
        ]
        client.count_documents.return_value = 2

        modal = DocumentFilterModal(client=client)
        app, _ = _make_app(temp_db_path, client)
        with (
            patch("haiku.rag.chat.app.HaikuRAG") as _stub_rag,
            _covering_returns(_stub_rag, client),
        ):
            async with app.run_test() as pilot:
                await app.push_screen(modal)
                await pilot.pause()

                boxes = list(modal.query(DocumentCheckbox))
                assert [str(b.label) for b in boxes] == [
                    "Report  (alpha)",
                    "Report  (beta)",
                ]

                boxes[0].value = True
                await pilot.pause()
                assert modal._selected == {("alpha", "id-x")}

                await modal._load_documents()
                rebuilt = list(modal.query(DocumentCheckbox))
                assert [b.value for b in rebuilt] == [True, False]

    def test_a_label_that_looks_like_markup_is_text(self):
        from haiku.rag.chat.widgets.document_filter_modal import (
            DocumentCheckbox,
            _labelled,
        )
        from haiku.rag.store.models.document import Document

        docs = [
            Document(
                id="id-one", content="", title="Report [/red]", source="alpha [/x]"
            )
        ]

        ((label, source, doc_id),) = _labelled(docs)
        box = DocumentCheckbox(label, source, doc_id, value=False)

        assert str(box.label) == "Report [/red]  (alpha [/x])"


def test_a_citation_title_that_looks_like_markup_is_text():
    from rich.text import Text

    from haiku.rag.chat.widgets.chat_history import CitationWidget
    from haiku.rag.store.models.citation import Citation

    citation = Citation(
        index=1,
        document_id="doc1",
        chunk_id="chunk1",
        document_uri="file:///doc [/blue].pdf",
        document_title="Report [/red]",
        headings=["Chapter [/dim]"],
        content="content",
        source="alpha [/x]",
    )

    widget = CitationWidget(citation, include_collection=True)

    title = Text.from_markup(str(widget.title)).plain
    assert "Report [/red]" in title
    assert "alpha [/x]" in title


class TestRenderingUnattributedPictures:
    @pytest.mark.asyncio
    async def test_a_sourceless_picture_citation_does_not_fail_the_answer(
        self, temp_db_path: Path
    ):
        """A citation without a source has no picture owner across databases.
        It renders with its figure markers."""
        from haiku.rag.chat.app import RAG_STATE_NAMESPACE
        from haiku.rag.store.models.citation import Citation

        covering = _make_mock_client()
        covering.covers_multiple = True
        covering.source_names = ("alpha", "beta")
        covering.reader_for = AsyncMock(return_value=None)
        covering.get_picture_bytes = AsyncMock(
            side_effect=AssertionError("asked a set for a picture it cannot place")
        )

        citation = Citation(
            document_id="d1",
            chunk_id="c1",
            content="body",
            document_uri="test://doc",
            picture_refs=["#/pictures/0"],
        )

        app, _ = _make_app(temp_db_path, covering)
        with (
            patch("haiku.rag.chat.app.HaikuRAG") as stub,
            _covering_returns(stub, covering),
        ):
            async with app.run_test() as pilot:
                app._state[RAG_STATE_NAMESPACE] = {
                    "citations": ["c1"],
                    "citation_index": {"c1": citation.model_dump(mode="json")},
                }
                from haiku.rag.chat.widgets.chat_history import ChatHistory

                await app._show_citations_and_programs(app.query_one(ChatHistory))
                await pilot.pause()

        covering.reader_for.assert_awaited_once_with(None)
        covering.get_picture_bytes.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_one_chunk_id_in_two_collections_keeps_its_own_pictures(
        self, temp_db_path: Path, monkeypatch
    ):
        """Chunk ids repeat between copies of a database, and both capabilities'
        citations are gathered into one mapping."""
        from io import BytesIO

        from PIL import Image as PILImage

        from haiku.rag.chat.app import ANALYSIS_STATE_NAMESPACE, RAG_STATE_NAMESPACE
        from haiku.rag.chat.widgets.chat_history import ChatHistory, CitationWidget
        from haiku.rag.store.models.citation import Citation

        def png(color: str) -> bytes:
            buffer = BytesIO()
            PILImage.new("RGB", (4, 4), color).save(buffer, format="PNG")
            return buffer.getvalue()

        pictures = {"alpha": png("red"), "beta": png("blue")}

        def reader(source):
            owner = AsyncMock()
            owner.get_picture_bytes = AsyncMock(return_value=pictures[source])
            return owner

        covering = _make_mock_client()
        covering.covers_multiple = True
        covering.source_names = ("alpha", "beta")
        covering.reader_for = AsyncMock(side_effect=reader)

        def cited(source: str) -> dict:
            return Citation(
                document_id="d1",
                chunk_id="c1",
                source=source,
                content="body",
                document_uri=f"test://{source}",
                picture_refs=["#/pictures/0"],
            ).model_dump(mode="json")

        seen: list[tuple[str | None, list[bytes] | None]] = []
        build = CitationWidget.__init__

        def capture(self, citation, picture_bytes=None, **kwargs):
            seen.append((citation.source, picture_bytes))
            build(self, citation, picture_bytes, **kwargs)

        monkeypatch.setattr(CitationWidget, "__init__", capture)

        app, _ = _make_app(temp_db_path, covering)
        with (
            patch("haiku.rag.chat.app.HaikuRAG") as stub,
            _covering_returns(stub, covering),
        ):
            async with app.run_test() as pilot:
                app._state[RAG_STATE_NAMESPACE] = {
                    "citations": ["c1"],
                    "citation_index": {"c1": cited("alpha")},
                }
                app._state[ANALYSIS_STATE_NAMESPACE] = {
                    "citations": ["c1"],
                    "citation_index": {"c1": cited("beta")},
                }

                await app._show_citations_and_programs(app.query_one(ChatHistory))
                await pilot.pause()

        assert seen == [
            ("alpha", [pictures["alpha"]]),
            ("beta", [pictures["beta"]]),
        ]


class TestNamingACitationsCollection:
    @staticmethod
    async def _titles(temp_db_path, covering, *sources: str | None) -> list[str]:
        """The collapsed titles of one citation per source, all named alike."""
        from haiku.rag.chat.app import RAG_STATE_NAMESPACE
        from haiku.rag.chat.widgets.chat_history import ChatHistory, CitationWidget
        from haiku.rag.store.models.citation import Citation

        index = {
            f"c{position}": Citation(
                document_id=f"d{position}",
                chunk_id=f"c{position}",
                source=source,
                content="body",
                document_uri="test://report",
                document_title="Quarterly report",
            ).model_dump(mode="json")
            for position, source in enumerate(sources)
        }

        app, _ = _make_app(temp_db_path, covering)
        with (
            patch("haiku.rag.chat.app.HaikuRAG") as stub,
            _covering_returns(stub, covering),
        ):
            async with app.run_test() as pilot:
                app._state[RAG_STATE_NAMESPACE] = {
                    "citations": list(index),
                    "citation_index": index,
                }
                await app._show_citations_and_programs(app.query_one(ChatHistory))
                await pilot.pause()
                return [str(w.title) for w in app.query(CitationWidget)]

    @pytest.mark.asyncio
    async def test_one_title_in_two_collections_reads_as_two_citations(
        self, temp_db_path: Path
    ):
        covering = _make_mock_client()
        covering.covers_multiple = True
        covering.source_names = ("alpha", "beta")

        titles = await self._titles(temp_db_path, covering, "alpha", "beta")

        assert len(set(titles)) == 2
        assert "alpha" in titles[0]
        assert "beta" in titles[1]

    @pytest.mark.asyncio
    async def test_one_named_database_is_not_named_on_its_citations(
        self, temp_db_path: Path
    ):
        covering = _make_mock_client()
        covering.covers_multiple = False
        covering.source_names = ("alpha",)

        [title] = await self._titles(temp_db_path, covering, "alpha")

        assert "alpha" not in title


class TestKeepingSelectionsReachable:
    """A selection applies whether or not the page shows it, and a checkbox is
    the only way to remove one."""

    @pytest.mark.asyncio
    async def test_every_selection_is_reachable_and_the_page_stays_bounded(
        self, temp_db_path: Path
    ):
        """Selections accumulate across searches and get their own listing,
        paged the same way."""
        from textual.widgets import Button, Static

        from haiku.rag.chat.widgets.document_filter_modal import (
            DOCUMENT_PAGE,
            DocumentCheckbox,
            DocumentFilterModal,
        )
        from haiku.rag.store.models.document import Document

        picked = [
            Document(id=f"sel-{i:04d}", content="", title=f"Selected {i:04d}")
            for i in range(DOCUMENT_PAGE + 20)
        ]
        by_id = {d.id: d for d in picked}
        matched = [
            Document(id=f"hit-{i}", content="", title=f"Hit {i}") for i in range(5)
        ]

        client = AsyncMock()
        client.covers_multiple = False
        client.source_names = ()
        client.count_documents.return_value = 5

        async def listing(limit=None, offset=0, filter=None):
            if filter and filter.startswith("id IN"):
                ids = filter[len("id IN (") : -1].replace("'", "").split(", ")
                return [by_id[i] for i in ids]
            return matched

        client.list_documents.side_effect = listing

        modal = DocumentFilterModal(
            client=client, selected=[(None, d.id or "") for d in picked]
        )
        app, _ = _make_app(temp_db_path, client)
        with (
            patch("haiku.rag.chat.app.HaikuRAG") as stub,
            _covering_returns(stub, client),
        ):
            async with app.run_test() as pilot:
                await app.push_screen(modal)
                await pilot.pause()

                # Results are the results: selections are not appended to them.
                assert len(list(modal.query(DocumentCheckbox))) == len(matched)

                await modal.on_button_pressed(
                    Button.Pressed(modal.query_one("#selected-btn", Button))
                )
                await pilot.pause()
                first = [b.doc_id for b in modal.query(DocumentCheckbox)]
                footer = str(modal.query_one("#filter-footer", Static).content)

                await modal.on_button_pressed(
                    Button.Pressed(modal.query_one("#next-btn", Button))
                )
                await pilot.pause()
                second = [b.doc_id for b in modal.query(DocumentCheckbox)]

        assert len(first) == DOCUMENT_PAGE
        assert "page 1 of 2" in footer
        # The rest are on the next page, so every selection can be removed.
        assert len(second) == 20
        assert set(first) | set(second) == {d.id for d in picked}

    @pytest.mark.asyncio
    async def test_deselecting_updates_the_selected_listing(self, temp_db_path: Path):
        """The listing is the selection, so removing one changes what it holds
        and how far it runs. Its last page can stop existing."""
        from textual.widgets import Button, Checkbox, Static

        from haiku.rag.chat.widgets.document_filter_modal import (
            DOCUMENT_PAGE,
            DocumentCheckbox,
            DocumentFilterModal,
        )
        from haiku.rag.store.models.document import Document

        picked = [
            Document(id=f"sel-{i:04d}", content="", title=f"Selected {i:04d}")
            for i in range(DOCUMENT_PAGE + 1)
        ]
        by_id = {d.id: d for d in picked}

        client = AsyncMock()
        client.covers_multiple = False
        client.source_names = ()
        client.count_documents.return_value = 0

        async def listing(limit=None, offset=0, filter=None):
            if filter and filter.startswith("id IN"):
                ids = filter[len("id IN (") : -1].replace("'", "").split(", ")
                return [by_id[i] for i in ids]
            return []

        client.list_documents.side_effect = listing

        modal = DocumentFilterModal(
            client=client, selected=[(None, d.id or "") for d in picked]
        )
        app, _ = _make_app(temp_db_path, client)
        with (
            patch("haiku.rag.chat.app.HaikuRAG") as stub,
            _covering_returns(stub, client),
        ):
            async with app.run_test() as pilot:
                await app.push_screen(modal)
                await pilot.pause()
                await modal.on_button_pressed(
                    Button.Pressed(modal.query_one("#selected-btn", Button))
                )
                await pilot.pause()
                await modal.on_button_pressed(
                    Button.Pressed(modal.query_one("#next-btn", Button))
                )
                await pilot.pause()

                [only] = list(modal.query(DocumentCheckbox))
                assert only.doc_id == "sel-0200"
                # Awaited directly: the handler reads the database, and a
                # single pause need not have flushed it.
                await modal.on_checkbox_changed(Checkbox.Changed(only, False))
                await pilot.pause()

                remaining = [b.doc_id for b in modal.query(DocumentCheckbox)]
                footer = str(modal.query_one("#filter-footer", Static).content)

        # The row is gone from the listing, not merely unchecked.
        assert "sel-0200" not in remaining
        assert len(remaining) == DOCUMENT_PAGE
        assert modal._selected == {(None, d.id) for d in picked} - {(None, "sel-0200")}
        # The page it was on no longer exists, so the modal does not report it.
        assert modal._page == 0
        assert "page" not in footer
        assert f"[bold]{DOCUMENT_PAGE}[/bold] document(s) selected" in footer

    @pytest.mark.asyncio
    async def test_the_results_listing_pages_too(self, temp_db_path: Path):
        """More documents match than one page holds; the rest are a page
        away."""
        from textual.widgets import Button

        from haiku.rag.chat.widgets.document_filter_modal import (
            DOCUMENT_PAGE,
            DocumentFilterModal,
        )
        from haiku.rag.store.models.document import Document

        client = AsyncMock()
        client.covers_multiple = False
        client.source_names = ()
        client.count_documents.return_value = DOCUMENT_PAGE * 2
        client.list_documents.return_value = [
            Document(id="d1", content="", title="One")
        ]

        modal = DocumentFilterModal(client=client)
        app, _ = _make_app(temp_db_path, client)
        with (
            patch("haiku.rag.chat.app.HaikuRAG") as stub,
            _covering_returns(stub, client),
        ):
            async with app.run_test() as pilot:
                await app.push_screen(modal)
                await pilot.pause()
                assert client.list_documents.await_args.kwargs["offset"] == 0

                await modal.on_button_pressed(
                    Button.Pressed(modal.query_one("#next-btn", Button))
                )
                await pilot.pause()

        assert client.list_documents.await_args.kwargs["offset"] == DOCUMENT_PAGE

    @pytest.mark.asyncio
    async def test_a_search_matching_nothing_says_so(self, temp_db_path: Path):
        """An empty list is indistinguishable from one still loading."""
        from textual.widgets import Static

        from haiku.rag.chat.widgets.document_filter_modal import DocumentFilterModal

        client = AsyncMock()
        client.covers_multiple = False
        client.source_names = ()
        client.list_documents.return_value = []
        client.count_documents.return_value = 0

        modal = DocumentFilterModal(client=client)
        app, _ = _make_app(temp_db_path, client)
        with (
            patch("haiku.rag.chat.app.HaikuRAG") as stub,
            _covering_returns(stub, client),
        ):
            async with app.run_test() as pilot:
                await app.push_screen(modal)
                await pilot.pause()

                empty = modal.query_one("#filter-empty", Static)
                assert "No documents match" in str(empty.content)

    @pytest.mark.asyncio
    async def test_typing_without_submitting_leaves_the_listing_alone(
        self, temp_db_path: Path
    ):
        """The listing is what the last submitted search asked for, and it
        pages. The term applies on enter, and the footer says so until then.
        """
        from textual.widgets import Button, Input, Static

        from haiku.rag.chat.widgets.document_filter_modal import (
            DOCUMENT_PAGE,
            DocumentCheckbox,
            DocumentFilterModal,
        )
        from haiku.rag.store.models.document import Document

        client = AsyncMock()
        client.covers_multiple = False
        client.source_names = ()
        client.list_documents.return_value = [
            Document(id="id-one", content="", title="Capital region"),
            Document(id="id-two", content="", title="Nobel laureates"),
        ]
        client.count_documents.return_value = DOCUMENT_PAGE * 2

        modal = DocumentFilterModal(client=client)
        app, _ = _make_app(temp_db_path, client)
        with (
            patch("haiku.rag.chat.app.HaikuRAG") as stub,
            _covering_returns(stub, client),
        ):
            async with app.run_test() as pilot:
                await app.push_screen(modal)
                await pilot.pause()
                footer = modal.query_one("#filter-footer", Static)
                assert "press enter to search" not in str(footer.content)

                search = modal.query_one("#filter-search", Input)
                search.value = "nobel"
                modal.on_input_changed(Input.Changed(search, "nobel"))

                assert "press enter to search" in str(footer.content)
                assert all(box.display for box in modal.query(DocumentCheckbox))

                await modal.on_button_pressed(
                    Button.Pressed(modal.query_one("#next-btn", Button))
                )
                await pilot.pause()

                paged = client.list_documents.await_args.kwargs
                assert paged["offset"] == DOCUMENT_PAGE
                assert paged["filter"] is None
                assert "press enter to search" in str(footer.content)

                await modal.on_input_submitted(Input.Submitted(search, search.value))
                await pilot.pause()

                submitted = client.list_documents.await_args.kwargs
                assert submitted["offset"] == 0
                assert "nobel" in submitted["filter"]
                assert "press enter to search" not in str(footer.content)


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

    @pytest.mark.asyncio
    async def test_submitting_a_search_reloads_the_page(self, temp_db_path: Path):
        """The typed term reaches the database and replaces what is shown."""
        from textual.widgets import Input

        from haiku.rag.chat.widgets.document_filter_modal import (
            DocumentCheckbox,
            DocumentFilterModal,
        )
        from haiku.rag.store.models.document import Document

        client = AsyncMock()
        client.covers_multiple = False
        client.source_names = ()
        client.list_documents.return_value = [
            Document(id="id-one", content="", title="Capital region"),
            Document(id="id-two", content="", title="Nobel laureates"),
        ]
        client.count_documents.return_value = 2

        modal = DocumentFilterModal(client=client)
        app, _ = _make_app(temp_db_path, client)
        with (
            patch("haiku.rag.chat.app.HaikuRAG") as _stub_rag,
            _covering_returns(_stub_rag, client),
        ):
            async with app.run_test() as pilot:
                await app.push_screen(modal)
                await pilot.pause()
                assert len(list(modal.query(DocumentCheckbox))) == 2

                client.list_documents.return_value = [
                    Document(id="id-two", content="", title="Nobel laureates"),
                ]
                client.count_documents.return_value = 1
                await modal.on_input_submitted(Input.Submitted(Input(), "Nobel"))
                await pilot.pause()

                assert client.list_documents.await_args is not None
                assert "nobel" in client.list_documents.await_args.kwargs["filter"]
                labels = [str(b.label) for b in modal.query(DocumentCheckbox)]
                assert labels == ["Nobel laureates"]
