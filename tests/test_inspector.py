import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image as PILImage
from typer.testing import CliRunner

from haiku.rag.cli import _cli as cli
from haiku.rag.store.models import Chunk, Document, SearchResult
from tests.conftest import for_path

runner = CliRunner()


def _png_b64(color: str = "red", size: tuple[int, int] = (8, 8)) -> str:
    img = PILImage.new("RGB", size, color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_client(*, vision: bool, image_data: dict[str, str] | None) -> MagicMock:
    """Stub HaikuRAG that returns one expanded SearchResult with the given attachments."""
    from haiku.rag.config import AppConfig

    config = AppConfig()
    config.qa.model.vision = vision

    expanded = SearchResult(
        content="expanded text incl. picture descriptions if any",
        score=0.5,
        chunk_id="chunk-1",
        document_id="doc-1",
        doc_item_refs=["#/texts/0"] + (list(image_data) if image_data else []),
        page_numbers=[1],
        labels=["paragraph"] + (["picture"] if image_data else []),
        image_data=image_data,
    )
    client = MagicMock()
    client._config = config
    client.expand_context = AsyncMock(return_value=[expanded])
    return client


def test_inspect_command():
    """Test inspect command launches inspector TUI."""
    with patch("haiku.rag.inspector.run_inspector") as mock_inspector:
        mock_inspector.return_value = None

        result = runner.invoke(cli, ["inspect"])

        assert result.exit_code == 0
        mock_inspector.assert_called_once()


@pytest.mark.asyncio
async def test_document_list_loads_initial_batch():
    """Test that DocumentList loads only the initial batch on startup."""
    from textual.app import App

    from haiku.rag.inspector.widgets.document_list import DocumentList

    # Create mock documents
    mock_docs = [
        Document(id=f"doc-{i}", content=f"Content {i}", title=f"Doc {i}")
        for i in range(50)
    ]

    class TestApp(App):
        def compose(self):
            yield DocumentList(id="doc-list")

    app = TestApp()
    async with app.run_test():
        doc_list = app.query_one(DocumentList)

        # Create mock client
        mock_client = AsyncMock()
        mock_client.list_documents = AsyncMock(return_value=mock_docs)

        await doc_list.load_documents(mock_client)

        # Should have called list_documents with a limit (not None)
        mock_client.list_documents.assert_called_once()
        call_kwargs = mock_client.list_documents.call_args
        # The limit should be set (not None) for initial load
        assert call_kwargs.kwargs.get("limit") is not None


@pytest.mark.asyncio
async def test_document_list_load_more():
    """Test that DocumentList can load more documents."""
    from textual.app import App

    from haiku.rag.inspector.widgets.document_list import DocumentList

    # Create mock documents - two batches
    batch1 = [
        Document(id=f"doc-{i}", content=f"Content {i}", title=f"Doc {i}")
        for i in range(50)
    ]
    batch2 = [
        Document(id=f"doc-{i}", content=f"Content {i}", title=f"Doc {i}")
        for i in range(50, 100)
    ]

    class TestApp(App):
        def compose(self):
            yield DocumentList(id="doc-list")

    app = TestApp()
    async with app.run_test():
        doc_list = app.query_one(DocumentList)

        mock_client = AsyncMock()
        mock_client.list_documents = AsyncMock(side_effect=[batch1, batch2])

        # Load initial batch
        await doc_list.load_documents(mock_client)
        assert len(doc_list.documents) == 50

        # Load more
        await doc_list.load_more(mock_client)
        assert len(doc_list.documents) == 100

        # Verify offset was used in second call
        second_call = mock_client.list_documents.call_args_list[1]
        assert second_call.kwargs.get("offset") == 50


@pytest.mark.asyncio
async def test_document_list_tracks_has_more():
    """Test that DocumentList tracks whether more documents are available."""
    from textual.app import App

    from haiku.rag.inspector.widgets.document_list import DocumentList

    # First batch returns full page, second returns partial
    batch1 = [
        Document(id=f"doc-{i}", content=f"Content {i}", title=f"Doc {i}")
        for i in range(50)
    ]
    batch2 = [
        Document(id=f"doc-{i}", content=f"Content {i}", title=f"Doc {i}")
        for i in range(50, 60)
    ]

    class TestApp(App):
        def compose(self):
            yield DocumentList(id="doc-list")

    app = TestApp()
    async with app.run_test():
        doc_list = app.query_one(DocumentList)

        mock_client = AsyncMock()
        mock_client.list_documents = AsyncMock(side_effect=[batch1, batch2])

        await doc_list.load_documents(mock_client)
        # After loading full batch, has_more should be True
        assert doc_list.has_more is True

        await doc_list.load_more(mock_client)
        # After loading partial batch (<50), has_more should be False
        assert doc_list.has_more is False


@pytest.mark.asyncio
async def test_detail_view_shows_chunk_metadata():
    """show_chunk renders metadata keys _format_provenance doesn't already
    cover, but not a duplicate of the standard fields it does (headings,
    page_numbers, labels, doc_item_refs)."""
    from textual.app import App

    from haiku.rag.inspector.widgets.detail_view import DetailView

    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        content="raw chunk text",
        metadata={"headings": ["Chapter 1"], "para_no": "12"},
    )

    class TestApp(App):
        def compose(self):
            yield DetailView(id="detail")

    app = TestApp()
    async with app.run_test():
        detail_view = app.query_one(DetailView)
        await detail_view.show_chunk(chunk)
        source = detail_view.content_widget.source
        assert "**Metadata:**" in source
        assert "para_no: 12" in source
        assert "headings:" not in source  # already shown as **Section:**


@pytest.mark.asyncio
async def test_detail_view_omits_metadata_block_when_only_standard_fields():
    """No **Metadata:** block at all when chunk.metadata holds nothing
    beyond what _format_provenance already renders."""
    from textual.app import App

    from haiku.rag.inspector.widgets.detail_view import DetailView

    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        content="raw chunk text",
        metadata={"headings": ["Chapter 1"], "page_numbers": [1]},
    )

    class TestApp(App):
        def compose(self):
            yield DetailView(id="detail")

    app = TestApp()
    async with app.run_test():
        detail_view = app.query_one(DetailView)
        await detail_view.show_chunk(chunk)
        source = detail_view.content_widget.source
        assert "**Metadata:**" not in source


@pytest.mark.asyncio
async def test_detail_view_shows_search_result_chunk_meta():
    """show_search_result renders SearchResult.chunk_meta's non-standard
    keys, the anchor chunk's own metadata carried through search/expansion."""
    from textual.app import App

    from haiku.rag.inspector.widgets.detail_view import DetailView

    chunk = Chunk(id="chunk-1", document_id="doc-1", content="raw chunk text")
    search_result = SearchResult(
        content="raw chunk text",
        score=0.5,
        chunk_id="chunk-1",
        doc_item_refs=[f"#/texts/{i}" for i in range(7)],
        chunk_meta={
            "para_no": "12",
            "doc_item_refs": [f"#/texts/{i}" for i in range(7)],
        },
    )

    class TestApp(App):
        def compose(self):
            yield DetailView(id="detail")

    app = TestApp()
    async with app.run_test():
        detail_view = app.query_one(DetailView)
        await detail_view.show_search_result(chunk, search_result)
        source = detail_view.content_widget.source
        assert "**Metadata:**" in source
        assert "para_no: 12" in source
        # _format_provenance's own 5-item truncation for doc_item_refs is
        # untouched by the filtered-out duplicate in chunk_meta.
        assert "+2 more" in source
        assert "doc_item_refs:" not in source


@pytest.mark.asyncio
async def test_context_modal_renders_pictures_when_vision_enabled():
    """ContextModal must mount one TextualImage per attached picture
    when qa.model.vision is True — that's what the LLM actually sees."""
    from textual.app import App
    from textual_image.widget import Image as TextualImage

    from haiku.rag.inspector.widgets.context_modal import ContextModal

    chunk = Chunk(
        id="chunk-1", document_id="doc-1", content="raw chunk text", metadata={}
    )
    client = _make_client(
        vision=True,
        image_data={"#/pictures/0": _png_b64("red"), "#/pictures/1": _png_b64("blue")},
    )

    class TestApp(App):
        async def on_mount(self) -> None:
            await self.push_screen(ContextModal(chunk=chunk, client=client))

    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        modal = app.screen
        images = list(modal.query(TextualImage))
        assert len(images) == 2


@pytest.mark.asyncio
async def test_context_modal_suppresses_pictures_when_vision_disabled():
    """ContextModal must NOT mount picture widgets when vision is off,
    even if expansion attached image_data — text-only models would never
    see those bytes, so the inspector shouldn't show them either."""
    from textual.app import App
    from textual_image.widget import Image as TextualImage

    from haiku.rag.inspector.widgets.context_modal import ContextModal

    chunk = Chunk(
        id="chunk-1", document_id="doc-1", content="raw chunk text", metadata={}
    )
    client = _make_client(
        vision=False,
        image_data={"#/pictures/0": _png_b64("red")},
    )

    class TestApp(App):
        async def on_mount(self) -> None:
            await self.push_screen(ContextModal(chunk=chunk, client=client))

    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        modal = app.screen
        assert list(modal.query(TextualImage)) == []


@pytest.mark.asyncio
async def test_inspector_open_failure_surfaces_real_error(tmp_path):
    """A failed database open must surface its own error, not an
    AttributeError from tearing down a client that never opened."""
    from haiku.rag.inspector.app import InspectorApp

    app = InspectorApp(scope=for_path(tmp_path / "missing.lancedb"), read_only=True)
    with pytest.raises(FileNotFoundError):
        async with app.run_test():
            pass


class TestReportedLocation:
    """A URI-backed database is constructed with a placeholder local path, so
    what the modal prints has to come from the configuration."""

    @staticmethod
    def _session(location: str):
        from haiku.rag.client.scope import DatabaseScope
        from haiku.rag.client.session import SingleDatabaseSession, default_db_path
        from haiku.rag.config.models import AppConfig, LanceDBConfig

        config = AppConfig(lancedb=LanceDBConfig(databases={"alpha": location}))
        [ref] = DatabaseScope.resolve(config, database_name="alpha").databases
        one, db_path = ref.connection(config)
        return SingleDatabaseSession(
            db_path if db_path is not None else default_db_path(one),
            one,
            source="alpha",
        )

    def test_a_named_remote_database_reports_its_uri(self):
        session = self._session("s3://bucket/alpha.lancedb")

        assert isinstance(session.db_path, Path)
        assert session.location == "s3://bucket/alpha.lancedb"

    def test_a_named_local_database_reports_its_path(self):
        session = self._session("/data/alpha.lancedb")

        assert session.location == Path("/data/alpha.lancedb")


class TestReportingReusesTheConnection:
    @pytest.mark.asyncio
    async def test_statistics_come_from_the_open_connection(
        self, tmp_path, monkeypatch
    ):
        """The client already holds a connection to the database being reported,
        so opening a second one would be an open for the same statistics."""
        from haiku.rag.inspector.widgets.info_modal import database_lines
        from haiku.rag.store.engine import ConnectionMode

        asked: list[object] = []

        async def fake_stats(db):
            asked.append(db)
            return {
                "settings": {"exists": False},
                "documents": {"num_rows": 1},
                "document_meta": {"num_rows": 1},
                "chunks": {"num_rows": 1},
            }

        monkeypatch.setattr("haiku.rag.store.info.get_database_stats", fake_stats)

        connection = object()
        client = MagicMock()
        client.store.db = connection
        client.store.db_path = tmp_path
        client.store.stored_settings = {}
        client.store._connection_mode = ConnectionMode.LOCAL

        lines = await database_lines(client)

        assert asked == [connection]
        assert any("documents" in line for line in lines)

    @pytest.mark.asyncio
    async def test_settings_come_from_the_store_that_parsed_them(
        self, tmp_path, monkeypatch
    ):
        """The store read and parsed the settings blob on open, so reporting
        reads it from there instead of querying the settings table again."""
        from haiku.rag.inspector.widgets.info_modal import database_lines
        from haiku.rag.store.engine import ConnectionMode

        async def fake_stats(db):  # noqa: ARG001
            return {
                "documents": {"num_rows": 1},
                "document_meta": {"num_rows": 1},
                "chunks": {"num_rows": 1},
            }

        monkeypatch.setattr("haiku.rag.store.info.get_database_stats", fake_stats)

        client = MagicMock()
        client.store.db_path = tmp_path
        client.store._connection_mode = ConnectionMode.LOCAL
        client.store.stored_settings = {
            "version": "1.2.3",
            "embeddings": {
                "model": {"provider": "ollama", "name": "embed", "vector_dim": 7}
            },
        }

        lines = await database_lines(client)

        assert any("1.2.3" in line for line in lines)
        assert any("ollama/embed (dim: 7)" in line for line in lines)


class TestReportingEachDatabase:
    @pytest.mark.asyncio
    async def test_a_database_that_cannot_be_opened_reports_itself(self):
        """One unreachable database must not cost the report on the others."""
        from haiku.rag.inspector.widgets.info_modal import InfoModal
        from haiku.rag.store.exceptions import SourceUnavailableError

        modal = InfoModal.__new__(InfoModal)
        client = AsyncMock()
        client.clients_for.side_effect = SourceUnavailableError(
            "database 'beta' could not be opened: OSError"
        )
        modal.client = client

        lines = await modal._report("beta")

        assert lines[0] == "[bold]beta[/bold]"
        assert "could not be opened" in lines[1]

    @pytest.mark.asyncio
    async def test_a_failure_with_bracketed_text_renders_literally(self):
        """Error text and database names render verbatim, not as Rich markup."""
        from rich.text import Text

        from haiku.rag.inspector.widgets.info_modal import InfoModal
        from haiku.rag.store.exceptions import SourceUnavailableError

        modal = InfoModal.__new__(InfoModal)
        client = AsyncMock()
        message = "database 'beta [prod]' could not be opened: [/red] [Errno 2]"
        client.clients_for.side_effect = SourceUnavailableError(message)
        modal.client = client

        lines = await modal._report("beta [prod]")

        assert Text.from_markup(lines[0]).plain == "beta [prod]"
        assert message in Text.from_markup(lines[1]).plain

    @pytest.mark.asyncio
    async def test_an_open_failure_with_bracketed_text_renders_literally(
        self, tmp_path
    ):
        """A stats-read failure renders its message verbatim, not as markup."""
        from rich.text import Text

        from haiku.rag.inspector.widgets.info_modal import database_lines
        from haiku.rag.store.engine import ConnectionMode

        client = MagicMock()
        client.store.db_path = tmp_path
        client.store._connection_mode = ConnectionMode.LOCAL
        message = "Schema error: no field named [vector, text] [/red]"
        type(client.store).db = property(MagicMock(side_effect=RuntimeError(message)))

        lines = await database_lines(client)

        assert message in Text.from_markup(lines[0]).plain
