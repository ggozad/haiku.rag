import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image as PILImage

from haiku.rag.store.models import Chunk, Document, SearchResult
from tests.conftest import for_path


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


async def test_document_list_loads_in_batches():
    """DocumentList loads one batch, loads the next at its offset, and reports
    no more documents after a partial batch."""
    from textual.app import App

    from haiku.rag.inspector.widgets.document_list import DocumentList

    full = [
        Document(id=f"doc-{i}", content=f"Content {i}", title=f"Doc {i}")
        for i in range(50)
    ]
    short = [
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
        mock_client.list_documents = AsyncMock(side_effect=[full, short])

        await doc_list.load_documents(mock_client)
        mock_client.list_documents.assert_called_once()
        assert mock_client.list_documents.call_args.kwargs.get("limit") is not None
        assert len(doc_list.documents) == 50
        assert doc_list.has_more is True

        await doc_list.load_more(mock_client)
        second_call = mock_client.list_documents.call_args_list[1]
        assert second_call.kwargs.get("offset") == 50
        assert len(doc_list.documents) == 60
        assert doc_list.has_more is False


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


@pytest.mark.parametrize(
    "vision, image_data, mounted",
    [
        (True, {"#/pictures/0": _png_b64("red"), "#/pictures/1": _png_b64("blue")}, 2),
        (False, {"#/pictures/0": _png_b64("red")}, 0),
    ],
)
async def test_context_modal_renders_pictures_only_with_vision(
    vision, image_data, mounted
):
    """ContextModal mounts one TextualImage per attached picture when
    qa.model.vision is set, and none when it is not."""
    from textual.app import App
    from textual_image.widget import Image as TextualImage

    from haiku.rag.inspector.widgets.context_modal import ContextModal

    chunk = Chunk(
        id="chunk-1", document_id="doc-1", content="raw chunk text", metadata={}
    )
    client = _make_client(vision=vision, image_data=image_data)

    class TestApp(App):
        async def on_mount(self) -> None:
            await self.push_screen(ContextModal(chunk=chunk, client=client))

    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        modal = app.screen
        assert len(list(modal.query(TextualImage))) == mounted


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
        from haiku.rag.client.session import SingleDatabaseSession
        from haiku.rag.config.models import AppConfig, LanceDBConfig

        config = AppConfig(lancedb=LanceDBConfig(databases={"alpha": location}))
        [ref] = DatabaseScope.resolve(config, database_name="alpha").databases
        return SingleDatabaseSession(ref, config)

    def test_a_named_remote_database_reports_its_uri(self):
        session = self._session("s3://bucket/alpha.lancedb")

        assert session.db_path is None
        assert session.location == "s3://bucket/alpha.lancedb"
        assert session.source == "alpha"

    def test_a_named_local_database_reports_its_path(self):
        session = self._session("/data/alpha.lancedb")

        assert session.location == Path("/data/alpha.lancedb")


class TestReportingReusesTheConnection:
    async def test_statistics_come_from_the_open_connection(
        self, tmp_path, monkeypatch
    ):
        """Statistics are read through the connection the client already
        holds."""
        from haiku.rag.inspector.widgets.info_modal import database_lines
        from haiku.rag.store.engine import ConnectionMode

        asked: list[object] = []

        async def fake_stats(db, tag=None):
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

    async def test_settings_come_from_the_store_that_parsed_them(
        self, tmp_path, monkeypatch
    ):
        """The store read and parsed the settings blob on open, and reporting
        reads it from there."""
        from haiku.rag.inspector.widgets.info_modal import database_lines
        from haiku.rag.store.engine import ConnectionMode

        async def fake_stats(db, tag=None):  # noqa: ARG001
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


class TestReportingAtATag:
    async def test_statistics_are_read_at_the_clients_tag(self, temp_db_path):
        from haiku.rag.client import HaikuRAG
        from haiku.rag.inspector.widgets.info_modal import database_lines
        from haiku.rag.store.engine import Store
        from haiku.rag.store.repositories.document import DocumentRepository

        async with Store(temp_db_path, create=True) as store:
            repo = DocumentRepository(store)
            await repo.create(Document(content="before the tag"))
            await store.create_tag("before")
            await repo.create(Document(content="after the tag"))

        async with HaikuRAG(temp_db_path, tag="before") as client:
            lines = await database_lines(client)

        assert any(
            line.startswith("[bold $accent]documents[/bold $accent]: 1 ")
            for line in lines
        )


class TestReportingEachDatabase:
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
