import base64
from io import BytesIO
from typing import TYPE_CHECKING

from PIL import Image as PILImage
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Markdown, Static
from textual_image.widget import Image as TextualImage

from haiku.rag.store.models import SearchResult

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG
    from haiku.rag.store.models import Chunk


class ContextModal(Screen):
    """Modal screen for displaying how a chunk appears to agents."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("c", "dismiss", "Close", show=True),
    ]

    CSS = """
    ContextModal {
        background: $surface;
        layout: vertical;
    }

    #context-header {
        dock: top;
        height: auto;
        padding: 1;
    }

    #context-content {
        height: 1fr;
        width: 100%;
        padding: 1;
    }

    #context-content Markdown {
        width: 100%;
    }

    #context-content Image {
        margin: 1 0;
        height: auto;
        max-height: 30;
    }

    .picture-caption {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(self, chunk: "Chunk", client: "HaikuRAG"):
        super().__init__()
        self.chunk = chunk
        self.client = client
        self._content_widget = Markdown("Loading...")

    def compose(self) -> ComposeResult:
        yield Static("[bold]Agent Context Format[/bold]", id="context-header")
        with VerticalScroll(id="context-content"):
            yield self._content_widget

    async def on_mount(self) -> None:
        chunk_meta = self.chunk.get_chunk_metadata()
        search_result = SearchResult(
            content=self.chunk.content,
            score=0.0,
            chunk_id=self.chunk.id,
            document_id=self.chunk.document_id,
            document_uri=self.chunk.document_uri,
            document_title=self.chunk.document_title,
            doc_item_refs=chunk_meta.doc_item_refs,
            page_numbers=chunk_meta.page_numbers,
            headings=chunk_meta.headings,
            labels=chunk_meta.labels,
        )

        expanded_results = await self.client.expand_context([search_result])
        expanded = expanded_results[0] if expanded_results else search_result

        vision = self.client._config.qa.model.vision
        attached = expanded.image_data or {}

        if vision and attached:
            preface = (
                f"*This is what the LLM receives. Vision is enabled — {len(attached)} "
                "picture(s) are attached as image content (rendered below).*"
            )
        elif vision:
            preface = (
                "*This is what the LLM receives. Vision is enabled, but no pictures "
                "are in this expanded context.*"
            )
        elif attached:
            preface = (
                f"*This is what the LLM receives. Vision is disabled — {len(attached)} "
                "picture(s) are suppressed; any VLM descriptions remain inline below.*"
            )
        else:
            preface = "*This is what the LLM receives.*"

        await self._content_widget.update(
            f"{preface}\n\n---\n\n{expanded.format_for_agent()}"
        )

        if vision and attached:
            scroll = self.query_one("#context-content", VerticalScroll)
            for self_ref, b64 in attached.items():
                try:
                    pil = PILImage.open(BytesIO(base64.b64decode(b64)))
                except Exception:
                    continue
                await scroll.mount(
                    Static(f"[b]{self_ref}[/b]", classes="picture-caption")
                )
                await scroll.mount(TextualImage(pil))

    async def action_dismiss(self, result=None) -> None:
        self.app.pop_screen()
