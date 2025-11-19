from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static

from haiku.rag.store.models import Chunk, Document


class DetailView(VerticalScroll):
    """Widget for displaying detailed content of documents or chunks."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.title_widget: Static | None = None
        self.content_widget: Markdown | None = None

    def compose(self) -> ComposeResult:
        """Compose the detail view."""
        self.title_widget = Static("[bold]Detail View[/bold]", classes="title")
        yield self.title_widget
        self.content_widget = Markdown("")
        yield self.content_widget

    async def show_document(self, document: Document) -> None:
        """Display document details.

        Args:
            document: Document to display
        """
        if self.title_widget and self.content_widget:
            title = document.title or document.uri or "Untitled Document"
            self.title_widget.update(f"[bold]Document: {title}[/bold]")

            # Build markdown content
            content_parts = []

            if document.id:
                content_parts.append(f"**ID:** `{document.id}`")
            if document.uri:
                content_parts.append(f"**URI:** `{document.uri}`")
            if document.metadata:
                metadata_str = "\n".join(
                    f"  - {k}: {v}" for k, v in document.metadata.items()
                )
                content_parts.append(f"**Metadata:**\n{metadata_str}")
            if document.created_at:
                content_parts.append(f"**Created:** {document.created_at}")
            if document.updated_at:
                content_parts.append(f"**Updated:** {document.updated_at}")

            content_parts.append("\n---\n")
            content_parts.append("**Content:**\n")
            content_parts.append(f"```\n{document.content}\n```")

            await self.content_widget.update("\n\n".join(content_parts))

    async def show_chunk(self, chunk: Chunk) -> None:
        """Display chunk details.

        Args:
            chunk: Chunk to display
        """
        if self.title_widget and self.content_widget:
            self.title_widget.update(f"[bold]Chunk {chunk.order}[/bold]")

            # Build markdown content
            content_parts = []

            if chunk.id:
                content_parts.append(f"**ID:** `{chunk.id}`")
            if chunk.document_id:
                content_parts.append(f"**Document ID:** `{chunk.document_id}`")
            if chunk.document_title:
                content_parts.append(f"**Document Title:** {chunk.document_title}")
            if chunk.document_uri:
                content_parts.append(f"**Document URI:** `{chunk.document_uri}`")
            content_parts.append(f"**Order:** {chunk.order}")
            if chunk.metadata:
                metadata_str = "\n".join(
                    f"  - {k}: {v}" for k, v in chunk.metadata.items()
                )
                content_parts.append(f"**Metadata:**\n{metadata_str}")
            if chunk.embedding:
                content_parts.append(
                    f"**Embedding:** {len(chunk.embedding)} dimensions"
                )

            content_parts.append("\n---\n")
            content_parts.append("**Content:**\n")
            content_parts.append(f"```\n{chunk.content}\n```")

            await self.content_widget.update("\n\n".join(content_parts))
