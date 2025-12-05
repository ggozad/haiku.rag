from typing import Protocol

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static

from haiku.rag.store.models import Chunk, Document, SearchResult


class ProvenanceData(Protocol):
    """Protocol for objects that have provenance metadata."""

    page_numbers: list[int]
    headings: list[str] | None
    labels: list[str]
    doc_item_refs: list[str]


class DetailView(VerticalScroll):  # pragma: no cover
    """Widget for displaying detailed content of documents or chunks."""

    can_focus = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.title_widget = Static("[bold]Detail View[/bold]", classes="title")
        self.content_widget = Markdown("")
        self.content_widget.can_focus = True

    def compose(self) -> ComposeResult:
        yield self.title_widget
        yield self.content_widget

    def _format_provenance(self, prov: ProvenanceData) -> list[str]:
        """Format provenance metadata as markdown lines."""
        parts: list[str] = []
        if prov.page_numbers:
            pages_str = ", ".join(str(p) for p in prov.page_numbers)
            parts.append(f"**Page(s):** {pages_str}")
        if prov.headings:
            headings_str = " > ".join(prov.headings)
            parts.append(f"**Section:** {headings_str}")
        if prov.labels:
            labels_str = ", ".join(prov.labels)
            parts.append(f"**Labels:** {labels_str}")
        if prov.doc_item_refs:
            refs_str = ", ".join(prov.doc_item_refs[:5])
            if len(prov.doc_item_refs) > 5:
                refs_str += f" ... (+{len(prov.doc_item_refs) - 5} more)"
            parts.append(f"**DocItem Refs:** `{refs_str}`")
        return parts

    async def show_document(self, document: Document) -> None:
        """Display document details."""
        title = document.title or document.uri or "Untitled Document"
        self.title_widget.update(f"[bold]Document: {title}[/bold]")

        content_parts: list[str] = []
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
        content_parts.append(document.content)

        await self.content_widget.update("\n\n".join(content_parts))

    async def show_chunk(self, chunk: Chunk) -> None:
        """Display chunk details."""
        self.title_widget.update(f"[bold]Chunk {chunk.order}[/bold]")

        content_parts: list[str] = []
        if chunk.id:
            content_parts.append(f"**ID:** `{chunk.id}`")
        if chunk.document_id:
            content_parts.append(f"**Document ID:** `{chunk.document_id}`")
        if chunk.document_title:
            content_parts.append(f"**Document Title:** {chunk.document_title}")
        if chunk.document_uri:
            content_parts.append(f"**Document URI:** `{chunk.document_uri}`")
        content_parts.append(f"**Order:** {chunk.order}")

        chunk_meta = chunk.get_chunk_metadata()
        content_parts.extend(self._format_provenance(chunk_meta))

        if chunk.embedding:
            content_parts.append(f"**Embedding:** {len(chunk.embedding)} dimensions")

        content_parts.append("\n---\n")
        content_parts.append(chunk.content)

        await self.content_widget.update("\n\n".join(content_parts))

    async def show_search_result(
        self, chunk: Chunk, search_result: SearchResult
    ) -> None:
        """Display chunk details with search result metadata."""
        self.title_widget.update(f"[bold]Chunk {chunk.order}[/bold]")

        content_parts: list[str] = []
        if chunk.id:
            content_parts.append(f"**ID:** `{chunk.id}`")
        if chunk.document_id:
            content_parts.append(f"**Document ID:** `{chunk.document_id}`")
        if search_result.document_title:
            content_parts.append(f"**Document Title:** {search_result.document_title}")
        if search_result.document_uri:
            content_parts.append(f"**Document URI:** `{search_result.document_uri}`")
        content_parts.append(f"**Order:** {chunk.order}")
        content_parts.append(f"**Score:** {search_result.score:.4f}")

        content_parts.extend(self._format_provenance(search_result))

        if chunk.embedding:
            content_parts.append(f"**Embedding:** {len(chunk.embedding)} dimensions")

        content_parts.append("\n---\n")
        content_parts.append(chunk.content)

        await self.content_widget.update("\n\n".join(content_parts))
