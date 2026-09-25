from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import RenderableType

    from haiku.rag.client import HaikuRAG
    from haiku.rag.store.models.citation import Citation


CITATION_PREVIEW_CHARS = 300


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def _citation_pages(c: "Citation") -> str | None:
    if not c.page_numbers:
        return None
    if len(c.page_numbers) == 1:
        return f"p. {c.page_numbers[0]}"
    return f"pp. {c.page_numbers[0]}-{c.page_numbers[-1]}"


def _citation_section(c: "Citation") -> str | None:
    if c.headings:
        return c.headings[-1]
    return None


def _citation_label(c: "Citation") -> str:
    if c.document_title and c.document_uri:
        return f"{c.document_title} ({c.document_uri})"
    return c.document_title or c.document_uri


def truncated(text: str, limit: int) -> str:
    """The first `limit` characters of `text`, with `…` appended when anything"""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


async def format_citations_rich(
    citations: "list[Citation]",
    client: "HaikuRAG | None" = None,
    full: bool = False,
) -> "list[RenderableType]":
    """Format citations as Rich renderables for terminal display."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    if not citations:
        return []

    renderables: list[RenderableType] = []
    renderables.append(Text(""))
    renderables.append(Text("Citations", style="bold green"))
    renderables.append(Text(""))

    for i, c in enumerate(citations):
        if i > 0:
            renderables.append(Text(""))
        idx = c.index if c.index is not None else (i + 1)

        header_parts: list[str] = [f"[{idx}] {_citation_label(c)}"]
        if c.source and client is not None and client.covers_multiple:
            header_parts.append(c.source)
        pages = _citation_pages(c)
        if pages:
            header_parts.append(pages)
        section = _citation_section(c)
        if section:
            header_parts.append(f"§{section}")
        header = Text(" — ".join(header_parts), style="bold")

        body: list[RenderableType] = []
        for ref in c.picture_refs:
            image_renderable = await _render_picture(
                client, c.document_id, ref, c.source
            )
            body.append(
                image_renderable
                if image_renderable
                else Text(f"[Figure: {ref}]", style="italic dim")
            )

        body.append(
            Text(c.content if full else truncated(c.content, CITATION_PREVIEW_CHARS))
        )

        footer = Text()
        footer.append("doc: ", style="dim")
        footer.append(c.document_id, style="dim cyan")
        footer.append("  chunk: ", style="dim")
        footer.append(c.chunk_id, style="dim cyan")

        panel = Panel(
            Group(*body),
            title=header,
            title_align="left",
            subtitle=footer,
            subtitle_align="left",
            border_style="dim",
        )
        renderables.append(panel)

    return renderables


async def _render_picture(
    client: "HaikuRAG | None", document_id: str, ref: str, source: str | None = None
) -> "RenderableType | None":
    """A picture as a Rich renderable, or None where it cannot be rendered."""
    if client is None:
        return None
    if source is None and client.covers_multiple:
        return None
    from io import BytesIO

    from PIL import Image as PILImage
    from textual_image.renderable import Image as RichImage

    data = await client.get_picture_bytes(document_id, ref, source)
    if not data:
        return None
    try:
        pil = PILImage.open(BytesIO(data))
        pil.load()
    except Exception:
        return None
    return RichImage(pil)
