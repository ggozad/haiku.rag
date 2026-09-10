"""Base class for document converters."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.config import AppConfig
    from haiku.rag.config.models import ModelConfig


def vlm_api_url(config: "AppConfig", model: "ModelConfig") -> str:
    """Construct the VLM chat-completions URL for a picture-description model."""
    if model.base_url:
        return f"{model.base_url.rstrip('/')}/v1/chat/completions"

    if model.provider == "ollama":
        return f"{config.providers.ollama.base_url.rstrip('/')}/v1/chat/completions"

    if model.provider == "openai":
        return "https://api.openai.com/v1/chat/completions"

    raise ValueError(f"Unsupported VLM provider: {model.provider}")


def vlm_api_headers(model: "ModelConfig") -> dict[str, str]:
    """Auth headers for the picture-description VLM endpoint. Docling posts to
    it directly, so the key travels as a header rather than through an SDK.

    The public OpenAI endpoint falls back to ``OPENAI_API_KEY``. A custom
    ``base_url`` never does: that key belongs to api.openai.com, not to
    whatever self-hosted server the model points at.
    """
    key = model.api_key
    if not key and model.provider == "openai" and not model.base_url:
        key = os.environ.get("OPENAI_API_KEY")
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


def vlm_api_params(model: "ModelConfig", max_tokens: int) -> dict[str, object]:
    """Request body fields docling posts alongside the picture."""
    from haiku.rag.utils import reasoning_effort

    params: dict[str, object] = {
        "model": model.name,
        "max_completion_tokens": max_tokens,
    }
    effort = reasoning_effort(model)
    if effort is not None:
        params["reasoning_effort"] = effort
    return params


def flatten_inline_groups(doc: "DoclingDocument") -> bool:
    """Replace every inline group of text runs with the one text item it renders as.

    A group whose runs are not all text, that carries more than one provenance
    record, or that another item refers into, is left as it is. Returns
    whether the document changed.
    """
    from docling_core.transforms.serializer.markdown import (
        MarkdownDocSerializer,
        MarkdownParams,
    )
    from docling_core.types.doc.document import (
        ContentLayer,
        DocItem,
        FloatingItem,
        InlineGroup,
        ProvenanceItem,
        RichTableCell,
        SectionHeaderItem,
        TableItem,
        TextItem,
        TitleItem,
    )
    from docling_core.types.doc.labels import DocItemLabel

    # The rendering is stored as an item's text, not re-parsed as markdown.
    linked = MarkdownDocSerializer(
        doc=doc, params=MarkdownParams(escape_underscores=False, escape_html=False)
    )
    plain = MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(
            escape_underscores=False, escape_html=False, include_hyperlinks=False
        ),
    )

    # A ref into a deleted run is renumbered onto another item.
    referenced: set[str] = set()
    for item, _ in doc.iterate_items(
        with_groups=True,
        traverse_pictures=True,
        included_content_layers=set(ContentLayer),
    ):
        if isinstance(item, DocItem):
            referenced.update(ref.cref for ref in item.comments)
        if isinstance(item, FloatingItem):
            referenced.update(
                ref.cref for ref in (*item.captions, *item.footnotes, *item.references)
            )
        if isinstance(item, TableItem):
            referenced.update(
                cell.ref.cref
                for cell in item.data.table_cells
                if isinstance(cell, RichTableCell)
            )

    flattened: list[
        tuple[InlineGroup, TextItem | None, ProvenanceItem | None, str]
    ] = []
    claimed: set[str] = set()
    for item, _ in doc.iterate_items(with_groups=True, traverse_pictures=True):
        if not isinstance(item, InlineGroup):
            continue
        children = [child.resolve(doc) for child in item.children]
        runs = [child for child in children if isinstance(child, TextItem)]
        if not runs or len(runs) != len(children):
            continue
        if item.self_ref in referenced or any(
            run.self_ref in referenced for run in runs
        ):
            continue
        # One provenance record survives the merge.
        provenance = [prov for run in runs for prov in run.prov]
        if len(provenance) > 1:
            continue
        # Paragraphs are parented to the item above them; an empty parent owns
        # its first group.
        parent = item.parent.resolve(doc) if item.parent else None
        owner = (
            parent
            if isinstance(parent, TextItem)
            and not parent.text
            and parent.self_ref not in claimed
            else None
        )
        if owner is not None:
            claimed.add(owner.self_ref)
        # A URL in a heading travels into breadcrumbs and chunk contextualization.
        heading = isinstance(owner, TitleItem | SectionHeaderItem)
        serializer = plain if heading else linked
        text = serializer.serialize(item=item).text
        flattened.append((item, owner, provenance[0] if provenance else None, text))

    for group, owner, prov, text in flattened:
        if owner is not None:
            owner.text = owner.orig = text
            if prov is not None and not owner.prov:
                owner.prov = [prov]
        else:
            doc.insert_text(
                sibling=group,
                label=DocItemLabel.TEXT,
                text=text,
                prov=prov,
                after=False,
            )

    if not flattened:
        return False

    doc.delete_items(node_items=[group for group, _, _, _ in flattened])
    return True


class DocumentConverter(ABC):
    """Abstract base class for document converters.

    Document converters are responsible for converting various document formats
    (PDF, DOCX, HTML, etc.) into DoclingDocument format for further processing.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of file extensions supported by this converter.

        Returns:
            List of file extensions (including the dot, e.g., [".pdf", ".docx"]).
        """
        pass

    @abstractmethod
    async def convert_file(
        self, path: Path, source_uri: str | None = None
    ) -> "DoclingDocument":
        """Convert a file to DoclingDocument format.

        Args:
            path: Path to the file to convert.
            source_uri: Optional origin URI (e.g. the URL the file was
                downloaded from) used by docling's HTML/Markdown backends to
                resolve relative `<img src="/path">` references. Ignored by
                converters that have no equivalent backend option (notably
                docling-serve).

        Returns:
            DoclingDocument representation of the file.

        Raises:
            ValueError: If the file cannot be converted.
        """
        pass

    SUPPORTED_FORMATS = ("md", "html", "plain")

    @abstractmethod
    async def convert_text(
        self,
        text: str,
        name: str = "content.md",
        format: str = "md",
        source_uri: str | None = None,
    ) -> "DoclingDocument":
        """Convert text content to DoclingDocument format.

        Args:
            text: The text content to convert.
            name: The name to use for the document (defaults to "content.md").
            format: The format of the text content ("md", "html", or "plain").
                Defaults to "md". Use "plain" for plain text without parsing.
            source_uri: Optional origin URI used by docling's HTML/Markdown
                backends to resolve relative image references.

        Returns:
            DoclingDocument representation of the text.

        Raises:
            ValueError: If the text cannot be converted or format is unsupported.
        """
        pass
