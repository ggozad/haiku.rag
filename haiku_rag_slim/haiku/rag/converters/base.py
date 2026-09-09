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

    docling's markdown and HTML backends model a paragraph carrying inline
    markup as an ``InlineGroup`` of runs, and leave the heading or list item
    that owns one with empty text. ``iterate_items()`` skips groups, so
    consumers see the runs as unrelated items and the owner as blank.

    A group whose runs are not all text (an inline picture) is left alone.

    Every group is serialized before any is replaced, and all are deleted in
    one call: a serializer caches the refs it excludes, and a deletion
    renumbers every ref in the document.

    Returns whether the document was changed.
    """
    from docling_core.transforms.serializer.markdown import (
        MarkdownDocSerializer,
        MarkdownParams,
    )
    from docling_core.types.doc.document import (
        InlineGroup,
        SectionHeaderItem,
        TextItem,
        TitleItem,
    )
    from docling_core.types.doc.labels import DocItemLabel

    linked = MarkdownDocSerializer(doc=doc)
    plain = MarkdownDocSerializer(
        doc=doc, params=MarkdownParams(include_hyperlinks=False)
    )

    flattened: list[tuple[InlineGroup, TextItem | None, TextItem, str]] = []
    for item, _ in doc.iterate_items(with_groups=True, traverse_pictures=True):
        if not isinstance(item, InlineGroup):
            continue
        runs = [child.resolve(doc) for child in item.children]
        if not runs or not all(isinstance(run, TextItem) for run in runs):
            continue
        # The backends parent a paragraph to the heading above it, so only an
        # empty owner is one the group carries the text of.
        parent = item.parent.resolve(doc) if item.parent else None
        owner = parent if isinstance(parent, TextItem) and not parent.text else None
        # A URL in a heading travels into breadcrumbs and chunk contextualization.
        heading = isinstance(owner, TitleItem | SectionHeaderItem)
        serializer = plain if heading else linked
        flattened.append((item, owner, runs[0], serializer.serialize(item=item).text))

    for group, owner, first_run, text in flattened:
        if owner is not None:
            owner.text = owner.orig = text
        else:
            doc.insert_text(
                sibling=group,
                label=DocItemLabel.TEXT,
                text=text,
                prov=first_run.prov[0] if first_run.prov else None,
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
