import logging
from typing import TYPE_CHECKING

from haiku.rag.config import AppConfig
from haiku.rag.store.models.document import Document

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

logger = logging.getLogger(__name__)


def extract_structural_title(docling_document: "DoclingDocument") -> str | None:
    """Extract a title from DoclingDocument structural metadata.

    Priority: FURNITURE TITLE > BODY TITLE > first SECTION_HEADER.
    """
    from docling_core.types.doc.document import ContentLayer
    from docling_core.types.doc.labels import DocItemLabel

    furniture_title = None
    body_title = None
    first_section_header = None

    for item in docling_document.texts:
        if item.label == DocItemLabel.TITLE:
            text = item.text.strip()
            if not text:
                continue
            if item.content_layer == ContentLayer.FURNITURE:
                furniture_title = text
            elif body_title is None:
                body_title = text
        elif item.label == DocItemLabel.SECTION_HEADER and first_section_header is None:
            text = item.text.strip()
            if text:
                first_section_header = text

    return furniture_title or body_title or first_section_header


async def generate_title_with_llm(config: AppConfig, content: str) -> str | None:
    """Generate a title using LLM from document content."""
    from pydantic_ai import Agent

    from haiku.rag.utils import get_model

    truncated = content[:2000]

    model = get_model(config.processing.title_model, config)
    agent: Agent[None, str] = Agent(
        model=model,
        output_type=str,
        instructions=(
            "Generate a concise, descriptive title for the following document. "
            "The title should be at most 10 words. "
            "Return ONLY the title text, nothing else."
        ),
    )
    result = await agent.run(truncated)
    title = result.output.strip()
    return title if title else None


async def resolve_title(
    config: AppConfig,
    docling_document: "DoclingDocument",
    content: str,
) -> str | None:
    """Auto-generate a title from document structure or LLM.

    Returns None if auto_title is disabled or generation fails.
    """
    if not config.processing.auto_title:
        return None

    structural = extract_structural_title(docling_document)
    if structural:
        return structural

    try:
        return await generate_title_with_llm(config, content)
    except Exception:
        logger.warning("LLM title generation failed during ingestion", exc_info=True)
        return None


async def generate_title(config: AppConfig, document: Document) -> str | None:
    """Generate a title for a document.

    Attempts structural extraction from the stored DoclingDocument, then falls
    back to LLM generation. Bypasses the auto_title config since this is an
    explicit call.

    Does NOT update the document — caller decides.
    """
    docling_doc = document.get_docling_document()
    content = document.content or ""

    if docling_doc is not None:
        structural = extract_structural_title(docling_doc)
        if structural:
            return structural

    return await generate_title_with_llm(config, content)
