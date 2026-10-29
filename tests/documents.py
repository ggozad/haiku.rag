from docling_core.types.doc.document import DoclingDocument, ImageRef
from docling_core.types.doc.labels import DocItemLabel
from PIL import Image

from haiku.rag.client.documents import DocumentImport
from haiku.rag.config import get_config
from haiku.rag.context import expand_with_items, window_for
from haiku.rag.store.models.chunk import Chunk


def docling_doc(name: str, text: str) -> DoclingDocument:
    doc = DoclingDocument(name=name)
    doc.add_text(label=DocItemLabel.TEXT, text=text)
    return doc


def document_import(name: str, text: str, **overrides) -> DocumentImport:
    """Build a DocumentImport with one pre-embedded chunk (no embedder call)."""
    dim = get_config().embeddings.model.vector_dim
    chunk = Chunk(content=text, embedding=[0.1] * dim, order=0)
    return DocumentImport(
        docling_document=docling_doc(name, text), chunks=[chunk], **overrides
    )


def docling_doc_with_picture() -> DoclingDocument:
    """A paragraph and one picture carrying real PNG bytes."""
    img = Image.new("RGB", (8, 8), "red")
    doc = DoclingDocument(name="test-with-picture")
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Hello world")
    doc.add_picture(image=ImageRef.from_pil(img, dpi=72))
    return doc


def add_picture_text(doc, parent, *texts: str) -> None:
    """Add text items under ``parent``, as docling does for text it OCRs inside a picture."""
    for text in texts:
        doc.add_text(label=DocItemLabel.TEXT, text=text, parent=parent)


async def fetch_and_expand(repo, document_id, results, max_chars):
    """Do the fetching `expand_context` does, for tests exercising the
    expansion logic rather than the batched fetch."""
    positions = (
        await repo.resolve_refs_grouped(
            {document_id: [ref for r in results for ref in r.doc_item_refs]}
        )
    ).get(document_id, {})
    items: list = []
    if positions:
        items = (
            await repo.get_items_in_ranges({document_id: window_for(positions)})
        ).get(document_id, [])
    return expand_with_items(results, max_chars, positions, items)
