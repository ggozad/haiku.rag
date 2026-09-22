from unittest.mock import patch

import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.models.document import Document


@pytest.mark.vcr()
async def test_client_visualize_chunk_no_document(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        chunk = Chunk(content="Orphan chunk", order=0)
        images = await client.visualize_chunk(chunk)
        assert images == []


@pytest.mark.vcr()
async def test_client_visualize_chunk_no_bounding_boxes(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(
            content="Simple text content without structure",
            uri="test://simple",
        )

        assert doc.id is not None
        assert doc.docling_document is not None

        chunks = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(chunks) >= 1

        images = await client.visualize_chunk(chunks[0])
        assert images == []


@pytest.mark.vcr()
async def test_client_visualize_chunk_with_pdf(temp_db_path, doclaynet_first_page_pdf):
    from PIL.Image import Image as PILImage

    from haiku.rag.config import AppConfig

    pdf_path = doclaynet_first_page_pdf
    config = AppConfig()
    config.processing.conversion_options.do_ocr = False

    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        doc = await client.create_document_from_source(pdf_path)
        assert isinstance(doc, Document)
        assert doc.id is not None
        assert doc.docling_document is not None

        chunks = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(chunks) > 0

        chunks_with_refs = [c for c in chunks if c.get_chunk_metadata().doc_item_refs]
        assert len(chunks_with_refs) > 0, "PDF should have chunks with doc_item_refs"

        images = await client.visualize_chunk(chunks_with_refs[0])

        assert isinstance(images, list)
        assert len(images) > 0, "PDF with page images should return visualizations"

        for img in images:
            assert isinstance(img, PILImage)


async def test_client_visualize_chunk_multi_page(temp_db_path):
    from docling_core.types.doc.base import BoundingBox, Size
    from docling_core.types.doc.document import (
        DoclingDocument,
        ImageRef,
        ProvenanceItem,
    )
    from docling_core.types.doc.labels import DocItemLabel
    from PIL import Image as PilImageModule
    from PIL.Image import Image as PILImage

    docling_doc = DoclingDocument(name="multi-page-test")
    page_size = Size(width=612.0, height=792.0)
    img1 = PilImageModule.new("RGB", (612, 792), color="white")
    img2 = PilImageModule.new("RGB", (612, 792), color="white")
    docling_doc.add_page(
        page_no=1, size=page_size, image=ImageRef.from_pil(img1, dpi=72)
    )
    docling_doc.add_page(
        page_no=2, size=page_size, image=ImageRef.from_pil(img2, dpi=72)
    )

    docling_doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="Content on page one.",
        prov=ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=50, t=700, r=550, b=650),
            charspan=(0, 20),
        ),
    )
    docling_doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="Content on page two.",
        prov=ProvenanceItem(
            page_no=2,
            bbox=BoundingBox(l=50, t=700, r=550, b=650),
            charspan=(0, 20),
        ),
    )

    chunks = [
        Chunk(
            content="Content on page one.\nContent on page two.",
            metadata={
                "doc_item_refs": ["#/texts/0", "#/texts/1"],
                "page_numbers": [1, 2],
                "labels": ["paragraph", "paragraph"],
            },
            order=0,
            embedding=[0.1] * 2560,
        )
    ]

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.import_document(docling_doc, chunks, uri="test://multi-page")

        stored_chunks = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(stored_chunks) == 1

        chunk = stored_chunks[0]
        images = await client.visualize_chunk(chunk)
        assert len(images) == 2

        for img in images:
            assert isinstance(img, PILImage)
            assert img.size == (612, 792)

        blank = PilImageModule.new("RGB", (612, 792), color="white")
        for img in images:
            assert img.tobytes() != blank.tobytes()


async def test_client_visualize_chunk_merged_chunks_union_pages(temp_db_path):
    """Visualizing all chunks of a merged result covers the union of their
    expansions, which a single constituent chunk alone does not reach."""
    from docling_core.types.doc.base import BoundingBox, Size
    from docling_core.types.doc.document import (
        DoclingDocument,
        ImageRef,
        ProvenanceItem,
    )
    from docling_core.types.doc.labels import DocItemLabel
    from PIL import Image as PilImageModule

    docling_doc = DoclingDocument(name="merged-viz-test")
    page_size = Size(width=612.0, height=792.0)
    for page_no in (1, 2):
        docling_doc.add_page(
            page_no=page_no,
            size=page_size,
            image=ImageRef.from_pil(
                PilImageModule.new("RGB", (612, 792), color="white"), dpi=72
            ),
        )

    # Each section stays within the expansion budget and its own boundary.
    layout = [
        (DocItemLabel.SECTION_HEADER, "Section One", 1),
        (DocItemLabel.PARAGRAPH, "Page one body. " + "x" * 3000, 1),
        (DocItemLabel.SECTION_HEADER, "Section Two", 2),
        (DocItemLabel.PARAGRAPH, "Page two body. " + "y" * 3000, 2),
    ]
    for i, (label, text, page_no) in enumerate(layout):
        docling_doc.add_text(
            label=label,
            text=text,
            prov=ProvenanceItem(
                page_no=page_no,
                bbox=BoundingBox(l=50, t=700 - (i % 2) * 100, r=550, b=650),
                charspan=(0, 20),
            ),
        )

    chunks = [
        Chunk(
            content="Page one body. " + "x" * 3000,
            metadata={
                "doc_item_refs": ["#/texts/1"],
                "page_numbers": [1],
                "labels": ["paragraph"],
            },
            order=0,
            embedding=[0.1] * 2560,
        ),
        Chunk(
            content="Page two body. " + "y" * 3000,
            metadata={
                "doc_item_refs": ["#/texts/3"],
                "page_numbers": [2],
                "labels": ["paragraph"],
            },
            order=1,
            embedding=[0.1] * 2560,
        ),
    ]

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.import_document(docling_doc, chunks, uri="test://merged")

        stored_chunks = await client.chunk_repository.get_by_document_id(doc.id)
        stored_chunks.sort(key=lambda c: c.order)
        assert len(stored_chunks) == 2
        c1, c2 = stored_chunks

        solo_images = await client.visualize_chunk(c1)
        assert len(solo_images) == 1

        merged_images = await client.visualize_chunk([c1, c2])
        assert len(merged_images) == 2


async def test_client_visualize_chunk_two_tone_highlights(temp_db_path):
    """Matched content draws stronger than context swept in by expansion."""
    from docling_core.types.doc.base import BoundingBox, Size
    from docling_core.types.doc.document import (
        DoclingDocument,
        ImageRef,
        ProvenanceItem,
    )
    from docling_core.types.doc.labels import DocItemLabel
    from PIL import Image as PilImageModule

    docling_doc = DoclingDocument(name="two-tone-test")
    page_size = Size(width=612.0, height=792.0)
    docling_doc.add_page(
        page_no=1,
        size=page_size,
        image=ImageRef.from_pil(
            PilImageModule.new("RGB", (612, 792), color="white"), dpi=72
        ),
    )

    # Three small paragraphs; the chunk matches only the middle one, so
    # expansion sweeps in its neighbors.
    for i in range(3):
        docling_doc.add_text(
            label=DocItemLabel.PARAGRAPH,
            text=f"Paragraph {i}.",
            prov=ProvenanceItem(
                page_no=1,
                bbox=BoundingBox(l=50, t=700 - i * 100, r=550, b=650 - i * 100),
                charspan=(0, 12),
            ),
        )

    chunks = [
        Chunk(
            content="Paragraph 1.",
            metadata={
                "doc_item_refs": ["#/texts/1"],
                "page_numbers": [1],
                "labels": ["paragraph"],
            },
            order=0,
            embedding=[0.1] * 2560,
        )
    ]

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.import_document(docling_doc, chunks, uri="test://two-tone")
        stored_chunks = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(stored_chunks) == 1

        images = await client.visualize_chunk(stored_chunks[0])
        assert len(images) == 1
        image = images[0]

        # Page dpi 72 == document coords, bottom-left origin flipped to
        # top-left: item i's box spans y = 92 + i * 100 .. 142 + i * 100.
        matched = image.getpixel((300, 217))  # inside #/texts/1
        swept = image.getpixel((300, 117))  # inside #/texts/0
        background = image.getpixel((300, 30))  # outside all boxes

        assert matched != background
        assert swept != background
        assert matched != swept


async def test_client_visualize_chunk_uses_given_refs(temp_db_path):
    """Explicit refs restrict visualization without expanding chunk context."""
    from docling_core.types.doc.base import BoundingBox, Size
    from docling_core.types.doc.document import (
        DoclingDocument,
        ImageRef,
        ProvenanceItem,
    )
    from docling_core.types.doc.labels import DocItemLabel
    from PIL import Image as PilImageModule

    docling_doc = DoclingDocument(name="refs-test")
    page_size = Size(width=612.0, height=792.0)
    for page_no in (1, 2):
        docling_doc.add_page(
            page_no=page_no,
            size=page_size,
            image=ImageRef.from_pil(
                PilImageModule.new("RGB", (612, 792), color="white"), dpi=72
            ),
        )
    docling_doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="Content on page one.",
        prov=ProvenanceItem(
            page_no=1, bbox=BoundingBox(l=50, t=700, r=550, b=650), charspan=(0, 20)
        ),
    )
    docling_doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="Content on page two.",
        prov=ProvenanceItem(
            page_no=2, bbox=BoundingBox(l=50, t=700, r=550, b=650), charspan=(0, 20)
        ),
    )

    chunks = [
        Chunk(
            content="Content on page one.\nContent on page two.",
            metadata={
                "doc_item_refs": ["#/texts/0", "#/texts/1"],
                "page_numbers": [1, 2],
                "labels": ["paragraph", "paragraph"],
            },
            order=0,
            embedding=[0.1] * 2560,
        )
    ]

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.import_document(docling_doc, chunks, uri="test://refs")
        chunk = (await client.chunk_repository.get_by_document_id(doc.id))[0]

        assert len(await client.visualize_chunk(chunk)) == 2
        assert len(await client.visualize_chunk(chunk, refs=["#/texts/0"])) == 1


async def test_client_visualize_chunk_no_expand_shows_only_chunk(temp_db_path):
    """expand=False draws only the chunk's own items, not the expanded section."""
    from docling_core.types.doc.base import BoundingBox, Size
    from docling_core.types.doc.document import (
        DoclingDocument,
        ImageRef,
        ProvenanceItem,
    )
    from docling_core.types.doc.labels import DocItemLabel
    from PIL import Image as PilImageModule

    docling_doc = DoclingDocument(name="no-expand-test")
    page_size = Size(width=612.0, height=792.0)
    for page_no in (1, 2):
        docling_doc.add_page(
            page_no=page_no,
            size=page_size,
            image=ImageRef.from_pil(
                PilImageModule.new("RGB", (612, 792), color="white"), dpi=72
            ),
        )
    for page_no in (1, 2):
        docling_doc.add_text(
            label=DocItemLabel.PARAGRAPH,
            text=f"Short paragraph on page {page_no}.",
            prov=ProvenanceItem(
                page_no=page_no,
                bbox=BoundingBox(l=50, t=700, r=550, b=650),
                charspan=(0, 20),
            ),
        )

    chunks = [
        Chunk(
            content="Short paragraph on page 1.",
            metadata={
                "doc_item_refs": ["#/texts/0"],
                "page_numbers": [1],
                "labels": ["paragraph"],
            },
            order=0,
            embedding=[0.1] * 2560,
        )
    ]

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.import_document(docling_doc, chunks, uri="test://no-expand")
        chunk = (await client.chunk_repository.get_by_document_id(doc.id))[0]

        assert len(await client.visualize_chunk(chunk)) == 2
        assert len(await client.visualize_chunk(chunk, expand=False)) == 1


def _bbox_doc(*, with_page_image: bool, pages: tuple[int, ...] = (1,)):
    """DoclingDocument with one paragraph per page, each carrying a bbox.

    ``with_page_image=False`` produces pages with no raster, so bounding boxes
    resolve but there is nothing to draw them on.
    """
    from docling_core.types.doc.base import BoundingBox, Size
    from docling_core.types.doc.document import ImageRef, ProvenanceItem
    from PIL import Image as PilImageModule

    doc = DoclingDocument(name="bbox-doc")
    size = Size(width=612.0, height=792.0)
    for page_no in pages:
        image = (
            ImageRef.from_pil(
                PilImageModule.new("RGB", (612, 792), color="white"), dpi=72
            )
            if with_page_image
            else None
        )
        doc.add_page(page_no=page_no, size=size, image=image)
        doc.add_text(
            label=DocItemLabel.PARAGRAPH,
            text=f"Content on page {page_no}.",
            prov=ProvenanceItem(
                page_no=page_no,
                bbox=BoundingBox(l=50, t=700, r=550, b=650),
                charspan=(0, 20),
            ),
        )
    return doc


async def test_visualize_chunk_returns_empty_without_page_rasters(temp_db_path):
    """Boxes resolve, but a document ingested without page images has nothing
    to render them onto."""
    docling_doc = _bbox_doc(with_page_image=False)
    chunks = [
        Chunk(
            content="Content on page 1.",
            metadata={
                "doc_item_refs": ["#/texts/0"],
                "page_numbers": [1],
                "labels": ["paragraph"],
            },
            order=0,
            embedding=[0.1] * 2560,
        )
    ]

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.import_document(docling_doc, chunks, uri="test://no-raster")
        stored = await client.chunk_repository.get_by_document_id(doc.id)

        assert await client.visualize_chunk(stored[0]) == []


async def test_visualize_chunk_skips_pages_without_a_raster(temp_db_path):
    """A document where only some pages carry a raster renders just those."""
    from docling_core.types.doc.base import BoundingBox, Size
    from docling_core.types.doc.document import ImageRef, ProvenanceItem
    from PIL import Image as PilImageModule

    docling_doc = DoclingDocument(name="mixed-rasters")
    size = Size(width=612.0, height=792.0)
    docling_doc.add_page(
        page_no=1,
        size=size,
        image=ImageRef.from_pil(
            PilImageModule.new("RGB", (612, 792), color="white"), dpi=72
        ),
    )
    docling_doc.add_page(page_no=2, size=size, image=None)
    for page_no in (1, 2):
        docling_doc.add_text(
            label=DocItemLabel.PARAGRAPH,
            text=f"Content on page {page_no}.",
            prov=ProvenanceItem(
                page_no=page_no,
                bbox=BoundingBox(l=50, t=700, r=550, b=650),
                charspan=(0, 20),
            ),
        )

    chunks = [
        Chunk(
            content="Content on page 1.\nContent on page 2.",
            metadata={
                "doc_item_refs": ["#/texts/0", "#/texts/1"],
                "page_numbers": [1, 2],
                "labels": ["paragraph", "paragraph"],
            },
            order=0,
            embedding=[0.1] * 2560,
        )
    ]

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.import_document(
            docling_doc, chunks, uri="test://mixed-rasters"
        )
        stored = await client.chunk_repository.get_by_document_id(doc.id)

        images = await client.visualize_chunk(stored[0])

    assert len(images) == 1


async def test_visualize_chunk_returns_empty_when_pages_row_missing(temp_db_path):
    docling_doc = _bbox_doc(with_page_image=True)
    chunks = [
        Chunk(
            content="Content on page 1.",
            metadata={
                "doc_item_refs": ["#/texts/0"],
                "page_numbers": [1],
                "labels": ["paragraph"],
            },
            order=0,
            embedding=[0.1] * 2560,
        )
    ]

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.import_document(docling_doc, chunks, uri="test://no-row")
        stored = await client.chunk_repository.get_by_document_id(doc.id)

        async def no_pages_row(document_id):
            return None

        client.document_repository.get_pages_data = no_pages_row  # type: ignore[method-assign]

        assert await client.visualize_chunk(stored[0]) == []


async def test_visualize_chunk_skips_box_on_unstored_page(temp_db_path):
    """Bounding boxes on unregistered pages are ignored."""
    from docling_core.types.doc.base import BoundingBox, Size
    from docling_core.types.doc.document import ImageRef, ProvenanceItem
    from PIL import Image as PilImageModule

    docling_doc = DoclingDocument(name="orphan-page-box")
    docling_doc.add_page(
        page_no=1,
        size=Size(width=612.0, height=792.0),
        image=ImageRef.from_pil(
            PilImageModule.new("RGB", (612, 792), color="white"), dpi=72
        ),
    )
    docling_doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="Content attributed to a page with no raster.",
        prov=ProvenanceItem(
            page_no=3,
            bbox=BoundingBox(l=50, t=700, r=550, b=650),
            charspan=(0, 20),
        ),
    )

    chunks = [
        Chunk(
            content="Content attributed to a page with no raster.",
            metadata={
                "doc_item_refs": ["#/texts/0"],
                "page_numbers": [3],
                "labels": ["paragraph"],
            },
            order=0,
            embedding=[0.1] * 2560,
        )
    ]

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.import_document(
            docling_doc, chunks, uri="test://orphan-page"
        )
        stored = await client.chunk_repository.get_by_document_id(doc.id)

        assert await client.visualize_chunk(stored[0]) == []


async def test_visualize_chunk_without_refs_falls_back_to_chunk_metadata(temp_db_path):
    """A chunk carrying no doc_item_refs has nothing to expand from."""
    docling_doc = _bbox_doc(with_page_image=True)
    chunks = [
        Chunk(
            content="Content on page 1.",
            metadata={"page_numbers": [1], "labels": ["paragraph"]},
            order=0,
            embedding=[0.1] * 2560,
        )
    ]

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.import_document(docling_doc, chunks, uri="test://no-refs")
        stored = await client.chunk_repository.get_by_document_id(doc.id)

        assert await client.visualize_chunk(stored[0]) == []


async def test_visualize_chunk_falls_back_when_expansion_drops_refs(temp_db_path):
    """If expansion returns results carrying no refs, the original search
    results' refs are used instead."""
    from haiku.rag.client import search as search_module

    docling_doc = _bbox_doc(with_page_image=True)
    chunks = [
        Chunk(
            content="Content on page 1.",
            metadata={
                "doc_item_refs": ["#/texts/0"],
                "page_numbers": [1],
                "labels": ["paragraph"],
            },
            order=0,
            embedding=[0.1] * 2560,
        )
    ]

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.import_document(docling_doc, chunks, uri="test://drops-refs")
        stored = await client.chunk_repository.get_by_document_id(doc.id)

        async def expansion_without_refs(_client, results):
            return [r.model_copy(update={"doc_item_refs": []}) for r in results]

        with patch.object(search_module, "expand_context", expansion_without_refs):
            images = await client.visualize_chunk(stored[0])

    assert len(images) == 1
