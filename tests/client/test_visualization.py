from unittest.mock import patch

import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.models.document import Document
from tests.platform import converts_pdf_locally


@converts_pdf_locally
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
        assert await client.document_repository.get_original(doc.id) == (
            pdf_path.read_bytes()
        )

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
    from PIL import Image as PilImageModule
    from PIL.Image import Image as PILImage

    docling_doc = _bbox_doc(pages=(1, 2))

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
            docling_doc, chunks, uri="test://multi-page", original=_blank_pdf(2)
        )

        stored_chunks = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(stored_chunks) == 1

        chunk = stored_chunks[0]
        images = await client.visualize_chunk(chunk)
        assert len(images) == 2

        for img in images:
            assert isinstance(img, PILImage)
            assert img.size == (1224, 1584)

        blank = PilImageModule.new("RGB", (1224, 1584), color="white")
        for img in images:
            assert img.tobytes() != blank.tobytes()


async def test_client_visualize_chunk_merged_chunks_union_pages(temp_db_path):
    """Visualizing all chunks of a merged result covers the union of their
    expansions, which a single constituent chunk alone does not reach."""
    from docling_core.types.doc.base import BoundingBox, Size
    from docling_core.types.doc.document import DoclingDocument, ProvenanceItem
    from docling_core.types.doc.labels import DocItemLabel

    docling_doc = DoclingDocument(name="merged-viz-test")
    page_size = Size(width=612.0, height=792.0)
    for page_no in (1, 2):
        docling_doc.add_page(page_no=page_no, size=page_size)

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
        doc = await client.import_document(
            docling_doc, chunks, uri="test://merged", original=_blank_pdf(2)
        )

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
    from docling_core.types.doc.document import DoclingDocument, ProvenanceItem
    from docling_core.types.doc.labels import DocItemLabel

    docling_doc = DoclingDocument(name="two-tone-test")
    docling_doc.add_page(page_no=1, size=Size(width=612.0, height=792.0))

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
        doc = await client.import_document(
            docling_doc, chunks, uri="test://two-tone", original=_blank_pdf(1)
        )
        stored_chunks = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(stored_chunks) == 1

        images = await client.visualize_chunk(stored_chunks[0])
        assert len(images) == 1
        image = images[0]

        # Rendered at 2 px per point, bottom-left origin flipped to top-left:
        # item i's box spans y = 184 + i * 200 .. 284 + i * 200.
        matched = image.getpixel((600, 434))  # inside #/texts/1
        swept = image.getpixel((600, 234))  # inside #/texts/0
        background = image.getpixel((600, 60))  # outside all boxes

        assert matched != background
        assert swept != background
        assert matched != swept


async def test_client_visualize_chunk_uses_given_refs(temp_db_path):
    """Explicit refs restrict visualization without expanding chunk context."""
    docling_doc = _bbox_doc(pages=(1, 2))

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
            docling_doc, chunks, uri="test://refs", original=_blank_pdf(2)
        )
        chunk = (await client.chunk_repository.get_by_document_id(doc.id))[0]

        assert len(await client.visualize_chunk(chunk)) == 2
        assert len(await client.visualize_chunk(chunk, refs=["#/texts/0"])) == 1


async def test_client_visualize_chunk_no_expand_shows_only_chunk(temp_db_path):
    """expand=False draws only the chunk's own items, not the expanded section."""
    docling_doc = _bbox_doc(pages=(1, 2))

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
        doc = await client.import_document(
            docling_doc, chunks, uri="test://no-expand", original=_blank_pdf(2)
        )
        chunk = (await client.chunk_repository.get_by_document_id(doc.id))[0]

        assert len(await client.visualize_chunk(chunk)) == 2
        assert len(await client.visualize_chunk(chunk, expand=False)) == 1


def _blank_pdf(pages: int) -> bytes:
    """A PDF of `pages` blank letter-size pages, the original of `_bbox_doc`."""
    from io import BytesIO

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    for _ in range(pages):
        pdf.new_page(612, 792)
    buffer = BytesIO()
    pdf.save(buffer)
    return buffer.getvalue()


def _bbox_doc(*, pages: tuple[int, ...] = (1,)):
    """DoclingDocument with one paragraph per page, each carrying a bbox."""
    from docling_core.types.doc.base import BoundingBox, Size
    from docling_core.types.doc.document import ProvenanceItem

    doc = DoclingDocument(name="bbox-doc")
    size = Size(width=612.0, height=792.0)
    for page_no in pages:
        doc.add_page(page_no=page_no, size=size)
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


async def test_visualize_chunk_returns_empty_without_an_original(temp_db_path):
    """Boxes resolve, but a document without an original has nothing to render
    them onto."""
    docling_doc = _bbox_doc()
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


async def test_visualize_chunk_skips_pages_the_original_lacks(temp_db_path):
    """A structure page past the original's last page renders nothing."""
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
            _bbox_doc(pages=(1, 2)), chunks, uri="test://short", original=_blank_pdf(1)
        )
        stored = await client.chunk_repository.get_by_document_id(doc.id)

        images = await client.visualize_chunk(stored[0])

    assert len(images) == 1


async def test_visualize_chunk_draws_on_a_page_the_structure_does_not_list(
    temp_db_path,
):
    """The original's pages, not the structure's, decide what renders."""
    from docling_core.types.doc.base import BoundingBox, Size
    from docling_core.types.doc.document import ProvenanceItem

    docling_doc = DoclingDocument(name="orphan-page-box")
    docling_doc.add_page(page_no=1, size=Size(width=612.0, height=792.0))
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
            docling_doc, chunks, uri="test://orphan-page", original=_blank_pdf(3)
        )
        stored = await client.chunk_repository.get_by_document_id(doc.id)

        assert len(await client.visualize_chunk(stored[0])) == 1


async def test_visualize_chunk_sizes_pages_from_the_original(temp_db_path):
    """A structure without pages, as migrated, grounds on its original."""
    docling_doc = _bbox_doc()
    docling_doc.pages = {}
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
        doc = await client.import_document(
            docling_doc, chunks, uri="test://no-pages", original=_blank_pdf(1)
        )
        stored = await client.chunk_repository.get_by_document_id(doc.id)

        [image] = await client.visualize_chunk(stored[0])

    # 2 px per point, top-left origin: the box spans y = 184 .. 284.
    assert image.size == (1224, 1584)
    assert image.getpixel((600, 234)) != image.getpixel((600, 60))


def _doc_without_provenance():
    doc = DoclingDocument(name="no-prov")
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Content on page 1.")
    return doc


@pytest.mark.parametrize(
    "docling_doc, metadata",
    [
        (
            _bbox_doc(),
            {"page_numbers": [1], "labels": ["paragraph"]},
        ),
        (
            _doc_without_provenance(),
            {"doc_item_refs": ["#/texts/0"], "labels": ["paragraph"]},
        ),
    ],
    ids=["no-refs", "refs-without-provenance"],
)
async def test_visualize_chunk_returns_empty_without_boxes(
    temp_db_path, docling_doc, metadata
):
    """A chunk whose items locate nothing on a page renders nothing."""
    chunks = [
        Chunk(
            content="Content on page 1.",
            metadata=metadata,
            order=0,
            embedding=[0.1] * 2560,
        )
    ]

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.import_document(docling_doc, chunks, uri="test://no-boxes")
        stored = await client.chunk_repository.get_by_document_id(doc.id)

        assert await client.visualize_chunk(stored[0]) == []


async def test_visualize_chunk_falls_back_when_expansion_drops_refs(temp_db_path):
    """If expansion returns results carrying no refs, the original search
    results' refs are used instead."""
    from haiku.rag.client import search as search_module

    docling_doc = _bbox_doc()
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
        doc = await client.import_document(
            docling_doc, chunks, uri="test://drops-refs", original=_blank_pdf(1)
        )
        stored = await client.chunk_repository.get_by_document_id(doc.id)

        async def expansion_without_refs(_client, results):
            return [r.model_copy(update={"doc_item_refs": []}) for r in results]

        with patch.object(search_module, "expand_context", expansion_without_refs):
            images = await client.visualize_chunk(stored[0])

    assert len(images) == 1


async def test_visualize_command_omits_the_document_line_without_a_uri(
    temp_db_path, monkeypatch
):
    from rich.console import Console

    from haiku.rag.app import HaikuRAGApp
    from haiku.rag.config import AppConfig
    from tests.conftest import for_path

    chunks = [
        Chunk(
            content="Content on page 1.",
            metadata={"doc_item_refs": ["#/texts/0"], "page_numbers": [1]},
            order=0,
            embedding=[0.1] * 2560,
        )
    ]
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.import_document(_bbox_doc(), chunks, original=_blank_pdf(1))
        stored = await client.chunk_repository.get_by_document_id(doc.id)
    assert stored[0].id is not None
    monkeypatch.setattr("textual_image.renderable.Image", lambda img: "<page image>")
    app = HaikuRAGApp(scope=for_path(temp_db_path), config=AppConfig())
    app.console = Console(record=True, width=200)

    await app.visualize_chunk(stored[0].id)

    printed = app.console.export_text()
    assert "Page 1/1" in printed
    assert "document:" not in printed


async def test_chunk_grounding_gives_the_boxes_by_page_with_the_original(
    temp_db_path,
):
    import hashlib

    from haiku.rag.client.search import GroundingBox
    from haiku.rag.store.models.chunk import BoundingBox

    chunks = [
        Chunk(
            content="Content on page 2.",
            metadata={"doc_item_refs": ["#/texts/1"], "page_numbers": [2]},
            order=0,
            embedding=[0.1] * 2560,
        )
    ]
    original = _blank_pdf(2)

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.import_document(
            _bbox_doc(pages=(1, 2)), chunks, uri="test://grounding", original=original
        )
        [chunk] = await client.chunk_repository.get_by_document_id(doc.id)

        grounding = await client.chunk_grounding(chunk)

    assert grounding is not None
    assert grounding.document_id == doc.id
    assert grounding.original == original
    assert grounding.original_sha256 == hashlib.sha256(original).hexdigest()

    def box(page_no: int) -> BoundingBox:
        return BoundingBox(page_no=page_no, left=50, top=700, right=550, bottom=650)

    assert grounding.pages == {
        1: [GroundingBox(box(1), matched=False)],
        2: [GroundingBox(box(2), matched=True)],
    }


async def test_chunk_grounding_is_none_without_an_original(temp_db_path):
    chunks = [
        Chunk(
            content="Content on page 1.",
            metadata={"doc_item_refs": ["#/texts/0"], "page_numbers": [1]},
            order=0,
            embedding=[0.1] * 2560,
        )
    ]
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.import_document(_bbox_doc(), chunks, uri="test://bare")
        [chunk] = await client.chunk_repository.get_by_document_id(doc.id)

        assert await client.chunk_grounding(chunk) is None
