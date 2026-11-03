from unittest.mock import MagicMock

from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import DoclingDocument, ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.config import AppConfig
from haiku.rag.locations import LOCATIONS_KEY, attach_locations, item_locations
from haiku.rag.store.models.chunk import Chunk, SearchResult
from haiku.rag.uri import uri_to_path


def _document() -> DoclingDocument:
    doc = DoclingDocument(name="book")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    doc.add_page(page_no=2, size=Size(width=600, height=800))
    for page, top in ((1, 700.0), (1, 500.0), (2, 300.0)):
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=f"text on {page}",
            prov=ProvenanceItem(
                page_no=page,
                bbox=BoundingBox(
                    l=50, t=top, r=550, b=top - 40, coord_origin=CoordOrigin.BOTTOMLEFT
                ),
                charspan=(0, 9),
            ),
        )
    doc.add_text(label=DocItemLabel.TEXT, text="no provenance")
    # Provenance on a page the document does not list.
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="stray",
        prov=ProvenanceItem(
            page_no=9, bbox=BoundingBox(l=0, t=1, r=1, b=0), charspan=(0, 5)
        ),
    )
    return doc


def test_locations_are_grouped_by_page_with_a_top_left_origin():
    refs = ["#/texts/2", "#/texts/0", "#/texts/1", "#/texts/3", "#/texts/4"]
    assert item_locations(_document(), refs) == [
        {
            "page": 1,
            "width": 600.0,
            "height": 800.0,
            "boxes": [[50.0, 100.0, 550.0, 140.0], [50.0, 300.0, 550.0, 340.0]],
        },
        {
            "page": 2,
            "width": 600.0,
            "height": 800.0,
            "boxes": [[50.0, 500.0, 550.0, 540.0]],
        },
    ]


def test_refs_that_do_not_resolve_are_skipped():
    refs = ["#/texts/99", "#/bogus/0", "not a ref", "#/texts/3"]
    assert item_locations(_document(), refs) == []


def test_chunks_get_locations_only_where_they_have_some():
    doc = _document()
    located = Chunk(content="a", metadata={"doc_item_refs": ["#/texts/2"]})
    unlocated = Chunk(content="b", metadata={"doc_item_refs": ["#/texts/3"]})
    bare = Chunk(content="c")
    attach_locations(doc, [located, unlocated, bare])
    assert located.metadata[LOCATIONS_KEY][0]["page"] == 2
    assert LOCATIONS_KEY not in unlocated.metadata
    assert bare.metadata == {}


async def test_chunking_records_locations(monkeypatch):
    from haiku.rag.client import processing

    doc = _document()

    class Chunker:
        async def chunk(self, document):  # noqa: ARG002
            return [Chunk(content="x", metadata={"doc_item_refs": ["#/texts/0"]})]

    monkeypatch.setattr("haiku.rag.chunkers.get_chunker", lambda config: Chunker())
    [chunk] = await processing.chunk(
        AppConfig(), doc, embedder=MagicMock(supports_images=False)
    )
    assert chunk.metadata[LOCATIONS_KEY][0]["boxes"] == [[50.0, 100.0, 550.0, 140.0]]


def test_the_agent_rendering_names_the_file_and_pages_on_request():
    result = SearchResult(
        content="passage",
        score=1.0,
        chunk_id="c1",
        document_uri="file:///Users/me/Guitar/book.pdf",
        page_numbers=[3, 4],
        chunk_meta={LOCATIONS_KEY: [{"page": 3}], "custom": 1},
    )
    plain = result.format_for_agent(include_chunk_meta=True)
    assert "Path:" not in plain
    assert '{"custom": 1}' in plain
    assert "locations" not in plain
    located = result.format_for_agent(include_location=True)
    assert f"Path: {uri_to_path('file:///Users/me/Guitar/book.pdf')}" in located
    assert "Pages: 3, 4" in located
    remote = SearchResult(content="p", score=1.0, document_uri="https://x/y")
    assert "Path:" not in remote.format_for_agent(include_location=True)
    assert "Pages:" not in remote.format_for_agent(include_location=True)
