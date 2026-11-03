import asyncio
from pathlib import Path

import pytest
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import DoclingDocument, ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig, LanceDBConfig
from haiku.rag.embeddings import EmbedderWrapper
from haiku.rag.locations import attach_locations
from haiku.rag.service.search import (
    SearchService,
    document_type,
    find_line,
    local_path,
    metadata_filter,
    query_terms,
    snippet,
)
from haiku.rag.store.exceptions import UnknownDatabaseError
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.uri import uri_to_path
from tests.conftest import writing

NOTES = """# Practice notes

Warm up slowly.
Travis picking keeps the thumb on alternating bass strings.
End with scales.
"""


@pytest.fixture(autouse=True)
def constant_embedder(monkeypatch):
    async def embed_documents(self, texts):
        return [[0.1] * self.vector_dim for _ in texts]

    async def embed_query(self, text):  # noqa: ARG001
        return [0.1] * self.vector_dim

    monkeypatch.setattr(EmbedderWrapper, "_embed_documents", embed_documents)
    monkeypatch.setattr(EmbedderWrapper, "_embed_query", embed_query)


def _book(texts: list[tuple[int, str]]) -> DoclingDocument:
    doc = DoclingDocument(name="book")
    for page in sorted({page for page, _ in texts}):
        doc.add_page(page_no=page, size=Size(width=600, height=800))
    for page, text in texts:
        doc.add_text(
            label=DocItemLabel.TEXT,
            text=text,
            prov=ProvenanceItem(
                page_no=page,
                bbox=BoundingBox(
                    l=50, t=700, r=550, b=600, coord_origin=CoordOrigin.BOTTOMLEFT
                ),
                charspan=(0, len(text)),
            ),
        )
    return doc


def _chunk(text: str, ref: str, page: int) -> Chunk:
    return Chunk(
        content=text,
        embedding=[0.1] * AppConfig().embeddings.model.vector_dim,
        metadata={
            "doc_item_refs": [ref],
            "labels": ["text"],
            "page_numbers": [page],
            "headings": ["Chapter 5", "Travis Picking"],
        },
    )


async def _seed(tmp_path: Path, config: AppConfig) -> dict[str, str]:
    book_text = "Travis picking is a fingerstyle technique named after Merle Travis."
    old_text = "Older chunk about travis picking, stored before locations existed."
    book = _book([(1, "Contents"), (40, book_text), (41, old_text)])
    book_chunks = [
        _chunk(book_text, "#/texts/1", 40),
        _chunk(old_text, "#/texts/2", 41),
    ]
    attach_locations(book, book_chunks[:1])
    notes_path = tmp_path / "notes.md"
    notes_path.write_text(NOTES, encoding="utf-8")
    notes = DoclingDocument(name="notes")
    notes.add_text(label=DocItemLabel.TEXT, text=NOTES)
    notes_chunk = Chunk(
        content="Travis picking keeps the thumb on alternating bass strings.",
        embedding=[0.1] * config.embeddings.model.vector_dim,
        metadata={"doc_item_refs": ["#/texts/0"], "labels": ["text"]},
    )
    ids = {}
    async with HaikuRAG(config=config, create=True, sources=["files"]) as rag:
        ids["book"] = (
            await rag.import_document(
                book,
                book_chunks,
                uri=(tmp_path / "Country Guitar.pdf").as_uri(),
                title="Country Guitar",
            )
        ).id
        ids["notes"] = (
            await rag.import_document(
                notes,
                [notes_chunk],
                uri=notes_path.as_uri(),
                title="Practice notes",
            )
        ).id
        for name, owner, content_type in (
            ("book", "guitar", "application/pdf"),
            ("notes", "notes", "text/markdown"),
        ):
            await _set_metadata(
                rag, ids[name], {"source_id": owner, "content_type": content_type}
            )
    return ids


async def _set_metadata(rag: HaikuRAG, document_id: str, metadata: dict) -> None:
    """Ingestion-owned keys, which an import does not take from its caller."""
    document = await rag.get_document_by_id(document_id)
    assert document is not None
    document.metadata = metadata
    await writing(rag).document_repository.update_meta(document)


@pytest.fixture
async def seeded(tmp_path):
    config = AppConfig(
        lancedb=LanceDBConfig(databases={"files": str(tmp_path / "files.lancedb")})
    )
    ids = await _seed(tmp_path, config)
    return config, ids


async def test_hits_locate_their_passage(tmp_path, seeded):
    config, ids = seeded
    service = SearchService(config)
    try:
        response = await service.search(
            "travis picking", limit=10, search_type="fts", folders=["guitar"]
        )
    finally:
        await service.aclose()

    assert response.query == "travis picking"
    assert response.took_ms >= 0
    by_page = {hit.page_numbers[0]: hit for hit in response.hits}
    assert sorted(by_page) == [40, 41]
    hit = by_page[40]
    assert hit.rank in (1, 2)
    assert hit.document_id == ids["book"]
    assert hit.database == "files"
    assert hit.folder_id == "guitar"
    assert hit.title == "Country Guitar"
    assert hit.path == str(tmp_path / "Country Guitar.pdf")
    assert hit.document_type == "pdf"
    assert hit.content_type == "application/pdf"
    assert hit.line is None
    assert hit.headings == ["Chapter 5", "Travis Picking"]
    assert hit.labels == ["text"]
    assert hit.snippet.startswith("Travis picking")
    assert hit.snippet_highlights[:2] == [(0, 6), (7, 14)]
    assert [p.model_dump() for p in hit.pages] == [
        {
            "page": 40,
            "width": 600.0,
            "height": 800.0,
            "boxes": [{"left": 50.0, "top": 100.0, "right": 550.0, "bottom": 200.0}],
        }
    ]
    # Stored before chunks recorded locations: read from the stored document.
    assert [p.page for p in by_page[41].pages] == [41]
    assert by_page[41].pages[0].boxes[0].top == 100.0


async def test_a_stored_document_is_read_once_for_its_locations(seeded):
    config, _ = seeded
    service = SearchService(config)
    try:
        first = await service.search("older chunk", limit=1, search_type="fts")
        second = await service.search("older chunk", limit=1, search_type="fts")
    finally:
        await service.aclose()
    assert first.hits[0].pages == second.hits[0].pages != []


async def test_a_document_stored_without_pages_has_no_locations(seeded):
    config, _ = seeded
    doc = DoclingDocument(name="loose")
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="pageless passage",
        prov=ProvenanceItem(
            page_no=1, bbox=BoundingBox(l=0, t=1, r=1, b=0), charspan=(0, 5)
        ),
    )
    async with HaikuRAG(config=config, sources=["files"]) as rag:
        await rag.import_document(
            doc, [_chunk("pageless passage", "#/texts/0", 1)], uri="file:///p.pdf"
        )
    service = SearchService(config)
    try:
        response = await service.search("pageless", limit=1, search_type="fts")
    finally:
        await service.aclose()
    assert response.hits[0].pages == []


async def test_closing_an_unused_service():
    service = SearchService(AppConfig())
    await service.aclose()
    assert not service.loaded


async def test_a_text_hit_names_its_line(tmp_path, seeded):
    config, _ = seeded
    service = SearchService(config)
    try:
        response = await service.search(
            "alternating bass", limit=5, search_type="fts", folders=["notes"]
        )
    finally:
        await service.aclose()
    [hit] = response.hits
    assert hit.document_type == "markdown"
    assert hit.line == 4
    assert hit.pages == []


async def test_search_within_a_document_and_with_context(seeded):
    config, ids = seeded
    service = SearchService(config)
    try:
        response = await service.search(
            "travis",
            limit=5,
            search_type="fts",
            document_id=ids["notes"],
            context=True,
            databases=["files"],
        )
    finally:
        await service.aclose()
    [hit] = response.hits
    assert hit.document_id == ids["notes"]
    assert hit.context is not None
    assert "Warm up slowly." in hit.context


async def test_an_unknown_database_is_refused(seeded):
    config, _ = seeded
    service = SearchService(config)
    try:
        with pytest.raises(UnknownDatabaseError):
            await service.search("x", limit=5, databases=["nope"])
    finally:
        await service.aclose()


async def test_the_client_closes_when_idle_and_reopens(seeded):
    config, _ = seeded
    config.ingester.service.search_idle_s = 0.1
    service = SearchService(config)
    await service.search("travis", limit=1, search_type="fts")
    assert service.loaded
    for _ in range(100):
        if not service.loaded:
            break
        await asyncio.sleep(0.02)
    assert not service.loaded
    await service.search("travis", limit=1, search_type="fts")
    assert service.loaded
    await service.aclose()
    assert not service.loaded


async def test_a_running_search_keeps_the_client_open(seeded):
    config, _ = seeded
    service = SearchService(config)
    async with service.client() as rag:
        service._idle()
        await asyncio.sleep(0.05)
        assert service.loaded
        assert await rag.count_documents() == 2
    await service.aclose()


async def test_work_while_idle_waits_for_no_search(seeded):
    config, _ = seeded
    service = SearchService(config)
    ran = []

    async def work():
        assert not service.loaded
        ran.append(1)

    async with service.client():
        assert not await service.while_idle(work)
    assert ran == []
    assert service.loaded
    assert await service.while_idle(work)
    assert ran == [1]
    await service.search("travis", limit=1, search_type="fts")
    assert service.loaded
    await service.aclose()


async def test_a_new_configuration_applies_once_searches_end(tmp_path, seeded):
    config, _ = seeded
    service = SearchService(config)
    async with service.client() as first:
        changed = config.model_copy(deep=True)
        service.reconfigure(changed)
        assert service.config is changed
        async with service.client() as during:
            assert during is first
    async with service.client() as after:
        assert after is not first
    await service.aclose()


async def test_located_documents_are_kept_a_few_at_a_time(tmp_path, seeded):
    config, _ = seeded
    service = SearchService(config)
    try:
        async with HaikuRAG(config=config, sources=["files"]) as rag:
            for n in range(6):
                doc = _book([(1, f"unlocated passage {n}")])
                await rag.import_document(
                    doc,
                    [_chunk(f"unlocated passage {n}", "#/texts/0", 1)],
                    uri=f"file:///tmp/u{n}.pdf",
                )
        response = await service.search("unlocated", limit=10, search_type="fts")
        assert len(response.hits) == 6
        assert all(hit.pages[0].page == 1 for hit in response.hits)
        # Without a stored title a hit is named after its file.
        assert sorted(hit.title or "" for hit in response.hits) == [
            f"u{n}.pdf" for n in range(6)
        ]
        assert len(service._docling) == 4
        # A document read again comes from the kept ones.
        again = await service.search("unlocated", limit=10, search_type="fts")
        assert len(again.hits) == 6
    finally:
        await service.aclose()


async def test_a_document_gone_from_the_store_has_no_locations(seeded, monkeypatch):
    config, _ = seeded
    from haiku.rag.store.repositories.document import DocumentRepository

    async def nothing(self, document_id):  # noqa: ARG001
        return None

    monkeypatch.setattr(DocumentRepository, "get_docling_data", nothing)
    service = SearchService(config)
    try:
        response = await service.search("older chunk", limit=1, search_type="fts")
    finally:
        await service.aclose()
    assert response.hits[0].page_numbers == [41]
    assert response.hits[0].pages == []


@pytest.mark.parametrize(
    ("uri", "content_type", "expected"),
    [
        ("file:///a/b.PDF", None, "pdf"),
        ("file:///a/b.docx", None, "word"),
        ("file:///a/b.pptx", None, "powerpoint"),
        ("file:///a/b.epub", None, "epub"),
        ("file:///a/b.md", None, "markdown"),
        ("file:///a/b.txt", None, "text"),
        ("file:///a/b.py", None, "code"),
        ("file:///a/b.png", None, "image"),
        ("file:///a/b.pdf#attachment=c.md", None, "pdf"),
        ("https://x/download", "application/pdf", "pdf"),
        ("https://x/page", "text/html", "other"),
        (None, None, "other"),
    ],
)
def test_document_types(uri, content_type, expected):
    assert document_type(uri, content_type) == expected


def test_local_paths():
    assert local_path("file:///a/b.pdf") == uri_to_path("file:///a/b.pdf")
    assert local_path("file:///a/b.pdf#attachment=c.md") is None
    assert local_path("https://x/y") is None
    assert local_path(None) is None


def test_query_terms():
    assert query_terms("The Travis picking, and TRAVIS a la pompe") == [
        "travis",
        "picking",
        "la",
        "pompe",
    ]


def test_snippets():
    assert snippet("short  text\nhere", ["text"], 300) == ("short text here", [(6, 10)])
    assert snippet("no match here", [], 300) == ("no match here", [])
    long = " ".join(f"word{n}" for n in range(200)) + " target " + "tail " * 50
    shown, highlights = snippet(long, ["target"], 120)
    assert shown.startswith("...") and shown.endswith("...")
    assert len(shown) <= 126
    [(start, end)] = highlights
    assert shown[start:end] == "target"
    head, _ = snippet(long, ["missing"], 50)
    assert head.startswith("word0") and head.endswith("...")
    tail = "x " * 10 + "target"
    assert snippet(tail, ["target"], 8)[0].endswith("target")


def test_find_line(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text(NOTES, encoding="utf-8")
    assert find_line(path, "Travis picking keeps the thumb") == 4
    assert find_line(path, "## x\n- ab\nEnd with scales.") == 5
    assert find_line(path, "nowhere in the file") is None
    assert find_line(tmp_path / "missing.md", "anything") is None
    big = tmp_path / "big.txt"
    big.write_bytes(b"x" * (4 * 1024 * 1024 + 1))
    assert find_line(big, "xxxx") is None


def test_metadata_filter_escapes_the_value():
    assert metadata_filter("source_id", "it's 50%_") == (
        "metadata LIKE '%\"source_id\": \"it''s 50\\%\\_\"%' ESCAPE '\\'"
    )
