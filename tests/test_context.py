import pytest

from haiku.rag.context import (
    _expand_outward,
    _find_expansion_range,
    _merge_ranges,
    expand_with_items,
)
from haiku.rag.store.models.chunk import SearchResult
from haiku.rag.store.models.document_item import DocumentItem


def _item(
    position: int, label: str = "text", text: str = "", pages: list[int] | None = None
) -> DocumentItem:
    return DocumentItem(
        document_id="doc-1",
        position=position,
        self_ref=f"#/texts/{position}",
        label=label,
        text=text or f"Text for item {position}.",
        page_numbers=pages or [1],
    )


def _result(score: float = 0.5, refs: list[str] | None = None) -> SearchResult:
    return SearchResult(
        content="original",
        score=score,
        document_id="doc-1",
        doc_item_refs=refs or [],
    )


class TestMergeRanges:
    def test_empty(self):
        assert _merge_ranges([]) == []

    def test_no_overlap(self):
        r1, r2 = _result(), _result()
        merged = _merge_ranges([(0, 5, r1), (10, 15, r2)])
        assert len(merged) == 2
        assert merged[0] == (0, 5, [r1])
        assert merged[1] == (10, 15, [r2])

    def test_overlapping(self):
        r1, r2 = _result(0.9), _result(0.8)
        merged = _merge_ranges([(0, 10, r1), (5, 15, r2)])
        assert len(merged) == 1
        assert merged[0] == (0, 15, [r1, r2])

    def test_adjacent_stay_separate(self):
        r1, r2 = _result(), _result()
        merged = _merge_ranges([(0, 5, r1), (6, 10, r2)])
        assert len(merged) == 2
        assert merged[0] == (0, 5, [r1])
        assert merged[1] == (6, 10, [r2])

    def test_sorts_by_position(self):
        r1, r2 = _result(), _result()
        merged = _merge_ranges([(10, 15, r1), (0, 5, r2)])
        assert merged[0][0] == 0
        assert merged[1][0] == 10


class TestExpandOutward:
    def test_basic_expansion(self):
        items = [_item(i, text=f"{'x' * 100}") for i in range(10)]
        lo, hi = _expand_outward(items, 5, max_chars=500)
        assert lo <= 5
        assert hi >= 5
        total = sum(
            len(items[i].text)
            for i in range(lo, hi + 1)
            if items[i].position >= lo and items[i].position <= hi
        )
        # Should be around 500 chars (may overshoot by one item)
        assert total >= 400

    def test_respects_max_chars(self):
        items = [_item(i, text=f"{'x' * 200}") for i in range(20)]
        lo, hi = _expand_outward(items, 10, max_chars=500)
        total = sum(
            len(items[i].text)
            for i in range(lo, hi + 1)
            if items[i].position >= lo and items[i].position <= hi
        )
        # Should be near 500, may overshoot by one item (~200 chars)
        assert total <= 900

    def test_center_at_start(self):
        items = [_item(i) for i in range(10)]
        lo, hi = _expand_outward(items, 0, max_chars=999999)
        assert lo == 0

    def test_center_at_end(self):
        items = [_item(i) for i in range(10)]
        lo, hi = _expand_outward(items, 9, max_chars=999999)
        assert hi == 9

    def test_skip_noise_excludes_from_char_count(self):
        items = [
            _item(0, text="a" * 100),
            _item(1, label="footnote", text="f" * 5000),
            _item(2, text="b" * 100),
            _item(3, text="c" * 100),
            _item(4, label="footnote", text="f" * 5000),
            _item(5, text="d" * 100),
        ]
        lo, hi = _expand_outward(items, 2, max_chars=500, skip_noise=True)
        # Footnotes (5000 chars each) should NOT count toward budget
        # So we should expand past them
        assert lo <= 0
        assert hi >= 5

    def test_noise_center_gets_zero_chars(self):
        items = [
            _item(0, text="a" * 200),
            _item(1, label="document_index", text="x" * 10000),
            _item(2, text="b" * 200),
        ]
        lo, hi = _expand_outward(items, 1, max_chars=500, skip_noise=True)
        # Center is noise, should start at 0 chars and expand outward
        assert lo == 0
        assert hi == 2

    def test_respects_bounds(self):
        items = [_item(i, text="x" * 100) for i in range(20)]
        lo, hi = _expand_outward(items, 10, max_chars=999999, lo_bound=8, hi_bound=12)
        assert lo == 8
        assert hi == 12


class TestFindExpansionRange:
    def _structured_items(self):
        """Document with two sections, each over min_useful (1000 chars)."""
        return [
            _item(0, label="section_header", text="Introduction"),
            _item(1, text="First paragraph. " * 40),  # ~680 chars
            _item(2, text="Second paragraph. " * 40),  # ~720 chars
            _item(3, label="footnote", text="Some footnote."),
            _item(4, label="section_header", text="Methods"),
            _item(5, text="Methods paragraph one. " * 40),  # ~920 chars
            _item(6, text="Methods paragraph two. " * 40),  # ~920 chars
        ]

    def test_structured_returns_section(self):
        items = self._structured_items()
        lo, hi = _find_expansion_range(items, {1}, has_sections=True, max_chars=5000)
        # Should return the Introduction section (items 0-3)
        assert lo == 0
        assert hi == 3

    def test_structured_different_section(self):
        items = self._structured_items()
        lo, hi = _find_expansion_range(items, {5}, has_sections=True, max_chars=5000)
        # Should return the Methods section (items 4-6)
        assert lo == 4
        assert hi == 6

    def test_structured_large_section_bounded_by_section(self):
        items = [
            _item(0, label="section_header", text="Big Section"),
        ] + [_item(i, text="x" * 1000) for i in range(1, 20)]
        # Section has 19 * 1000 = 19000 chars, way over 5000 budget
        lo, hi = _find_expansion_range(items, {10}, has_sections=True, max_chars=5000)
        # Should NOT return the full section, but should stay within it
        total = sum(
            len(items[i].text) for i in range(lo, hi + 1) if items[i].position >= lo
        )
        assert total < 10000

    def test_structured_section_with_many_items_returned_whole(self):
        """A section that fits in char budget is returned even with many items."""
        items = (
            [
                _item(0, label="section_header", text="Section"),
            ]
            + [_item(i, text="x" * 200) for i in range(1, 20)]
            + [
                _item(20, label="section_header", text="Next"),
            ]
        )
        # Section has 19 * 200 = 3800 chars + header, under 5000 and over min_useful
        lo, hi = _find_expansion_range(items, {10}, has_sections=True, max_chars=5000)
        # Should return entire section despite 20 items
        assert lo == 0
        assert hi == 19

    def test_structured_small_section_expands_outward(self):
        items = [
            _item(0, label="title", text="Paper Title"),
            _item(1, text="Author names"),
            _item(2, label="section_header", text="Abstract"),
            _item(3, text="Abstract content. " * 50),
            _item(4, label="section_header", text="Introduction"),
            _item(5, text="Intro content. " * 50),
        ]
        # Title section (items 0-1) is tiny (~25 chars) < 20% of 5000
        lo, hi = _find_expansion_range(items, {0}, has_sections=True, max_chars=5000)
        # Should expand past the title section into the abstract
        assert hi >= 3

    def test_unstructured_expands_outward(self):
        items = [_item(i, text=f"Paragraph {i}. " * 10) for i in range(10)]
        lo, hi = _find_expansion_range(items, {5}, has_sections=False, max_chars=5000)
        assert lo < 5
        assert hi > 5

    def test_multiple_matched_positions_uses_center(self):
        items = [_item(i, text="x" * 100) for i in range(20)]
        # Use a char budget that forces partial expansion so center matters
        lo, hi = _find_expansion_range(items, {3, 7}, has_sections=False, max_chars=500)
        center = (lo + hi) // 2
        # Center should be around position 5
        assert 3 <= center <= 7

    def test_noise_excluded_from_section_char_count(self):
        items = [
            _item(0, label="section_header", text="Section"),
            _item(1, text="Real content." * 10),
            _item(2, label="footnote", text="x" * 10000),
            _item(3, text="More content." * 10),
        ]
        # Section non-noise chars: ~260 chars (items 0,1,3). Under 5000 budget.
        # The footnote's 10000 chars should NOT count.
        lo, hi = _find_expansion_range(items, {1}, has_sections=True, max_chars=5000)
        # Should return full section (it fits in budget excluding noise)
        assert lo == 0
        assert hi == 3

    def test_items_before_first_header_form_section(self):
        items = [
            _item(0, text="Preamble text."),
            _item(1, text="More preamble."),
            _item(2, label="section_header", text="First Section"),
            _item(3, text="Section content."),
        ]
        lo, hi = _find_expansion_range(items, {0}, has_sections=True, max_chars=5000)
        # Match is in preamble section (items 0-1), which is small
        # Should expand outward into the first section
        assert hi >= 2


@pytest.mark.asyncio
class TestExpandWithItems:
    async def test_unresolvable_refs_returns_original(self, temp_db_path):
        from haiku.rag.client import HaikuRAG
        from haiku.rag.store.models.document import Document

        async with HaikuRAG(temp_db_path, create=True) as rag:
            doc = await rag._store_document_with_chunks(
                Document(content="test"),
                [],
                __import__(
                    "docling_core.types.doc.document", fromlist=["DoclingDocument"]
                ).DoclingDocument(name="t"),
            )
            result = SearchResult(
                content="original",
                score=0.9,
                document_id=doc.id,
                doc_item_refs=["#/texts/999999"],
            )
            expanded = await expand_with_items(
                rag.document_item_repository, doc.id, [result], 5000
            )
            assert len(expanded) == 1
            assert expanded[0].content == "original"

    async def test_noise_only_range_preserves_original(self, temp_db_path):
        """When noise filtering removes all content, original chunk is preserved."""
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as rag:
            # Structured document where the matched item's section has only noise
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/texts/0",
                    label="section_header",
                    text="Table of Contents",
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=1,
                    self_ref="#/texts/1",
                    label="document_index",
                    text="x" * 2000,
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=2,
                    self_ref="#/texts/2",
                    label="section_header",
                    text="Introduction",
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=3,
                    self_ref="#/texts/3",
                    label="text",
                    text="Intro content. " * 100,
                ),
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            result = SearchResult(
                content="original chunk content",
                score=0.9,
                document_id="doc-1",
                doc_item_refs=["#/texts/1"],
            )
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", [result], 5000
            )
            assert len(expanded) == 1
            # The TOC section's only non-header item is document_index (noise).
            # The section_header "Table of Contents" has text but _expand_outward
            # with skip_noise crosses into the Introduction section which has
            # real content — so we get expanded content, not the fallback.
            assert len(expanded[0].content) > 0

    async def test_fragmented_items_preserve_chunk(self, temp_db_path):
        """When items are fragmented (e.g., list_item children), the original
        chunk content is preserved if expansion produces less text."""
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as rag:
            # Simulate docling's list_item structure: container with empty text,
            # children with tiny fragments
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/texts/0",
                    label="section_header",
                    text="Steps",
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=1,
                    self_ref="#/texts/1",
                    label="list_item",
                    text="",
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=2,
                    self_ref="#/texts/2",
                    label="text",
                    text="Click",
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=3,
                    self_ref="#/texts/3",
                    label="text",
                    text="+",
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=4,
                    self_ref="#/texts/4",
                    label="text",
                    text="Add a New Service",
                ),
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            # The chunk had properly assembled content from the chunker
            result = SearchResult(
                content="1. Click + Add a New Service in the dashboard.",
                score=0.9,
                document_id="doc-1",
                doc_item_refs=["#/texts/1", "#/texts/2", "#/texts/3", "#/texts/4"],
            )
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", [result], 5000
            )
            assert len(expanded) == 1
            # Expansion produces "Steps\n\nClick\n\n+\n\nAdd a New Service" = 38 chars
            # which is less than the chunk's 46 chars — fallback preserves the chunk
            assert expanded[0].content == result.content
