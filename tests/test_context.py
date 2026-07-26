import pytest

from haiku.rag.client.documents import _store_document_with_chunks
from haiku.rag.context import (
    _clip_to_budget,
    _evidence_anchors,
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

    def test_picture_in_small_section_stays_section_bounded(self):
        items = [
            _item(0, label="section_header", text="Chapter 1"),
            _item(1, text="Chapter 1 prose. " * 100),
            _item(2, label="section_header", text="Figure heading"),
            _item(3, label="picture", text="Diagram description."),
            _item(4, label="caption", text="Figure 2-3. Balance arm."),
            _item(5, label="section_header", text="Chapter 3"),
            _item(6, text="Chapter 3 prose. " * 100),
        ]
        # Figure section (items 2-4) is far under 20% of 5000 chars.
        lo, hi = _find_expansion_range(items, {3}, has_sections=True, max_chars=5000)
        # Never crosses either header
        assert (lo, hi) == (2, 4)

    def test_table_in_small_section_stays_section_bounded(self):
        items = [
            _item(0, label="section_header", text="Chapter 1"),
            _item(1, text="Chapter 1 prose. " * 100),
            _item(2, label="section_header", text="Table heading"),
            _item(3, label="table", text="Header | Value"),
            _item(4, label="section_header", text="Chapter 3"),
            _item(5, text="Chapter 3 prose. " * 100),
        ]
        lo, hi = _find_expansion_range(items, {3}, has_sections=True, max_chars=5000)
        assert (lo, hi) == (2, 3)

    def test_text_in_small_section_still_expands_outward(self):
        items = [
            _item(0, label="section_header", text="Chapter 1"),
            _item(1, text="Chapter 1 prose. " * 100),
            _item(2, label="section_header", text="Short note"),
            _item(3, text="A brief remark."),
            _item(4, label="section_header", text="Chapter 3"),
            _item(5, text="Chapter 3 prose. " * 100),
        ]
        lo, hi = _find_expansion_range(items, {3}, has_sections=True, max_chars=5000)
        # Text hits keep growing across section boundaries
        assert lo < 2 or hi > 3


class TestEvidenceAnchors:
    def test_empty_content(self):
        assert _evidence_anchors("", 5000) == []

    def test_anchors_never_exceed_budget(self):
        # Even when max_chars is smaller than the minimum anchor length, no
        # anchor may exceed the window it has to fit inside.
        anchors = _evidence_anchors("z" * 1000, 50)
        assert anchors
        assert all(len(a) <= 50 for a in anchors)

    def test_tiny_content_uses_full_text(self):
        anchors = _evidence_anchors("short", 5000)
        assert anchors == ["short"]

    def test_long_content_offers_full_and_central_slice(self):
        content = "L" * 500 + "M" * 500
        anchors = _evidence_anchors(content, 5000)
        assert content in anchors
        # A strictly shorter central slice is also offered for drift tolerance.
        assert any(a != content and a in content for a in anchors)


class TestClipToBudget:
    def test_zero_budget_returns_empty(self):
        result = SearchResult(content="anything", score=0.9, document_id="d")
        assert _clip_to_budget("some long content", [result], 0) == ""

    def test_no_anchor_found_falls_back_to_prefix(self):
        content = "xyz" * 1000
        result = SearchResult(content="NOTPRESENT", score=0.9, document_id="d")
        assert _clip_to_budget(content, [result], 100) == content[:100]

    def test_centers_window_on_evidence(self):
        marker = "UNIQUE_MATCH_TEXT"
        content = "A" * 300_000 + marker + "B" * 300_000
        result = SearchResult(content=marker, score=0.9, document_id="d")
        clipped = _clip_to_budget(content, [result], 10_000)
        assert len(clipped) <= 10_000
        assert marker in clipped


@pytest.mark.asyncio
class TestExpandWithItems:
    async def test_unresolvable_refs_returns_original(self, temp_db_path):
        from haiku.rag.client import HaikuRAG
        from haiku.rag.store.models.document import Document

        async with HaikuRAG(temp_db_path, create=True) as rag:
            doc = await _store_document_with_chunks(
                rag,
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
            assert doc.id is not None
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

    async def test_picture_expansion_stays_within_section_pages(self, temp_db_path):
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as rag:
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/texts/0",
                    label="section_header",
                    text="Chapter 1",
                    page_numbers=[10],
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=1,
                    self_ref="#/texts/1",
                    label="text",
                    text="Chapter 1 prose. " * 200,
                    page_numbers=[10],
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=2,
                    self_ref="#/texts/2",
                    label="section_header",
                    text="Figure heading",
                    page_numbers=[13],
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=3,
                    self_ref="#/pictures/0",
                    label="picture",
                    text="Diagram description.",
                    page_numbers=[13],
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=4,
                    self_ref="#/texts/3",
                    label="caption",
                    text="Figure 2-3. Balance arm.",
                    page_numbers=[13],
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=5,
                    self_ref="#/texts/4",
                    label="section_header",
                    text="Chapter 3",
                    page_numbers=[15],
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=6,
                    self_ref="#/texts/5",
                    label="text",
                    text="Chapter 3 prose. " * 200,
                    page_numbers=[15],
                ),
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            result = SearchResult(
                content="Diagram description.",
                score=0.9,
                document_id="doc-1",
                doc_item_refs=["#/pictures/0"],
                page_numbers=[13],
            )
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", [result], 5000
            )
            assert len(expanded) == 1
            assert expanded[0].page_numbers == [13]
            assert "Chapter 1 prose" not in expanded[0].content
            assert "Chapter 3 prose" not in expanded[0].content

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

    async def test_oversized_item_clipped_to_budget(self, temp_db_path):
        """A single huge item (e.g. a whole spreadsheet as one table) is clipped
        to the budget, centered on the matched text."""
        from haiku.rag.client import HaikuRAG

        marker = "UNIQUE_MATCH_TEXT"
        async with HaikuRAG(temp_db_path, create=True) as rag:
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/tables/0",
                    label="table",
                    text="A" * 300_000 + marker + "B" * 300_000,
                ),
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            result = SearchResult(
                content=marker,
                score=0.9,
                document_id="doc-1",
                doc_item_refs=["#/tables/0"],
            )
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", [result], 10_000
            )
            assert len(expanded) == 1
            assert len(expanded[0].content) <= 10_000
            assert marker in expanded[0].content

    async def test_giant_neighbor_does_not_blow_budget(self, temp_db_path):
        """An adjacent giant item swept in by expansion cannot blow the budget,
        and the matched item's text survives."""
        from haiku.rag.client import HaikuRAG

        matched_text = "matched small text here"
        async with HaikuRAG(temp_db_path, create=True) as rag:
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/texts/0",
                    label="text",
                    text=matched_text,
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=1,
                    self_ref="#/tables/1",
                    label="table",
                    text="C" * 600_000,
                ),
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            result = SearchResult(
                content=matched_text,
                score=0.9,
                document_id="doc-1",
                doc_item_refs=["#/texts/0"],
            )
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", [result], 10_000
            )
            assert len(expanded) == 1
            assert len(expanded[0].content) <= 10_000
            assert matched_text in expanded[0].content

    async def test_small_items_under_budget_not_clipped(self, temp_db_path):
        """Normal small-item expansion is untouched — clipping never triggers."""
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as rag:
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/texts/0",
                    label="text",
                    text="First paragraph here.",
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=1,
                    self_ref="#/texts/1",
                    label="text",
                    text="Second matched paragraph.",
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=2,
                    self_ref="#/texts/2",
                    label="text",
                    text="Third paragraph here.",
                ),
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            result = SearchResult(
                content="Second matched paragraph.",
                score=0.9,
                document_id="doc-1",
                doc_item_refs=["#/texts/1"],
            )
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", [result], 5000
            )
            assert len(expanded) == 1
            assert len(expanded[0].content) < 5000
            # All three items expanded in, nothing truncated.
            assert "First paragraph here." in expanded[0].content
            assert "Second matched paragraph." in expanded[0].content
            assert "Third paragraph here." in expanded[0].content

    async def test_original_chunk_larger_than_budget_is_capped(self, temp_db_path):
        """When the floor restores an original chunk bigger than the budget, the
        hard cap still wins (returning less than the original chunk)."""
        from haiku.rag.client import HaikuRAG

        marker = "CENTRAL_MARKER_" + "Z" * 200
        big_chunk = "A" * 10_000 + marker + "A" * 10_000
        async with HaikuRAG(temp_db_path, create=True) as rag:
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/texts/0",
                    label="text",
                    text="tiny",
                ),
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            result = SearchResult(
                content=big_chunk,
                score=0.9,
                document_id="doc-1",
                doc_item_refs=["#/texts/0"],
            )
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", [result], 5000
            )
            assert len(expanded) == 1
            assert len(expanded[0].content) <= 5000
            assert marker in expanded[0].content

    async def test_solo_result_carries_own_chunk_id(self, temp_db_path):
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as rag:
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=i,
                    self_ref=f"#/texts/{i}",
                    label="text",
                    text=f"Paragraph {i}. " * 10,
                )
                for i in range(5)
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            result = SearchResult(
                content="Paragraph 2.",
                score=0.9,
                chunk_id="c1",
                document_id="doc-1",
                doc_item_refs=["#/texts/2"],
            )
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", [result], 5000
            )
            assert len(expanded) == 1
            assert expanded[0].chunk_ids == ["c1"]

    async def test_merged_results_carry_all_chunk_ids(self, temp_db_path):
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as rag:
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=i,
                    self_ref=f"#/texts/{i}",
                    label="text",
                    text=f"Paragraph {i}. " * 10,
                )
                for i in range(5)
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            r1 = SearchResult(
                content="Paragraph 1.",
                score=0.9,
                chunk_id="c1",
                document_id="doc-1",
                doc_item_refs=["#/texts/1"],
            )
            r2 = SearchResult(
                content="Paragraph 3.",
                score=0.85,
                chunk_id="c2",
                document_id="doc-1",
                doc_item_refs=["#/texts/3"],
            )
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", [r1, r2], 5000
            )
            # Ranges around positions 1 and 3 overlap → one merged result.
            assert len(expanded) == 1
            assert expanded[0].chunk_id == "c1"
            assert expanded[0].chunk_ids == ["c1", "c2"]
            # Sibling chunk ids are plumbing for visualization, never shown
            # to the model.
            assert "c2" not in expanded[0].format_for_agent()

    async def test_expanded_result_carries_document_meta(self, temp_db_path):
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as rag:
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=i,
                    self_ref=f"#/texts/{i}",
                    label="text",
                    text=f"Paragraph {i}. " * 10,
                )
                for i in range(5)
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            r1 = SearchResult(
                content="Paragraph 1.",
                score=0.9,
                chunk_id="c1",
                document_id="doc-1",
                doc_item_refs=["#/texts/1"],
                document_meta={"source_url": "https://example.org/report/view"},
            )
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", [r1], 5000
            )
            assert len(expanded) == 1
            assert expanded[0].document_meta == {
                "source_url": "https://example.org/report/view"
            }

    async def test_merged_anchor_is_highest_scoring_constituent(self, temp_db_path):
        """A merged result's chunk_id anchors on the best-scoring constituent,
        not whichever chunk sits earliest in the document."""
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as rag:
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=i,
                    self_ref=f"#/texts/{i}",
                    label="text",
                    text=f"Paragraph {i}. " * 10,
                )
                for i in range(5)
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            # earlier in the document, lower score
            r_early = SearchResult(
                content="Paragraph 1.",
                score=0.40,
                chunk_id="c-early",
                document_id="doc-1",
                doc_item_refs=["#/texts/1"],
            )
            # later in the document, higher score — the real hit
            r_best = SearchResult(
                content="Paragraph 3.",
                score=0.95,
                chunk_id="c-best",
                document_id="doc-1",
                doc_item_refs=["#/texts/3"],
            )
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", [r_early, r_best], 5000
            )
            assert len(expanded) == 1
            assert expanded[0].chunk_id == "c-best"
            assert expanded[0].score == 0.95
            # provenance still lists both
            assert set(expanded[0].chunk_ids) == {"c-early", "c-best"}

    async def test_clipped_merge_that_evicts_a_constituent_splits(self, temp_db_path):
        """A merged group whose budget clip would evict a constituent's evidence
        is split back into per-result windows: no retrieved result is dropped.
        Each split result keeps its own evidence and metadata describing only
        its visible content."""
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as rag:
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/texts/0",
                    label="text",
                    text="LOWMARK " + "a" * 400,
                    page_numbers=[1],
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=1,
                    self_ref="#/texts/1",
                    label="text",
                    text="b" * 400,
                    page_numbers=[2],
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=2,
                    self_ref="#/texts/2",
                    label="text",
                    text="c" * 400 + " HIGHMARK",
                    page_numbers=[3],
                ),
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            r_low = SearchResult(
                content="LOWMARK " + "a" * 400,
                score=0.4,
                chunk_id="c-low",
                document_id="doc-1",
                doc_item_refs=["#/texts/0"],
                page_numbers=[1],
            )
            r_high = SearchResult(
                content="c" * 400 + " HIGHMARK",
                score=0.9,
                chunk_id="c-high",
                document_id="doc-1",
                doc_item_refs=["#/texts/2"],
                page_numbers=[3],
            )
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", [r_low, r_high], 500
            )
            # The clip window around HIGHMARK cannot contain LOWMARK's item,
            # so the group splits instead of dropping r_low.
            assert len(expanded) == 2
            by_chunk = {e.chunk_id: e for e in expanded}
            e_high = by_chunk["c-high"]
            assert "HIGHMARK" in e_high.content
            assert "LOWMARK" not in e_high.content
            assert 3 in e_high.page_numbers
            # per-result metadata still describes only the visible content
            assert 1 not in e_high.page_numbers
            assert "#/texts/0" not in e_high.doc_item_refs
            assert e_high.chunk_ids == ["c-high"]
            e_low = by_chunk["c-low"]
            assert "LOWMARK" in e_low.content
            assert 1 in e_low.page_numbers
            assert e_low.chunk_ids == ["c-low"]

    async def test_split_results_each_keep_own_evidence_and_budget(self, temp_db_path):
        """Close matches on different pages at a small budget: the merged
        window cannot afford both, so each hit gets its own clipped window.
        Every input result's page survives across the output results."""
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as rag:
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=pos,
                    self_ref=f"#/texts/{pos}",
                    label="text",
                    text=f"i{pos:02d}" + "x" * 17,
                    page_numbers=[1 if pos < 3 else 2],
                )
                for pos in range(8)
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            inputs = [
                SearchResult(
                    content=items[1].text,
                    score=0.5,
                    document_id="doc-1",
                    doc_item_refs=["#/texts/1"],
                    page_numbers=[1],
                ),
                SearchResult(
                    content=items[5].text,
                    score=0.9,
                    document_id="doc-1",
                    doc_item_refs=["#/texts/5"],
                    page_numbers=[2],
                ),
            ]
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", inputs, 100
            )
            assert len(expanded) == 2
            for e in expanded:
                assert len(e.content) <= 100
            contents = " || ".join(e.content for e in expanded)
            assert items[1].text in contents
            assert items[5].text in contents
            input_pages = {p for r in inputs for p in r.page_numbers}
            output_pages = {p for e in expanded for p in e.page_numbers}
            assert input_pages <= output_pages

    async def test_clipped_merge_stays_merged_when_all_evidence_survives(
        self, temp_db_path
    ):
        """A merged group is clipped but the window still contains every
        constituent's evidence: no split, one merged result."""
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as rag:
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=pos,
                    self_ref=f"#/texts/{pos}",
                    label="text",
                    text=f"i{pos:02d}" + "y" * 97,
                )
                for pos in range(10)
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            r1 = SearchResult(
                content=items[4].text,
                score=0.5,
                chunk_id="c1",
                document_id="doc-1",
                doc_item_refs=["#/texts/4"],
            )
            r2 = SearchResult(
                content=items[5].text,
                score=0.9,
                chunk_id="c2",
                document_id="doc-1",
                doc_item_refs=["#/texts/5"],
            )
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", [r1, r2], 400
            )
            assert len(expanded) == 1
            e = expanded[0]
            assert len(e.content) <= 400
            assert items[4].text in e.content
            assert items[5].text in e.content
            assert set(e.chunk_ids) == {"c1", "c2"}

    async def test_fragmented_merge_splits_instead_of_dropping(self, temp_db_path):
        """When fragmented item text triggers the chunk-content fallback for a
        merged group, the group splits so the non-primary hit is not dropped."""
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as rag:
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/texts/0",
                    label="text",
                    text="frag a",
                    page_numbers=[1],
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=1,
                    self_ref="#/texts/1",
                    label="text",
                    text="frag b",
                    page_numbers=[2],
                ),
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            r1 = SearchResult(
                content="A" * 500,
                score=0.9,
                chunk_id="c1",
                document_id="doc-1",
                doc_item_refs=["#/texts/0"],
                page_numbers=[1],
            )
            r2 = SearchResult(
                content="B" * 400,
                score=0.5,
                chunk_id="c2",
                document_id="doc-1",
                doc_item_refs=["#/texts/1"],
                page_numbers=[2],
            )
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", [r1, r2], 5000
            )
            assert len(expanded) == 2
            by_chunk = {e.chunk_id: e for e in expanded}
            assert by_chunk["c1"].content == "A" * 500
            assert by_chunk["c1"].page_numbers == [1]
            assert by_chunk["c2"].content == "B" * 400
            assert by_chunk["c2"].page_numbers == [2]

    async def test_surviving_refs_fill_missing_item_pages_from_input(
        self, temp_db_path
    ):
        """When a visible item has missing page metadata, use the input
        result's page_numbers as a floor for that surviving ref."""
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as rag:
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/texts/0",
                    label="text",
                    text="Visible item with missing item-table pages.",
                    page_numbers=[],
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=1,
                    self_ref="#/texts/1",
                    label="text",
                    text="Visible item with stored item-table pages.",
                    page_numbers=[8],
                ),
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            r_missing_item_page = SearchResult(
                content=items[0].text,
                score=0.9,
                chunk_id="c-missing",
                document_id="doc-1",
                doc_item_refs=["#/texts/0"],
                page_numbers=[7],
            )
            r_with_item_page = SearchResult(
                content=items[1].text,
                score=0.8,
                chunk_id="c-present",
                document_id="doc-1",
                doc_item_refs=["#/texts/1"],
                page_numbers=[8],
            )
            expanded = await expand_with_items(
                rag.document_item_repository,
                "doc-1",
                [r_missing_item_page, r_with_item_page],
                5000,
            )

            assert len(expanded) == 1
            assert set(expanded[0].doc_item_refs) == {"#/texts/0", "#/texts/1"}
            assert expanded[0].page_numbers == [7, 8]

    async def test_input_pages_not_added_for_clipped_out_refs(self, temp_db_path):
        """Input page metadata is not blindly unioned when only some of a
        constituent's refs survive the clip window."""
        from haiku.rag.client import HaikuRAG

        item0_text = "LEFTMARK " + "a" * 91
        item1_text = "RIGHTMARK " + "b" * 70
        async with HaikuRAG(temp_db_path, create=True) as rag:
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/texts/0",
                    label="text",
                    text=item0_text,
                    page_numbers=[1],
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=1,
                    self_ref="#/texts/1",
                    label="text",
                    text=item1_text,
                    page_numbers=[2],
                ),
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            result = SearchResult(
                content=item0_text,
                score=0.9,
                chunk_id="c-both",
                document_id="doc-1",
                doc_item_refs=["#/texts/0", "#/texts/1"],
                page_numbers=[1, 2],
            )
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", [result], 100
            )

            assert len(expanded) == 1
            assert "LEFTMARK" in expanded[0].content
            assert "RIGHTMARK" not in expanded[0].content
            assert expanded[0].doc_item_refs == ["#/texts/0"]
            assert expanded[0].page_numbers == [1]

    async def test_fuzzy_match_preserves_central_marker(self, temp_db_path):
        """The chunk's text need not be verbatim in the joined item text: a clean
        central marker is still located via the central-slice anchor."""
        from haiku.rag.client import HaikuRAG

        marker = "M" * 2000
        async with HaikuRAG(temp_db_path, create=True) as rag:
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/tables/0",
                    label="table",
                    text="A" * 300_000 + marker + "B" * 300_000,
                ),
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            # Chunk content has edge formatting that is NOT verbatim in the item
            # text, but the central marker is identical.
            result = SearchResult(
                content="<edge>" + marker + "</edge>",
                score=0.9,
                document_id="doc-1",
                doc_item_refs=["#/tables/0"],
            )
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", [result], 10_000
            )
            assert len(expanded) == 1
            assert len(expanded[0].content) <= 10_000
            assert "M" * 500 in expanded[0].content


@pytest.mark.asyncio
class TestExpandWithItemsPictureBytes:
    """Picture bytes only ride along for refs present in the pre-expansion
    chunk. Pictures swept in by section expansion are still referenced in
    ``doc_item_refs`` for cross-referencing but their image_data is not
    re-fetched — keeps the multimodal payload bounded.
    """

    async def _populate(self, rag):
        items = [
            DocumentItem(
                document_id="doc-1",
                position=0,
                self_ref="#/texts/0",
                label="section_header",
                text="Section 1",
            ),
            DocumentItem(
                document_id="doc-1",
                position=1,
                self_ref="#/texts/1",
                label="text",
                text="Paragraph in section 1.",
            ),
            DocumentItem(
                document_id="doc-1",
                position=2,
                self_ref="#/pictures/0",
                label="picture",
                text="Figure 1 caption.",
            ),
            DocumentItem(
                document_id="doc-1",
                position=3,
                self_ref="#/texts/2",
                label="text",
                text="Another paragraph after the figure.",
            ),
        ]
        await rag.document_item_repository.create_items("doc-1", items)

    async def test_pre_expansion_picture_bytes_preserved(self, temp_db_path):
        """A picture chunk's image_data survives expansion."""
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as rag:
            await self._populate(rag)
            result = SearchResult(
                content="Figure 1 caption.",
                score=0.9,
                document_id="doc-1",
                doc_item_refs=["#/pictures/0"],
                image_data={"#/pictures/0": "BASE64BYTES"},
                picture_captions={"#/pictures/0": "Figure 1 caption."},
            )
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", [result], 5000
            )
            assert len(expanded) == 1
            assert expanded[0].image_data == {"#/pictures/0": "BASE64BYTES"}
            assert expanded[0].picture_captions == {"#/pictures/0": "Figure 1 caption."}

    async def test_merged_results_union_image_data(self, temp_db_path):
        """When two results' ranges merge, their pre-expansion image_data
        is unioned onto the merged output."""
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as rag:
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=i,
                    self_ref=f"#/pictures/{i}" if i in (1, 3) else f"#/texts/{i}",
                    label="picture" if i in (1, 3) else "text",
                    text=f"Item {i}.",
                )
                for i in range(5)
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            r1 = SearchResult(
                content="Item 1.",
                score=0.9,
                document_id="doc-1",
                doc_item_refs=["#/pictures/1"],
                image_data={"#/pictures/1": "A"},
            )
            r2 = SearchResult(
                content="Item 3.",
                score=0.85,
                document_id="doc-1",
                doc_item_refs=["#/pictures/3"],
                image_data={"#/pictures/3": "B"},
            )
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", [r1, r2], 5000
            )
            # Ranges around positions 1 and 3 overlap → one merged result.
            assert len(expanded) == 1
            assert expanded[0].image_data == {
                "#/pictures/1": "A",
                "#/pictures/3": "B",
            }

    async def test_split_results_carry_only_own_picture_bytes(self, temp_db_path):
        """When a clipped merge splits, each result ships only the image bytes
        its own window shows — the model must not receive an image the
        citation and visualization omit."""
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as rag:
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/pictures/0",
                    label="picture",
                    text="LOWPIC " + "a" * 400,
                    page_numbers=[1],
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=1,
                    self_ref="#/texts/1",
                    label="text",
                    text="b" * 400,
                    page_numbers=[2],
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=2,
                    self_ref="#/pictures/1",
                    label="picture",
                    text="c" * 400 + " HIGHPIC",
                    page_numbers=[3],
                ),
            ]
            await rag.document_item_repository.create_items("doc-1", items)

            r_low = SearchResult(
                content="LOWPIC " + "a" * 400,
                score=0.4,
                chunk_id="c-low",
                document_id="doc-1",
                doc_item_refs=["#/pictures/0"],
                image_data={"#/pictures/0": "LOWBYTES"},
            )
            r_high = SearchResult(
                content="c" * 400 + " HIGHPIC",
                score=0.9,
                chunk_id="c-high",
                document_id="doc-1",
                doc_item_refs=["#/pictures/1"],
                image_data={"#/pictures/1": "HIGHBYTES"},
            )
            expanded = await expand_with_items(
                rag.document_item_repository, "doc-1", [r_low, r_high], 500
            )
            assert len(expanded) == 2
            by_chunk = {e.chunk_id: e for e in expanded}
            e_high = by_chunk["c-high"]
            assert "#/pictures/0" not in e_high.doc_item_refs
            assert e_high.image_data == {"#/pictures/1": "HIGHBYTES"}
            e_low = by_chunk["c-low"]
            assert e_low.image_data == {"#/pictures/0": "LOWBYTES"}
            assert "HIGHBYTES" not in (e_low.image_data or {}).values()


class TestSpanInWindow:
    def test_zero_width_span_is_inside_when_position_is_in_window(self):
        from haiku.rag.context import _span_in_window
        from haiku.rag.store.models.document_item import DocumentItem

        item = DocumentItem(
            document_id="d1", position=0, self_ref="#/pictures/0", label="picture"
        )
        # A picture occupies no characters, so containment is by position.
        assert _span_in_window((10, 10, item), 0, 20) is True
        assert _span_in_window((30, 30, item), 0, 20) is False


@pytest.mark.asyncio
class TestExpandWithItemsWindowEdges:
    async def test_empty_window_returns_original_results(
        self, temp_db_path, monkeypatch
    ):
        """Refs resolve but the surrounding window comes back empty, so there is
        nothing to expand from."""
        from haiku.rag.client import HaikuRAG
        from haiku.rag.store.models.document import Document

        async with HaikuRAG(temp_db_path, create=True) as rag:
            doc = await rag.document_repository.create(
                Document(content="body", uri="test://window")
            )
            assert doc.id is not None
            await rag.document_item_repository.create_items(
                doc.id,
                [
                    DocumentItem(
                        document_id=doc.id,
                        position=0,
                        self_ref="#/texts/0",
                        label="paragraph",
                        text="body",
                        page_numbers=[1],
                    )
                ],
            )

            async def no_window(*_args, **_kwargs):
                return []

            monkeypatch.setattr(
                rag.document_item_repository, "get_items_in_range", no_window
            )

            result = SearchResult(
                content="original",
                score=0.9,
                document_id=doc.id,
                doc_item_refs=["#/texts/0"],
            )
            expanded = await expand_with_items(
                rag.document_item_repository, doc.id, [result], 5000
            )

            assert [r.content for r in expanded] == ["original"]

    async def test_result_with_unmatched_refs_passes_through(self, temp_db_path):
        """Two results share a document; the one whose refs resolve is expanded
        and the other is returned unchanged."""
        from haiku.rag.client import HaikuRAG
        from haiku.rag.store.models.document import Document

        async with HaikuRAG(temp_db_path, create=True) as rag:
            doc = await rag.document_repository.create(
                Document(content="body", uri="test://mixed")
            )
            assert doc.id is not None
            await rag.document_item_repository.create_items(
                doc.id,
                [
                    DocumentItem(
                        document_id=doc.id,
                        position=i,
                        self_ref=f"#/texts/{i}",
                        label="paragraph",
                        text=f"paragraph {i}",
                        page_numbers=[1],
                    )
                    for i in range(2)
                ],
            )

            resolvable = SearchResult(
                content="paragraph 0",
                score=0.9,
                document_id=doc.id,
                doc_item_refs=["#/texts/0"],
            )
            unmatched = SearchResult(
                content="untouched",
                score=0.5,
                document_id=doc.id,
                doc_item_refs=["#/texts/404"],
            )

            expanded = await expand_with_items(
                rag.document_item_repository, doc.id, [resolvable, unmatched], 5000
            )

            assert "untouched" in [r.content for r in expanded]


def test_build_result_skips_positions_with_no_item():
    """A sparse position map (items removed or never stored) leaves gaps in the
    range; those positions contribute nothing."""
    from haiku.rag.context import _build_result

    original = SearchResult(content="p0", score=0.9, document_id="d1")
    # Positions 1 and 2 in the 0..3 range carry no item.
    pos_to_item = {
        0: DocumentItem(
            document_id="d1",
            position=0,
            self_ref="#/texts/0",
            label="paragraph",
            text="first",
            page_numbers=[1],
        ),
        3: DocumentItem(
            document_id="d1",
            position=3,
            self_ref="#/texts/3",
            label="paragraph",
            text="last",
            page_numbers=[1],
        ),
    }

    built = _build_result(0, 3, [original], pos_to_item, False, 5000)

    assert built.content == "first\n\nlast"
