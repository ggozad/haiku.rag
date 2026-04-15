from haiku.rag.context import (
    _expand_outward,
    _find_expansion_range,
    _merge_ranges,
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

    def test_adjacent(self):
        r1, r2 = _result(), _result()
        merged = _merge_ranges([(0, 5, r1), (6, 10, r2)])
        assert len(merged) == 1
        assert merged[0] == (0, 10, [r1, r2])

    def test_sorts_by_position(self):
        r1, r2 = _result(), _result()
        merged = _merge_ranges([(10, 15, r1), (0, 5, r2)])
        assert merged[0][0] == 0
        assert merged[1][0] == 10


class TestExpandOutward:
    def test_basic_expansion(self):
        items = [_item(i, text=f"{'x' * 100}") for i in range(10)]
        lo, hi = _expand_outward(items, 5, max_items=10, max_chars=500)
        assert lo <= 5
        assert hi >= 5
        total = sum(
            len(items[i].text)
            for i in range(lo, hi + 1)
            if items[i].position >= lo and items[i].position <= hi
        )
        # Should be around 500 chars (may overshoot by one item)
        assert total >= 400

    def test_respects_max_items(self):
        items = [_item(i, text="x") for i in range(100)]
        lo, hi = _expand_outward(items, 50, max_items=5, max_chars=999999)
        count = hi - lo + 1
        # May overshoot by 1-2 items due to alternating expansion
        assert count <= 7

    def test_respects_max_chars(self):
        items = [_item(i, text=f"{'x' * 200}") for i in range(20)]
        lo, hi = _expand_outward(items, 10, max_items=999, max_chars=500)
        total = sum(
            len(items[i].text)
            for i in range(lo, hi + 1)
            if items[i].position >= lo and items[i].position <= hi
        )
        # Should be near 500, may overshoot by one item (~200 chars)
        assert total <= 900

    def test_center_at_start(self):
        items = [_item(i) for i in range(10)]
        lo, hi = _expand_outward(items, 0, max_items=5, max_chars=999999)
        assert lo == 0

    def test_center_at_end(self):
        items = [_item(i) for i in range(10)]
        lo, hi = _expand_outward(items, 9, max_items=5, max_chars=999999)
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
        lo, hi = _expand_outward(items, 2, max_items=10, max_chars=500, skip_noise=True)
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
        lo, hi = _expand_outward(items, 1, max_items=10, max_chars=500, skip_noise=True)
        # Center is noise, should start at 0 chars and expand outward
        assert lo == 0
        assert hi == 2


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
        lo, hi = _find_expansion_range(
            items, {1}, has_sections=True, max_items=20, max_chars=5000
        )
        # Should return the Introduction section (items 0-3)
        assert lo == 0
        assert hi == 3

    def test_structured_different_section(self):
        items = self._structured_items()
        lo, hi = _find_expansion_range(
            items, {5}, has_sections=True, max_items=20, max_chars=5000
        )
        # Should return the Methods section (items 4-6)
        assert lo == 4
        assert hi == 6

    def test_structured_large_section_falls_back_to_outward(self):
        items = [
            _item(0, label="section_header", text="Big Section"),
        ] + [_item(i, text="x" * 1000) for i in range(1, 20)]
        # Section has 19 * 1000 = 19000 chars, way over 5000 budget
        lo, hi = _find_expansion_range(
            items, {10}, has_sections=True, max_items=50, max_chars=5000
        )
        # Should NOT return the full section
        total = sum(
            len(items[i].text) for i in range(lo, hi + 1) if items[i].position >= lo
        )
        assert total < 10000

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
        lo, hi = _find_expansion_range(
            items, {0}, has_sections=True, max_items=20, max_chars=5000
        )
        # Should expand past the title section into the abstract
        assert hi >= 3

    def test_unstructured_expands_outward(self):
        items = [_item(i, text=f"Paragraph {i}. " * 10) for i in range(10)]
        lo, hi = _find_expansion_range(
            items, {5}, has_sections=False, max_items=20, max_chars=5000
        )
        assert lo < 5
        assert hi > 5

    def test_multiple_matched_positions_uses_center(self):
        items = [_item(i, text="x" * 100) for i in range(20)]
        # Match at positions 3 and 7, center should be index for position 5 (median)
        lo, hi = _find_expansion_range(
            items, {3, 7}, has_sections=False, max_items=5, max_chars=999999
        )
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
        lo, hi = _find_expansion_range(
            items, {1}, has_sections=True, max_items=20, max_chars=5000
        )
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
        lo, hi = _find_expansion_range(
            items, {0}, has_sections=True, max_items=20, max_chars=5000
        )
        # Match is in preamble section (items 0-1), which is small
        # Should expand outward into the first section
        assert hi >= 2
