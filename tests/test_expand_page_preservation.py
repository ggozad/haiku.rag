"""Page-number behaviour of context expansion for multi-result documents.

These tests exercise ``expand_with_items`` directly against a hand-built
item table (a fake repository implementing only the two methods the
algorithm calls: ``resolve_refs`` and ``get_items_in_range``). This keeps
them deterministic — no embedder, no LLM, no VCR — so the page arithmetic
is the only thing under test.

Two scenarios, mirroring the two regimes the merge/clip logic falls into:

* Matches FAR apart (page 1 + page 100): expansion ranges never overlap,
  so the results stay separate and every page survives — but a single wide
  item window spanning both matches is fetched.
* Matches CLOSE together on different pages: expansion ranges overlap and
  merge; the budget clip then centres on the higher-scoring chunk and the
  lower-scoring result's page is clipped away.
"""

import pytest

from haiku.rag.context import expand_with_items
from haiku.rag.store.models import SearchResult
from haiku.rag.store.models.document_item import DocumentItem

DOC_ID = "doc-1"


class FakeItemRepo:
    """Minimal DocumentItemRepository stand-in over an in-memory item list.

    Records the (start, end) ranges passed to ``get_items_in_range`` so a
    test can assert how wide a window the algorithm fetched.
    """

    def __init__(self, items: list[DocumentItem]) -> None:
        self.items = items
        self.range_calls: list[tuple[int, int]] = []

    async def resolve_refs(self, document_id: str, refs: list[str]) -> dict[str, int]:
        wanted = set(refs)
        return {
            item.self_ref: item.position
            for item in self.items
            if item.document_id == document_id and item.self_ref in wanted
        }

    async def get_items_in_range(
        self, document_id: str, start: int, end: int
    ) -> list[DocumentItem]:
        self.range_calls.append((start, end))
        found = [
            item
            for item in self.items
            if item.document_id == document_id and start <= item.position <= end
        ]
        found.sort(key=lambda i: i.position)
        return found


def _item(position: int, page: int, n: int = 20) -> DocumentItem:
    """An ``n``-char, uniquely-texted plain-text item on a given page."""
    prefix = f"item{position:03d}-"
    text = (prefix + "x" * max(0, n - len(prefix)))[:n]
    return DocumentItem(
        document_id=DOC_ID,
        position=position,
        self_ref=f"#/texts/{position}",
        label="text",
        text=text,
        page_numbers=[page],
    )


async def test_far_apart_matches_preserve_both_pages():
    """Page 1 + page 100 matches: no merge, both pages survive.

    The two matches sit ~150 items apart, so their expansion ranges cannot
    overlap and ``_merge_ranges`` leaves them separate. Each result keeps its
    own page. Also asserts the fetched item window spans the full gap between
    the matches (the cost of far-apart matches).
    """
    # Positions 0..79 are page 1; 80..159 are page 100.
    items = [_item(pos, 1 if pos < 80 else 100) for pos in range(160)]
    repo = FakeItemRepo(items)

    results = [
        SearchResult(
            content=items[2].text,
            score=0.9,
            document_id=DOC_ID,
            doc_item_refs=[items[2].self_ref],
            page_numbers=[1],
        ),
        SearchResult(
            content=items[150].text,
            score=0.8,
            document_id=DOC_ID,
            doc_item_refs=[items[150].self_ref],
            page_numbers=[100],
        ),
    ]

    expanded = await expand_with_items(repo, DOC_ID, results, max_chars=200)

    # Ranges don't overlap -> two independent results.
    assert len(expanded) == 2
    pages = {tuple(r.page_numbers) for r in expanded}
    assert pages == {(1,), (100,)}, pages

    # A single window was fetched spanning from before the first match to
    # after the last: the whole ~150-item gap, not two small windows.
    assert len(repo.range_calls) == 1
    start, end = repo.range_calls[0]
    assert start <= 2 and end >= 150


async def test_close_matches_on_different_pages_drop_a_page():
    """Adjacent matches on different pages: merge + clip drops the low page.

    Two matches four items apart expand into overlapping ranges that merge
    into one. The merged content exceeds ``max_chars``, so the clip window
    centres on the higher-scoring chunk (page 2) and the lower-scoring
    chunk's item (the only page-1 item) is clipped out — so page 1, present
    on an original search result, is absent from the expanded output.
    """
    # Only positions 0..2 are page 1; the rest page 2. The page-1 match sits
    # at the low edge so the clip window (centred on the page-2 match) drops
    # every page-1 item.
    items = [_item(pos, 1 if pos < 3 else 2) for pos in range(8)]
    repo = FakeItemRepo(items)

    low_page_result = SearchResult(
        content=items[1].text,
        score=0.5,  # lower score -> not the clip anchor
        document_id=DOC_ID,
        doc_item_refs=[items[1].self_ref],
        page_numbers=[1],
    )
    high_page_result = SearchResult(
        content=items[5].text,
        score=0.9,  # higher score -> clip centres here
        document_id=DOC_ID,
        doc_item_refs=[items[5].self_ref],
        page_numbers=[2],
    )

    expanded = await expand_with_items(
        repo, DOC_ID, [low_page_result, high_page_result], max_chars=100
    )

    # Overlapping ranges merged into a single result.
    assert len(expanded) == 1

    result = expanded[0]
    # Page 1 was present on an input result but is dropped by the clip.
    assert result.page_numbers == [2]
    assert 1 not in result.page_numbers


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known bug: expanded page_numbers are recomputed from surviving "
        "item spans, so a clipped-out input result's page is dropped. Fix "
        "should union each output result's pages with the pages of all its "
        "constituent inputs. Remove this xfail when that lands."
    ),
)
async def test_expansion_pages_superset_of_inputs():
    """Regression guard: expansion must never drop an input result's page.

    Same close-match layout as ``test_close_matches_on_different_pages_drop_a
    _page``: a page-1 match and a page-2 match merge and the merged content
    is clipped. The contract we *want* is that the union of every expanded
    result's ``page_numbers`` covers every page present on the inputs — the
    surrounding context is added, never at the cost of a page the search
    already surfaced.

    Currently ``xfail`` because page 1 is dropped; it flips to a hard failure
    (``strict=True``) once the union-with-inputs fix lands, prompting removal
    of this marker.
    """
    items = [_item(pos, 1 if pos < 3 else 2) for pos in range(8)]
    repo = FakeItemRepo(items)

    inputs = [
        SearchResult(
            content=items[1].text,
            score=0.5,
            document_id=DOC_ID,
            doc_item_refs=[items[1].self_ref],
            page_numbers=[1],
        ),
        SearchResult(
            content=items[5].text,
            score=0.9,
            document_id=DOC_ID,
            doc_item_refs=[items[5].self_ref],
            page_numbers=[2],
        ),
    ]

    expanded = await expand_with_items(repo, DOC_ID, inputs, max_chars=100)

    input_pages = {p for r in inputs for p in r.page_numbers}
    output_pages = {p for r in expanded for p in r.page_numbers}
    missing = input_pages - output_pages
    assert not missing, f"expansion dropped input pages: {sorted(missing)}"


@pytest.mark.parametrize(
    ("r1_pos", "r2_pos", "expected_results", "expected_pages"),
    [
        # Matched items adjacent (no item between): expansion ranges overlap
        # and merge; the ~200-char merged span is clipped to 100 around the
        # higher-scoring page-2 chunk, evicting the page-1 chunk.
        pytest.param(3, 4, 1, [2], id="adjacent-merges-and-drops-page-1"),
        # One full item sits between the matches, mid-document: each match is
        # already ~99/100 chars full, so it expands one step *downward* (away
        # from the other match). The ranges never touch, stay separate, and
        # both pages survive.
        pytest.param(2, 4, 2, [1, 2], id="gap-stays-separate-keeps-both"),
    ],
)
async def test_budget_100_layout_determines_page_drop(
    r1_pos, r2_pos, expected_results, expected_pages
):
    """At a 100-char budget with ~99-char chunks, page loss is layout-driven.

    A 99-char chunk nearly fills the 100-char budget, so ``_expand_outward``
    takes a single step before the budget is spent. Whether that step makes
    the two matches' ranges overlap — and therefore whether merge+clip drops
    a page — depends purely on item adjacency, not on the chunks' character
    distance. Pages are assigned ``1`` for positions < 4 and ``2`` otherwise,
    so ``r1`` (lower score) is always page 1 and ``r2`` (higher score) page 2.
    """
    items = [_item(pos, 1 if pos < 4 else 2, n=99) for pos in range(7)]
    repo = FakeItemRepo(items)

    inputs = [
        SearchResult(
            content=items[r1_pos].text,
            score=0.5,  # lower score
            document_id=DOC_ID,
            doc_item_refs=[items[r1_pos].self_ref],
            page_numbers=items[r1_pos].page_numbers,
        ),
        SearchResult(
            content=items[r2_pos].text,
            score=0.9,  # higher score -> clip anchor when merged
            document_id=DOC_ID,
            doc_item_refs=[items[r2_pos].self_ref],
            page_numbers=items[r2_pos].page_numbers,
        ),
    ]

    expanded = await expand_with_items(repo, DOC_ID, inputs, max_chars=100)

    assert len(expanded) == expected_results
    output_pages = sorted({p for r in expanded for p in r.page_numbers})
    assert output_pages == expected_pages
