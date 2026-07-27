"""Section-bounded context expansion for search results.

Expands search results with surrounding content from the document using
the document_items table. The algorithm adapts to document structure:

For STRUCTURED documents (containing section_header or title labels):
  1. Resolve matched doc_item_refs to positions in the items table
  2. Find section boundaries around each match (section_header/title labels)
  3. If the section fits within the char budget, include it entirely
  4. If the section exceeds the char budget, expand item-by-item from the
     match center outward, bounded by section edges
  5. If the section is too small (under 20% of max_context_chars), expand
     item-by-item crossing into adjacent sections until the budget is filled.
     This lets small sections (e.g., title+authors) grow into neighboring
     content. Picture and table matches are exempt: they return their
     enclosing section as-is, never crossing section boundaries.
  6. Merge overlapping ranges from multiple results in the same document,
     but only when every constituent's matched evidence survives in the
     final budget-clipped window; otherwise the group is split back into
     per-result windows so no retrieved result is dropped. Adjacent but
     non-overlapping ranges stay separate to preserve section independence.

For UNSTRUCTURED documents (no section headers):
  Expand outward item-by-item from the match center until the character
  budget is filled. No noise filtering (unstructured docs typically only
  have text items).

In both cases:
  - max_context_chars caps total characters per expanded result
  - Noise labels (footnote, page_header, page_footer, document_index) are
    excluded from content AND budget counting in structured documents
  - Results without doc_item_refs pass through unexpanded
"""

from haiku.rag.store.models.chunk import SearchResult
from haiku.rag.store.models.document_item import DocumentItem
from haiku.rag.store.repositories.document_item import DocumentItemRepository

_NOISE_LABELS = {"footnote", "page_header", "page_footer", "document_index"}
_SECTION_BOUNDARY_LABELS = {"section_header", "title"}

# Labels whose pertinent unit is the item plus its own section: expansion
# never crosses section boundaries for these matches.
_TIGHT_LABELS = {"picture", "table"}

# Sections with fewer chars than this fraction of max_context_chars are
# considered too small — expansion falls through to item-by-item outward
# growth, which naturally crosses into adjacent sections.
_MIN_SECTION_BUDGET_RATIO = 0.2


def _merge_ranges(
    ranges: list[tuple[int, int, SearchResult]],
) -> list[tuple[int, int, list[SearchResult]]]:
    """Merge overlapping ranges. Adjacent but non-overlapping ranges stay separate."""
    if not ranges:
        return []

    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged: list[tuple[int, int, list[SearchResult]]] = []
    cur_min, cur_max, cur_results = (
        sorted_ranges[0][0],
        sorted_ranges[0][1],
        [sorted_ranges[0][2]],
    )

    for min_idx, max_idx, result in sorted_ranges[1:]:
        if cur_max >= min_idx:  # Truly overlapping
            cur_max = max(cur_max, max_idx)
            cur_results.append(result)
        else:
            merged.append((cur_min, cur_max, cur_results))
            cur_min, cur_max, cur_results = min_idx, max_idx, [result]

    merged.append((cur_min, cur_max, cur_results))
    return merged


_MIN_ANCHOR = 128


def _evidence_anchors(content: str, max_chars: int) -> list[str]:
    """Substrings of a matched chunk used to locate it inside expanded content.

    Tries the exact chunk text first, then a substantial central slice that
    tolerates edge formatting drift between the chunk and the joined item text.
    Every anchor is bounded by ``max_chars`` so it always fits inside the window
    the caller returns, and is never shorter than ``_MIN_ANCHOR`` (unless the
    chunk itself is) so a short, common substring can't anchor by accident.
    """
    if not content:
        return []
    anchors: list[str] = []
    if len(content) <= max_chars:
        anchors.append(content)
    if max_chars > 0:
        min_anchor = min(_MIN_ANCHOR, max_chars, len(content))
        target = min(max_chars // 2, len(content) // 2)
        target = min(max(target, min_anchor), max_chars, len(content))
        if target < len(content):
            mid = len(content) // 2
            start = max(0, mid - target // 2)
            anchors.append(content[start : start + target])
    return anchors


def _clip_window(
    content: str, results: list[SearchResult], max_chars: int
) -> tuple[int, int]:
    """Return the ``[start, end)`` char window to keep when clipping ``content``.

    Anchors on the first locatable result in ``results`` order (the primary
    chunk that supplies the expanded result's identity) and returns a
    ``max_chars``-wide window centered on it. Falls back to a prefix window only
    when no anchor is locatable (heavy drift).
    """
    if max_chars <= 0:
        return (0, 0)
    evidence_start, evidence_len = -1, 0
    for result in results:
        for anchor in _evidence_anchors(result.content, max_chars):
            idx = content.find(anchor)
            if idx != -1:
                evidence_start, evidence_len = idx, len(anchor)
                break
        if evidence_start != -1:
            break
    if evidence_start == -1:
        return (0, min(len(content), max_chars))
    center = evidence_start + evidence_len // 2
    start = max(0, center - max_chars // 2)
    end = min(len(content), start + max_chars)
    start = max(0, end - max_chars)
    return (start, end)


def _clip_to_budget(content: str, results: list[SearchResult], max_chars: int) -> str:
    """Clip expanded content to ``max_chars``, keeping the matched evidence."""
    start, end = _clip_window(content, results, max_chars)
    return content[start:end]


def _collect_meta(
    spans: list[tuple[int, int, DocumentItem]],
) -> tuple[set[int], list[str], set[str]]:
    """Union the page numbers, refs, and labels of the given item spans."""
    pages: set[int] = set()
    refs: list[str] = []
    labels: set[str] = set()
    for _start, _end, item in spans:
        refs.append(item.self_ref)
        if item.label:
            labels.add(item.label)
        pages.update(item.page_numbers)
    return pages, refs, labels


def _span_in_window(
    span: tuple[int, int, DocumentItem], win_start: int, win_end: int
) -> bool:
    """Whether an item's char span overlaps the ``[win_start, win_end]`` clip window."""
    start, end, _item = span
    if start == end:  # zero-width picture position
        return win_start <= start <= win_end
    return start < win_end and end > win_start


def _expand_outward(
    items: list[DocumentItem],
    center_idx: int,
    max_chars: int,
    skip_noise: bool = False,
    lo_bound: int = 0,
    hi_bound: int | None = None,
) -> tuple[int, int]:
    """Expand item-by-item outward from center until char budget is filled.

    When skip_noise is True, noise labels are excluded from char counting
    (used in structured documents so footnotes don't consume budget).

    lo_bound and hi_bound constrain expansion (e.g., to section edges).
    """
    if hi_bound is None:
        hi_bound = len(items) - 1
    lo = hi = center_idx
    center_is_noise = skip_noise and items[center_idx].label in _NOISE_LABELS
    char_count = 0 if center_is_noise else len(items[center_idx].text)

    while char_count < max_chars:
        grew = False
        if lo > lo_bound:
            lo -= 1
            if not (skip_noise and items[lo].label in _NOISE_LABELS):
                char_count += len(items[lo].text)
            grew = True
        if hi < hi_bound and char_count < max_chars:
            hi += 1
            if not (skip_noise and items[hi].label in _NOISE_LABELS):
                char_count += len(items[hi].text)
            grew = True
        if not grew:
            break

    return (items[lo].position, items[hi].position)


def _find_expansion_range(
    items: list[DocumentItem],
    matched_positions: set[int],
    has_sections: bool,
    max_chars: int,
) -> tuple[int, int]:
    """Find the expansion range for matched positions within a window of items."""
    pos_to_idx = {item.position: i for i, item in enumerate(items)}
    matched_indices = sorted(pos_to_idx[p] for p in matched_positions)
    center_idx = matched_indices[len(matched_indices) // 2]

    if not has_sections:
        return _expand_outward(items, center_idx, max_chars)

    # Build section spans: [(start_idx, end_idx), ...]
    headers = [
        i for i, item in enumerate(items) if item.label in _SECTION_BOUNDARY_LABELS
    ]
    sections: list[tuple[int, int]] = []
    if headers[0] > 0:
        sections.append((0, headers[0] - 1))
    for j, h in enumerate(headers):
        end = headers[j + 1] - 1 if j + 1 < len(headers) else len(items) - 1
        sections.append((h, end))

    # Find which section contains the center match
    current = 0
    for j, (start, end) in enumerate(sections):
        if start <= center_idx <= end:
            current = j
            break

    sec_start, sec_end = sections[current]
    sec_chars = sum(
        len(items[i].text)
        for i in range(sec_start, sec_end + 1)
        if items[i].label not in _NOISE_LABELS
    )

    min_useful = int(max_chars * _MIN_SECTION_BUDGET_RATIO)

    if sec_chars <= max_chars and sec_chars >= min_useful:
        # Section fits in char budget — return it regardless of item count
        return (items[sec_start].position, items[sec_end].position)

    if sec_chars > max_chars:
        # Section too large — expand outward bounded by section edges
        return _expand_outward(
            items,
            center_idx,
            max_chars,
            skip_noise=True,
            lo_bound=sec_start,
            hi_bound=sec_end,
        )

    # Picture/table hits stay section-bounded: their pertinent unit is the
    # figure or table plus its section, never neighboring sections.
    if any(items[i].label in _TIGHT_LABELS for i in matched_indices):
        return (items[sec_start].position, items[sec_end].position)

    # Section too small (e.g., title+authors) — expand across boundaries
    return _expand_outward(items, center_idx, max_chars, skip_noise=True)


def _group_lost_constituent(built: SearchResult, group: list[SearchResult]) -> bool:
    """Whether any constituent's matched refs were entirely evicted from ``built``.

    Fires when the budget clip (or the fragmented-content fallback) left a
    constituent with none of its own items in the built result — its evidence
    would be silently dropped if the group stayed merged.
    """
    surviving = set(built.doc_item_refs)
    return any(
        r.doc_item_refs and not surviving.intersection(r.doc_item_refs) for r in group
    )


def _add_input_pages_for_surviving_refs(
    pages: set[int], refs: list[str], original_results: list[SearchResult]
) -> None:
    """Fill missing item-table pages from inputs whose own refs all survived."""
    surviving = set(refs)
    for result in original_results:
        if not result.page_numbers or not result.doc_item_refs:
            continue
        if set(result.doc_item_refs) <= surviving:
            pages.update(result.page_numbers)


def _build_result(
    range_start: int,
    range_end: int,
    original_results: list[SearchResult],
    pos_to_item: dict[int, DocumentItem],
    has_sections: bool,
    max_chars: int,
) -> SearchResult:
    """Build one expanded result from the items in ``[range_start, range_end]``."""
    content_parts: list[str] = []
    # Char span of each contributing item within the joined content, so
    # metadata can be narrowed to whatever survives a budget clip.
    item_spans: list[tuple[int, int, DocumentItem]] = []
    cursor = 0
    separator = "\n\n"

    for pos in range(range_start, range_end + 1):
        item = pos_to_item.get(pos)
        if item is None:
            continue
        if has_sections and item.label in _NOISE_LABELS:
            continue
        if item.text:
            if content_parts:
                cursor += len(separator)
            start = cursor
            content_parts.append(item.text)
            cursor += len(item.text)
            item_spans.append((start, cursor, item))
        elif item.label == "picture":
            # Pictures may legitimately have empty text (no VLM
            # description configured). Keep their self_ref so the
            # downstream image_data lookup can still attach bytes. They
            # occupy a zero-width position in reading order.
            item_spans.append((cursor, cursor, item))

    all_headings: list[str] = []
    for r in original_results:
        if r.headings:
            all_headings.extend(h for h in r.headings if h not in all_headings)

    # Anchor identity (chunk_id, content/refs fallbacks) on the
    # best-scoring constituent — the chunk that earned the result its
    # rank — rather than whichever sits earliest in the document.
    first = max(original_results, key=lambda r: r.score)

    chunk_ids: list[str] = []
    for r in original_results:
        if r.chunk_id and r.chunk_id not in chunk_ids:
            chunk_ids.append(r.chunk_id)

    joined = separator.join(content_parts)

    # Expansion should never return less content than the original chunk.
    # This can happen when item texts are fragmented (e.g., docling splits
    # formatted HTML list items into many small text nodes); fall back to
    # the chunk's own content, described by the chunk's own metadata.
    if len(joined) < len(first.content):
        base_content, base_spans = first.content, None
    else:
        base_content, base_spans = joined, item_spans

    if len(base_content) > max_chars:
        # Clip to the budget, anchored on the primary (highest-scoring)
        # chunk, and narrow metadata to whatever survives the window.
        win_start, win_end = _clip_window(
            base_content, [first, *original_results], max_chars
        )
        expanded_content = base_content[win_start:win_end]
        if base_spans is None:
            pages, refs, labels = (
                set(first.page_numbers),
                list(first.doc_item_refs),
                set(first.labels),
            )
        else:
            pages, refs, labels = _collect_meta(
                [s for s in base_spans if _span_in_window(s, win_start, win_end)]
            )
    else:
        expanded_content = base_content
        if base_spans is None:
            pages, refs, labels = (
                set(first.page_numbers),
                list(first.doc_item_refs),
                set(first.labels),
            )
        else:
            pages, refs, labels = _collect_meta(base_spans)

    _add_input_pages_for_surviving_refs(pages, refs, original_results)

    # Carry image_data and picture_captions from the originally retrieved
    # chunks, but only for constituents whose refs survive the window — a
    # chunk clipped out of the budget must not still ship its image to the
    # model. Pictures swept in by section expansion are referenced in
    # ``refs`` but their bytes are never re-fetched, so the multimodal
    # payload stays bounded to what was actually retrieved and shown.
    surviving_refs = set(refs)
    merged_image_data: dict[str, str] = {}
    merged_captions: dict[str, str] = {}
    merged_annotations: dict[str, None] = {}
    for r in original_results:
        if r.doc_item_refs and not surviving_refs.intersection(r.doc_item_refs):
            continue
        if r.image_data:
            merged_image_data.update(r.image_data)
        if r.picture_captions:
            merged_captions.update(r.picture_captions)
        if r.annotations:
            merged_annotations.update(dict.fromkeys(r.annotations))

    return SearchResult(
        content=expanded_content,
        score=max(r.score for r in original_results),
        chunk_id=first.chunk_id,
        chunk_ids=chunk_ids,
        document_id=first.document_id,
        document_uri=first.document_uri,
        document_title=first.document_title,
        document_meta=first.document_meta,
        doc_item_refs=refs or first.doc_item_refs,
        page_numbers=sorted(pages) or first.page_numbers,
        headings=all_headings or None,
        labels=sorted(labels) or first.labels,
        image_data=merged_image_data or None,
        picture_captions=merged_captions,
        annotations=list(merged_annotations) or None,
    )


_WINDOW_MARGIN = 100


async def expand_with_items(
    document_item_repository: DocumentItemRepository,
    document_id: str,
    results: list[SearchResult],
    max_chars: int,
) -> list[SearchResult]:
    """Expand results using the document_items table."""
    all_refs = []
    for result in results:
        all_refs.extend(result.doc_item_refs)

    ref_positions = await document_item_repository.resolve_refs(document_id, all_refs)
    if not ref_positions:
        return results

    # Fetch a window of items around matched positions. The margin must be
    # wide enough to find section boundaries (the nearest section_header/title
    # above and below the match).
    all_positions = sorted(ref_positions.values())
    window_margin = _WINDOW_MARGIN
    window_start = max(0, min(all_positions) - window_margin)
    window_end = max(all_positions) + window_margin
    window_items = await document_item_repository.get_items_in_range(
        document_id, window_start, window_end
    )

    if not window_items:
        return results

    has_sections = any(item.label in _SECTION_BOUNDARY_LABELS for item in window_items)

    # Compute expansion ranges per result
    ranges: list[tuple[int, int, SearchResult]] = []
    passthrough: list[SearchResult] = []

    for result in results:
        matched = {ref_positions[r] for r in result.doc_item_refs if r in ref_positions}
        if not matched:
            passthrough.append(result)
            continue

        lo, hi = _find_expansion_range(window_items, matched, has_sections, max_chars)
        ranges.append((lo, hi, result))

    merged = _merge_ranges(ranges)
    constituent_range = {id(result): (lo, hi) for lo, hi, result in ranges}

    # Build results from the window items
    pos_to_item = {item.position: item for item in window_items}
    final_results: list[SearchResult] = []
    for range_start, range_end, group in merged:
        built = _build_result(
            range_start, range_end, group, pos_to_item, has_sections, max_chars
        )
        if len(group) > 1 and _group_lost_constituent(built, group):
            # The merged window cannot afford every constituent's evidence:
            # un-merge so no retrieved result is dropped from the group. Each
            # constituent gets its own window, clipped around its own anchor.
            for result in group:
                lo, hi = constituent_range[id(result)]
                final_results.append(
                    _build_result(
                        lo, hi, [result], pos_to_item, has_sections, max_chars
                    )
                )
            continue
        final_results.append(built)

    return final_results + passthrough
