"""Section-bounded context expansion for search results.

Expands search results with surrounding content from the document using
the document_items table. The algorithm adapts to document structure:

For STRUCTURED documents (containing section_header or title labels):
  1. Resolve matched doc_item_refs to positions in the items table
  2. Find section boundaries around each match (section_header/title labels)
  3. If the section fits within the budget, include it entirely
  4. If the section exceeds the budget, OR the section is too small (under
     20% of max_context_chars), expand item-by-item from the match center
     outward, skipping noise labels. This lets small sections (e.g., a
     title+authors area) grow into the next section's content.
  5. Merge overlapping ranges from multiple results in the same document

For UNSTRUCTURED documents (no section headers):
  Expand outward item-by-item from the match center until the character
  budget is filled. No noise filtering (unstructured docs typically only
  have text items).

In both cases:
  - max_context_chars caps total characters per expanded result
  - max_context_items caps total items per expanded result
  - Noise labels (footnote, page_header, page_footer, document_index) are
    excluded from content AND budget counting in structured documents
  - Results without doc_item_refs pass through unexpanded
"""

from haiku.rag.store.models.chunk import SearchResult
from haiku.rag.store.models.document_item import DocumentItem
from haiku.rag.store.repositories.document_item import DocumentItemRepository

_NOISE_LABELS = {"footnote", "page_header", "page_footer", "document_index"}
_SECTION_BOUNDARY_LABELS = {"section_header", "title"}

# Sections with fewer chars than this fraction of max_context_chars are
# considered too small — expansion falls through to item-by-item outward
# growth, which naturally crosses into adjacent sections.
_MIN_SECTION_BUDGET_RATIO = 0.2


def _merge_ranges(
    ranges: list[tuple[int, int, SearchResult]],
) -> list[tuple[int, int, list[SearchResult]]]:
    """Merge overlapping or adjacent ranges."""
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
        if cur_max >= min_idx - 1:  # Overlapping or adjacent
            cur_max = max(cur_max, max_idx)
            cur_results.append(result)
        else:
            merged.append((cur_min, cur_max, cur_results))
            cur_min, cur_max, cur_results = min_idx, max_idx, [result]

    merged.append((cur_min, cur_max, cur_results))
    return merged


def _expand_outward(
    items: list[DocumentItem],
    center_idx: int,
    max_items: int,
    max_chars: int,
    skip_noise: bool = False,
) -> tuple[int, int]:
    """Expand item-by-item outward from center until budget is filled.

    When skip_noise is True, noise labels are excluded from char counting
    (used in structured documents so footnotes don't consume budget).
    """
    lo = hi = center_idx
    center_is_noise = skip_noise and items[center_idx].label in _NOISE_LABELS
    char_count = 0 if center_is_noise else len(items[center_idx].text)

    while char_count < max_chars and hi - lo + 1 < max_items:
        grew = False
        if lo > 0:
            lo -= 1
            if not (skip_noise and items[lo].label in _NOISE_LABELS):
                char_count += len(items[lo].text)
            grew = True
        if hi < len(items) - 1 and char_count < max_chars:
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
    max_items: int,
    max_chars: int,
) -> tuple[int, int]:
    """Find the expansion range for matched positions within a window of items."""
    pos_to_idx = {item.position: i for i, item in enumerate(items)}
    matched_indices = sorted(pos_to_idx[p] for p in matched_positions)
    center_idx = matched_indices[len(matched_indices) // 2]

    if not has_sections:
        return _expand_outward(items, center_idx, max_items, max_chars)

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

    # Section fits nicely in the budget — return it as-is
    min_useful = int(max_chars * _MIN_SECTION_BUDGET_RATIO)
    if min_useful <= sec_chars <= max_chars and sec_end - sec_start + 1 <= max_items:
        return (items[sec_start].position, items[sec_end].position)

    # Section is too large or too small — expand item-by-item from center.
    # For too-large sections this stays within budget.
    # For too-small sections (e.g., title+authors) this naturally grows
    # into adjacent sections until the budget is filled.
    return _expand_outward(items, center_idx, max_items, max_chars, skip_noise=True)


async def expand_with_items(
    document_item_repository: DocumentItemRepository,
    document_id: str,
    results: list[SearchResult],
    max_items: int,
    max_chars: int,
) -> list[SearchResult]:
    """Expand results using the document_items table."""
    all_refs = []
    for result in results:
        all_refs.extend(result.doc_item_refs)

    ref_positions = await document_item_repository.resolve_refs(document_id, all_refs)
    if not ref_positions:
        return results

    # Fetch all items for the document to ensure we always detect section
    # structure correctly, regardless of where the match falls.
    item_count = await document_item_repository.get_item_count(document_id)
    window_items = await document_item_repository.get_items_in_range(
        document_id, 0, item_count
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

        lo, hi = _find_expansion_range(
            window_items, matched, has_sections, max_items, max_chars
        )
        ranges.append((lo, hi, result))

    merged = _merge_ranges(ranges)

    # Build results from the window items
    pos_to_item = {item.position: item for item in window_items}
    final_results: list[SearchResult] = []
    for range_start, range_end, original_results in merged:
        content_parts: list[str] = []
        refs: list[str] = []
        pages: set[int] = set()
        labels: set[str] = set()

        for pos in range(range_start, range_end + 1):
            item = pos_to_item.get(pos)
            if item is None:
                continue
            if has_sections and item.label in _NOISE_LABELS:
                continue
            if item.text:
                content_parts.append(item.text)
                refs.append(item.self_ref)
                if item.label:
                    labels.add(item.label)
                pages.update(item.page_numbers)

        all_headings: list[str] = []
        for r in original_results:
            if r.headings:
                all_headings.extend(h for h in r.headings if h not in all_headings)

        first = original_results[0]

        # If noise filtering removed all content, preserve the original
        expanded_content = "\n\n".join(content_parts)
        if not expanded_content:
            expanded_content = first.content

        final_results.append(
            SearchResult(
                content=expanded_content,
                score=max(r.score for r in original_results),
                chunk_id=first.chunk_id,
                document_id=first.document_id,
                document_uri=first.document_uri,
                document_title=first.document_title,
                doc_item_refs=refs or first.doc_item_refs,
                page_numbers=sorted(pages) or first.page_numbers,
                headings=all_headings or None,
                labels=sorted(labels) or first.labels,
            )
        )

    return final_results + passthrough
