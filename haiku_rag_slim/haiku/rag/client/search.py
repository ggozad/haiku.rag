import base64
from collections.abc import Sequence
from typing import TYPE_CHECKING

from haiku.rag.store.models.chunk import (
    Chunk,
    SearchResult,
    SearchType,
    qualified_id,
)
from haiku.rag.store.models.document_item import PICTURE_REF_PREFIX
from haiku.rag.utils import gather_all

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from haiku.rag.client import HaikuRAG
    from haiku.rag.client.session import FederatedSession, SingleDatabaseSession


async def search(
    client: "HaikuRAG",
    query: "str | bytes | PILImage.Image",
    limit: int | None = None,
    search_type: SearchType | None = None,
    filter: str | None = None,
    include_images: bool = True,
) -> list[SearchResult]:
    """Search for relevant chunks with optional reranking.

    Args:
        client: The HaikuRAG client (provides config + chunk repository).
        query: Text (``str``) or image (``bytes`` / ``PIL.Image.Image``).
            Image queries require a multimodal embedder and run vector-only.
        limit: Maximum number of results to return. Defaults to config.search.limit.
        search_type: "vector", "fts", or "hybrid".
            Applicable only for text queries, where the default is "hybrid".
        filter: Optional SQL WHERE clause to filter documents before searching chunks.
        include_images: When True, populate ``SearchResult.image_data`` with
            base64 picture bytes for picture-labeled chunks.

    Returns:
        List of SearchResult objects ordered by relevance.
    """
    if limit is None:
        limit = client._config.search.limit

    resolved = _resolved_search_type(query, search_type)
    # The repository embeds late, so a filter matching nothing never embeds.
    query_vector = (
        None if isinstance(query, str) else await _embed_query(client, query, resolved)
    )
    candidates = await client.chunk_repository.search(
        query=query if isinstance(query, str) else "",
        limit=_fetch_limit(client, query, limit),
        search_type=resolved,
        filter=filter,
        query_vector=query_vector,
    )
    chunk_results = await _rank(client, query, candidates, limit)

    results = [SearchResult.from_chunk(chunk, score) for chunk, score in chunk_results]
    results = _dedup_picture_chunks(results)

    if include_images:
        await _populate_image_data(client, results)

    return results


async def search_sources(
    client: "HaikuRAG",
    query: "str | bytes | PILImage.Image",
    limit: int | None = None,
    search_type: SearchType | None = None,
    filter: str | None = None,
    include_images: bool = True,
    sources: list[str] | None = None,
) -> list[SearchResult]:
    """Search several databases and fuse their results into one ranked list.

    Fetch, fuse, truncate, then enrich: enrichment runs on the survivors through
    the database each came from, so its cost is that of a single-database
    search.
    """
    if limit is None:
        limit = client._config.search.limit

    names = list(client.source_names) if sources is None else list(sources)
    if not names:
        return []
    selected = await client.clients_for(names)
    if len(selected) == 1:
        # One database is an ordinary search: fusion would replace its hybrid
        # scores with ranks, and embedding up front would defeat the late embed.
        return await selected[0].search(
            query, limit, search_type, filter, include_images
        )
    resolved = _resolved_search_type(query, search_type)
    if resolved != "fts":
        client._require_one_embedder(selected)

    # One over-fetch decision, one query vector and one reranker for the whole
    # set: the databases share an embedder, and deciding per database would have
    # each consult its own reranker.
    fetch_limit = _fetch_limit(client, query, limit)
    query_vector = await _embed_query(selected[0], query, resolved)
    text = query if isinstance(query, str) else ""
    per_source = await gather_all(
        *(
            c.chunk_repository.search(
                query=text,
                limit=fetch_limit,
                search_type=resolved,
                filter=filter,
                query_vector=query_vector,
            )
            for c in selected
        )
    )

    ranked = await _fuse(client, selected, query, per_source, limit)

    results: list[SearchResult] = []
    for owner, chunk, score in ranked:
        result = SearchResult.from_chunk(chunk, score)
        result.source = owner.source
        results.append(result)
    results = _dedup_picture_chunks(results)

    if include_images:
        by_owner: dict[str, list[SearchResult]] = {}
        for result in results:
            if result.source:
                by_owner.setdefault(result.source, []).append(result)
        owners = await client.clients_for(list(by_owner))
        await gather_all(
            *(
                _populate_image_data(owner, by_owner[name])
                for name, owner in zip(by_owner, owners, strict=True)
            )
        )

    return results


async def _fuse(
    federator: "HaikuRAG",
    clients: list["HaikuRAG"],
    query: "str | bytes | PILImage.Image",
    per_source: list[list[tuple[Chunk, float]]],
    limit: int,
) -> list[tuple["HaikuRAG", Chunk, float]]:
    """One ranked list from several, keeping each candidate's owner.

    A configured reranker scores the union directly, which is what makes ranking
    across databases tractable: it compares query against document and does not
    care where a candidate came from. Without one, reciprocal rank fusion over the
    per-database rankings, since scores from separate indexes are not comparable.
    """
    owned = [
        (client, chunk, score)
        for client, candidates in zip(clients, per_source, strict=True)
        for chunk, score in candidates
    ]
    if not owned:
        return []

    # An image query has no text for a reranker to score against, and the check
    # precedes `reranker`, which builds the reranker on first access.
    if isinstance(query, str):
        reranker = federator.reranker
        if reranker is not None:
            chunks = [chunk for _, chunk, _ in owned]
            if federator._config.reranking.multimodal:
                await gather_all(
                    *(
                        _attach_picture_data(
                            c, [chunk for owner, chunk, _ in owned if owner is c]
                        )
                        for c in clients
                    )
                )
            reranked = await reranker.rerank(query, chunks, top_n=limit)
            # Identity, since chunk ids repeat between copies of a database.
            owner_of = {id(chunk): client for client, chunk, _ in owned}
            if any(id(chunk) not in owner_of for chunk, _ in reranked):
                raise ValueError(
                    f"{type(reranker).__name__} returned chunks that are not the "
                    "ones it was given, so the database each came from is lost; "
                    "a reranker must return objects from the list passed to it"
                )
            return [(owner_of[id(chunk)], chunk, score) for chunk, score in reranked]

    scored: list[tuple[float, HaikuRAG, Chunk]] = []
    for client, candidates in zip(clients, per_source, strict=True):
        for rank, (chunk, _) in enumerate(candidates):
            scored.append((1.0 / (_RRF_K + rank + 1), client, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(client, chunk, score) for score, client, chunk in scored[:limit]]


# Reciprocal rank fusion's smoothing constant, the value the literature uses.
_RRF_K = 60


# Candidates per requested result when a reranker will re-order them.
_RERANK_OVERFETCH = 10


def _fetch_limit(
    client: "HaikuRAG",
    query: "str | bytes | PILImage.Image",
    limit: int,
) -> int:
    """How many candidates to fetch per database.

    Only a text query with a reranker over-fetches: an image query keeps its
    vector ranking, and the type is checked before `reranker`, which loads model
    weights for a local one on first access.
    """
    if not isinstance(query, str):
        return limit
    return limit * _RERANK_OVERFETCH if client.reranker else limit


def _resolved_search_type(
    query: "str | bytes | PILImage.Image", search_type: SearchType | None
) -> SearchType:
    """The search actually run for this query.

    An image query has no text to match against, so it is vector-only whatever
    the caller asked for; a text query defaults to hybrid.
    """
    if not isinstance(query, str):
        return "vector"
    return search_type or "hybrid"


async def _embed_query(
    client: "HaikuRAG", query: "str | bytes | PILImage.Image", search_type: SearchType
) -> list[float] | None:
    """The query as a vector, or None when the search needs no vector.

    The caller computes it once for however many databases the search covers:
    the databases in a selection share an embedder. `search_type` is the
    resolved one, so only a text query ever reaches this as full-text.
    """
    if search_type == "fts":
        return None
    if isinstance(query, str):
        return await client.embedder.embed_query(query)

    embedder = client.embedder
    if not embedder.supports_images:
        raise ValueError(
            "Image queries require a multimodal embedder. Set "
            "embeddings.model.multimodal: true on a vllm, voyageai, or cohere "
            "model."
        )
    return await embedder.embed_image(query)


async def _rank(
    client: "HaikuRAG",
    query: "str | bytes | PILImage.Image",
    candidates: list[tuple[Chunk, float]],
    limit: int,
) -> list[tuple[Chunk, float]]:
    """Order candidates and cut them to `limit`.

    An image query carries no text for a reranker to score against, so its
    candidates keep the vector ranking. Its type is checked before
    `client.reranker`, which builds the reranker on first access and loads model
    weights for a local one.
    """
    if not isinstance(query, str):
        return candidates[:limit]

    reranker = client.reranker
    if reranker is None:
        return candidates[:limit]

    chunks = [chunk for chunk, _ in candidates]
    if client._config.reranking.multimodal:
        await _attach_picture_data(client, chunks)
    return await reranker.rerank(query, chunks, top_n=limit)


async def _attach_picture_data(client: "HaikuRAG", chunks: list[Chunk]) -> None:
    """Attach picture bytes to synthetic picture chunks in-place; a multimodal
    reranker scores the pixels.

    One query however many documents the candidates span, which matters here
    more than anywhere: reranking fetches `limit * 10` candidates.
    """
    by_doc: dict[str, list[tuple[Chunk, str]]] = {}
    for chunk in chunks:
        if chunk.document_id is None:
            continue
        refs = chunk.get_chunk_metadata().doc_item_refs
        if len(refs) == 1 and refs[0].startswith(PICTURE_REF_PREFIX):
            by_doc.setdefault(chunk.document_id, []).append((chunk, refs[0]))

    bytes_by_document, _ = await client.document_item_repository.get_pictures_grouped(
        {doc_id: [ref for _, ref in pairs] for doc_id, pairs in by_doc.items()}
    )
    for doc_id, doc_chunks in by_doc.items():
        bytes_by_ref = bytes_by_document.get(doc_id, {})
        for chunk, ref in doc_chunks:
            data = bytes_by_ref.get(ref)
            if data:
                chunk._picture_data = data


def _dedup_picture_chunks(results: list[SearchResult]) -> list[SearchResult]:
    """Collapse duplicate picture-only chunks to one result per ``self_ref``.

    Keyed by database as well, since a database copied from another holds the same
    document id: collapsing across them would drop one of two real results.

    A single picture can produce two chunks for the same self_ref: one whose
    vector is the text embedding of the picture's description, and one whose
    vector is the image embedding of the picture's bytes. Both can rank for
    the same query. When two results share a single picture self_ref as
    their only ref, keep the higher-scoring one. Wider chunks that span the
    picture plus surrounding items pass through untouched.
    """
    seen: dict[tuple[str | None, str | None, str], int] = {}
    keep: list[bool] = [True] * len(results)
    for i, r in enumerate(results):
        if len(r.doc_item_refs) == 1 and r.doc_item_refs[0].startswith(
            PICTURE_REF_PREFIX
        ):
            key = (r.source, r.document_id, r.doc_item_refs[0])
            prior = seen.get(key)
            if prior is None:
                seen[key] = i
            elif r.score > results[prior].score:
                keep[prior] = False
                seen[key] = i
            else:
                keep[i] = False
    return [r for r, k in zip(results, keep) if k]


async def _populate_image_data(client: "HaikuRAG", results: list[SearchResult]) -> None:
    """Attach base64 picture bytes to ``SearchResult.image_data`` in-place.

    A result carries a picture when its refs include the picture directly, or
    when they include the picture's caption — the common case where a prose
    chunk carrying a figure's caption ranks while the picture is its own chunk.
    Costs a fixed number of reads however many documents the result set spans.
    """
    repo = client.document_item_repository
    by_doc: dict[str, list[SearchResult]] = {}
    for r in results:
        if r.document_id and r.doc_item_refs:
            by_doc.setdefault(r.document_id, []).append(r)
    if not by_doc:
        return

    refs_by_document = {
        doc_id: list({ref for r in doc_results for ref in r.doc_item_refs})
        for doc_id, doc_results in by_doc.items()
    }
    captions_to_pictures = await repo.get_caption_picture_refs_grouped(refs_by_document)

    # Which pictures each result wants, and which to fetch per document.
    result_pictures: list[tuple[SearchResult, list[str]]] = []
    wanted: dict[str, list[str]] = {}
    for doc_id, doc_results in by_doc.items():
        caption_to_picture = captions_to_pictures.get(doc_id, {})
        seen: set[str] = set()
        for r in doc_results:
            pictures: list[str] = []
            for ref in r.doc_item_refs:
                picture = (
                    ref
                    if ref.startswith(PICTURE_REF_PREFIX)
                    else caption_to_picture.get(ref)
                )
                if picture and picture not in pictures:
                    pictures.append(picture)
            if pictures:
                result_pictures.append((r, pictures))
                for picture in pictures:
                    if picture not in seen:
                        wanted.setdefault(doc_id, []).append(picture)
                        seen.add(picture)
    if not wanted:
        return

    bytes_by_document, captions_by_document = await repo.get_pictures_grouped(
        wanted, with_text=True
    )
    if not bytes_by_document:
        return

    for r, pictures in result_pictures:
        bytes_by_ref = bytes_by_document.get(r.document_id or "", {})
        captions_by_ref = captions_by_document.get(r.document_id or "", {})
        attached: dict[str, str] = {}
        captions: dict[str, str] = {}
        for ref in pictures:
            blob = bytes_by_ref.get(ref)
            if blob:
                attached[ref] = base64.b64encode(blob).decode("ascii")
                caption = captions_by_ref.get(ref)
                if caption:
                    captions[ref] = caption
        if attached:
            r.image_data = attached
        if captions:
            r.picture_captions = captions


async def expand_sources(
    federated: "FederatedSession",
    search_results: list[SearchResult],
) -> list[SearchResult]:
    """Expand results drawn from several databases, each through its own.

    A result naming no database passes through unexpanded: it cannot be placed,
    which is the case for results a caller built by hand.
    """
    by_source: dict[str, list[SearchResult]] = {}
    unsourced: list[SearchResult] = []
    for result in search_results:
        if result.source:
            by_source.setdefault(result.source, []).append(result)
        else:
            unsourced.append(result)
    names = list(by_source)
    sessions = await federated.sessions_for(names)
    expanded_groups = await gather_all(
        *(
            expand_context(session, by_source[name])
            for name, session in zip(names, sessions, strict=True)
        )
    )
    merged = unsourced + [r for group in expanded_groups for r in group]
    # Grouping by database must not become the tiebreak: fused scores tie often,
    # so equal scores keep the order they were fused in.
    arrival = {
        qualified_id(result.source, result.chunk_id): rank
        for rank, result in enumerate(search_results)
        if result.chunk_id
    }

    def fused_rank(result: SearchResult) -> int:
        return min(
            (
                arrival[key]
                for cid in (result.chunk_id, *result.chunk_ids)
                if (key := qualified_id(result.source, cid)) in arrival
            ),
            default=len(arrival),
        )

    merged.sort(key=lambda r: (-r.score, fused_rank(r)))
    return merged


async def expand_context(
    session: "SingleDatabaseSession",
    search_results: list[SearchResult],
) -> list[SearchResult]:
    """Expand search results with surrounding content from the document.

    Uses the document_items table for section-bounded expansion.
    See haiku.rag.context for the algorithm description.

    Results without doc_item_refs pass through unexpanded. This happens when
    chunks were created without docling metadata (e.g., custom chunks passed
    to import_document).
    """
    from haiku.rag.context import expand_with_items, window_for

    max_chars = session.config.search.max_context_chars

    # Group by document_id for efficient processing
    document_groups: dict[str | None, list[SearchResult]] = {}
    for result in search_results:
        doc_id = result.document_id
        if doc_id not in document_groups:
            document_groups[doc_id] = []
        document_groups[doc_id].append(result)

    expanded_results = []
    expandable = {
        doc_id: doc_results
        for doc_id, doc_results in document_groups.items()
        if doc_id is not None and any(r.doc_item_refs for r in doc_results)
    }
    repo = session.document_item_repository
    positions_by_document = await repo.resolve_refs_grouped(
        {
            doc_id: [ref for r in doc_results for ref in r.doc_item_refs]
            for doc_id, doc_results in expandable.items()
        }
    )
    windows = {
        doc_id: window_for(positions)
        for doc_id, positions in positions_by_document.items()
        if positions
    }
    items_by_document = await repo.get_items_in_ranges(windows)

    # In document_groups order: the score sort below is stable, so assembling
    # expandable and passthrough documents in separate passes would reorder
    # equal-scored results.
    for doc_id, doc_results in document_groups.items():
        if doc_id not in expandable:
            expanded_results.extend(doc_results)
            continue
        expanded_results.extend(
            expand_with_items(
                doc_results,
                max_chars,
                positions_by_document.get(doc_id, {}),
                items_by_document.get(doc_id, []),
            )
        )

    expanded_results.sort(key=lambda r: r.score, reverse=True)
    # image_data and picture_captions are preserved through expansion by
    # expand_with_items — we deliberately do not re-attach bytes for refs
    # introduced by section expansion, so the multimodal payload stays
    # bounded by what was originally retrieved.
    return expanded_results


async def visualize_chunk(
    session: "SingleDatabaseSession",
    chunk: "Chunk | Sequence[Chunk]",
    refs: list[str] | None = None,
    expand: bool = True,
) -> list:
    """Render page images with bounding box highlights for one or more chunks.

    When ``refs`` is given (the ``doc_item_refs`` of the citation, i.e. the
    exact items the model saw), bounding boxes are resolved from them directly
    so the visualization matches the cited context precisely. Otherwise, with
    ``expand=True`` (default) the chunks' context is re-expanded to recover the
    surrounding section; with ``expand=False`` only the chunks' own items are
    drawn, so the visualization shows just the retrieved chunk with no context.

    The chunks' own items draw in a strong highlight; the remaining items draw
    fainter, so the matched content stands out from its surrounding context.
    Chunks from a different document than the first are ignored.

    Returns a list of PIL Image objects, one per page with bounding boxes.
    Empty list if no bounding boxes or page images available.
    """
    from copy import deepcopy

    from PIL import ImageDraw

    from haiku.rag.store.models.chunk import ChunkMetadata

    chunks = [chunk] if isinstance(chunk, Chunk) else list(chunk)
    if not chunks:
        return []
    document_id = chunks[0].document_id
    if not document_id:
        return []
    chunks = [c for c in chunks if c.document_id == document_id]

    doc = await session.document_repository.get_docling_data(document_id)
    if not doc:
        return []

    docling_doc = doc.get_docling_document()
    if not docling_doc:
        return []

    matched_refs = {r for c in chunks for r in c.get_chunk_metadata().doc_item_refs}

    if refs is not None:
        all_refs = list(refs)
    elif not expand:
        # Chunk-only: draw just the retrieved chunks' own items, no context.
        all_refs = list(matched_refs)
    else:
        # No stored context: re-expand the chunks to recover their section.
        search_results = [
            SearchResult(
                content=c.content,
                score=1.0,
                chunk_id=c.id,
                document_id=c.document_id,
                doc_item_refs=meta.doc_item_refs,
                page_numbers=meta.page_numbers,
            )
            for c in chunks
            if (meta := c.get_chunk_metadata()).doc_item_refs
        ]
        if search_results:
            expanded = await expand_context(session, search_results)
            all_refs = []
            for result in expanded:
                all_refs.extend(r for r in result.doc_item_refs if r not in all_refs)
            if not all_refs:
                all_refs = [r for sr in search_results for r in sr.doc_item_refs]
        else:
            all_refs = list(chunks[0].get_chunk_metadata().doc_item_refs)

    matched_draw = [r for r in all_refs if r in matched_refs]
    swept_refs = [r for r in all_refs if r not in matched_refs]

    matched_boxes = ChunkMetadata(doc_item_refs=matched_draw).resolve_bounding_boxes(
        docling_doc
    )
    swept_boxes = ChunkMetadata(doc_item_refs=swept_refs).resolve_bounding_boxes(
        docling_doc
    )
    if not matched_boxes and not swept_boxes:
        return []

    # Group bounding boxes by page; swept boxes first so matched draw on top
    boxes_by_page: dict[int, list] = {}
    for bbox, is_matched in [(b, False) for b in swept_boxes] + [
        (b, True) for b in matched_boxes
    ]:
        if bbox.page_no not in boxes_by_page:
            boxes_by_page[bbox.page_no] = []
        boxes_by_page[bbox.page_no].append((bbox, is_matched))

    # Load only the needed page images
    pages_doc = await session.document_repository.get_pages_data(document_id)
    if not pages_doc:
        return []
    page_images = pages_doc.get_page_images(list(boxes_by_page.keys()))

    images = []
    for page_no in sorted(boxes_by_page.keys()):
        if page_no not in page_images:
            continue

        page = page_images[page_no]
        if page.image is None or page.image.pil_image is None:
            continue

        pil_image = page.image.pil_image
        page_height = page.size.height

        # Scale factor: image pixels vs document coordinates
        scale_x = pil_image.width / page.size.width
        scale_y = pil_image.height / page.size.height

        image = deepcopy(pil_image)
        draw = ImageDraw.Draw(image, "RGBA")

        for bbox, is_matched in boxes_by_page[page_no]:
            # Document coords are bottom-left origin; PIL uses top-left
            x0 = bbox.left * scale_x
            y0 = (page_height - bbox.top) * scale_y
            x1 = bbox.right * scale_x
            y1 = (page_height - bbox.bottom) * scale_y

            if y0 > y1:
                y0, y1 = y1, y0

            if is_matched:
                fill_color = (255, 150, 0, 55)  # Orange, matched content
                outline_color = (240, 130, 0, 150)  # Orange outline
            else:
                fill_color = (255, 255, 0, 40)  # Yellow, surrounding context
                outline_color = (255, 165, 0, 100)

            draw.rectangle([(x0, y0), (x1, y1)], fill=fill_color, outline=None)
            draw.rectangle([(x0, y0), (x1, y1)], outline=outline_color, width=1)

        images.append(image)

    return images
