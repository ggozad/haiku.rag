import base64
from typing import TYPE_CHECKING

from haiku.rag.store.models.chunk import Chunk, SearchResult, SearchType
from haiku.rag.store.models.document_item import PICTURE_REF_PREFIX

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from haiku.rag.client import HaikuRAG


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

    if isinstance(query, str):
        if search_type is None:
            search_type = "hybrid"

        reranker = client.reranker

        if reranker is None:
            chunk_results = await client.chunk_repository.search(
                query, limit, search_type, filter
            )
        else:
            search_limit = limit * 10
            raw_results = await client.chunk_repository.search(
                query, search_limit, search_type, filter
            )
            chunks = [chunk for chunk, _ in raw_results]
            chunk_results = await reranker.rerank(query, chunks, top_n=limit)
    else:
        from haiku.rag.embeddings import get_embedder

        embedder = get_embedder(client._config)
        if not embedder.supports_images:
            raise ValueError(
                "Image queries require a multimodal embedder. Configure "
                "provider='vllm' (or another image-capable provider)."
            )
        query_vector = await embedder.embed_image(query)
        chunk_results = await client.chunk_repository.search(
            query="",
            limit=limit,
            filter=filter,
            query_vector=query_vector,
        )

    results = [SearchResult.from_chunk(chunk, score) for chunk, score in chunk_results]
    results = _dedup_picture_chunks(results)

    if include_images:
        await _populate_image_data(client, results)

    return results


def _dedup_picture_chunks(results: list[SearchResult]) -> list[SearchResult]:
    """Collapse duplicate picture-only chunks to one result per ``self_ref``.

    A single picture can produce two chunks for the same self_ref: one whose
    vector is the text embedding of the picture's description, and one whose
    vector is the image embedding of the picture's bytes. Both can rank for
    the same query. When two results share a single picture self_ref as
    their only ref, keep the higher-scoring one. Wider chunks that span the
    picture plus surrounding items pass through untouched.
    """
    seen: dict[tuple[str | None, str], int] = {}
    keep: list[bool] = [True] * len(results)
    for i, r in enumerate(results):
        if len(r.doc_item_refs) == 1 and r.doc_item_refs[0].startswith(
            PICTURE_REF_PREFIX
        ):
            key = (r.document_id, r.doc_item_refs[0])
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

    Groups results by document_id and batches one picture-bytes lookup per
    document so a result set spanning N documents costs N reads, not one per
    picture. Only refs starting with ``PICTURE_REF_PREFIX`` are queried.
    """
    by_doc: dict[str, list[SearchResult]] = {}
    for r in results:
        if not r.document_id:
            continue
        if not any(ref.startswith(PICTURE_REF_PREFIX) for ref in r.doc_item_refs):
            continue
        by_doc.setdefault(r.document_id, []).append(r)

    for doc_id, doc_results in by_doc.items():
        wanted: list[str] = []
        seen: set[str] = set()
        for r in doc_results:
            for ref in r.doc_item_refs:
                if ref.startswith(PICTURE_REF_PREFIX) and ref not in seen:
                    wanted.append(ref)
                    seen.add(ref)
        if not wanted:
            continue
        bytes_by_ref = await client.document_item_repository.get_pictures_for_chunk(
            doc_id, wanted
        )
        if not bytes_by_ref:
            continue
        captions_by_ref = await client.document_item_repository.get_text_for_refs(
            doc_id, list(bytes_by_ref.keys())
        )
        for r in doc_results:
            attached: dict[str, str] = {}
            captions: dict[str, str] = {}
            for ref in r.doc_item_refs:
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


async def expand_context(
    client: "HaikuRAG",
    search_results: list[SearchResult],
) -> list[SearchResult]:
    """Expand search results with surrounding content from the document.

    Uses the document_items table for section-bounded expansion.
    See haiku.rag.context for the algorithm description.

    Results without doc_item_refs pass through unexpanded. This happens when
    chunks were created without docling metadata (e.g., custom chunks passed
    to import_document).
    """
    from haiku.rag.context import expand_with_items

    max_chars = client._config.search.max_context_chars

    # Group by document_id for efficient processing
    document_groups: dict[str | None, list[SearchResult]] = {}
    for result in search_results:
        doc_id = result.document_id
        if doc_id not in document_groups:
            document_groups[doc_id] = []
        document_groups[doc_id].append(result)

    expanded_results = []

    for doc_id, doc_results in document_groups.items():
        if doc_id is None:
            expanded_results.extend(doc_results)
            continue

        has_refs = any(r.doc_item_refs for r in doc_results)
        if not has_refs:
            expanded_results.extend(doc_results)
            continue

        expanded = await expand_with_items(
            client.document_item_repository,
            doc_id,
            doc_results,
            max_chars,
        )
        expanded_results.extend(expanded)

    expanded_results.sort(key=lambda r: r.score, reverse=True)
    # image_data and picture_captions are preserved through expansion by
    # expand_with_items — we deliberately do not re-attach bytes for refs
    # introduced by section expansion, so the multimodal payload stays
    # bounded by what was originally retrieved.
    return expanded_results


async def visualize_chunk(client: "HaikuRAG", chunk: Chunk) -> list:
    """Render page images with bounding box highlights for a chunk.

    Expands the chunk's context to find the full section, then resolves
    bounding boxes from all items in the expanded range. This ensures
    visualization covers all pages the expanded content spans.

    Returns a list of PIL Image objects, one per page with bounding boxes.
    Empty list if no bounding boxes or page images available.
    """
    from copy import deepcopy

    from PIL import ImageDraw

    from haiku.rag.store.models.chunk import ChunkMetadata

    if not chunk.document_id:
        return []

    doc = await client.document_repository.get_docling_data(chunk.document_id)
    if not doc:
        return []

    docling_doc = doc.get_docling_document()
    if not docling_doc:
        return []

    # Expand context to get all doc_item_refs in the section
    chunk_meta = chunk.get_chunk_metadata()
    if chunk_meta.doc_item_refs:
        search_result = SearchResult(
            content=chunk.content,
            score=1.0,
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            doc_item_refs=chunk_meta.doc_item_refs,
            page_numbers=chunk_meta.page_numbers,
        )
        expanded = await expand_context(client, [search_result])
        refs = expanded[0].doc_item_refs if expanded else chunk_meta.doc_item_refs
        meta = ChunkMetadata(doc_item_refs=refs)
    else:
        meta = chunk_meta
    bounding_boxes = meta.resolve_bounding_boxes(docling_doc)
    if not bounding_boxes:
        return []

    # Group bounding boxes by page
    boxes_by_page: dict[int, list] = {}
    for bbox in bounding_boxes:
        if bbox.page_no not in boxes_by_page:
            boxes_by_page[bbox.page_no] = []
        boxes_by_page[bbox.page_no].append(bbox)

    # Load only the needed page images
    pages_doc = await client.document_repository.get_pages_data(chunk.document_id)
    if not pages_doc:
        return []
    page_images = pages_doc.get_page_images(list(boxes_by_page.keys()))

    # Render each page with its bounding boxes
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

        for bbox in boxes_by_page[page_no]:
            # Document coords are bottom-left origin; PIL uses top-left
            x0 = bbox.left * scale_x
            y0 = (page_height - bbox.top) * scale_y
            x1 = bbox.right * scale_x
            y1 = (page_height - bbox.bottom) * scale_y

            if y0 > y1:
                y0, y1 = y1, y0

            fill_color = (255, 255, 0, 40)  # Yellow with transparency
            outline_color = (255, 165, 0, 100)  # Orange outline

            draw.rectangle([(x0, y0), (x1, y1)], fill=fill_color, outline=None)
            draw.rectangle([(x0, y0), (x1, y1)], outline=outline_color, width=1)

        images.append(image)

    return images
