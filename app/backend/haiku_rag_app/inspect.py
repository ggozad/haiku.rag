import asyncio
import json
from typing import Any

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from haiku.rag.client import HaikuRAG
from haiku.rag.context import build_toc
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.schema import REQUIRED_TABLES
from haiku.rag.utils.sql import escape_sql_string, title_or_uri_filter
from haiku_rag_app.chat import MAX_IMAGE_BYTES, checked_image


async def _reader(request: Request) -> HaikuRAG:
    return await request.app.state.databases.reader(request.path_params["source"])


def _int(request: Request, name: str, default: int, most: int) -> int:
    raw = request.query_params.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise HTTPException(400, f"{name} must be a whole number") from error
    if not 0 <= value <= most:
        raise HTTPException(400, f"{name} must be between 0 and {most}")
    return value


def _document_filter(request: Request) -> str | None:
    """The title-or-URI text box and the raw `WHERE` clause, both applied."""
    clauses = [
        clause
        for clause in (
            title_or_uri_filter(request.query_params.get("q", "")),
            (request.query_params.get("filter") or "").strip() or None,
        )
        if clause
    ]
    return " AND ".join(f"({clause})" for clause in clauses) or None


async def _document(owner: HaikuRAG, document_id: str):
    document = await owner.document_repository.get_by_id(document_id)
    if document is None:
        raise HTTPException(404, f"no document {document_id!r}")
    return document


def _chunk_row(chunk: Chunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "order": chunk.order,
        "content": chunk.content,
        "metadata": chunk.metadata,
    }


async def list_documents(request: Request) -> JSONResponse:
    owner = await _reader(request)
    limit = _int(request, "limit", 50, 500)
    offset = _int(request, "offset", 0, 10**9)
    document_filter = _document_filter(request)
    try:
        documents = await owner.list_documents(limit, offset, document_filter)
        total = await owner.count_documents(document_filter)
    except Exception as error:
        if document_filter is None:
            raise HTTPException(503, str(error)) from error
        raise HTTPException(400, f"the filter failed: {error}") from error
    return JSONResponse(
        {
            "total": total,
            "documents": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "uri": doc.uri,
                    "source_id": doc.metadata.get("source_id"),
                    "content_type": doc.metadata.get("content_type"),
                    "created_at": doc.created_at.isoformat(),
                    "updated_at": doc.updated_at.isoformat(),
                }
                for doc in documents
            ],
        }
    )


async def get_document(request: Request) -> JSONResponse:
    owner = await _reader(request)
    document = await _document(owner, request.path_params["document_id"])
    return JSONResponse(
        {
            "id": document.id,
            "title": document.title,
            "uri": document.uri,
            "metadata": document.metadata,
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat(),
        }
    )


async def _items_and_index(owner: HaikuRAG, document_id: str):
    """The document's items in reading order, and for each item the chunks
    holding it in document order."""
    await _document(owner, document_id)
    items = await owner.document_item_repository.get_all_items(document_id)
    index = await owner.chunk_repository.get_chunk_ids_by_self_ref_grouped(
        [document_id]
    )
    rows = (
        await owner.store.chunks_table.query()
        .select(["id", "order"])
        .where(f"document_id = '{escape_sql_string(document_id)}'")
        .to_list()
    )
    order = {row["id"]: row["order"] for row in rows}
    chunk_index = {
        ref: sorted(ids, key=lambda chunk_id: order.get(chunk_id, 0))
        for ref, ids in index.get(document_id, {}).items()
    }
    return sorted(items, key=lambda item: item.position), chunk_index


async def document_items(request: Request) -> JSONResponse:
    """The document's items in reading order, each with the chunks holding it.

    A picture is referenced by its `self_ref`; its bytes come from the picture
    route.
    """
    owner = await _reader(request)
    items, chunk_index = await _items_and_index(
        owner, request.path_params["document_id"]
    )
    return JSONResponse(
        {
            "items": [
                {
                    "self_ref": item.self_ref,
                    "label": item.label,
                    "text": item.text,
                    "page_numbers": item.page_numbers,
                    "heading_level": item.heading_level,
                    "chunk_ids": chunk_index.get(item.self_ref, []),
                }
                for item in items
            ]
        }
    )


async def document_outline(request: Request) -> JSONResponse:
    owner = await _reader(request)
    items, chunk_index = await _items_and_index(
        owner, request.path_params["document_id"]
    )
    return JSONResponse({"outline": build_toc(items, chunk_index)})


async def document_chunks(request: Request) -> JSONResponse:
    owner = await _reader(request)
    document_id = request.path_params["document_id"]
    await _document(owner, document_id)
    limit = _int(request, "limit", 50, 500)
    offset = _int(request, "offset", 0, 10**9)
    chunks = await owner.chunk_repository.get_by_document_id(
        document_id, limit=limit, offset=offset
    )
    total = await owner.chunk_repository.count_by_document_id(document_id)
    return JSONResponse(
        {"total": total, "chunks": [_chunk_row(chunk) for chunk in chunks]}
    )


async def get_chunk(request: Request) -> JSONResponse:
    owner = await _reader(request)
    chunk = await owner.chunk_repository.get_by_id(request.path_params["chunk_id"])
    if chunk is None:
        raise HTTPException(404, f"no chunk {request.path_params['chunk_id']!r}")
    return JSONResponse(_chunk_row(chunk))


PREFIX = "/api/inspect/{source}"

routes = [
    Route(f"{PREFIX}/documents", list_documents, methods=["GET"]),
    Route(f"{PREFIX}/documents/{{document_id}}", get_document, methods=["GET"]),
    Route(f"{PREFIX}/documents/{{document_id}}/items", document_items, methods=["GET"]),
    Route(
        f"{PREFIX}/documents/{{document_id}}/outline",
        document_outline,
        methods=["GET"],
    ),
    Route(
        f"{PREFIX}/documents/{{document_id}}/chunks",
        document_chunks,
        methods=["GET"],
    ),
    Route(f"{PREFIX}/chunks/{{chunk_id}}", get_chunk, methods=["GET"]),
]


async def _chunk(owner: HaikuRAG, chunk_id: str) -> Chunk:
    chunk = await owner.chunk_repository.get_by_id(chunk_id)
    if chunk is None:
        raise HTTPException(404, f"no chunk {chunk_id!r}")
    return chunk


async def chunk_context(request: Request) -> JSONResponse:
    """The chunk as the agent receives it from a search: expanded to its
    context, with the pictures a search attaches."""
    from haiku.rag.client.search import populate_image_data
    from haiku.rag.store.models.chunk import SearchResult

    source = request.path_params["source"]
    owner = await _reader(request)
    chunk = await _chunk(owner, request.path_params["chunk_id"])
    assert chunk.document_id is not None
    document = await _document(owner, chunk.document_id)
    chunk.document_uri, chunk.document_title = document.uri, document.title
    chunk.document_meta = document.metadata
    result = SearchResult.from_chunk(chunk, score=1.0)
    result.source = source
    await populate_image_data(owner, [result])
    [expanded] = await owner.expand_context([result])
    return JSONResponse(
        {
            "rendering": expanded.format_for_agent(include_document_id=True),
            "pictures": sorted((expanded.image_data or {}).keys()),
            "document_id": chunk.document_id,
        }
    )


def _result_row(result) -> dict[str, Any]:
    return {
        "source": result.source,
        "chunk_id": result.chunk_id,
        "document_id": result.document_id,
        "document_title": result.document_title,
        "document_uri": result.document_uri,
        "score": result.score,
        "page_numbers": result.page_numbers,
        "headings": result.headings,
        "content": result.content,
    }


async def similar_chunks(request: Request) -> JSONResponse:
    """The chunks a search with this chunk's content returns, itself left out."""
    owner = await _reader(request)
    chunk = await _chunk(owner, request.path_params["chunk_id"])
    limit = _int(request, "limit", 10, 50)
    results = await owner.search(chunk.content, limit=limit + 1)
    for result in results:
        result.source = request.path_params["source"]
    return JSONResponse(
        {"results": [_result_row(r) for r in results if r.chunk_id != chunk.id][:limit]}
    )


async def _page_chunks(owner: HaikuRAG, document_id: str):
    """For each page an item covers, the chunk-held items on it and the chunks
    holding them, both empty on a page no chunk holds anything of."""
    items, chunk_index = await _items_and_index(owner, document_id)
    pages: dict[int, dict[str, Any]] = {}
    for item in items:
        held_by = chunk_index.get(item.self_ref, [])
        for page in item.page_numbers:
            entry = pages.setdefault(page, {"refs": [], "chunk_ids": [], "items": {}})
            if not held_by:
                continue
            entry["refs"].append(item.self_ref)
            entry["items"][item.self_ref] = item.page_numbers
            for chunk_id in held_by:
                if chunk_id not in entry["chunk_ids"]:
                    entry["chunk_ids"].append(chunk_id)
    return pages


async def document_pages(request: Request) -> JSONResponse:
    owner = await _reader(request)
    pages = await _page_chunks(owner, request.path_params["document_id"])
    return JSONResponse(
        {
            "pages": [
                {"page": page, "chunk_ids": pages[page]["chunk_ids"]}
                for page in sorted(pages)
            ]
        }
    )


async def page_image(request: Request) -> Response:
    """A page with every chunk on it drawn, as PNG."""
    from io import BytesIO

    owner = await _reader(request)
    document_id = request.path_params["document_id"]
    try:
        page = int(request.path_params["page"])
    except ValueError as error:
        raise HTTPException(400, "the page must be a whole number") from error
    pages = await _page_chunks(owner, document_id)
    if page not in pages:
        raise HTTPException(404, f"no chunk is drawn on page {page}")
    entry = pages[page]
    if not entry["chunk_ids"]:
        return await _plain_page(owner, document_id, page)
    chunks = [await _chunk(owner, chunk_id) for chunk_id in entry["chunk_ids"]]
    images = await owner.visualize_chunk(chunks, entry["refs"])
    # One image per page the drawn items touch, in page order.
    touched = sorted({p for numbers in entry["items"].values() for p in numbers})
    if len(images) != len(touched):
        raise HTTPException(404, f"page {page} has no image")
    buffer = BytesIO()
    images[touched.index(page)].save(buffer, format="PNG")
    return Response(buffer.getvalue(), media_type="image/png")


async def _plain_page(owner: HaikuRAG, document_id: str, page: int) -> Response:
    from io import BytesIO

    stored = await owner.document_repository.get_pages_data(document_id)
    item = stored.get_page_images([page]).get(page) if stored else None
    if item is None or item.image is None or item.image.pil_image is None:
        raise HTTPException(404, f"page {page} has no image")
    buffer = BytesIO()
    item.image.pil_image.save(buffer, format="PNG")
    return Response(buffer.getvalue(), media_type="image/png")


async def document_pictures(request: Request) -> JSONResponse:
    """The document's pictures with caption, description and holding chunk."""
    owner = await _reader(request)
    document_id = request.path_params["document_id"]
    _, chunk_index = await _items_and_index(owner, document_id)
    stored = await owner.document_repository.get_docling_data(document_id)
    doc = stored.get_docling_document() if stored else None
    pictures = (
        [
            {
                "self_ref": item.self_ref,
                "caption": item.caption_text(doc) or None,
                "description": item.meta.description.text
                if item.meta and item.meta.description
                else None,
                "page_numbers": sorted({p.page_no for p in item.prov}),
                "chunk_id": next(iter(chunk_index.get(item.self_ref, [])), None),
            }
            for item in doc.pictures
        ]
        if doc
        else []
    )
    return JSONResponse({"pictures": pictures})


# One base64 image at its limit, and room for the rest of the query.
MAX_SEARCH_BODY = (MAX_IMAGE_BYTES + 2) // 3 * 4 + 64 * 1024


async def search(request: Request) -> JSONResponse:
    """A search across the databases, by text or by image, with the agent's
    rendering of the expanded results."""
    raw = bytearray()
    async for part in request.stream():
        raw += part
        if len(raw) > MAX_SEARCH_BODY:
            raise HTTPException(413, f"a search body exceeds {MAX_SEARCH_BODY} bytes")
    try:
        body = json.loads(raw)
    except ValueError as error:
        raise HTTPException(400, "the body must be JSON") from error
    if not isinstance(body, dict):
        raise HTTPException(400, "the body must be a JSON object")
    databases = request.app.state.databases
    client = await databases.covering()
    limit = body.get("limit", 10)
    if not isinstance(limit, int) or not 1 <= limit <= 50:
        raise HTTPException(400, "limit must be between 1 and 50")
    search_type = body.get("search_type", "hybrid")
    if search_type not in ("hybrid", "vector", "fts"):
        raise HTTPException(400, "search_type is hybrid, vector or fts")
    sources = body.get("sources")
    if sources is not None and not (
        isinstance(sources, list) and all(isinstance(s, str) for s in sources)
    ):
        raise HTTPException(400, "sources must be a list of database names")
    query: str | bytes
    if body.get("image"):
        if not client.embedder.supports_images:
            raise HTTPException(400, "the embedder does not embed images")
        query = await asyncio.to_thread(checked_image, body["image"])
    elif isinstance(body.get("query"), str) and body["query"].strip():
        query = body["query"].strip()
    else:
        raise HTTPException(400, "give a query or an image")
    try:
        results = await client.search(
            query,
            limit=limit,
            search_type=search_type,
            filter=body.get("filter") or None,
            sources=sources,
        )
        expanded = await client.expand_context(results)
    except Exception as error:
        raise HTTPException(400, f"the search failed: {error}") from error
    multiple = databases.scope.covers_multiple
    return JSONResponse(
        {
            "results": [_result_row(r) for r in results],
            "rendering": "\n\n".join(
                r.format_for_agent(rank, len(expanded), include_collection=multiple)
                for rank, r in enumerate(expanded, start=1)
            ),
        }
    )


async def database_info(request: Request) -> JSONResponse:
    """What `haiku-rag info` reports, read without opening a store, and the
    tags and version history when the database opens."""
    from haiku.rag.store.info import gather_database_info

    name = request.path_params["source"]
    databases = request.app.state.databases
    ref = next((r for r in databases.scope.databases if r.name == name), None)
    if ref is None:
        raise HTTPException(404, f"unknown database {name!r}")
    info = await gather_database_info(ref.location, databases.config, tag=ref.tag)
    body: dict[str, Any] = {"info": info.model_dump(mode="json"), "tags": None}
    try:
        owner = await databases.reader(name)
        tags = await owner.store.list_tags()
        history = {}
        for table in REQUIRED_TABLES:
            versions = await owner.store.list_table_versions(table)
            history[table] = [
                {
                    "version": v["version"],
                    "timestamp": str(v["timestamp"]),
                    "tags": sorted(
                        tag
                        for tag, t in tags.items()
                        if t.tables.get(table) == v["version"]
                    ),
                }
                for v in sorted(versions, key=lambda v: v["version"], reverse=True)[:20]
            ]
        body["tags"] = {
            tag: {
                "complete": t.complete,
                "tables": t.tables,
                "missing_tables": t.missing_tables,
            }
            for tag, t in tags.items()
        }
        body["history"] = history
    except Exception as error:
        detail = error.detail if isinstance(error, HTTPException) else str(error)
        body["unavailable"] = detail
    return JSONResponse(body)


routes += [
    Route(f"{PREFIX}/chunks/{{chunk_id}}/context", chunk_context, methods=["GET"]),
    Route(f"{PREFIX}/chunks/{{chunk_id}}/similar", similar_chunks, methods=["GET"]),
    Route(f"{PREFIX}/documents/{{document_id}}/pages", document_pages, methods=["GET"]),
    Route(
        f"{PREFIX}/documents/{{document_id}}/pictures",
        document_pictures,
        methods=["GET"],
    ),
    Route(
        f"{PREFIX}/documents/{{document_id}}/pages/{{page}}",
        page_image,
        methods=["GET"],
    ),
    Route(f"{PREFIX}/info", database_info, methods=["GET"]),
    Route("/api/search", search, methods=["POST"]),
]
