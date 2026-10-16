from typing import Any

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from haiku.rag.client import HaikuRAG
from haiku.rag.context import build_toc
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.utils.sql import escape_sql_string, title_or_uri_filter


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
