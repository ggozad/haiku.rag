import asyncio
import base64
import json
from io import BytesIO
from typing import Any

from PIL import Image
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models.chunk import Chunk

MAX_IMAGE_EDGE = 2048
MAX_IMAGE_BYTES = 16 * 1024**2


def checked_image(value: Any, declared: str | None = None) -> bytes:
    """The bytes of a base64 image the page would send, decoded in full.

    Refuses anything else with a 400. `declared`, when given, must be the
    detected media type. Decoding is CPU work: run it off the request loop.
    """
    if not isinstance(value, str):
        raise HTTPException(400, "an image is not valid base64")
    if len(value) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
        raise HTTPException(400, f"an image exceeds {MAX_IMAGE_BYTES} bytes")
    try:
        raw = base64.b64decode(value, validate=True)
    except ValueError as error:
        raise HTTPException(400, "an image is not valid base64") from error
    try:
        with Image.open(BytesIO(raw)) as image:
            detected = Image.MIME.get(image.format or "")
            if detected is None or (declared is not None and detected != declared):
                raise HTTPException(
                    400, f"an image is {detected or image.format}, not {declared}"
                )
            if max(image.size) > MAX_IMAGE_EDGE:
                raise HTTPException(
                    400,
                    f"an image is larger than {MAX_IMAGE_EDGE} px on its long edge",
                )
            image.load()
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(400, "an image is not an image") from error
    return raw


async def picture(request: Request) -> Response:
    """A picture's stored bytes, with the media type read from them."""
    params = request.query_params
    document_id, ref = params.get("document_id"), params.get("ref")
    if not document_id or not ref:
        raise HTTPException(400, "document_id and ref are required")
    owner = await request.state.databases.reader(params.get("source"))
    data = await owner.get_picture_bytes(document_id, ref)
    if data is None:
        raise HTTPException(404, f"no picture {ref!r} in document {document_id!r}")
    try:
        with Image.open(BytesIO(data)) as image:
            media_type = Image.MIME.get(image.format or "")
    except OSError:
        media_type = None
    if media_type is None:
        raise HTTPException(415, "the stored picture is in no known image format")
    return Response(data, media_type=media_type)


async def visualize(request: Request) -> JSONResponse:
    """The boxes highlighting chunks, by page, with the original to draw them on.

    `chunk_ids` is comma-separated, the constituents of a merged citation, and
    `refs` the JSON list of the items the model saw, drawn instead of
    re-expanding. Box coordinates have a bottom-left origin, in the page size
    given beside them.
    """
    owner = await request.state.databases.reader(request.query_params.get("source"))
    refs: list[str] | None = None
    if refs_param := request.query_params.get("refs"):
        try:
            parsed = json.loads(refs_param)
        except ValueError as error:
            raise HTTPException(400, "refs must be a JSON list") from error
        if not isinstance(parsed, list):
            raise HTTPException(400, "refs must be a JSON list")
        refs = [str(ref) for ref in parsed]

    chunks = []
    for chunk_id in request.path_params["chunk_ids"].split(","):
        chunk = await owner.chunk_repository.get_by_id(chunk_id)
        if chunk is not None:
            chunks.append(chunk)
    if not chunks:
        raise HTTPException(404, "chunk not found")
    return await grounding_response(owner, chunks, refs)


async def grounding_response(
    owner: HaikuRAG, chunks: list[Chunk], refs: list[str] | None
) -> JSONResponse:
    """The chunks' boxes by page with their original, or why there are none."""
    from haiku.rag.utils.images import original_layout

    grounding = await owner.chunk_grounding(chunks, refs) if chunks else None
    if grounding is not None:
        layout = await asyncio.to_thread(original_layout, grounding.original)
        pages = [
            {
                "page": n,
                "width": layout.sizes[n][0],
                "height": layout.sizes[n][1],
                "boxes": [
                    {
                        "l": box.left,
                        "t": box.top,
                        "r": box.right,
                        "b": box.bottom,
                        "matched": matched,
                    }
                    for box, matched in grounding.pages[n]
                ],
            }
            for n in sorted(grounding.pages)
            if n in layout.sizes
        ]
        if pages:
            return JSONResponse(
                {
                    "source": owner.source,
                    "document_id": grounding.document_id,
                    "original": {"md5": grounding.original_md5, "pdf": layout.pdf},
                    "pages": pages,
                    "message": None,
                }
            )
    return JSONResponse(
        {
            "source": owner.source,
            "document_id": None,
            "original": None,
            "pages": [],
            "message": "No visual grounding available",
        }
    )


routes = [
    Route("/api/pictures", picture, methods=["GET"]),
    Route("/api/visualize/{chunk_ids}", visualize, methods=["GET"]),
]
