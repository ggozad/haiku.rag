import base64
from collections.abc import Set as AbstractSet
from io import BytesIO

from PIL import Image
from pydantic_ai.messages import BinaryContent

from haiku.rag.store.models import SearchResult


def image_media_type(data: bytes) -> str:
    """Media type of raw image bytes, PNG for anything unrecognized."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def image_data_uri(data: bytes) -> str:
    """Raw image bytes as a ``data:`` URI carrying their sniffed media type."""
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{image_media_type(data)};base64,{encoded}"


def image_binary_content(data: bytes) -> "BinaryContent":
    """Wrap raw image bytes as BinaryContent with the sniffed media type."""
    from io import BytesIO

    from PIL import Image as PILImage
    from PIL import UnidentifiedImageError
    from pydantic_ai.messages import BinaryContent

    try:
        fmt = PILImage.open(BytesIO(data)).format or "PNG"
    except UnidentifiedImageError as e:
        raise ValueError("data is not a recognizable image") from e
    return BinaryContent(data=data, media_type=f"image/{fmt.lower()}")


RETRIEVED_IMAGE_TAG = "[haiku.rag/retrieved-image]"
CITED_EVIDENCE = "for cited evidence"
"""Marks the label of a picture cited in an earlier question."""
PictureKey = tuple[str | None, str | None, str]


def picture_keys(result: SearchResult) -> frozenset[PictureKey]:
    """The identity of every picture this result carries."""
    return frozenset(
        (result.source, result.document_id, self_ref)
        for self_ref in (result.image_data or {})
    )


def decode_picture(data: bytes, self_ref: str) -> BinaryContent | None:
    """Wrap picture bytes for the wire, or return nothing if they will not decode."""
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
    except Exception:
        return None
    return BinaryContent(data=data, media_type="image/png", identifier=self_ref)


def collect_pictures(
    results: list[SearchResult], exclude: AbstractSet[PictureKey] = frozenset()
) -> tuple[list[tuple[str | None, str | None, str, BinaryContent]], set[PictureKey]]:
    """Every distinct, decodable picture attached to ``results``, in order."""
    collected: list[tuple[str | None, str | None, str, BinaryContent]] = []
    seen: set[PictureKey] = set(exclude)
    emitted: set[PictureKey] = set()
    for result in results:
        if not result.image_data:
            continue
        for self_ref, b64 in result.image_data.items():
            key = (result.source, result.document_id, self_ref)
            if key in seen:
                continue
            picture = decode_picture(base64.b64decode(b64), self_ref)
            if picture is None:
                continue
            collected.append((result.source, result.chunk_id, self_ref, picture))
            seen.add(key)
            emitted.add(key)
    return collected, emitted


def build_image_content_from_results(
    results: list[SearchResult],
    include_collection: bool = False,
    exclude: AbstractSet[PictureKey] = frozenset(),
) -> tuple[list[str | BinaryContent], set[PictureKey]]:
    """Decode and validate picture bytes attached to search results, labelled."""
    from haiku.rag.utils.formatting import format_time_range

    collected, emitted = collect_pictures(results, exclude)
    frame_times = {
        (r.source, r.chunk_id, ref): at
        for r in results
        for ref, at in r.picture_times.items()
    }
    content: list[str | BinaryContent] = []
    total = len(collected)
    for position, (source, chunk_id, self_ref, picture) in enumerate(collected, 1):
        collection = f"Collection: {source}. " if include_collection and source else ""
        at = frame_times.get((source, chunk_id, self_ref))
        kind = (
            f"Video frame at {format_time_range((at, at))}, {position} of {total}"
            if at is not None
            else f"Page image {position} of {total}"
        )
        content.append(
            f"{kind}, retrieved from the knowledge base "
            f"for search result [{chunk_id}] ({self_ref}). {collection}"
            f"Not provided by the user. {RETRIEVED_IMAGE_TAG}"
        )
        content.append(picture)
    return content, emitted
