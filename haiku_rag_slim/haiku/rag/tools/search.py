import base64
from collections.abc import Set as AbstractSet
from io import BytesIO

from PIL import Image
from pydantic_ai.messages import BinaryContent

from haiku.rag.store.models import SearchResult

RETRIEVED_IMAGE_TAG = "[haiku.rag/retrieved-image]"
"""Tag every label we attach to a retrieved picture ends with.

Identifies our own pictures on the wire without inferring ownership from position,
which is wrong as soon as two tools' results arrive in one request. Deliberately not
a phrase: a user writing "retrieved from the knowledge base for my report" above
their own picture had it removed, along with their text.
"""


PictureKey = tuple[str | None, str | None, str]
"""Identity of one attached picture: (source, document_id, self_ref).

``self_ref`` alone collides across documents, and a copy of a document in
another collection carries its own pictures.
"""


def picture_keys(result: SearchResult) -> frozenset[PictureKey]:
    """The identity of every picture this result carries."""
    return frozenset(
        (result.source, result.document_id, self_ref)
        for self_ref in (result.image_data or {})
    )


def decode_picture(data: bytes, self_ref: str) -> BinaryContent | None:
    """Wrap picture bytes for the wire, or return nothing if they will not decode.

    The model adapter renders one vision placeholder per ``BinaryContent``, so
    emitting one for an image the server cannot decode leaves the processor with an
    off-by-one count.
    """
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
    except Exception:
        return None
    return BinaryContent(data=data, media_type="image/png", identifier=self_ref)


def collect_pictures(
    results: list[SearchResult], exclude: AbstractSet[PictureKey] = frozenset()
) -> tuple[list[tuple[str | None, str | None, str, BinaryContent]], set[PictureKey]]:
    """Every distinct, decodable picture attached to ``results``, in order.

    Returns ``(source, chunk_id, self_ref, picture)`` per picture and the
    ``PictureKey`` of each. Dedup keyed on ``PictureKey`` so the same picture in
    different chunks is emitted once, and a copy in another collection is its
    own; ``exclude`` seeds that dedup with pictures already sent. Pictures that
    fail ``PIL.Image.verify()`` are skipped.
    """
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
    """Decode and validate picture bytes attached to search results, labelled.

    Returns the labelled content and the ``PictureKey`` of every picture it
    emitted, as ``collect_pictures`` decides them. An undecodable picture is
    skipped because the model adapter renders one vision placeholder per
    ``BinaryContent``, so emitting one for an image the server can't decode
    leaves the processor with an off-by-one count.

    Every picture is preceded by a line naming the result it belongs to.
    ``ToolReturn.content`` reaches the model as a user-role message, so
    retrieved pictures are otherwise indistinguishable from ones the user
    attached, and models narrate them as part of the question: unlabelled,
    gemma4-26b answered about a figure from an unrelated document, and with a
    single note ahead of the batch it still called them "images in the prompt".
    The label also names the chunk to cite for a figure, which
    ``BinaryContent.identifier`` cannot do — it does not survive serialization
    to the vision API.
    """
    collected, emitted = collect_pictures(results, exclude)
    content: list[str | BinaryContent] = []
    total = len(collected)
    for position, (source, chunk_id, self_ref, picture) in enumerate(collected, 1):
        collection = f"Collection: {source}. " if include_collection and source else ""
        content.append(
            f"Page image {position} of {total}, retrieved from the knowledge base "
            f"for search result [{chunk_id}] ({self_ref}). {collection}"
            f"Not provided by the user. {RETRIEVED_IMAGE_TAG}"
        )
        content.append(picture)
    return content, emitted
