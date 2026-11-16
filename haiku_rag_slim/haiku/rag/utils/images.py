import base64
from collections.abc import Set as AbstractSet
from io import BytesIO
from typing import NamedTuple

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic_ai.messages import BinaryContent

from haiku.rag.store.models import SearchResult

# Formats whose converted pages are the file's own pages, so the file renders them.
ORIGINAL_EXTENSIONS = frozenset(
    {".pdf", ".bmp", ".jpeg", ".jpg", ".png", ".tiff", ".webp"}
)


class OriginalPage(NamedTuple):
    """A rendered page and its size in the structure's coordinates."""

    image: Image.Image
    width: float
    height: float


def render_original_pages(
    data: bytes, page_numbers: list[int], scale: float
) -> dict[int, OriginalPage]:
    """Render the requested 1-based pages of an original file.

    An image's frames are its pages, returned at their own size and oriented
    and flattened as docling reads them. Anything else renders as a PDF at
    ``scale`` pixels per point. Pages past the end are left out.
    """
    try:
        image = Image.open(BytesIO(data))
    except UnidentifiedImageError:
        return _render_pdf_pages(data, page_numbers, scale)
    with image:
        frames = getattr(image, "n_frames", 1)
        pages = {}
        for n in page_numbers:
            if 0 < n <= frames:
                image.seek(n - 1)
                pages[n] = _as_docling_reads(image)
        return pages


class OriginalLayout(NamedTuple):
    """Whether an original is a PDF, and each 1-based page's size in the
    structure's coordinates."""

    pdf: bool
    sizes: dict[int, tuple[float, float]]


def original_layout(data: bytes) -> OriginalLayout:
    """The page sizes from `render_original_pages`, without rendering a PDF."""
    try:
        image = Image.open(BytesIO(data))
    except UnidentifiedImageError:
        import pypdfium2 as pdfium

        from haiku.rag.converters.pdf_split import PDFIUM_LOCK

        with PDFIUM_LOCK:
            pdf = pdfium.PdfDocument(data)
            try:
                return OriginalLayout(
                    True, {n + 1: pdf.get_page_size(n) for n in range(len(pdf))}
                )
            finally:
                pdf.close()
    with image:
        sizes = {}
        for n in range(getattr(image, "n_frames", 1)):
            image.seek(n)
            sizes[n + 1] = _docling_size(image, ImageOps.exif_transpose(image))
        return OriginalLayout(False, sizes)


def _docling_size(frame: Image.Image, oriented: Image.Image) -> tuple[float, float]:
    """The oriented frame's size in points from its DPI, the axes swapped when
    orienting it turned it, as docling sizes it."""
    dpi = frame.info.get("dpi")
    dpi_x, dpi_y = (72.0, 72.0) if dpi in (None, (1, 1)) else map(float, dpi)
    if oriented.size != frame.size:
        dpi_x, dpi_y = dpi_y, dpi_x
    return oriented.width * 72 / dpi_x, oriented.height * 72 / dpi_y


def _as_docling_reads(frame: Image.Image) -> OriginalPage:
    """The frame oriented and flattened on white, sized as docling sizes it."""
    oriented = ImageOps.exif_transpose(frame)
    size = _docling_size(frame, oriented)
    if oriented.mode in ("RGBA", "LA", "PA") or "transparency" in oriented.info:
        rgba = oriented.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        oriented = Image.alpha_composite(background, rgba)
    return OriginalPage(oriented.convert("RGB"), *size)


def _render_pdf_pages(
    data: bytes, page_numbers: list[int], scale: float
) -> dict[int, OriginalPage]:
    import pypdfium2 as pdfium

    from haiku.rag.converters.pdf_split import PDFIUM_LOCK

    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(data)
        try:
            return {
                n: OriginalPage(
                    pdf[n - 1].render(scale=scale).to_pil().convert("RGB"),
                    *pdf[n - 1].get_size(),
                )
                for n in page_numbers
                if 0 < n <= len(pdf)
            }
        finally:
            pdf.close()


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
    return BinaryContent(
        data=data, media_type=image_media_type(data), identifier=self_ref
    )


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
