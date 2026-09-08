import json

try:  # pragma: no cover
    from compression.zstd import (  # ty: ignore[unresolved-import]
        compress as _zstd_compress,  # type: ignore[import-not-found]
    )
    from compression.zstd import (  # ty: ignore[unresolved-import]
        decompress as _zstd_decompress,  # type: ignore[import-not-found]
    )
except ImportError:
    from zstandard import ZstdCompressor, ZstdDecompressor, get_frame_parameters

    # ZstdCompressor/ZstdDecompressor are not thread-safe: each wraps a single
    # reused ZSTD_CCtx/ZSTD_DCtx, and concurrent .compress()/.decompress() calls
    # corrupt that context and segfault in the C backend. Ingestion drives this
    # path from multiple worker threads (asyncio.to_thread in
    # _prepare_document_from_docling), so construct a fresh instance per call
    # rather than sharing a module-level singleton.
    def _zstd_compress(data: bytes) -> bytes:
        return ZstdCompressor().compress(data)

    def _zstd_decompress(data: bytes) -> bytes:
        content_size = get_frame_parameters(data).content_size
        return ZstdDecompressor().decompress(data, max_output_size=content_size)


def compress_json(json_str: str) -> bytes:
    """Compress a JSON string with zstd."""
    return _zstd_compress(json_str.encode("utf-8"))


def decompress_json(data: bytes) -> str:
    """Decompress zstd-compressed data to a JSON string."""
    return _zstd_decompress(data).decode("utf-8")


# docling-core encodes page and picture images with `cv2.imencode(".png", ...)`,
# which passes no compression level, so OpenCV uses 1. Remove both callers of
# this once docling-project/docling-core#758 ships.
_PNG_COMPRESS_LEVEL = 6


def recompress_png(data: bytes) -> bytes:
    """Re-encode PNG bytes at zlib level 6, returning them unchanged on failure.

    Idempotent, and never larger than its input. Every exception returns the
    input: PIL raises `DecompressionBombError`, which derives from `Exception`
    rather than `OSError`, above `MAX_IMAGE_PIXELS`.
    """
    from io import BytesIO

    from PIL import Image

    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG":
                return data
            buffer = BytesIO()
            image.save(buffer, format="PNG", compress_level=_PNG_COMPRESS_LEVEL)
    except Exception:
        return data
    recompressed = buffer.getvalue()
    return recompressed if len(recompressed) < len(data) else data


def recompress_png_data_uri(uri: str) -> str:
    """Re-encode the PNG payload of a ``data:`` URI, or return it unchanged."""
    import base64

    if not uri.startswith("data:image/png;base64,"):
        return uri
    head, _, encoded = uri.partition(",")
    try:
        raw = base64.b64decode(encoded, validate=False)
    except ValueError:
        return uri
    recompressed = recompress_png(raw)
    if recompressed is raw:
        return uri
    return f"{head},{base64.b64encode(recompressed).decode('ascii')}"


def compress_docling_split(data: dict) -> tuple[bytes, bytes | None]:
    """Split a DoclingDocument dict into structure and pages, compress both with zstd.

    Picture image URIs are stripped from the structure blob — they are stored on
    the corresponding ``document_items.picture_data`` rows and don't need to be
    duplicated inside the structure JSON. ``ImageRef.uri`` is required when the
    field is present, so each picture's ``image`` is set to ``None`` rather than
    partially mutated to keep the JSON re-validating cleanly.

    Mutates ``data`` in place (pops ``pages``, nulls picture images); callers
    pass a freshly built dict (``model_dump`` / ``json.loads`` output), so this
    never touches a live DoclingDocument.

    Returns:
        Tuple of (structure_bytes, pages_bytes). pages_bytes is None if the
        document has no page images.
    """
    pages = data.pop("pages", None)

    for page in (pages or {}).values():
        image = page.get("image") if isinstance(page, dict) else None
        if isinstance(image, dict) and isinstance(image.get("uri"), str):
            image["uri"] = recompress_png_data_uri(image["uri"])

    for picture in data.get("pictures") or []:
        if isinstance(picture, dict):
            picture["image"] = None

    structure_bytes = compress_json(json.dumps(data))
    pages_bytes = compress_json(json.dumps(pages)) if pages else None

    return structure_bytes, pages_bytes
