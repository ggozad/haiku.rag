import json

try:  # pragma: no cover
    from compression.zstd import (  # ty: ignore[import-not-found]
        compress as _zstd_compress,  # type: ignore[import-not-found]
    )
    from compression.zstd import (  # ty: ignore[import-not-found]
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


# docling-core encodes picture images with `cv2.imencode(".png", ...)`, which
# passes no compression level, so OpenCV uses 1. Remove the caller of this once
# docling-project/docling-core#758 ships.
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


def compress_docling(data: dict) -> bytes:
    """Compress a DoclingDocument dict with zstd, without its images.

    Pages keep their size and lose their raster. Picture images are stored on
    the ``document_items.picture_data`` rows. ``ImageRef.uri`` is required when
    the field is present, so each ``image`` is set to ``None`` rather than
    partially mutated, which keeps the JSON re-validating.

    Mutates ``data`` in place: callers pass a freshly built dict
    (``model_dump`` / ``json.loads`` output), never a live DoclingDocument.
    """
    for page in (data.get("pages") or {}).values():
        page["image"] = None

    for picture in data.get("pictures") or []:
        if isinstance(picture, dict):
            picture["image"] = None

    return compress_json(json.dumps(data))
