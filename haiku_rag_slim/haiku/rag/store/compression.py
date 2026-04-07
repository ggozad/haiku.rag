import json

try:
    from compression.zstd import (  # ty: ignore[unresolved-import]
        compress as _zstd_compress,  # type: ignore[import-not-found]
    )
    from compression.zstd import (  # ty: ignore[unresolved-import]
        decompress as _zstd_decompress,  # type: ignore[import-not-found]
    )
except ImportError:
    from zstandard import ZstdCompressor, ZstdDecompressor

    _zstd_compressor = ZstdCompressor()
    _zstd_decompressor = ZstdDecompressor()

    def _zstd_compress(data: bytes) -> bytes:
        return _zstd_compressor.compress(data)

    def _zstd_decompress(data: bytes) -> bytes:
        return _zstd_decompressor.decompress(data, max_output_size=len(data) * 20)


def compress_json(json_str: str) -> bytes:
    """Compress a JSON string with zstd."""
    return _zstd_compress(json_str.encode("utf-8"))


def decompress_json(data: bytes) -> str:
    """Decompress zstd-compressed data to a JSON string."""
    return _zstd_decompress(data).decode("utf-8")


def compress_docling_split(json_str: str) -> tuple[bytes, bytes | None]:
    """Split a DoclingDocument JSON into structure and pages, compress both with zstd.

    Returns:
        Tuple of (structure_bytes, pages_bytes). pages_bytes is None if the
        document has no page images.
    """
    data = json.loads(json_str)
    pages = data.pop("pages", None)

    structure_bytes = _zstd_compress(json.dumps(data).encode("utf-8"))

    pages_bytes = None
    if pages:
        pages_bytes = _zstd_compress(json.dumps(pages).encode("utf-8"))

    return structure_bytes, pages_bytes
