import gzip

# Gzip magic number: first two bytes of any gzip stream
_GZIP_MAGIC = b"\x1f\x8b"


def compress_json(json_str: str, *, enabled: bool = True) -> bytes:
    """Compress a JSON string, optionally with gzip.

    Args:
        json_str: The JSON string to compress.
        enabled: If False, returns raw UTF-8 bytes without compression.
    """
    data = json_str.encode("utf-8")
    if enabled:
        return gzip.compress(data)
    return data


def decompress_json(data: bytes) -> str:
    """Decompress data to a JSON string.

    Automatically detects gzip-compressed data via magic bytes,
    so it handles both compressed and uncompressed storage.
    """
    if data[:2] == _GZIP_MAGIC:
        return gzip.decompress(data).decode("utf-8")
    return data.decode("utf-8")
