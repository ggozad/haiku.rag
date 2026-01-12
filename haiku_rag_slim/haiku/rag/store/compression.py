import gzip


def compress_json(json_str: str) -> bytes:
    """Compress a JSON string with gzip."""
    return gzip.compress(json_str.encode("utf-8"))


def decompress_json(data: bytes) -> str:
    """Decompress gzip-compressed data to a JSON string."""
    return gzip.decompress(data).decode("utf-8")
