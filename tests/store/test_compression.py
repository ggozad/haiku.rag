import json
from concurrent.futures import ThreadPoolExecutor

from haiku.rag.store.compression import (
    compress_docling_split,
    compress_json,
    decompress_json,
)


class TestJsonCompression:
    def test_compress_json_roundtrip(self):
        json_str = '{"key": "value", "number": 42, "nested": {"a": 1}}'
        compressed = compress_json(json_str)
        decompressed = decompress_json(compressed)
        assert decompressed == json_str

    def test_compress_json_with_unicode(self):
        json_str = '{"message": "Hello, 世界! 🌍"}'
        compressed = compress_json(json_str)
        decompressed = decompress_json(compressed)
        assert decompressed == json_str

    def test_compress_json_produces_zstd(self):
        compressed = compress_json('{"test": true}')
        assert compressed[:4] == b"\x28\xb5\x2f\xfd"

    def test_concurrent_compress_decompress_is_safe(self):
        """Compression must be safe under concurrent threads.

        Ingestion offloads compression to worker threads via
        asyncio.to_thread; sharing a single zstandard compressor/decompressor
        across threads corrupts its internal C context and segfaults the
        process. Hammer both paths from many threads to guard against a
        regression to module-level singletons.
        """
        payloads = [
            json.dumps({"i": i, "text": f"document body {i} " * 200}) for i in range(64)
        ]

        def roundtrip(json_str: str) -> str:
            return decompress_json(compress_json(json_str))

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(roundtrip, payloads * 8))

        assert results == (payloads * 8)


class TestDoclingCompressionSplit:
    def test_split_with_pages(self):
        data = {
            "name": "test_doc",
            "texts": [{"text": "hello"}],
            "pages": {"1": {"image": "base64data"}, "2": {"image": "more"}},
        }
        structure_bytes, pages_bytes = compress_docling_split(data)

        assert structure_bytes is not None
        assert pages_bytes is not None

        # Structure should not contain pages
        structure = json.loads(decompress_json(structure_bytes))
        assert "pages" not in structure
        assert structure["name"] == "test_doc"
        assert structure["texts"] == [{"text": "hello"}]

        # Pages should contain only pages
        pages = json.loads(decompress_json(pages_bytes))
        assert "1" in pages
        assert "2" in pages

    def test_split_without_pages(self):
        data = {"name": "test_doc", "texts": []}
        structure_bytes, pages_bytes = compress_docling_split(data)

        assert structure_bytes is not None
        assert pages_bytes is None

        structure = json.loads(decompress_json(structure_bytes))
        assert structure["name"] == "test_doc"

    def test_split_with_empty_pages(self):
        data = {"name": "test_doc", "texts": [], "pages": {}}
        structure_bytes, pages_bytes = compress_docling_split(data)

        assert structure_bytes is not None
        assert pages_bytes is None
