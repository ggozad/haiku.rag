import json

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


class TestDoclingCompressionSplit:
    def test_split_with_pages(self):
        data = {
            "name": "test_doc",
            "texts": [{"text": "hello"}],
            "pages": {"1": {"image": "base64data"}, "2": {"image": "more"}},
        }
        json_str = json.dumps(data)
        structure_bytes, pages_bytes = compress_docling_split(json_str)

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
        json_str = json.dumps(data)
        structure_bytes, pages_bytes = compress_docling_split(json_str)

        assert structure_bytes is not None
        assert pages_bytes is None

        structure = json.loads(decompress_json(structure_bytes))
        assert structure["name"] == "test_doc"

    def test_split_with_empty_pages(self):
        data = {"name": "test_doc", "texts": [], "pages": {}}
        json_str = json.dumps(data)
        structure_bytes, pages_bytes = compress_docling_split(json_str)

        assert structure_bytes is not None
        assert pages_bytes is None
