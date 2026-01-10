from haiku.rag.store.compression import compress_json, decompress_json


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
