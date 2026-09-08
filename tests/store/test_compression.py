import json
from concurrent.futures import ThreadPoolExecutor

from haiku.rag.store.compression import (
    compress_docling_split,
    compress_json,
    decompress_json,
    recompress_png,
    recompress_png_data_uri,
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


def _cv2_page_png() -> bytes:
    """A PNG encoded the way docling-core does, with no compression level."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(0)
    page = np.full((400, 320, 3), 255, dtype=np.uint8)
    for y in range(20, 380, 18):
        page[y : y + 7, 20 : 20 + int(rng.integers(80, 280))] = 20
    ok, buffer = cv2.imencode(".png", page)
    assert ok
    return buffer.tobytes()


class TestPngRecompression:
    def test_recompresses_a_cv2_encoded_png(self):
        """The stored image shrinks without touching a pixel."""
        from io import BytesIO

        from PIL import Image

        original = _cv2_page_png()
        smaller = recompress_png(original)

        assert len(smaller) < len(original)
        with Image.open(BytesIO(original)) as a, Image.open(BytesIO(smaller)) as b:
            assert a.size == b.size
            assert a.convert("RGB").tobytes() == b.convert("RGB").tobytes()

    def test_is_idempotent(self):
        """Applying it to an already-recompressed image changes nothing."""
        once = recompress_png(_cv2_page_png())
        assert recompress_png(once) == once

    def test_keeps_unreadable_data(self):
        """Undecodable bytes are stored as they are."""
        assert recompress_png(b"not a png") == b"not a png"

    def test_keeps_a_non_png_image(self):
        """Only PNG is re-encoded."""
        from io import BytesIO

        from PIL import Image

        buffer = BytesIO()
        Image.new("RGB", (8, 8), "white").save(buffer, format="JPEG")
        jpeg = buffer.getvalue()
        assert recompress_png(jpeg) == jpeg

    def test_data_uri_payload_shrinks_and_prefix_survives(self):
        import base64

        raw = _cv2_page_png()
        uri = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
        out = recompress_png_data_uri(uri)

        assert out.startswith("data:image/png;base64,")
        assert len(base64.b64decode(out.partition(",")[2])) < len(raw)

    def test_data_uri_with_undecodable_payload_is_untouched(self):
        """A malformed payload is left alone."""
        uri = "data:image/png;base64,A"
        assert recompress_png_data_uri(uri) == uri

    def test_data_uri_that_cannot_shrink_is_returned_unchanged(self):
        """A payload PIL cannot read keeps its original base64 exactly."""
        import base64

        uri = "data:image/png;base64," + base64.b64encode(b"not a png").decode("ascii")
        assert recompress_png_data_uri(uri) == uri

    def test_data_uri_of_another_type_is_untouched(self):
        uri = "data:image/jpeg;base64,AAAA"
        assert recompress_png_data_uri(uri) == uri

    def test_file_uri_is_untouched(self):
        assert recompress_png_data_uri("file:///tmp/page.png") == "file:///tmp/page.png"

    def test_split_recompresses_page_images(self):
        """`compress_docling_split` shrinks the page blob it stores."""
        import base64

        raw = _cv2_page_png()
        uri = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
        data = {
            "name": "doc",
            "pages": {"1": {"page_no": 1, "size": {}, "image": {"uri": uri}}},
        }
        _, pages_bytes = compress_docling_split(data)

        assert pages_bytes is not None
        stored = json.loads(decompress_json(pages_bytes))["1"]["image"]["uri"]
        assert len(base64.b64decode(stored.partition(",")[2])) < len(raw)

    def test_split_tolerates_a_page_without_an_image(self):
        data = {"name": "doc", "pages": {"1": {"page_no": 1, "size": {}}}}
        _, pages_bytes = compress_docling_split(data)
        assert pages_bytes is not None
