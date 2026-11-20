import json
from concurrent.futures import ThreadPoolExecutor

from haiku.rag.store.compression import (
    compress_docling,
    compress_json,
    decompress_json,
    recompress_png,
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


class TestDoclingCompression:
    def test_keeps_pages_without_their_images(self):
        data = {
            "name": "test_doc",
            "texts": [{"text": "hello"}],
            "pages": {
                "1": {
                    "page_no": 1,
                    "size": {"width": 612.0, "height": 792.0},
                    "image": {"uri": "data:image/png;base64,AAAA"},
                },
                "2": {"page_no": 2, "size": {"width": 792.0, "height": 612.0}},
            },
        }

        structure = json.loads(decompress_json(compress_docling(data)))

        assert structure == {
            "name": "test_doc",
            "texts": [{"text": "hello"}],
            "pages": {
                "1": {
                    "page_no": 1,
                    "size": {"width": 612.0, "height": 792.0},
                    "image": None,
                },
                "2": {
                    "page_no": 2,
                    "size": {"width": 792.0, "height": 612.0},
                    "image": None,
                },
            },
        }

    def test_without_pages(self):
        data = {"name": "test_doc", "texts": []}

        assert json.loads(decompress_json(compress_docling(data))) == data

    def test_drops_picture_images_and_keeps_other_entries(self):
        data = {
            "name": "test_doc",
            "pictures": [{"image": {"uri": "data:image/png;base64,AAAA"}}, None],
        }

        structure = json.loads(decompress_json(compress_docling(data)))

        assert structure["pictures"] == [{"image": None}, None]


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
