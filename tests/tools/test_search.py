import pytest

from haiku.rag.store.models import SearchResult


def _png_b64(color: str = "red") -> str:
    import base64
    from io import BytesIO

    from PIL import Image as PILImage

    buf = BytesIO()
    PILImage.new("RGB", (4, 4), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TestBuildImageContentFromResults:
    """Picture bytes are attached once per (document, self_ref) pair, and labelled."""

    def test_results_without_image_data_contribute_nothing(self):
        from haiku.rag.tools.search import build_image_content_from_results

        results = [
            SearchResult(content="text only", score=0.5, chunk_id="c1", image_data=None)
        ]

        assert build_image_content_from_results(results) == ([], set())

    def test_duplicate_document_and_ref_is_attached_once(self):
        from pydantic_ai.messages import BinaryContent

        from haiku.rag.tools.search import build_image_content_from_results

        shared = {"#/pictures/0": _png_b64()}
        results = [
            SearchResult(
                content="a",
                score=0.9,
                chunk_id="c1",
                document_id="doc-1",
                image_data=shared,
            ),
            SearchResult(
                content="b",
                score=0.8,
                chunk_id="c2",
                document_id="doc-1",
                image_data=shared,
            ),
        ]

        content, _ = build_image_content_from_results(results)

        images = [item for item in content if isinstance(item, BinaryContent)]
        assert len(images) == 1

    def test_the_same_ref_in_different_documents_attaches_each_picture(self):
        import base64

        from pydantic_ai.messages import BinaryContent

        from haiku.rag.tools.search import build_image_content_from_results

        pictures = [_png_b64("red"), _png_b64("blue")]
        results = [
            SearchResult(
                content=f"Figure from document {index}",
                score=1.0,
                chunk_id=f"chunk-{index}",
                document_id=f"doc-{index}",
                image_data={"#/pictures/0": picture},
            )
            for index, picture in enumerate(pictures)
        ]

        content, emitted = build_image_content_from_results(results)

        images = [item for item in content if isinstance(item, BinaryContent)]
        assert [image.data for image in images] == [
            base64.b64decode(picture) for picture in pictures
        ]
        assert emitted == {
            (None, "doc-0", "#/pictures/0"),
            (None, "doc-1", "#/pictures/0"),
        }

    @pytest.mark.parametrize("include_valid", [True, False])
    def test_undecodable_pictures_emit_neither_images_nor_labels(
        self, include_valid: bool
    ):
        import base64

        from pydantic_ai.messages import BinaryContent

        from haiku.rag.tools.search import build_image_content_from_results

        valid = _png_b64()
        pictures = {"#/pictures/0": valid} if include_valid else {}
        pictures["#/pictures/1"] = base64.b64encode(
            b"\x89PNG\r\n\x1a\ngarbage"
        ).decode()
        result = SearchResult(
            content="Figures",
            score=1.0,
            chunk_id="c1",
            document_id="doc-1",
            image_data=pictures,
        )

        content, emitted = build_image_content_from_results([result])

        if include_valid:
            label, image = content
            assert isinstance(label, str)
            assert "#/pictures/0" in label
            assert "#/pictures/1" not in label
            assert "1 of 1" in label
            assert isinstance(image, BinaryContent)
            assert image.data == base64.b64decode(valid)
            assert image.identifier == "#/pictures/0"
            assert emitted == {(None, "doc-1", "#/pictures/0")}
        else:
            assert content == []
            assert emitted == set()

    @staticmethod
    def _one_picture_in_two_collections():
        """A document copied into another collection keeps its ids and refs."""
        shared = {"#/pictures/0": _png_b64()}
        return [
            SearchResult(
                content="a",
                score=0.9,
                chunk_id="c1",
                document_id="doc-1",
                source="papers",
                image_data=shared,
            ),
            SearchResult(
                content="a",
                score=0.8,
                chunk_id="c1",
                document_id="doc-1",
                source="wiki",
                image_data=shared,
            ),
        ]

    def test_the_same_picture_in_two_collections_is_attached_from_each(self):
        from pydantic_ai.messages import BinaryContent

        from haiku.rag.tools.search import build_image_content_from_results

        content, _ = build_image_content_from_results(
            self._one_picture_in_two_collections()
        )

        images = [item for item in content if isinstance(item, BinaryContent)]
        assert len(images) == 2

    def test_each_image_is_labelled_with_the_collection_it_came_from(self):
        """Nothing else tells the two apart: same chunk id, same document, same
        reference."""
        from haiku.rag.tools.search import build_image_content_from_results

        content, _ = build_image_content_from_results(
            self._one_picture_in_two_collections(), include_collection=True
        )

        labels = [item for item in content if isinstance(item, str)]
        assert "Collection: papers." in labels[0]
        assert "Collection: wiki." in labels[1]

    def test_an_unasked_for_collection_is_not_named_on_an_image(self):
        from haiku.rag.tools.search import build_image_content_from_results

        content, _ = build_image_content_from_results(
            self._one_picture_in_two_collections()
        )

        assert not [
            item for item in content if isinstance(item, str) and "Collection" in item
        ]

    def test_each_image_is_labelled_with_the_result_it_belongs_to(self):
        """Label every picture, not just the batch.

        ``ToolReturn.content`` reaches the model as a user-role message, and one
        leading note does not override that: with a single note on the wire,
        gemma4-26b still reasoned "the user also provided images in the prompt".
        A label adjacent to each picture also names the chunk to cite for it,
        which ``BinaryContent.identifier`` cannot do — it does not survive
        serialization to the vision API.
        """
        from pydantic_ai.messages import BinaryContent

        from haiku.rag.tools.search import build_image_content_from_results

        results = [
            SearchResult(
                content="a",
                score=0.9,
                chunk_id="c1",
                document_id="doc-1",
                image_data={"#/pictures/0": _png_b64()},
            ),
            SearchResult(
                content="b",
                score=0.8,
                chunk_id="c2",
                document_id="doc-2",
                image_data={"#/pictures/3": _png_b64()},
            ),
        ]

        content, _ = build_image_content_from_results(results)

        # label, image, label, image — each picture preceded by its own line.
        assert [type(item) is str for item in content] == [True, False, True, False]
        assert isinstance(content[1], BinaryContent)
        assert isinstance(content[3], BinaryContent)

        first, second = content[0], content[2]
        assert isinstance(first, str) and isinstance(second, str)
        assert "c1" in first and "#/pictures/0" in first
        assert "c2" in second and "#/pictures/3" in second
        assert "1 of 2" in first and "2 of 2" in second
        for label in (first, second):
            assert "not provided by the user" in label.lower()
