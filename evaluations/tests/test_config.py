from pathlib import Path
from unittest.mock import patch

from evaluations.config import DatasetSpec, DocumentPayload, RetrievalSample


def _make_spec(**kwargs: object) -> DatasetSpec:
    defaults: dict[str, object] = {
        "key": "test",
        "db_filename": "test.lancedb",
        "document_loader": lambda: None,
        "document_mapper": lambda doc: None,
        "qa_loader": lambda: None,
        "qa_case_builder": lambda idx, doc: None,
    }
    defaults.update(kwargs)
    return DatasetSpec(**defaults)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


class TestDatasetSpecDbPath:
    def test_override_path_takes_precedence(self) -> None:
        spec = _make_spec()
        override = Path("/tmp/custom.lancedb")
        assert spec.db_path(override) == override

    def test_default_uses_data_dir(self) -> None:
        spec = _make_spec(db_filename="mydb.lancedb")
        with patch(
            "haiku.rag.utils.get_default_data_dir",
            return_value=Path("/home/user/.local/share/haiku.rag"),
        ):
            result = spec.db_path()
        assert result == Path(
            "/home/user/.local/share/haiku.rag/evaluations/dbs/mydb.lancedb"
        )

    def test_none_override_uses_default(self) -> None:
        spec = _make_spec(db_filename="other.lancedb")
        with patch(
            "haiku.rag.utils.get_default_data_dir",
            return_value=Path("/data"),
        ):
            result = spec.db_path(None)
        assert result == Path("/data/evaluations/dbs/other.lancedb")


class TestDatasetSpecDefaults:
    def test_optional_fields_default_to_none(self) -> None:
        spec = _make_spec()
        assert spec.retrieval_loader is None
        assert spec.retrieval_mapper is None
        assert spec.retrieval_evaluator is None
        assert spec.document_limit is None


class TestDocumentPayload:
    def test_defaults(self) -> None:
        payload = DocumentPayload(uri="test://doc")
        assert payload.content is None
        assert payload.title is None
        assert payload.metadata is None
        assert payload.format == "md"
        assert payload.source_path is None

    def test_all_fields(self) -> None:
        payload = DocumentPayload(
            uri="test://doc",
            content="hello",
            title="Title",
            metadata={"k": "v"},
            format="html",
            source_path=Path("/tmp/doc.pdf"),
        )
        assert payload.uri == "test://doc"
        assert payload.content == "hello"
        assert payload.source_path == Path("/tmp/doc.pdf")


class TestRetrievalSample:
    def test_defaults(self) -> None:
        sample = RetrievalSample(question="q?", expected_uris=("u1",))
        assert sample.skip is False
        assert sample.source_type is None

    def test_all_fields(self) -> None:
        sample = RetrievalSample(
            question="q?",
            expected_uris=("u1", "u2"),
            skip=True,
            source_type="image",
        )
        assert sample.skip is True
        assert sample.source_type == "image"
