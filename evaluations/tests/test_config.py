from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from evaluations.config import (
    ConversationInput,
    DatasetSpec,
    DocumentPayload,
    RetrievalSample,
    Turn,
)


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
        assert spec.retrieval_evaluators is None
        assert spec.citation_evaluator is None
        assert spec.document_limit is None
        assert spec.retrieval_limit == 5


class TestConversationInput:
    def _conversation(self) -> ConversationInput:
        return ConversationInput(
            turns=[
                Turn(speaker="user", text="who takes photos of planes?"),
                Turn(speaker="agent", text="Ground-to-air photographers."),
                Turn(speaker="user", text="No, I meant photos in the air."),
            ]
        )

    def test_question_is_last_turn(self) -> None:
        assert self._conversation().question == "No, I meant photos in the air."

    def test_prefix_excludes_last_turn(self) -> None:
        prefix = self._conversation().prefix
        assert [t.speaker for t in prefix] == ["user", "agent"]

    def test_transcript_renders_speaker_lines(self) -> None:
        assert self._conversation().transcript == (
            "user: who takes photos of planes?\n"
            "agent: Ground-to-air photographers.\n"
            "user: No, I meant photos in the air."
        )

    def test_single_turn_has_empty_prefix(self) -> None:
        conversation = ConversationInput(turns=[Turn(speaker="user", text="hi")])
        assert conversation.prefix == []
        assert conversation.question == "hi"

    def test_must_end_with_user_turn(self) -> None:
        with pytest.raises(ValidationError, match="user turn"):
            ConversationInput(
                turns=[
                    Turn(speaker="user", text="q"),
                    Turn(speaker="agent", text="a"),
                ]
            )

    def test_must_have_turns(self) -> None:
        with pytest.raises(ValidationError, match="user turn"):
            ConversationInput(turns=[])


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


class TestCoversASet:
    """A run over `lancedb.databases` must pass no path, since a path names one
    database and wins over the configured set."""

    def test_a_configured_set_is_covered(self):
        from haiku.rag.config.models import AppConfig, LanceDBConfig

        from evaluations.datasets import DATASETS

        spec = next(iter(DATASETS.values()))
        config = AppConfig(
            lancedb=LanceDBConfig(databases={"a": "/a.lancedb", "b": "/b.lancedb"})
        )

        assert spec.uses_configured_databases(config) is True

    def test_a_named_path_overrides_the_set(self):
        """`--db` is documented as an override, so it names the one database to
        evaluate even when the configuration names several."""
        from pathlib import Path as _Path

        from haiku.rag.config.models import AppConfig, LanceDBConfig

        from evaluations.datasets import DATASETS

        spec = next(iter(DATASETS.values()))
        config = AppConfig(
            lancedb=LanceDBConfig(databases={"a": "/a.lancedb", "b": "/b.lancedb"})
        )

        assert spec.uses_configured_databases(config, _Path("/chosen.lancedb")) is False

    def test_one_database_is_not_a_set(self):
        from haiku.rag.config.models import AppConfig

        from evaluations.datasets import DATASETS

        spec = next(iter(DATASETS.values()))

        assert spec.uses_configured_databases(AppConfig()) is False
