from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from evaluations.config import (
    ConversationInput,
    DocumentPayload,
    RetrievalSample,
    Turn,
)
from tests.specs import stub_spec


class TestDatasetSpecDbPath:
    def test_override_path_takes_precedence(self) -> None:
        spec = stub_spec()
        override = Path("/tmp/custom.lancedb")
        assert spec.db_path(override) == override

    def test_default_uses_data_dir(self) -> None:
        spec = stub_spec(db_filename="mydb.lancedb")
        with patch(
            "haiku.rag.utils.paths.get_default_data_dir",
            return_value=Path("/home/user/.local/share/haiku.rag"),
        ):
            result = spec.db_path()
        assert result == Path(
            "/home/user/.local/share/haiku.rag/evaluations/dbs/mydb.lancedb"
        )


class TestDatasetSpecDefaults:
    def test_optional_fields_default_to_none(self) -> None:
        spec = stub_spec()
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

    @pytest.mark.parametrize(
        "turns",
        [
            [Turn(speaker="user", text="q"), Turn(speaker="agent", text="a")],
            [],
        ],
        ids=["ends-with-agent", "empty"],
    )
    def test_must_end_with_user_turn(self, turns: list[Turn]) -> None:
        with pytest.raises(ValidationError, match="user turn"):
            ConversationInput(turns=turns)


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
    """A run over `lancedb.databases` passes no path: a path names one
    database."""

    @pytest.mark.parametrize(
        ("databases", "path", "expected"),
        [
            ({"a": "/a.lancedb", "b": "/b.lancedb"}, None, True),
            ({"a": "/a.lancedb", "b": "/b.lancedb"}, Path("/chosen.lancedb"), False),
            ({"a": "/a.lancedb"}, None, True),
            (None, None, False),
        ],
        ids=[
            "configured-set",
            "named-path-overrides-the-set",
            "set-of-one-is-still-configured",
            "no-database-named",
        ],
    )
    def test_covers_the_configured_set(
        self, databases: dict[str, str] | None, path: Path | None, expected: bool
    ) -> None:
        from evaluations.datasets import DATASETS
        from haiku.rag.config.models import AppConfig, LanceDBConfig

        spec = next(iter(DATASETS.values()))
        config = (
            AppConfig(lancedb=LanceDBConfig(databases=databases))
            if databases
            else AppConfig()
        )

        assert spec.uses_configured_databases(config, path) is expected
