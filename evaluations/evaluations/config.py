from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset
from pydantic import BaseModel, model_validator
from pydantic_evals import Case
from pydantic_evals.evaluators import Evaluator


class Turn(BaseModel):
    speaker: Literal["user", "agent"]
    text: str


class ConversationInput(BaseModel):
    """A conversation prefix plus the final user question (the last turn)."""

    turns: list[Turn]

    @model_validator(mode="after")
    def _ends_with_user_turn(self) -> "ConversationInput":
        if not self.turns or self.turns[-1].speaker != "user":
            raise ValueError("conversation must end with a user turn")
        return self

    @property
    def question(self) -> str:
        return self.turns[-1].text

    @property
    def prefix(self) -> list[Turn]:
        return self.turns[:-1]

    @property
    def transcript(self) -> str:
        return "\n".join(f"{turn.speaker}: {turn.text}" for turn in self.turns)


class ScopedQuestion(BaseModel):
    """A question and the databases it may draw on.

    The task function receives only a case's inputs, never its metadata, so a
    per-case scope has to travel in the inputs. `sources=[]` covers no database
    and `None` covers every one the client covers.
    """

    question: str
    sources: list[str] | None = None


@dataclass
class DocumentPayload:
    uri: str
    content: str | None = None
    title: str | None = None
    metadata: dict[str, Any] | None = None
    format: str = "md"
    source_path: Path | None = None


@dataclass
class RetrievalSample:
    question: str
    expected_uris: tuple[str, ...]
    skip: bool = False
    source_type: str | None = None


DocumentLoader = Callable[[], Dataset]
DocumentMapper = Callable[[Mapping[str, Any]], DocumentPayload | None]
RetrievalLoader = Callable[[], Dataset]
RetrievalMapper = Callable[[Mapping[str, Any]], RetrievalSample | None]
CaseBuilder = Callable[[int, Mapping[str, Any]], Case[Any, Any, dict[str, Any]]]


@dataclass
class DatasetSpec:
    key: str
    db_filename: str
    document_loader: DocumentLoader
    document_mapper: DocumentMapper
    qa_loader: DocumentLoader
    qa_case_builder: CaseBuilder
    retrieval_loader: RetrievalLoader | None = None
    retrieval_mapper: RetrievalMapper | None = None
    retrieval_evaluators: list[Evaluator] | None = None
    citation_evaluator: Evaluator | None = None
    qa_evaluator: Evaluator | None = None
    document_limit: int | None = None
    retrieval_limit: int = 5
    ingest_batch_size: int | None = None
    live: bool = False
    compaction: bool = False
    experiment_metadata: dict[str, Any] | None = None
    # Called with the report's cases after the run prints, for datasets that
    # report something the shared summary cannot express (e.g. hard gates).
    report_hook: Callable[[list[Any]], None] | None = None

    def uses_configured_databases(
        self, config, override_path: Path | None = None
    ) -> bool:
        """Whether `lancedb.databases` places the databases to evaluate over.

        A path names one database and wins over the configuration, both when it
        comes from `--db` and when the client resolves it. True for a mapping of
        one, which is a configured database like any other and keeps its name.
        """
        return bool(config.lancedb.databases) and override_path is None

    def db_path(self, override_path: Path | None = None) -> Path:
        """Get the database path.

        Args:
            override_path: Optional path to override the default database location.

        Returns:
            The database path to use.
        """
        if override_path is not None:
            return override_path

        from haiku.rag.utils import get_default_data_dir

        data_dir = get_default_data_dir()
        return data_dir / "evaluations" / "dbs" / self.db_filename
