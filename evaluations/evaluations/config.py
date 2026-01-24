from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset
from pydantic_evals import Case
from pydantic_evals.evaluators import Evaluator


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
CaseBuilder = Callable[[int, Mapping[str, Any]], Case[str, str, dict[str, str]]]


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
    retrieval_evaluator: Evaluator | None = None
    document_limit: int | None = None
    system_prompt: str | None = None

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
