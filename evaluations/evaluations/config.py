from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset
from pydantic_evals import Case


@dataclass
class DocumentPayload:
    uri: str
    content: str
    title: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class RetrievalSample:
    question: str
    expected_uris: tuple[str, ...]
    skip: bool = False


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
    document_limit: int | None = None

    @property
    def db_path(self) -> Path:
        return Path(__file__).parent / "data" / self.db_filename
