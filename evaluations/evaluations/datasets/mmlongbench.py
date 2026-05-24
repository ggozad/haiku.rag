import ast
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from pydantic_evals import Case

from evaluations.config import DatasetSpec, DocumentPayload, RetrievalSample
from evaluations.evaluators import MAPEvaluator

logger = logging.getLogger(__name__)

REPO_ID = "yubo2333/MMLongBench-Doc"
PDF_SUBDIR = "documents"
_LIST_FIELDS = ("evidence_pages", "evidence_sources")


def get_cache_dir() -> Path:
    cache_dir = Path.home() / ".cache" / "haiku.rag" / "evaluations" / "mmlongbench"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def ensure_pdfs_downloaded() -> Path:
    cache_dir = get_cache_dir()
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=f"{PDF_SUBDIR}/*.pdf",
        local_dir=cache_dir,
    )
    return cache_dir / PDF_SUBDIR


def _parse_list_field(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    return ast.literal_eval(value)


def _load_hf_qa_split() -> list[dict[str, Any]]:
    dataset = load_dataset(REPO_ID, split="train")
    return [dict(row) for row in dataset]


_qa_records: list[dict[str, Any]] | None = None


def load_qa_records() -> list[dict[str, Any]]:
    global _qa_records
    if _qa_records is not None:
        return _qa_records
    rows = _load_hf_qa_split()
    for row in rows:
        for field in _LIST_FIELDS:
            row[field] = _parse_list_field(row.get(field))
    _qa_records = rows
    return _qa_records


def load_mmlb_corpus() -> Dataset:
    pdf_dir = ensure_pdfs_downloaded()
    records = load_qa_records()
    doc_types: dict[str, str] = {}
    for row in records:
        doc_id = row["doc_id"]
        if doc_id not in doc_types:
            doc_types[doc_id] = row.get("doc_type", "")
    corpus = [
        {"doc_id": doc_id, "doc_type": doc_type}
        for doc_id, doc_type in doc_types.items()
        if (pdf_dir / doc_id).exists()
    ]
    return Dataset.from_list(corpus)


def map_mmlb_document(doc: Mapping[str, Any]) -> DocumentPayload | None:
    doc_id = doc["doc_id"]
    pdf_path = get_cache_dir() / PDF_SUBDIR / doc_id
    if not pdf_path.exists():
        logger.warning(f"PDF not found in cache: {doc_id}")
        return None
    return DocumentPayload(
        uri=doc_id,
        source_path=pdf_path,
        title=doc_id,
        metadata={"doc_type": doc.get("doc_type", "")},
    )


def load_mmlb_qa() -> Dataset:
    return Dataset.from_list(load_qa_records())


def build_mmlb_case(
    index: int, doc: Mapping[str, Any]
) -> Case[str, str, dict[str, str]]:
    evidence_pages = list(doc.get("evidence_pages") or [])
    evidence_sources = list(doc.get("evidence_sources") or [])
    metadata: dict[str, str] = {
        "case_index": str(index),
        "doc_id": doc["doc_id"],
        "doc_type": doc.get("doc_type", ""),
        "answer_format": doc.get("answer_format", ""),
        "evidence_pages": str(evidence_pages),
        "evidence_sources": ",".join(str(s) for s in evidence_sources),
    }
    return Case(
        name=f"{index}_{doc['doc_id']}",
        inputs=doc["question"],
        expected_output=doc["answer"],
        metadata=metadata,
    )


def load_mmlb_retrieval() -> Dataset:
    records = []
    for row in load_qa_records():
        if not row.get("evidence_pages"):
            continue
        records.append(row)
    return Dataset.from_list(records)


def map_mmlb_retrieval(doc: Mapping[str, Any]) -> RetrievalSample | None:
    evidence_pages = doc.get("evidence_pages") or []
    if not evidence_pages:
        return None
    sources = doc.get("evidence_sources") or []
    source_type = ",".join(str(s) for s in sources) if sources else None
    return RetrievalSample(
        question=doc["question"],
        expected_uris=(doc["doc_id"],),
        source_type=source_type,
    )


MMLONGBENCH_SPEC = DatasetSpec(
    key="mmlongbench",
    db_filename="mmlongbench.lancedb",
    document_loader=load_mmlb_corpus,
    document_mapper=map_mmlb_document,
    qa_loader=load_mmlb_qa,
    qa_case_builder=build_mmlb_case,
    retrieval_loader=load_mmlb_retrieval,
    retrieval_mapper=map_mmlb_retrieval,
    retrieval_evaluator=MAPEvaluator(),
)
