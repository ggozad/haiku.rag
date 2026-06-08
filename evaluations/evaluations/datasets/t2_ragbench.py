import json
import shutil
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

from datasets import Dataset
from huggingface_hub import hf_hub_download
from pydantic_evals import Case

from evaluations.config import DatasetSpec, DocumentPayload, RetrievalSample
from evaluations.evaluators import MAPEvaluator, NumberMatchEvaluator

REPO_ID = "G4KMU/t2-ragbench"

# Per-subset layout: which metadata files hold the rows, and the repo prefix the
# PDF lives under ({split} is filled from the row).
_SUBSETS: dict[str, dict[str, Any]] = {
    "FinQA": {
        "metadata": tuple(
            f"data/FinQA/{s}/metadata.jsonl" for s in ("dev", "test", "train")
        ),
        "pdf_prefix": "data/FinQA/{split}/",
    },
    "TAT-DQA": {
        "metadata": tuple(
            f"data/TAT-DQA/{s}/metadata.jsonl" for s in ("dev", "test", "train")
        ),
        "pdf_prefix": "data/TAT-DQA/{split}/",
    },
}


def get_cache_dir() -> Path:
    cache_dir = Path.home() / ".cache" / "haiku.rag" / "evaluations" / "t2_pdfs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


_rows_cache: dict[str, list[dict[str, Any]]] = {}


def _load_rows(subset: str) -> list[dict[str, Any]]:
    cached = _rows_cache.get(subset)
    if cached is not None:
        return cached

    rows: list[dict[str, Any]] = []
    for metadata_file in _SUBSETS[subset]["metadata"]:
        path = hf_hub_download(REPO_ID, metadata_file, repo_type="dataset")
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append({**json.loads(line), "subset": subset})

    _rows_cache[subset] = rows
    return rows


def download_t2_pdf(subset: str, split: str, file_name: str) -> Path:
    # hf_hub_download may return a content-addressed blob path with no suffix;
    # the converter dispatches on extension, so materialize a real ``.pdf`` file.
    dest = get_cache_dir() / f"{subset}_{split}_{file_name.replace('/', '_')}"
    if dest.exists():
        return dest

    repo_path = _SUBSETS[subset]["pdf_prefix"].format(split=split) + file_name
    src = hf_hub_download(REPO_ID, repo_path, repo_type="dataset")
    shutil.copyfile(src, dest)
    return dest


def load_t2_corpus(subset: str) -> Dataset:
    seen: set[str] = set()
    docs: list[dict[str, Any]] = []
    for row in _load_rows(subset):
        context_id = row["context_id"]
        if context_id in seen:
            continue
        seen.add(context_id)
        docs.append(row)
    return Dataset.from_list(docs)


def load_t2_qa(subset: str) -> Dataset:
    return Dataset.from_list(_load_rows(subset))


def map_t2_document(doc: Mapping[str, Any]) -> DocumentPayload | None:
    pdf_path = download_t2_pdf(doc["subset"], doc["split"], doc["file_name"])

    metadata: dict[str, Any] = {"file_name": doc["file_name"]}
    for key in ("company_name", "company_symbol", "report_year", "company_sector"):
        value = doc.get(key)
        if value is not None:
            metadata[key] = value

    title_parts = [
        str(doc[key]) for key in ("company_name", "report_year") if doc.get(key)
    ]
    title = " ".join(title_parts) if title_parts else doc["context_id"]

    return DocumentPayload(
        uri=doc["context_id"],
        source_path=pdf_path,
        title=title,
        metadata=metadata,
    )


def map_t2_retrieval(doc: Mapping[str, Any]) -> RetrievalSample | None:
    return RetrievalSample(
        question=doc["question"],
        expected_uris=(doc["context_id"],),
    )


def build_t2_case(index: int, doc: Mapping[str, Any]) -> Case[str, str, dict[str, str]]:
    metadata = {
        "case_index": str(index),
        "context_id": doc["context_id"],
        "id": doc["id"],
    }

    return Case(
        name=f"{index}_{doc['id']}",
        inputs=doc["question"],
        expected_output=str(doc["program_answer"]),
        metadata=metadata,
    )


def _t2_spec(subset: str, key: str, db_filename: str) -> DatasetSpec:
    return DatasetSpec(
        key=key,
        db_filename=db_filename,
        document_loader=partial(load_t2_corpus, subset),
        document_mapper=map_t2_document,
        qa_loader=partial(load_t2_qa, subset),
        qa_case_builder=build_t2_case,
        retrieval_loader=partial(load_t2_qa, subset),
        retrieval_mapper=map_t2_retrieval,
        retrieval_evaluator=MAPEvaluator(),
        qa_evaluator=NumberMatchEvaluator(),
    )


T2_FINQA_SPEC = _t2_spec(
    subset="FinQA",
    key="t2_finqa",
    db_filename="t2_ragbench_finqa.lancedb",
)

T2_TATDQA_SPEC = _t2_spec(
    subset="TAT-DQA",
    key="t2_tatdqa",
    db_filename="t2_ragbench_tatdqa.lancedb",
)
