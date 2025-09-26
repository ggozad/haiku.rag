import json
from collections.abc import Iterable, Mapping
from typing import Any, cast

from datasets import Dataset, DatasetDict, load_dataset
from pydantic_evals import Case

from evaluations.config import DatasetSpec, DocumentPayload, RetrievalSample


def load_wix_corpus() -> Dataset:
    dataset_dict = cast(DatasetDict, load_dataset("Wix/WixQA", "wix_kb_corpus"))
    return cast(Dataset, dataset_dict["train"])


def map_wix_document(doc: Mapping[str, Any]) -> DocumentPayload:
    article_id = doc.get("id")
    url = doc.get("url")
    uri = str(article_id) if article_id is not None else str(url)

    metadata: dict[str, str] = {}
    if article_id is not None:
        metadata["article_id"] = str(article_id)
    if url:
        metadata["url"] = str(url)

    return DocumentPayload(
        uri=uri,
        content=doc["contents"],
        title=doc.get("title"),
        metadata=metadata or None,
    )


def load_wix_qa() -> Dataset:
    dataset_dict = cast(DatasetDict, load_dataset("Wix/WixQA", "wixqa_expertwritten"))
    return cast(Dataset, dataset_dict["train"])


def map_wix_retrieval(doc: Mapping[str, Any]) -> RetrievalSample | None:
    article_ids: Iterable[int | str] | None = doc.get("article_ids")
    if not article_ids:
        return None

    expected_uris = tuple(str(article_id) for article_id in article_ids)
    return RetrievalSample(
        question=doc["question"],
        expected_uris=expected_uris,
    )


def build_wix_case(
    index: int, doc: Mapping[str, Any]
) -> Case[str, str, dict[str, str]]:
    article_ids = tuple(str(article_id) for article_id in doc.get("article_ids") or [])
    joined_ids = "-".join(article_ids)
    case_name = f"{index}_{joined_ids}" if joined_ids else f"case_{index}"

    metadata = {
        "case_index": str(index),
        "document_ids": json.dumps(article_ids),
    }

    return Case(
        name=case_name,
        inputs=doc["question"],
        expected_output=doc["answer"],
        metadata=metadata,
    )


WIX_SPEC = DatasetSpec(
    key="wix",
    db_filename="wix.lancedb",
    document_loader=load_wix_corpus,
    document_mapper=map_wix_document,
    qa_loader=load_wix_qa,
    qa_case_builder=build_wix_case,
    retrieval_loader=load_wix_qa,
    retrieval_mapper=map_wix_retrieval,
)
