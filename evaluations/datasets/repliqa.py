from collections.abc import Mapping
from typing import Any, cast

from datasets import Dataset, DatasetDict, load_dataset
from pydantic_evals import Case

from evaluations.config import DatasetSpec, DocumentPayload, RetrievalSample


def load_repliqa_corpus() -> Dataset:
    dataset_dict = cast(DatasetDict, load_dataset("ServiceNow/repliqa"))
    dataset = cast(Dataset, dataset_dict["repliqa_3"])
    return dataset.filter(lambda doc: doc["document_topic"] == "News Stories")


def map_repliqa_document(doc: Mapping[str, Any]) -> DocumentPayload:
    return DocumentPayload(
        uri=str(doc["document_id"]),
        content=doc["document_extracted"],
    )


def map_repliqa_retrieval(doc: Mapping[str, Any]) -> RetrievalSample | None:
    expected_answer = doc["answer"]
    if expected_answer == "The answer is not found in the document.":
        return None
    return RetrievalSample(
        question=doc["question"],
        expected_uris=(str(doc["document_id"]),),
    )


def build_repliqa_case(
    index: int, doc: Mapping[str, Any]
) -> Case[str, str, dict[str, str]]:
    document_id = doc["document_id"]
    case_name = f"{index}_{document_id}" if document_id is not None else f"case_{index}"
    return Case(
        name=case_name,
        inputs=doc["question"],
        expected_output=doc["answer"],
        metadata={
            "document_id": str(document_id),
            "case_index": str(index),
        },
    )


REPLIQ_SPEC = DatasetSpec(
    key="repliqa",
    db_filename="repliqa.lancedb",
    document_loader=load_repliqa_corpus,
    document_mapper=map_repliqa_document,
    qa_loader=load_repliqa_corpus,
    qa_case_builder=build_repliqa_case,
    retrieval_loader=load_repliqa_corpus,
    retrieval_mapper=map_repliqa_retrieval,
)
