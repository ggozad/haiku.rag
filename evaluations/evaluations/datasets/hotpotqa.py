from collections.abc import Mapping
from typing import Any, cast

from datasets import Dataset, load_dataset
from pydantic_evals import Case

from evaluations.config import DatasetSpec, DocumentPayload, RetrievalSample
from evaluations.evaluators import MAPEvaluator


def load_hotpotqa_validation() -> Dataset:
    dataset_dict = load_dataset("hotpotqa/hotpot_qa", "distractor")
    return dataset_dict["validation"]


def extract_unique_documents(dataset: Dataset) -> list[dict[str, Any]]:
    """Extract unique documents from all context paragraphs, deduplicated by title."""
    seen_titles: set[str] = set()
    documents: list[dict[str, Any]] = []

    for sample in dataset:
        sample = cast(Mapping[str, Any], sample)
        context = sample["context"]
        titles = context["title"]
        sentences_list = context["sentences"]

        for title, sentences in zip(titles, sentences_list):
            if title in seen_titles:
                continue
            seen_titles.add(title)
            content = " ".join(sentences)
            documents.append({"title": title, "content": content})

    return documents


_cached_documents: list[dict[str, Any]] | None = None


def load_hotpotqa_documents() -> list[dict[str, Any]]:
    """Load and cache unique documents from HotpotQA."""
    global _cached_documents
    if _cached_documents is None:
        dataset = load_hotpotqa_validation()
        _cached_documents = extract_unique_documents(dataset)
    return _cached_documents


def document_loader() -> Dataset:
    """Return documents as a Dataset-like iterable."""
    docs = load_hotpotqa_documents()
    return Dataset.from_list(docs)


def map_hotpotqa_document(doc: Mapping[str, Any]) -> DocumentPayload:
    return DocumentPayload(
        uri=doc["title"],
        content=doc["content"],
        title=doc["title"],
    )


def map_hotpotqa_retrieval(doc: Mapping[str, Any]) -> RetrievalSample | None:
    supporting_facts = doc["supporting_facts"]
    titles = supporting_facts["title"]
    if not titles:
        return None

    unique_titles = tuple(dict.fromkeys(titles))
    return RetrievalSample(
        question=doc["question"],
        expected_uris=unique_titles,
    )


def build_hotpotqa_case(
    index: int, doc: Mapping[str, Any]
) -> Case[str, str, dict[str, str]]:
    question_id = doc["id"]
    question_type = doc["type"]
    level = doc["level"]

    case_name = f"{index}_{question_id}"

    return Case(
        name=case_name,
        inputs=doc["question"],
        expected_output=doc["answer"],
        metadata={
            "question_id": str(question_id),
            "type": str(question_type),
            "level": str(level),
            "case_index": str(index),
        },
    )


HOTPOTQA_SPEC = DatasetSpec(
    key="hotpotqa",
    db_filename="hotpotqa.lancedb",
    document_loader=document_loader,
    document_mapper=map_hotpotqa_document,
    qa_loader=load_hotpotqa_validation,
    qa_case_builder=build_hotpotqa_case,
    retrieval_loader=load_hotpotqa_validation,
    retrieval_mapper=map_hotpotqa_retrieval,
    retrieval_evaluator=MAPEvaluator(),
)
