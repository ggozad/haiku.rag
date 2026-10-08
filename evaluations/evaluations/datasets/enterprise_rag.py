import ast
from collections.abc import Mapping
from typing import Any

from datasets import Dataset, load_dataset
from pydantic_evals import Case

from evaluations.config import DatasetSpec, DocumentPayload, RetrievalSample
from evaluations.evaluators import CitationMAPEvaluator, RecallEvaluator

REPO_ID = "onyx-dot-app/EnterpriseRAG-Bench"


def load_enterprise_rag_documents() -> Dataset:
    return load_dataset(REPO_ID, "documents")["test"]


def load_enterprise_rag_questions() -> Dataset:
    return load_dataset(REPO_ID, "questions")["test"]


def unescape_newlines(text: str) -> str:
    """Turn literal `\\n` sequences into newlines in text that has none."""
    if "\n" in text:
        return text
    return text.replace("\\n", "\n")


def decode_message_list(content: str) -> str:
    """Join a thread stored as the `repr` of a list of messages, as most gmail
    documents are, and return any other content unchanged."""
    if not content.startswith(("['", '["')):
        return content
    try:
        messages = ast.literal_eval(content)
    except (SyntaxError, ValueError):
        return content
    if not isinstance(messages, list) or not all(isinstance(m, str) for m in messages):
        return content
    return "\n\n".join(unescape_newlines(m).strip("\n") for m in messages)


def map_enterprise_rag_document(doc: Mapping[str, Any]) -> DocumentPayload:
    return DocumentPayload(
        uri=doc["doc_id"],
        content=unescape_newlines(decode_message_list(doc["content"])),
        title=doc["title"],
        metadata={"source_type": doc["source_type"]},
    )


def map_enterprise_rag_retrieval(doc: Mapping[str, Any]) -> RetrievalSample | None:
    if not doc["expected_doc_ids"]:
        return None
    return RetrievalSample(
        question=doc["question"],
        expected_uris=tuple(dict.fromkeys(doc["expected_doc_ids"])),
    )


def build_enterprise_rag_case(
    index: int, doc: Mapping[str, Any]
) -> Case[str, str, dict[str, Any]]:
    return Case(
        name=f"{index}_{doc['question_id']}",
        inputs=doc["question"],
        expected_output=doc["gold_answer"],
        metadata={
            "case_index": str(index),
            "question_id": doc["question_id"],
            "question_type": doc["question_type"],
            "source_types": list(doc["source_types"]),
            "answer_facts": list(doc["answer_facts"]),
            "answerability": "UNANSWERABLE"
            if doc["question_type"] == "info_not_found"
            else "ANSWERABLE",
        },
    )


ENTERPRISE_RAG_SPEC = DatasetSpec(
    key="enterprise_rag",
    db_filename="enterprise_rag.lancedb",
    document_loader=load_enterprise_rag_documents,
    document_mapper=map_enterprise_rag_document,
    qa_loader=load_enterprise_rag_questions,
    qa_case_builder=build_enterprise_rag_case,
    retrieval_loader=load_enterprise_rag_questions,
    retrieval_mapper=map_enterprise_rag_retrieval,
    retrieval_evaluators=[RecallEvaluator(k=10)],
    retrieval_limit=10,
    citation_evaluator=CitationMAPEvaluator(),
    ingest_batch_size=512,
)
