import json
from collections.abc import Iterable, Mapping
from typing import Any

from datasets import Dataset, load_dataset
from pydantic_evals import Case

from evaluations.config import DatasetSpec, DocumentPayload, RetrievalSample
from evaluations.evaluators import MAPEvaluator

WIX_SUPPORT_PROMPT = """You are a WIX technical support expert helping users with questions about the WIX platform.

Your process:
1. When a user asks a question, use the search_documents tool to find relevant information
2. Search with specific keywords and phrases from the user's question
3. Review the search results ordered by relevance (rank 1 = most relevant)
4. If you need additional context, perform follow-up searches with different keywords
5. Provide a short and to the point comprehensive answer based only on the retrieved documents

Guidelines:
- Base your answers strictly on the provided document content
- Quote or reference specific information when possible
- If multiple documents contain relevant information, synthesize them coherently
- Indicate when information is incomplete or when you need to search for additional context
- If the retrieved documents don't contain sufficient information, clearly state: "I cannot find enough information in the knowledge base to answer this question."
- For complex questions, consider breaking them down and performing multiple searches
- Stick to the answer, do not ellaborate or provide context unless explicitly asked for it.

Be concise, and always maintain accuracy over completeness. Prefer short, direct answers that are well-supported by the documents.
"""


def load_wix_corpus() -> Dataset:
    dataset_dict = load_dataset("Wix/WixQA", "wix_kb_corpus")
    return dataset_dict["train"]


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
        content=doc["html_content"],
        title=doc.get("title"),
        metadata=metadata or None,
        format="html",
    )


def load_wix_qa() -> Dataset:
    dataset_dict = load_dataset("Wix/WixQA", "wixqa_expertwritten")
    return dataset_dict["train"]


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
    retrieval_evaluator=MAPEvaluator(),
    system_prompt=WIX_SUPPORT_PROMPT,
)
