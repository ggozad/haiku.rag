import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
from datasets import Dataset
from huggingface_hub import hf_hub_download
from pydantic_evals import Case

from evaluations.config import DatasetSpec, DocumentPayload, RetrievalSample
from evaluations.evaluators import MAPEvaluator

logger = logging.getLogger(__name__)

ORB_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions using a document knowledge base.

Process:
1. Call search_documents with relevant keywords from the question
2. Review the results ordered by relevance
3. If needed, perform follow-up searches with different keywords (max 3 total)
4. Provide a concise answer based strictly on the retrieved content

The search tool returns results like:
[chunk_abc123] [rank 1 of 5]
Source: "Document Title" > Section > Subsection
Type: paragraph
Content:
The actual text content here...

[chunk_def456] [rank 2 of 5]
Source: "Another Document"
Type: table
Content:
| Column 1 | Column 2 |
...

Each result includes:
- chunk_id in brackets and rank position (rank 1 = most relevant)
- Source: document title and section hierarchy (when available)
- Type: content type like paragraph, table, code, list_item (when available)
- Content: the actual text

In your response, include the chunk IDs you used in cited_chunks.

Guidelines:
- Base answers strictly on retrieved content - do not use external knowledge
- Use the Source and Type metadata to understand context
- If multiple results are relevant, synthesize them coherently
- If information is insufficient, say: "I cannot find enough information in the knowledge base to answer this question."
- Be concise and direct - avoid elaboration unless asked
- Results are ordered by relevance, with rank 1 being most relevant
- IMPORTANT: Do not use LaTeX notation (like \\(...\\) or $...$) in your answers. Use plain text or Unicode math symbols instead.
"""

REPO_ID = "vectara/open_ragbench"
PDF_SUBDIR = "pdf/arxiv"


def get_cache_dir() -> Path:
    cache_dir = Path.home() / ".cache" / "haiku.rag" / "evaluations" / "arxiv_pdfs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_metadata_file(filename: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=f"{PDF_SUBDIR}/{filename}",
            repo_type="dataset",
        )
    )


def load_pdf_urls() -> dict[str, str]:
    path = download_metadata_file("pdf_urls.json")
    with open(path) as f:
        return json.load(f)


def load_queries() -> dict[str, dict[str, str]]:
    path = download_metadata_file("queries.json")
    with open(path) as f:
        return json.load(f)


def load_qrels() -> dict[str, dict[str, Any]]:
    path = download_metadata_file("qrels.json")
    with open(path) as f:
        return json.load(f)


def load_answers() -> dict[str, str]:
    path = download_metadata_file("answers.json")
    with open(path) as f:
        return json.load(f)


def download_pdf(paper_id: str, url: str, cache_dir: Path) -> Path | None:
    pdf_path = cache_dir / f"{paper_id}.pdf"
    if pdf_path.exists():
        return pdf_path

    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            pdf_path.write_bytes(response.content)
            return pdf_path
    except Exception as e:
        logger.warning(f"Failed to download PDF {paper_id}: {e}")
        return None


def download_all_pdfs(pdf_urls: dict[str, str]) -> dict[str, Path]:
    cache_dir = get_cache_dir()
    downloaded = {}

    for paper_id, url in pdf_urls.items():
        pdf_path = download_pdf(paper_id, url, cache_dir)
        if pdf_path is not None:
            downloaded[paper_id] = pdf_path

    logger.info(f"Downloaded {len(downloaded)}/{len(pdf_urls)} PDFs")
    return downloaded


_pdf_urls: dict[str, str] | None = None
_queries: dict[str, dict[str, str]] | None = None
_qrels: dict[str, dict[str, Any]] | None = None
_answers: dict[str, str] | None = None


def ensure_metadata_loaded() -> None:
    global _pdf_urls, _queries, _qrels, _answers
    if _pdf_urls is None:
        _pdf_urls = load_pdf_urls()
    if _queries is None:
        _queries = load_queries()
    if _qrels is None:
        _qrels = load_qrels()
    if _answers is None:
        _answers = load_answers()


def load_orb_corpus() -> Dataset:
    ensure_metadata_loaded()
    assert _pdf_urls is not None

    # Return paper IDs and URLs - PDFs are downloaded lazily during mapping
    records = [
        {"paper_id": paper_id, "pdf_url": url} for paper_id, url in _pdf_urls.items()
    ]

    return Dataset.from_list(records)


def map_orb_document(doc: Mapping[str, Any]) -> DocumentPayload | None:
    paper_id = doc["paper_id"]
    pdf_url = doc["pdf_url"]

    # Download PDF lazily
    cache_dir = get_cache_dir()
    pdf_path = download_pdf(paper_id, pdf_url, cache_dir)

    if pdf_path is None:
        return None

    return DocumentPayload(
        uri=paper_id,
        source_path=pdf_path,
        title=paper_id,
        metadata={"arxiv_id": paper_id},
    )


def load_orb_qa() -> Dataset:
    ensure_metadata_loaded()
    assert _queries is not None
    assert _answers is not None

    records = []
    for query_id, query_data in _queries.items():
        answer = _answers.get(query_id, "")
        records.append(
            {
                "query_id": query_id,
                "query": query_data["query"],
                "type": query_data["type"],
                "source": query_data["source"],
                "answer": answer,
            }
        )

    return Dataset.from_list(records)


def build_orb_case(
    index: int, doc: Mapping[str, Any]
) -> Case[str, str, dict[str, str]]:
    metadata = {
        "case_index": str(index),
        "query_id": doc["query_id"],
        "query_type": doc["type"],
        "query_source": doc["source"],
    }

    return Case(
        name=f"{index}_{doc['query_id'][:8]}",
        inputs=doc["query"],
        expected_output=doc["answer"],
        metadata=metadata,
    )


def load_orb_retrieval() -> Dataset:
    ensure_metadata_loaded()
    assert _pdf_urls is not None
    assert _queries is not None
    assert _qrels is not None

    records = []
    for query_id, query_data in _queries.items():
        qrel = _qrels.get(query_id)
        if qrel is None:
            continue

        doc_id = qrel.get("doc_id")
        if doc_id is None or doc_id not in _pdf_urls:
            continue

        records.append(
            {
                "query_id": query_id,
                "query": query_data["query"],
                "type": query_data["type"],
                "source": query_data["source"],
                "doc_id": doc_id,
            }
        )

    return Dataset.from_list(records)


def map_orb_retrieval(doc: Mapping[str, Any]) -> RetrievalSample | None:
    return RetrievalSample(
        question=doc["query"],
        expected_uris=(doc["doc_id"],),
        source_type=doc.get("source"),
    )


def is_multimodal_query(source: str) -> bool:
    return "image" in source


OPEN_RAG_BENCH_SPEC = DatasetSpec(
    key="orb",
    db_filename="open_rag_bench.lancedb",
    document_loader=load_orb_corpus,
    document_mapper=map_orb_document,
    qa_loader=load_orb_qa,
    qa_case_builder=build_orb_case,
    retrieval_loader=load_orb_retrieval,
    retrieval_mapper=map_orb_retrieval,
    retrieval_evaluator=MAPEvaluator(),
    system_prompt=ORB_SYSTEM_PROMPT,
)
