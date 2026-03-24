from pathlib import Path
from typing import Any

from haiku.rag.agents.research.models import Citation
from haiku.rag.tools.document import DocumentInfo
from haiku.rag.tools.qa import QAHistoryEntry


async def find_relevant_prior_qa(
    qa_history: list[QAHistoryEntry],
    query: str,
    config: Any,
) -> list[QAHistoryEntry]:
    from haiku.rag.embeddings import get_embedder
    from haiku.rag.tools.qa import PRIOR_ANSWER_RELEVANCE_THRESHOLD
    from haiku.rag.utils import cosine_similarity

    if not qa_history:
        return []

    embedder = get_embedder(config)
    query_embedding = await embedder.embed_query(query)

    to_embed = []
    to_embed_indices = []
    for i, qa in enumerate(qa_history):
        if qa.question_embedding is None:
            to_embed.append(qa.question)
            to_embed_indices.append(i)

    if to_embed:
        new_embeddings = await embedder.embed_documents(to_embed)
        for i, idx in enumerate(to_embed_indices):
            qa_history[idx].question_embedding = new_embeddings[i]

    matches = []
    for qa in qa_history:
        if qa.question_embedding is not None:
            similarity = cosine_similarity(query_embedding, qa.question_embedding)
            if similarity >= PRIOR_ANSWER_RELEVANCE_THRESHOLD:
                matches.append(qa)

    return matches


async def skill_search(
    db_path: Path,
    config: Any,
    query: str,
    limit: int | None = None,
    document_filter: str | None = None,
) -> tuple[str, list]:
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(db_path, config=config, read_only=True) as rag:
        results = await rag.search(
            query,
            limit=limit,
            filter=document_filter,
        )
        results = await rag.expand_context(results)

    formatted = "\n\n---\n\n".join(
        r.format_for_agent(rank=i + 1, total=len(results))
        for i, r in enumerate(results)
    )
    return formatted, list(results)


async def skill_list_documents(
    db_path: Path,
    config: Any,
    limit: int | None = None,
    offset: int | None = None,
) -> list[dict[str, Any]]:
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(db_path, config=config, read_only=True) as rag:
        documents = await rag.list_documents(limit, offset)
        return [
            {
                "id": doc.id,
                "title": doc.title,
                "uri": doc.uri,
                "metadata": doc.metadata,
                "created_at": str(doc.created_at),
                "updated_at": str(doc.updated_at),
            }
            for doc in documents
        ]


async def skill_get_document(
    db_path: Path,
    config: Any,
    query: str,
) -> dict[str, Any] | None:
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(db_path, config=config, read_only=True) as rag:
        document = await rag.resolve_document(query)
        if document is None:
            return None
        return {
            "id": document.id,
            "content": document.content,
            "title": document.title,
            "uri": document.uri,
            "metadata": document.metadata,
            "created_at": str(document.created_at),
            "updated_at": str(document.updated_at),
        }


async def skill_ask(
    db_path: Path,
    config: Any,
    question: str,
    qa_history: list[QAHistoryEntry] | None = None,
    document_filter: str | None = None,
) -> tuple[str, list[Citation]]:
    from haiku.rag.client import HaikuRAG
    from haiku.rag.utils import format_citations

    ask_question = question
    if qa_history:
        matches = await find_relevant_prior_qa(qa_history, question, config)
        if matches:
            prior_parts = []
            for qa in matches:
                part = f"Q: {qa.question}\nA: {qa.answer}"
                if qa.citations:
                    part += "\n" + format_citations(qa.citations)
                prior_parts.append(part)
            ask_question = (
                "Context from prior questions in this session:\n\n"
                + "\n\n---\n\n".join(prior_parts)
                + "\n\n---\n\nCurrent question: "
                + question
            )

    async with HaikuRAG(db_path, config=config, read_only=True) as rag:
        answer, citations = await rag.ask(
            ask_question,
            filter=document_filter,
        )

    return answer, citations


async def skill_research(
    db_path: Path,
    config: Any,
    question: str,
    document_filter: str | None = None,
) -> tuple[str, str, str]:
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(db_path, config=config, read_only=True) as rag:
        report = await rag.research(question, filter=document_filter)

    parts = [
        f"# {report.title}",
        f"\n## Executive Summary\n{report.executive_summary}",
    ]
    if report.main_findings:
        parts.append("\n## Main Findings")
        for finding in report.main_findings:
            parts.append(f"- {finding}")
    if report.conclusions:
        parts.append("\n## Conclusions")
        for conclusion in report.conclusions:
            parts.append(f"- {conclusion}")
    if report.limitations:
        parts.append("\n## Limitations")
        for limitation in report.limitations:
            parts.append(f"- {limitation}")
    if report.recommendations:
        parts.append("\n## Recommendations")
        for rec in report.recommendations:
            parts.append(f"- {rec}")
    parts.append(f"\n## Sources\n{report.sources_summary}")

    formatted = "\n".join(parts)
    return formatted, report.title, report.executive_summary


async def skill_analyze(
    db_path: Path,
    config: Any,
    question: str,
    document: str | None = None,
    filter: str | None = None,
) -> tuple[str, str, str | None]:
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(db_path, config=config, read_only=True) as rag:
        documents = [document] if document else None
        result = await rag.rlm(question, documents=documents, filter=filter)
        output = result.answer
        if result.program:
            output += f"\n\nProgram:\n{result.program}"

    return output, result.answer, result.program


def update_documents_state(
    documents_state: list[DocumentInfo],
    doc_dicts: list[dict[str, Any]],
) -> None:
    for doc_dict in doc_dicts:
        doc_info = DocumentInfo(
            id=str(doc_dict["id"]),
            title=doc_dict["title"] or "Untitled",
            uri=doc_dict.get("uri") or "",
            created=doc_dict.get("created_at", ""),
        )
        if not any(d.id == doc_info.id for d in documents_state):
            documents_state.append(doc_info)
