from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from pydantic_ai.models import Model

from haiku.rag.agents.research.models import Citation
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models.chunk import SearchResult
from haiku.skills import run_skill
from haiku.skills.models import Skill

SkillFactory = Callable[..., Skill]


class _RagLikeState(Protocol):
    document_filter: str | None
    citation_index: dict[str, Citation]
    citations: list[str]
    searches: dict[str, list[SearchResult]]


@dataclass
class SkillRunResult:
    answer: str
    cited_uris: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)
    searched_uris: list[str] = field(default_factory=list)
    n_searches: int = 0


async def run_skill_question(
    skill_factory: SkillFactory,
    db_path: Path,
    config: AppConfig,
    question: str,
    skill_model: str | Model,
    document_filter: str | None = None,
    request_limit: int | None = None,
) -> SkillRunResult:
    """Run a single question through a skill and return answer + retrieval data.

    Builds the skill via ``skill_factory(db_path=..., config=...)`` and
    invokes it with a fresh state instance derived from
    ``skill.state_type``. After the run, citations and searched documents
    are extracted from the state for downstream eval scoring.

    The skill must produce a state with RAG-skill-shaped fields (citation
    index, searches, optional document filter) — i.e. ``RAGState`` or
    ``AnalysisState`` from ``haiku.rag.skills``.
    """
    skill = skill_factory(db_path=db_path, config=config)
    if request_limit is not None:
        skill.request_limit = request_limit

    if skill.state_type is None:
        raise ValueError(f"Skill {skill.metadata.name!r} has no state_type")
    state = skill.state_type()
    typed = cast(_RagLikeState, state)
    if document_filter is not None:
        typed.document_filter = document_filter

    answer, _, _ = await run_skill(skill_model, skill, question, state=state)

    cited_chunk_ids: list[str] = list(typed.citations)
    seen_cited: set[str] = set()
    cited_uris: list[str] = []
    for chunk_id in cited_chunk_ids:
        citation = typed.citation_index.get(chunk_id)
        if citation is None:
            continue
        if citation.document_uri not in seen_cited:
            seen_cited.add(citation.document_uri)
            cited_uris.append(citation.document_uri)

    seen_searched: set[str] = set()
    searched_uris: list[str] = []
    for results in typed.searches.values():
        for result in results:
            uri = result.document_uri
            if uri and uri not in seen_searched:
                seen_searched.add(uri)
                searched_uris.append(uri)

    return SkillRunResult(
        answer=answer,
        cited_uris=cited_uris,
        cited_chunk_ids=cited_chunk_ids,
        searched_uris=searched_uris,
        n_searches=len(typed.searches),
    )
