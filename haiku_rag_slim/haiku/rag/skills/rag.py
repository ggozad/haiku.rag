import os
from functools import cache
from pathlib import Path

from pydantic import BaseModel, Field

from haiku.rag.agents.research.models import Citation
from haiku.rag.config.models import AppConfig
from haiku.rag.skills._tools import ResearchEntry
from haiku.rag.store.models.chunk import SearchResult
from haiku.rag.tools.document import DocumentInfo
from haiku.rag.tools.qa import QAHistoryEntry
from haiku.skills.models import Skill, SkillMetadata, SkillSource, StateMetadata
from haiku.skills.parser import parse_skill_md

AGENT_PREAMBLE = """You are a helpful research assistant powered by haiku.rag, a knowledge base system.

CRITICAL RULES:
1. For greetings or casual chat: respond directly WITHOUT using any tools
2. NEVER make up information - always use skills to get facts from the knowledge base
3. When a skill returns citations, always include them in your response
"""

_RAG_TOOLS = ["search", "list_documents", "get_document", "ask", "research"]


class RAGState(BaseModel):
    citations: list[Citation] = Field(default_factory=list)
    qa_history: list[QAHistoryEntry] = Field(default_factory=list)
    document_filter: str | None = None
    searches: dict[str, list[SearchResult]] = Field(default_factory=dict)
    documents: list[DocumentInfo] = Field(default_factory=list)
    reports: list[ResearchEntry] = Field(default_factory=list)


STATE_TYPE = RAGState
STATE_NAMESPACE = "rag"

_skill_path = Path(__file__).parent / "rag"


@cache
def skill_metadata() -> SkillMetadata:
    metadata, _ = parse_skill_md(_skill_path / "SKILL.md")
    return metadata


@cache
def instructions() -> str | None:
    _, instr = parse_skill_md(_skill_path / "SKILL.md")
    return instr


def state_metadata() -> StateMetadata:
    return StateMetadata(
        namespace=STATE_NAMESPACE,
        type=STATE_TYPE,
        schema=STATE_TYPE.model_json_schema(),
    )


def create_skill(
    db_path: Path | None = None,
    config: AppConfig | None = None,
) -> Skill:
    """Create a RAG skill for searching and analyzing documents.

    Args:
        db_path: Path to the LanceDB database. Resolved from:
            1. This argument
            2. HAIKU_RAG_DB environment variable
            3. haiku.rag default (config.storage.data_dir / "haiku.rag.lancedb")
        config: haiku.rag AppConfig instance. If None, uses get_config().
    """
    from haiku.rag.config import get_config
    from haiku.rag.skills._tools import create_skill_tools

    if config is None:
        config = get_config()

    if db_path is None:
        env_db = os.environ.get("HAIKU_RAG_DB")
        if env_db:
            db_path = Path(env_db).expanduser()
        else:
            db_path = config.storage.data_dir / "haiku.rag.lancedb"

    tools = create_skill_tools(db_path, config, RAGState, _RAG_TOOLS)

    return Skill(
        metadata=skill_metadata(),
        source=SkillSource.ENTRYPOINT,
        path=_skill_path,
        instructions=instructions(),
        tools=list(tools.values()),
        state_type=STATE_TYPE,
        state_namespace=STATE_NAMESPACE,
    )
