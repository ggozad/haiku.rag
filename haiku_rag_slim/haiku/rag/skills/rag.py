import os
from functools import cache
from pathlib import Path

from pydantic import BaseModel, Field

from haiku.rag.agents.research.models import Citation
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models.chunk import SearchResult
from haiku.skills.models import Skill, SkillMetadata, SkillSource, StateMetadata
from haiku.skills.parser import parse_skill_md

AGENT_PREAMBLE = """You are a helpful research assistant powered by haiku.rag, a knowledge base system.

CRITICAL RULES:
1. For greetings or casual chat: respond directly WITHOUT using any tools
2. NEVER make up information - always use skills to get facts from the knowledge base
3. When a skill returns citations, always include them in your response
"""

_RAG_TOOLS = ["search", "list_documents", "get_document", "cite"]


def get_agent_preamble(config: AppConfig) -> str:
    """Build the main agent preamble, prepending domain_preamble if configured."""
    if config.prompts.domain_preamble:
        return f"{config.prompts.domain_preamble}\n\n{AGENT_PREAMBLE}"
    return AGENT_PREAMBLE


class RAGState(BaseModel):
    citation_index: dict[str, Citation] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
    document_filter: str | None = None
    searches: dict[str, list[SearchResult]] = Field(default_factory=dict)


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
    from haiku.rag.skills._deps import RAGRunDeps, make_rag_lifespan
    from haiku.rag.skills._tools import create_skill_extras, create_skill_tools

    if config is None:
        config = get_config()

    if db_path is None:
        env_db = os.environ.get("HAIKU_RAG_DB")
        if env_db:
            db_path = Path(env_db).expanduser()
        else:
            db_path = config.storage.data_dir / "haiku.rag.lancedb"

    tools = create_skill_tools(db_path, config, RAGState, _RAG_TOOLS)
    extras = create_skill_extras(db_path, config)

    skill_instructions = instructions()
    if config.prompts.domain_preamble and skill_instructions:
        skill_instructions = f"{config.prompts.domain_preamble}\n\n{skill_instructions}"

    return Skill(
        metadata=skill_metadata(),
        source=SkillSource.ENTRYPOINT,
        path=_skill_path,
        instructions=skill_instructions,
        tools=list(tools.values()),
        extras=extras,
        state_type=STATE_TYPE,
        state_namespace=STATE_NAMESPACE,
        deps_type=RAGRunDeps,
        lifespan=make_rag_lifespan(db_path, config),
    )
