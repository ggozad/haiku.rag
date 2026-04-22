from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from haiku.rag.config.models import AppConfig
from haiku.skills.state import SkillRunDeps

if TYPE_CHECKING:
    from haiku.rag.agents.analysis.sandbox import Sandbox
    from haiku.rag.client import HaikuRAG


@dataclass
class RAGRunDeps(SkillRunDeps):
    rag: "HaikuRAG | None" = None
    search_count: int = 0


@dataclass
class AnalysisRunDeps(RAGRunDeps):
    sandbox: "Sandbox | None" = None


def make_rag_lifespan(db_path: Path, config: AppConfig):
    @asynccontextmanager
    async def lifespan(deps: RAGRunDeps) -> AsyncIterator[None]:
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(db_path, config=config, read_only=True) as rag:
            deps.rag = rag
            deps.search_count = 0
            yield

    return lifespan
