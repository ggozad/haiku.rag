from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext
from pydantic_ai.messages import ToolReturn
from pydantic_ai.toolsets import FunctionToolset

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG

from haiku.rag.capabilities._base import (
    EvidenceState,
    RAGCapabilityBase,
    resolve_db_path,
)
from haiku.rag.config.models import AppConfig

AGENT_PREAMBLE = """You are a helpful research assistant powered by haiku.rag, a knowledge base system.

CRITICAL RULES:
1. For greetings or casual chat: respond directly WITHOUT using any tools
2. NEVER make up information - always use capabilities to get facts from the knowledge base
3. When a capability returns citations, always include them in your response
"""

STATE_NAMESPACE = "rag"
_CAPABILITY_ID = "haiku-rag"
_TOOL_NAMES = frozenset({"rag_search", "rag_cite"})
_instructions_path = Path(__file__).parent / "instructions" / "rag.md"


class RAGState(EvidenceState):
    """The RAG capability carries nothing beyond the shared evidence fields."""


@cache
def instructions() -> str:
    return _instructions_path.read_text().strip()


@dataclass
class RAGCapability(RAGCapabilityBase[RAGState]):
    """Deferred, native Pydantic AI capability for grounded RAG queries."""

    @classmethod
    def from_spec(
        cls,
        db_path: Path | None = None,
        config: AppConfig | None = None,
        *,
        defer_loading: bool = True,
        request_limit: int | None = 20,
        vision: bool | None = None,
    ) -> "RAGCapability":
        """Build from an agent spec, mirroring the factory's serializable arguments.

        A live ``HaikuRAG`` client cannot be written in a spec, so ``rag`` is
        absent here. ``config`` arrives as a mapping and is validated.
        """
        return create_capability(
            db_path,
            AppConfig.model_validate(config) if config is not None else None,
            defer_loading=defer_loading,
            request_limit=request_limit,
            vision=vision,
        )

    def get_toolset(self) -> FunctionToolset[Any]:
        async def rag_search(
            ctx: RunContext[Any], query: str, limit: int | None = None
        ) -> str | ToolReturn:
            """Search the knowledge base using hybrid vector and full-text search."""
            return await self._with_state(self._search(query, limit))

        async def rag_cite(ctx: RunContext[Any], chunk_ids: list[str]) -> Any:
            """Register exact search-result chunk IDs as citations for the answer."""
            return await self._with_state(self._cite(chunk_ids))

        return FunctionToolset(
            [rag_search, rag_cite],
            id=_CAPABILITY_ID,
            max_retries=3,
            sequential=True,
        )


def create_capability(
    db_path: Path | str | None = None,
    config: AppConfig | None = None,
    *,
    defer_loading: bool = True,
    rag: "HaikuRAG | None" = None,
    request_limit: int | None = 20,
    vision: bool | None = None,
) -> RAGCapability:
    """Create a native Pydantic AI RAG capability.

    ``vision`` gates whether picture chunks are attached to search results as
    images, and should reflect the model the hosting agent actually runs.
    Defaults to ``config.qa.model.vision``.
    """
    if config is None:
        from haiku.rag.config import get_config

        config = get_config()
    return RAGCapability(
        db_path=resolve_db_path(db_path, config),
        config=config,
        borrowed_rag=rag,
        state_type=RAGState,
        state_namespace=STATE_NAMESPACE,
        instruction_text=instructions(),
        vision=config.qa.model.vision if vision is None else vision,
        tool_names=_TOOL_NAMES,
        request_limit=request_limit,
        id=_CAPABILITY_ID,
        description=(
            "Search the haiku.rag knowledge base and cite evidence for grounded answers."
        ),
        defer_loading=defer_loading,
    )


__all__ = [
    "AGENT_PREAMBLE",
    "RAGCapability",
    "RAGState",
    "STATE_NAMESPACE",
    "create_capability",
    "instructions",
]
