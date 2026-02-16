from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic_ai import FunctionToolset

from haiku.rag.config.models import AppConfig
from haiku.rag.tools.context import ToolContext, prepare_context
from haiku.rag.tools.prompts import build_tools_prompt

FEATURE_SEARCH = "search"
FEATURE_DOCUMENTS = "documents"
FEATURE_QA = "qa"
FEATURE_ANALYSIS = "analysis"


@dataclass(frozen=True)
class Toolkit:
    """Bundled toolsets, prompt, and context factory for haiku.rag agents.

    Created via build_toolkit(). Provides everything needed to compose
    an agent with haiku.rag toolsets and create matching ToolContexts.
    """

    toolsets: list[FunctionToolset] = field(default_factory=list)
    prompt: str = ""
    features: list[str] = field(default_factory=list)

    def create_context(self, state_key: str | None = None) -> ToolContext:
        """Create a ToolContext with namespaces matching this toolkit's features.

        Args:
            state_key: Optional AG-UI state key to set on the context.

        Returns:
            A prepared ToolContext.
        """
        context = ToolContext()
        prepare_context(context, features=self.features, state_key=state_key)
        return context

    def prepare(self, context: ToolContext, state_key: str | None = None) -> None:
        """Register namespaces on an existing ToolContext for this toolkit's features.

        Idempotent — safe to call multiple times on the same context.

        Args:
            context: ToolContext to prepare.
            state_key: Optional AG-UI state key to set on the context.
        """
        prepare_context(context, features=self.features, state_key=state_key)


def build_toolkit(
    config: AppConfig,
    features: list[str] | None = None,
    base_filter: str | None = None,
    expand_context: bool = True,
    on_qa_complete: Callable | None = None,
) -> Toolkit:
    """Build a Toolkit with toolsets, prompt, and context factory for the given features.

    Args:
        config: Application configuration.
        features: List of features to enable. Defaults to ["search", "documents"].
        base_filter: Optional base SQL WHERE clause applied to all toolset factories.
        expand_context: Whether to expand search results with surrounding context.
        on_qa_complete: Optional callback invoked after each QA cycle.

    Returns:
        A Toolkit ready for agent composition.
    """
    if features is None:
        features = [FEATURE_SEARCH, FEATURE_DOCUMENTS]

    toolsets: list[FunctionToolset] = []

    if FEATURE_SEARCH in features:
        from haiku.rag.tools.search import create_search_toolset

        toolsets.append(
            create_search_toolset(
                config, expand_context=expand_context, base_filter=base_filter
            )
        )

    if FEATURE_DOCUMENTS in features:
        from haiku.rag.tools.document import create_document_toolset

        toolsets.append(create_document_toolset(config, base_filter=base_filter))

    if FEATURE_QA in features:
        from haiku.rag.tools.qa import create_qa_toolset

        toolsets.append(
            create_qa_toolset(
                config, base_filter=base_filter, on_ask_complete=on_qa_complete
            )
        )

    if FEATURE_ANALYSIS in features:
        from haiku.rag.tools.analysis import create_analysis_toolset

        toolsets.append(create_analysis_toolset(config, base_filter=base_filter))

    prompt = build_tools_prompt(features)

    return Toolkit(toolsets=toolsets, prompt=prompt, features=features)
