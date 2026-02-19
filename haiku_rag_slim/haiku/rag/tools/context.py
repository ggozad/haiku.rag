from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG


@runtime_checkable
class RAGDeps(Protocol):
    """Contract for toolset dependencies injected via RunContext.

    Any deps object passed to an agent using haiku.rag toolsets must
    provide these attributes.
    """

    client: "HaikuRAG"
