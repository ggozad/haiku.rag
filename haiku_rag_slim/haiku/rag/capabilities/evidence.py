from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai import RunContext

from haiku.rag.capabilities._base import RAGCapabilityBase
from haiku.rag.capabilities.ledger import CapabilityEvidenceRecord
from haiku.rag.store.models.citation import Citation


@dataclass(frozen=True)
class DiscoveredEvidence:
    """One evidence capability's records, as another capability found them.

    Read-only and rebuilt per request: whoever discovers these merges them into a
    view and persists nothing about evidence itself.
    """

    capability: str
    record: CapabilityEvidenceRecord
    citations: Mapping[str, Citation]
    tool_names: frozenset[str]
    cite_available: bool


def discover_evidence(ctx: RunContext[Any]) -> list[DiscoveredEvidence]:
    """Read what each evidence capability recorded, without writing anything.

    Discovery runs one way through the run's capability registry, so no capability
    holds a reference to another, and a host running one, both, or neither needs no
    wiring change. The registry holds the per-run instances, which are the ones
    carrying state; the registered objects never do. That includes a deferred
    capability the model has not loaded, whose record is simply empty.
    """
    discovered = [
        DiscoveredEvidence(
            capability=capability.state_namespace,
            record=cast(CapabilityEvidenceRecord, cast(Any, capability.state).evidence),
            citations=cast(Any, capability.state).citation_index,
            tool_names=frozenset(capability.evidence_tool_names()),
            cite_available=capability.cite_available,
        )
        for capability in ctx.capabilities.values()
        if isinstance(capability, RAGCapabilityBase)
    ]
    return sorted(discovered, key=lambda evidence: evidence.capability)


def question_in_progress(evidence: list[DiscoveredEvidence]) -> int:
    """The identity every evidence capability agrees this question has.

    They all derive it from the same history, so they agree; taking the maximum
    rather than a first entry keeps the result independent of ordering.
    """
    return max((found.record.question or 0 for found in evidence), default=0)


__all__ = [
    "DiscoveredEvidence",
    "discover_evidence",
    "question_in_progress",
]
