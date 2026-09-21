import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from haiku.rag.ingester.queue.models import SyncRow
from haiku.rag.telemetry import logfire

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG
    from haiku.rag.ingester.queue.repository import SyncStateRepo

logger = logging.getLogger(__name__)

ATTRIBUTION_DOCS = "https://ggozad.github.io/haiku.rag/ingester/"


@dataclass
class SourceReconciliation:
    """What one source's sync_state and the document store disagreed on."""

    source_id: str
    recovered: int = 0
    invalidated: int = 0
    attributed: int = 0

    @property
    def drifted(self) -> bool:
        return bool(self.recovered or self.invalidated or self.attributed)


@dataclass
class _StoreIndex:
    """The document store as reconciliation reads it, from one listing."""

    uris: set[str] = field(default_factory=set)
    owner_by_uri: dict[str, str] = field(default_factory=dict)
    id_by_uri: dict[str, str] = field(default_factory=dict)

    @classmethod
    async def read(cls, rag: "HaikuRAG") -> "_StoreIndex":
        index = cls()
        for doc in await rag.list_documents():
            if doc.uri is None or doc.id is None:
                continue
            index.uris.add(doc.uri)
            index.id_by_uri[doc.uri] = doc.id
            owner = (doc.metadata or {}).get("source_id")
            if owner is not None:
                index.owner_by_uri[doc.uri] = owner
        return index

    @property
    def unattributed(self) -> set[str]:
        return self.uris - set(self.owner_by_uri)


@dataclass
class _SourceState:
    """One source's side of the comparison, read once."""

    known: set[str]
    revisioned: set[str]
    ingested: set[str]


def _warn_unconfigured(index: _StoreIndex, configured: set[str]) -> None:
    counts: dict[str, int] = {}
    for owner in index.owner_by_uri.values():
        if owner not in configured:
            counts[owner] = counts.get(owner, 0) + 1
    for source_id, count in sorted(counts.items()):
        logger.warning(
            "%d document(s) belong to source %s, which is not configured; "
            "they are not swept and not reconciled",
            count,
            source_id,
        )


def _claims(
    index: _StoreIndex, states: dict[str, _SourceState]
) -> dict[str, list[str]]:
    """Which sources successfully wrote each unattributed document."""
    claims: dict[str, list[str]] = {}
    for source_id, state in states.items():
        for uri in state.ingested & index.unattributed:
            claims.setdefault(uri, []).append(source_id)
    return claims


def _resolve_claims(claims: dict[str, list[str]]) -> dict[str, list[str]]:
    """Document ids to attribute, by source. A URI several sources wrote is
    left unattributed: their configured order is not evidence of ownership."""
    resolved: dict[str, list[str]] = {}
    for uri, claimants in sorted(claims.items()):
        if len(claimants) > 1:
            logger.warning(
                "%s was ingested by %s; leaving it unattributed until the "
                "overlapping sources are separated",
                uri,
                ", ".join(sorted(claimants)),
            )
            continue
        resolved.setdefault(claimants[0], []).append(uri)
    return resolved


async def reconcile(
    rag: "HaikuRAG", sync: "SyncStateRepo", source_ids: list[str]
) -> list[SourceReconciliation]:
    """Bring sync_state back in step with the document store.

    Three disagreements, each repaired in the direction the surviving record
    supports: a document a source owns with no sync_state row (the next sweep
    decides whether it is still at the source), a revision for a URI the store
    no longer holds (cleared, so the next sweep re-ingests), and a document
    with no source_id that a source successfully wrote (attributed, which is
    how a database written before attribution existed acquires it).

    Evidence of a successful write is `last_ingested_at`, never a revision: a
    revision is also written for a permanently failed job, so reading it as
    success would attribute documents the source never produced and would
    clear the marker that stops a dead URI being re-enqueued every sweep.
    """
    with logfire.span("ingester.reconcile") as span:
        index = await _StoreIndex.read(rag)
        span.set_attribute("documents", len(index.uris))
        _warn_unconfigured(index, set(source_ids))

        states = {
            source_id: _SourceState(
                known=await sync.list_known_uris(source_id),
                revisioned=set(await sync.get_revision_snapshot(source_id)),
                ingested=await sync.list_ingested_uris(source_id),
            )
            for source_id in source_ids
        }
        attributions = _resolve_claims(_claims(index, states))

        reports = []
        for source_id, state in states.items():
            report = SourceReconciliation(source_id=source_id)
            owned = {
                uri for uri, owner in index.owner_by_uri.items() if owner == source_id
            }

            recover = sorted(owned - state.known)
            if recover:
                await sync.batch_upsert(
                    [SyncRow(source_id, uri, None, None, False) for uri in recover]
                )
                report.recovered = len(recover)

            stale = sorted((state.ingested & state.revisioned) - index.uris)
            if stale:
                await sync.invalidate(source_id, stale)
                report.invalidated = len(stale)

            attribute = attributions.get(source_id, [])
            if attribute:
                await rag.set_document_source(
                    [index.id_by_uri[uri] for uri in attribute], source_id
                )
                report.attributed = len(attribute)

            if report.drifted:
                logger.info(
                    "Reconciled %s against the document store: "
                    "%d recovered, %d invalidated, %d attributed",
                    source_id,
                    report.recovered,
                    report.invalidated,
                    report.attributed,
                )
            reports.append(report)

        attributed = sum(r.attributed for r in reports)
        # The listing predates the attribution pass, so subtract what this run
        # claimed. What is left includes URIs several sources ingested.
        unattributed = len(index.unattributed) - attributed
        if unattributed:
            logger.warning(
                "%d document(s) remain without source attribution. Review "
                "whether they are intentionally unmanaged or have ambiguous "
                "or lost ownership. See %s",
                unattributed,
                ATTRIBUTION_DOCS,
            )

        span.set_attribute("recovered", sum(r.recovered for r in reports))
        span.set_attribute("invalidated", sum(r.invalidated for r in reports))
        span.set_attribute("attributed", attributed)
        span.set_attribute("unattributed", unattributed)
        return reports
