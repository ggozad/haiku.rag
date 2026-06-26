import asyncio
import json
from enum import StrEnum
from pathlib import Path

import httpx
import numpy as np
from pydantic import BaseModel, Field

from haiku.rag.config import AppConfig
from haiku.rag.config.models import DuplicateDetectionConfig
from haiku.rag.store.engine import (
    REQUIRED_TABLES,
    Store,
    connect_lancedb,
    get_database_stats,
)
from haiku.rag.store.repositories.settings import SettingsRepository
from haiku.rag.store.upgrades import get_pending_upgrades

# Cap how many offending ids we collect per check; doctor is a summary, not a dump.
_SAMPLE_LIMIT = 5

# API providers and the environment variable that carries their key.
_PROVIDER_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "cohere": "CO_API_KEY",
    "voyageai": "VOYAGE_API_KEY",
    "jina": "JINA_API_KEY",
    "zeroentropy": "ZEROENTROPY_API_KEY",
}

# Providers backed by in-process local models — no endpoint to probe.
_LOCAL_PROVIDERS = {"sentence-transformers", "mxbai", "cross-encoder", "jina-local"}

# Item labels that never yield a standalone chunk: pictures (handled via the
# image path), headings (folded into chunk context, not embedded alone), and
# page furniture. A document whose only items carry these labels is expected to
# have no chunks.
_NON_BODY_LABELS = {
    "picture",
    "section_header",
    "title",
    "page_header",
    "page_footer",
    "caption",
}

# Operators care whether an endpoint answers now, not eventually.
_PROBE_TIMEOUT_S = 2.0


class Severity(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


class CheckResult(BaseModel):
    name: str
    severity: Severity
    message: str
    remediation: str | None = None
    details: list[str] = Field(default_factory=list)


class DoctorReport(BaseModel):
    results: list[CheckResult] = Field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(r.severity is Severity.FAIL for r in self.results)

    def count(self, severity: Severity) -> int:
        return sum(1 for r in self.results if r.severity is severity)


def _sample(ids: list[str]) -> list[str]:
    """Cap a list of offending ids for display, noting how many were elided."""
    if len(ids) <= _SAMPLE_LIMIT:
        return list(ids)
    extra = len(ids) - _SAMPLE_LIMIT
    return [*ids[:_SAMPLE_LIMIT], f"... (+{extra} more)"]


def _active_models(config: AppConfig) -> list[tuple[str, str, str | None]]:
    """(provider, name, base_url) for every model role the config activates.

    Picture-description and title models are only included when their feature
    is enabled (``processing.pictures == "description"`` / ``auto_title``), so
    doctor checks exactly the providers the next ingest will use.
    """
    models = [
        (
            config.embeddings.model.provider,
            config.embeddings.model.name,
            config.embeddings.model.base_url,
        )
    ]
    for model in (config.reranking.model, config.qa.model, config.analysis.model):
        if model is not None:
            models.append((model.provider, model.name, model.base_url))

    proc = config.processing
    if proc.pictures == "description":
        pd = proc.conversion_options.picture_description.model
        models.append((pd.provider, pd.name, pd.base_url))
    if proc.auto_title:
        tm = proc.title_model
        models.append((tm.provider, tm.name, tm.base_url))
    return models


def _check_api_keys(config: AppConfig, environ: dict[str, str]) -> CheckResult:
    # A custom base_url points at a self-hosted OpenAI-compatible endpoint that
    # uses a placeholder key, so the SaaS key is only required when a provider
    # is used without one. Reachability of custom endpoints is the probe's job.
    need_key = {
        provider
        for provider, _name, base_url in _active_models(config)
        if not base_url and provider in _PROVIDER_ENV_VARS
    }
    missing = [
        f"{provider} ({_PROVIDER_ENV_VARS[provider]})"
        for provider in sorted(need_key)
        if not environ.get(_PROVIDER_ENV_VARS[provider])
    ]
    if missing:
        return CheckResult(
            name="api_keys",
            severity=Severity.FAIL,
            message="Configured providers are missing their API key.",
            remediation="Set the listed environment variables.",
            details=missing,
        )
    return CheckResult(
        name="api_keys",
        severity=Severity.OK,
        message="API keys present for all configured providers.",
    )


def _check_tables_present(stats: dict) -> CheckResult:
    missing = [name for name in REQUIRED_TABLES if not stats[name]["exists"]]
    if missing:
        return CheckResult(
            name="tables_present",
            severity=Severity.FAIL,
            message="Required tables are missing.",
            remediation="Run 'haiku-rag init' for a new database or 'haiku-rag migrate'.",
            details=missing,
        )
    return CheckResult(
        name="tables_present",
        severity=Severity.OK,
        message="All required tables are present.",
    )


def _classify_unchunked(
    no_chunk_ids: set[str],
    labels_by_doc: dict[str, set[str]],
    supports_images: bool,
) -> list[CheckResult]:
    """Classify chunk-less documents by what they hold.

    A document with body-text items but no chunks is always a problem. A
    picture-only document is a problem under a multimodal embedder (its picture
    chunks are missing) and an indexing gap under a text-only embedder (which
    cannot embed images). A document carrying only headings/furniture (or no
    items at all) is expected to have no chunks.
    """
    text_docs: list[str] = []
    picture_docs: list[str] = []
    for doc_id in no_chunk_ids:
        labels = labels_by_doc.get(doc_id, set())
        if any(label not in _NON_BODY_LABELS for label in labels):
            text_docs.append(doc_id)
        elif "picture" in labels:
            picture_docs.append(doc_id)

    results: list[CheckResult] = []
    if text_docs:
        results.append(
            CheckResult(
                name="documents_text_no_chunks",
                severity=Severity.WARN,
                message=f"{len(text_docs)} document(s) have text content but no chunks.",
                remediation="haiku-rag rebuild",
                details=_sample(sorted(text_docs)),
            )
        )
    if picture_docs and supports_images:
        results.append(
            CheckResult(
                name="documents_pictures_no_chunks",
                severity=Severity.WARN,
                message=f"{len(picture_docs)} document(s) with pictures have no chunks.",
                remediation="haiku-rag rebuild",
                details=_sample(sorted(picture_docs)),
            )
        )
    elif picture_docs:
        results.append(
            CheckResult(
                name="documents_images_unsearchable",
                severity=Severity.WARN,
                message=(
                    f"{len(picture_docs)} image-only document(s) have no chunks; "
                    "a text-only embedder cannot index images."
                ),
                remediation=(
                    "Set embeddings.model.multimodal: true on a vllm, voyageai, or "
                    "cohere model and rebuild to index images."
                ),
                details=_sample(sorted(picture_docs)),
            )
        )
    if not results:
        results.append(
            CheckResult(
                name="documents_without_chunks",
                severity=Severity.OK,
                message="Every document with content has chunks.",
            )
        )
    return results


async def _column_values(table, column: str) -> list:
    rows = await table.query().select([column]).to_list()
    return [row[column] for row in rows]


# Backstop on how many centroid candidate pairs we verify, bounding memory and
# runtime on a pathologically self-similar corpus.
MAX_CANDIDATE_PAIRS = 200_000


class _DuplicateFamily(BaseModel):
    members: list[str]
    superset: str
    pairs: list[tuple[str, str, float, float]]


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _containment(source: np.ndarray, target: np.ndarray, twin: float) -> float:
    """Fraction of ``source`` chunks with a near-identical chunk in ``target``."""
    return float(((source @ target.T).max(axis=1) >= twin).mean())


def _duplicate_families(
    doc_vectors: dict[str, np.ndarray], cfg: DuplicateDetectionConfig
) -> list[_DuplicateFamily]:
    """Cluster documents that share most of their chunks (revisions of one another).

    Block-then-verify: cheap centroid similarity proposes candidate document
    pairs, then directed chunk-overlap containment confirms them. Returns one
    entry per connected component of confirmed pairs.
    """
    # Normalize, drop unembedded (zero) vectors, apply the small-document floor.
    normalized: dict[str, np.ndarray] = {}
    for doc_id, matrix in doc_vectors.items():
        m = np.asarray(matrix, dtype=float)
        if m.ndim != 2 or m.shape[0] == 0:
            continue
        norms = np.linalg.norm(m, axis=1)
        m = m[norms > 0]
        if m.shape[0] < cfg.min_chunks:
            continue
        normalized[doc_id] = m / np.linalg.norm(m, axis=1)[:, None]
    if len(normalized) < 2:
        return []

    order = sorted(normalized)
    centroids = np.array([_unit(normalized[d].mean(axis=0)) for d in order])

    # Stage 1: centroid candidate pairs, block-wise to avoid a full D×D matrix.
    candidates: list[tuple[int, int]] = []
    block = 512
    for start in range(0, len(order), block):
        sims = centroids[start : start + block] @ centroids.T
        for row in range(sims.shape[0]):
            gi = start + row
            above = np.nonzero(sims[row, gi + 1 :] >= cfg.candidate_threshold)[0]
            candidates.extend((gi, gi + 1 + int(j)) for j in above)
        if len(candidates) >= MAX_CANDIDATE_PAIRS:
            candidates = candidates[:MAX_CANDIDATE_PAIRS]
            break

    # Stage 2: confirm candidates with directed chunk-overlap containment.
    adjacency: dict[int, set[int]] = {}
    edges: dict[tuple[int, int], tuple[float, float]] = {}
    for i, j in candidates:
        a_to_b = _containment(
            normalized[order[i]], normalized[order[j]], cfg.twin_similarity
        )
        b_to_a = _containment(
            normalized[order[j]], normalized[order[i]], cfg.twin_similarity
        )
        if max(a_to_b, b_to_a) >= cfg.containment_threshold:
            adjacency.setdefault(i, set()).add(j)
            adjacency.setdefault(j, set()).add(i)
            edges[(i, j)] = (a_to_b, b_to_a)
    if not edges:
        return []

    # Cluster confirmed pairs into families (connected components).
    families: list[_DuplicateFamily] = []
    seen: set[int] = set()
    for node in adjacency:
        if node in seen:
            continue
        component: set[int] = set()
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            component.add(cur)
            stack.extend(adjacency[cur] - seen)
        members = sorted(order[i] for i in component)
        # Largest document (most chunks) is the likely superset; smallest id on a tie.
        superset = min(members, key=lambda d: (-normalized[d].shape[0], d))
        pairs = sorted(
            (order[i], order[j], round(ab, 3), round(ba, 3))
            for (i, j), (ab, ba) in edges.items()
            if i in component and j in component
        )
        families.append(
            _DuplicateFamily(members=members, superset=superset, pairs=pairs)
        )
    return sorted(families, key=lambda f: f.members)


def _check_duplicate_documents(
    doc_vectors: dict[str, np.ndarray],
    uri_by_doc: dict[str, str | None],
    title_by_doc: dict[str, str | None],
    cfg: DuplicateDetectionConfig,
) -> CheckResult:
    families = _duplicate_families(doc_vectors, cfg)
    if not families:
        return CheckResult(
            name="duplicate_documents",
            severity=Severity.OK,
            message="No near-duplicate documents detected.",
        )

    def label(doc_id: str) -> str:
        return uri_by_doc.get(doc_id) or title_by_doc.get(doc_id) or doc_id

    lines: list[str] = []
    for family in families:
        members = ", ".join(label(m) for m in family.members)
        overlaps = "; ".join(
            f"{label(a)}→{label(b)}: {ab:.2f}, {label(b)}→{label(a)}: {ba:.2f}"
            for a, b, ab, ba in family.pairs
        )
        lines.append(f"[{members}] keep≈{label(family.superset)} ({overlaps})")

    total_docs = sum(len(f.members) for f in families)
    return CheckResult(
        name="duplicate_documents",
        severity=Severity.WARN,
        message=(
            f"{len(families)} group(s) of documents with substantial chunk overlap "
            f"(potential duplicates/revisions), {total_docs} documents."
        ),
        remediation=(
            "Review each group and remove redundant revisions; overlap may be intentional."
        ),
        details=_sample(lines),
    )


async def run_db_checks(
    store: Store, config: AppConfig, stats: dict
) -> list[CheckResult]:
    """Referential and content-integrity checks against an open read-only Store.

    Assumes all required tables exist (the caller short-circuits otherwise).
    """
    results: list[CheckResult] = []

    doc_ids = set(await _column_values(store.documents_table, "id"))
    meta_rows = (
        await store.document_meta_table.query()
        .select(["document_id", "metadata", "uri", "title"])
        .to_list()
    )
    meta_doc_ids = {row["document_id"] for row in meta_rows}
    content_type_by_doc = {
        row["document_id"]: json.loads(row.get("metadata") or "{}").get(
            "content_type", ""
        )
        for row in meta_rows
    }
    uri_by_doc = {row["document_id"]: row.get("uri") for row in meta_rows}
    title_by_doc = {row["document_id"]: row.get("title") for row in meta_rows}

    chunk_rows = (
        await store.chunks_table.query()
        .select(["id", "document_id", "metadata"])
        .to_list()
    )
    chunk_doc_ids = {row["document_id"] for row in chunk_rows}

    item_rows = (
        await store.document_items_table.query()
        .select(["document_id", "self_ref", "label"])
        .to_list()
    )
    item_doc_ids = {row["document_id"] for row in item_rows}
    self_refs_by_doc: dict[str, set[str]] = {}
    labels_by_doc: dict[str, set[str]] = {}
    for row in item_rows:
        self_refs_by_doc.setdefault(row["document_id"], set()).add(row["self_ref"])
        labels_by_doc.setdefault(row["document_id"], set()).add(row["label"])

    # documents <-> document_meta must be 1:1.
    orphan_docs = doc_ids - meta_doc_ids
    orphan_meta = meta_doc_ids - doc_ids
    if orphan_docs or orphan_meta:
        details = [f"document with no meta: {d}" for d in _sample(sorted(orphan_docs))]
        details += [f"meta with no document: {d}" for d in _sample(sorted(orphan_meta))]
        results.append(
            CheckResult(
                name="document_meta_parity",
                severity=Severity.FAIL,
                message="documents and document_meta are out of sync.",
                remediation="haiku-rag rebuild",
                details=details,
            )
        )
    else:
        results.append(
            CheckResult(
                name="document_meta_parity",
                severity=Severity.OK,
                message="documents and document_meta are consistent.",
            )
        )

    # Orphaned chunks / items reference a document that no longer exists.
    orphan_chunk_docs = chunk_doc_ids - doc_ids
    results.append(
        CheckResult(
            name="orphaned_chunks",
            severity=Severity.FAIL if orphan_chunk_docs else Severity.OK,
            message=(
                "Chunks reference missing documents."
                if orphan_chunk_docs
                else "No orphaned chunks."
            ),
            remediation="haiku-rag rebuild" if orphan_chunk_docs else None,
            details=_sample(sorted(orphan_chunk_docs)),
        )
    )

    orphan_item_docs = item_doc_ids - doc_ids
    results.append(
        CheckResult(
            name="orphaned_document_items",
            severity=Severity.FAIL if orphan_item_docs else Severity.OK,
            message=(
                "Document items reference missing documents."
                if orphan_item_docs
                else "No orphaned document items."
            ),
            remediation="haiku-rag rebuild" if orphan_item_docs else None,
            details=_sample(sorted(orphan_item_docs)),
        )
    )

    # Documents with no chunks, classified by what they contain and whether the
    # embedder can index images.
    results += _classify_unchunked(
        doc_ids - chunk_doc_ids, labels_by_doc, store.embedder.supports_images
    )

    # A chunked document must have items; one without them is corrupt. Empty
    # documents legitimately have neither, so only flag the chunked ones.
    docs_missing_items = (doc_ids & chunk_doc_ids) - item_doc_ids
    results.append(
        CheckResult(
            name="documents_without_items",
            severity=Severity.WARN if docs_missing_items else Severity.OK,
            message=(
                f"{len(docs_missing_items)} chunked document(s) have no document items."
                if docs_missing_items
                else "Every chunked document has document items."
            ),
            remediation="haiku-rag rebuild" if docs_missing_items else None,
            details=_sample(sorted(docs_missing_items)),
        )
    )

    # Chunk metadata may reference self_refs that do not exist for that document.
    dangling: list[str] = []
    for row in chunk_rows:
        refs = json.loads(row.get("metadata") or "{}").get("doc_item_refs") or []
        known = self_refs_by_doc.get(row["document_id"], set())
        if any(ref not in known for ref in refs):
            dangling.append(row["id"])
    results.append(
        CheckResult(
            name="dangling_doc_item_refs",
            severity=Severity.FAIL if dangling else Severity.OK,
            message=(
                f"{len(dangling)} chunk(s) reference missing document items."
                if dangling
                else "All chunk doc_item_refs resolve."
            ),
            remediation="haiku-rag rebuild" if dangling else None,
            details=_sample(dangling),
        )
    )

    # Vector dimension consistency and unembedded (all-zero) vectors share one
    # scan of the vector column — the heaviest check on large corpora.
    arrow = (
        await store.chunks_table.query()
        .select(["id", "vector", "document_id"])
        .to_arrow()
    )
    stored = await SettingsRepository(store).get_current_settings()
    stored_dim = stored.get("embeddings", {}).get("model", {}).get("vector_dim")
    actual_dim = arrow.schema.field("vector").type.list_size
    if stored_dim and stored_dim != actual_dim:
        results.append(
            CheckResult(
                name="vector_dimension",
                severity=Severity.FAIL,
                message=(
                    f"Chunk vector size {actual_dim} does not match stored "
                    f"vector_dim {stored_dim}."
                ),
                remediation="haiku-rag rebuild",
            )
        )
    else:
        results.append(
            CheckResult(
                name="vector_dimension",
                severity=Severity.OK,
                message=f"Chunk vectors are {actual_dim}-dimensional.",
            )
        )

    ids = arrow.column("id").to_pylist()
    vectors = np.asarray(arrow.column("vector").to_pylist(), dtype=float)
    zero_ids: list[str] = []
    if vectors.size:
        zero_ids = [ids[i] for i in np.nonzero(~vectors.any(axis=1))[0]]
    results.append(
        CheckResult(
            name="unembedded_chunks",
            severity=Severity.WARN if zero_ids else Severity.OK,
            message=(
                f"{len(zero_ids)} chunk(s) have an all-zero (unembedded) vector."
                if zero_ids
                else "All chunks are embedded."
            ),
            remediation="haiku-rag rebuild --embed-only" if zero_ids else None,
            details=_sample(zero_ids),
        )
    )

    # Near-duplicate documents (revisions sharing most chunks), grouped from the
    # same vector scan rather than a second pass.
    chunk_doc_ids_ordered = arrow.column("document_id").to_pylist()
    indices_by_doc: dict[str, list[int]] = {}
    for index, doc_id in enumerate(chunk_doc_ids_ordered):
        indices_by_doc.setdefault(doc_id, []).append(index)
    doc_vectors = {doc_id: vectors[idx] for doc_id, idx in indices_by_doc.items()}
    results.append(
        _check_duplicate_documents(
            doc_vectors, uri_by_doc, title_by_doc, config.doctor.duplicates
        )
    )

    # Pictures from image/PDF sources should carry raster bytes. Pictures that
    # are external image references in a text document (markdown, HTML) have no
    # embedded bytes by nature, so a missing raster there is expected.
    missing_picture_docs = [
        row["document_id"]
        for row in await store.document_items_table.query()
        .select(["document_id"])
        .where("label = 'picture' AND picture_data IS NULL")
        .to_list()
    ]
    real_missing = [
        doc_id
        for doc_id in missing_picture_docs
        if not content_type_by_doc.get(doc_id, "").startswith("text/")
    ]
    results.append(
        CheckResult(
            name="picture_data",
            severity=Severity.WARN if real_missing else Severity.OK,
            message=(
                f"{len(real_missing)} picture item(s) in image/PDF documents "
                "have no image data."
                if real_missing
                else "Pictures that should carry image data have it."
            ),
            remediation="haiku-rag rebuild" if real_missing else None,
            details=_sample(sorted(set(real_missing))),
        )
    )

    # Settings must hold exactly one canonical row.
    total_settings = await store.settings_table.count_rows()
    canonical = len(
        await store.settings_table.query().where("id = 'settings'").to_list()
    )
    if total_settings == 0 or canonical != 1:
        results.append(
            CheckResult(
                name="settings_row",
                severity=Severity.FAIL,
                message=(
                    f"Expected exactly one 'settings' row, found {canonical} "
                    f"(of {total_settings} total)."
                ),
                remediation="haiku-rag migrate",
            )
        )
    else:
        results.append(
            CheckResult(
                name="settings_row",
                severity=Severity.OK,
                message="Settings row is present.",
            )
        )

    results.append(_check_embedding_drift(stored, config))

    stored_version = str(stored.get("version", "unknown"))
    pending = (
        get_pending_upgrades(stored_version) if stored_version != "unknown" else []
    )
    results.append(
        CheckResult(
            name="pending_migrations",
            severity=Severity.WARN if pending else Severity.OK,
            message=(
                f"{len(pending)} migration(s) pending (db version {stored_version})."
                if pending
                else f"Database is up to date (version {stored_version})."
            ),
            remediation="haiku-rag migrate" if pending else None,
            details=[f"{step.version}: {step.description or ''}" for step in pending],
        )
    )

    results.append(_check_vector_index(stats))

    return results


def _check_embedding_drift(stored: dict, config: AppConfig) -> CheckResult:
    stored_model = stored.get("embeddings", {}).get("model", {})
    current_model = config.embeddings.model
    if not stored_model:
        return CheckResult(
            name="embedding_drift",
            severity=Severity.OK,
            message="No stored embedding identity to compare.",
        )

    stored_dim = stored_model.get("vector_dim")
    if stored_dim and stored_dim != current_model.vector_dim:
        return CheckResult(
            name="embedding_drift",
            severity=Severity.FAIL,
            message=(
                f"Embedding vector_dim differs: stored {stored_dim} -> "
                f"config {current_model.vector_dim}."
            ),
            remediation="haiku-rag rebuild",
        )

    drift: list[str] = []
    if stored_model.get("provider") not in (None, current_model.provider):
        drift.append(
            f"provider: {stored_model['provider']} -> {current_model.provider}"
        )
    if stored_model.get("name") not in (None, current_model.name):
        drift.append(f"name: {stored_model['name']} -> {current_model.name}")
    if drift:
        return CheckResult(
            name="embedding_drift",
            severity=Severity.WARN,
            message="Embedding identity differs from config (vector_dim matches).",
            remediation="haiku-rag rebuild --set-embedder",
            details=drift,
        )
    return CheckResult(
        name="embedding_drift",
        severity=Severity.OK,
        message="Embedding identity matches the stored settings.",
    )


def _check_vector_index(stats: dict) -> CheckResult:
    chunks = stats["chunks"]
    num_chunks = chunks.get("num_rows", 0)
    if not chunks.get("has_vector_index"):
        if num_chunks >= 100_000:
            return CheckResult(
                name="vector_index",
                severity=Severity.WARN,
                message=(
                    "No vector index on a large collection; "
                    "similarity search scans every chunk and may be slow."
                ),
                remediation="haiku-rag create-index",
            )
        return CheckResult(
            name="vector_index",
            severity=Severity.OK,
            message="No vector index; similarity search is exact (brute-force).",
        )
    unindexed = chunks.get("num_unindexed_rows", 0)
    if unindexed > 0:
        return CheckResult(
            name="vector_index",
            severity=Severity.WARN,
            message=f"{unindexed} chunk(s) are not in the vector index.",
            remediation="haiku-rag create-index",
        )
    return CheckResult(
        name="vector_index",
        severity=Severity.OK,
        message="Vector index covers all chunks.",
    )


def _resolve_endpoint(
    provider: str, base_url: str | None, ollama_base: str
) -> tuple[str, str, str] | str | None:
    """Map a model's provider to a probe target.

    Returns ``(probe_url, kind, display)``, the literal ``"local"`` for an
    in-process model, or ``None`` for a SaaS provider covered by the API-key
    check.
    """
    if provider == "ollama":
        base = (base_url or ollama_base).rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3].rstrip("/")
        return f"{base}/api/tags", "ollama", base
    if provider == "vllm":
        base = (base_url or "http://localhost:8000/v1").rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return f"{base}/models", "openai", base
    if provider == "openai" and base_url:
        base = base_url.rstrip("/")
        return f"{base}/models", "openai", base
    if provider in _LOCAL_PROVIDERS:
        return "local"
    return None


def _provider_targets(
    config: AppConfig,
) -> tuple[dict[str, dict], set[str]]:
    """Collect probe targets (keyed by probe URL) and local-only providers."""
    targets: dict[str, dict] = {}
    local: set[str] = set()
    ollama_base = config.providers.ollama.base_url

    def add_model(provider: str, name: str, base_url: str | None) -> None:
        resolved = _resolve_endpoint(provider, base_url, ollama_base)
        if resolved is None:
            return
        if resolved == "local":
            local.add(provider)
            return
        probe_url, kind, display = resolved
        entry = targets.setdefault(
            probe_url, {"kind": kind, "display": display, "models": set()}
        )
        if name:
            entry["models"].add(name)

    proc = config.processing
    if proc.converter == "docling-serve" or proc.chunker == "docling-serve":
        for url in config.providers.docling_serve.base_urls:
            base = url.rstrip("/")
            targets.setdefault(
                f"{base}/health",
                {"kind": "docling-serve", "display": base, "models": set()},
            )

    for provider, name, base_url in _active_models(config):
        add_model(provider, name, base_url)

    return targets, local


def _model_present(expected: str, available: set[str]) -> bool:
    if expected in available:
        return True
    if ":" not in expected:
        return any(a.split(":", 1)[0] == expected for a in available)
    return False


async def _probe_endpoint(
    client: httpx.AsyncClient, url: str
) -> tuple[bool, str | None, dict | None]:
    try:
        response = await client.get(url)
    except httpx.HTTPError as exc:
        return False, str(exc), None
    if not response.is_success:
        return False, f"HTTP {response.status_code}", None
    try:
        return True, None, response.json()
    except ValueError:
        return True, None, None


def _endpoint_result(
    entry: dict, reachable: bool, error: str | None, payload: dict | None
) -> CheckResult:
    kind = entry["kind"]
    display = entry["display"]
    name = f"provider:{display}"
    if not reachable:
        return CheckResult(
            name=name,
            severity=Severity.FAIL,
            message=f"{kind} at {display} is unreachable.",
            remediation="Start the service or fix the configured base_url.",
            details=[error] if error else [],
        )
    if kind == "ollama":
        available = {m.get("name", "") for m in (payload or {}).get("models", [])}
        missing = [
            model
            for model in sorted(entry["models"])
            if not _model_present(model, available)
        ]
        if missing:
            return CheckResult(
                name=name,
                severity=Severity.WARN,
                message=f"ollama at {display} is reachable but missing model(s).",
                remediation="ollama pull <model>",
                details=missing,
            )
    return CheckResult(
        name=name,
        severity=Severity.OK,
        message=f"{kind} at {display} is reachable.",
    )


async def run_provider_checks(config: AppConfig) -> list[CheckResult]:
    """Probe the external endpoints the current config actually uses."""
    targets, local = _provider_targets(config)

    results: list[CheckResult] = []
    if targets:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S) as client:
            probes = await asyncio.gather(
                *(_probe_endpoint(client, url) for url in targets)
            )
        for url, (reachable, error, payload) in zip(targets, probes):
            results.append(_endpoint_result(targets[url], reachable, error, payload))

    for provider in sorted(local):
        results.append(
            CheckResult(
                name=f"provider:{provider}",
                severity=Severity.OK,
                message=f"{provider}: local model, nothing to probe.",
            )
        )
    return results


async def run_doctor(
    config: AppConfig, db_path: Path, environ: dict[str, str]
) -> DoctorReport:
    """Open the database read-only and run every diagnostic check.

    Opens with validation and migration checks skipped so a drifted or
    pre-migration database can still be diagnosed rather than refusing to open.
    """
    db = await connect_lancedb(config, db_path)
    stats = await get_database_stats(db)

    results: list[CheckResult] = []
    if not any(entry["exists"] for entry in stats.values()):
        results.append(
            CheckResult(
                name="tables_present",
                severity=Severity.FAIL,
                message="Database is empty.",
                remediation="haiku-rag init",
            )
        )
    else:
        results.append(_check_tables_present(stats))
        missing = [name for name in REQUIRED_TABLES if not stats[name]["exists"]]
        if not missing:
            async with Store(
                db_path,
                config=config,
                skip_validation=True,
                read_only=True,
                skip_migration_check=True,
            ) as store:
                results += await run_db_checks(store, config, stats)

    results.append(_check_api_keys(config, environ))
    results += await run_provider_checks(config)
    return DoctorReport(results=results)
