import json
from enum import StrEnum
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from haiku.rag.config import AppConfig
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


def _configured_providers(config: AppConfig) -> set[str]:
    """Providers referenced by the current config across every model role."""
    providers = {config.embeddings.model.provider}
    for model in (
        config.reranking.model,
        config.qa.model,
        config.analysis.model,
    ):
        if model is not None:
            providers.add(model.provider)
    return providers


def _check_api_keys(config: AppConfig, environ: dict[str, str]) -> CheckResult:
    missing: list[str] = []
    for provider in sorted(_configured_providers(config)):
        env_var = _PROVIDER_ENV_VARS.get(provider)
        if env_var and not environ.get(env_var):
            missing.append(f"{provider} ({env_var})")
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


async def _column_values(table, column: str) -> list:
    rows = await table.query().select([column]).to_list()
    return [row[column] for row in rows]


async def run_db_checks(
    store: Store, config: AppConfig, stats: dict
) -> list[CheckResult]:
    """Referential and content-integrity checks against an open read-only Store.

    Assumes all required tables exist (the caller short-circuits otherwise).
    """
    results: list[CheckResult] = []

    doc_ids = set(await _column_values(store.documents_table, "id"))
    meta_doc_ids = set(await _column_values(store.document_meta_table, "document_id"))

    chunk_rows = (
        await store.chunks_table.query()
        .select(["id", "document_id", "metadata"])
        .to_list()
    )
    chunk_doc_ids = {row["document_id"] for row in chunk_rows}

    item_rows = (
        await store.document_items_table.query()
        .select(["document_id", "self_ref"])
        .to_list()
    )
    item_doc_ids = {row["document_id"] for row in item_rows}
    self_refs_by_doc: dict[str, set[str]] = {}
    for row in item_rows:
        self_refs_by_doc.setdefault(row["document_id"], set()).add(row["self_ref"])

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

    # Documents that never produced chunks / items.
    docs_without_chunks = doc_ids - chunk_doc_ids
    results.append(
        CheckResult(
            name="documents_without_chunks",
            severity=Severity.WARN if docs_without_chunks else Severity.OK,
            message=(
                f"{len(docs_without_chunks)} document(s) have no chunks."
                if docs_without_chunks
                else "Every document has chunks."
            ),
            remediation="haiku-rag rebuild" if docs_without_chunks else None,
            details=_sample(sorted(docs_without_chunks)),
        )
    )

    docs_without_items = doc_ids - item_doc_ids
    results.append(
        CheckResult(
            name="documents_without_items",
            severity=Severity.WARN if docs_without_items else Severity.OK,
            message=(
                f"{len(docs_without_items)} document(s) have no document items."
                if docs_without_items
                else "Every document has document items."
            ),
            remediation="haiku-rag rebuild" if docs_without_items else None,
            details=_sample(sorted(docs_without_items)),
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
    arrow = await store.chunks_table.query().select(["id", "vector"]).to_arrow()
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

    # Pictures should carry their raster bytes after extraction.
    total_pictures = await store.document_items_table.count_rows("label = 'picture'")
    missing_pictures = len(
        await store.document_items_table.query()
        .select(["self_ref"])
        .where("label = 'picture' AND picture_data IS NULL")
        .to_list()
    )
    results.append(
        CheckResult(
            name="picture_data",
            severity=Severity.WARN if missing_pictures else Severity.OK,
            message=(
                f"{missing_pictures} of {total_pictures} picture item(s) "
                "have no image data."
                if missing_pictures
                else f"All {total_pictures} picture item(s) have image data."
            ),
            remediation="haiku-rag rebuild" if missing_pictures else None,
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
        if num_chunks >= 256:
            return CheckResult(
                name="vector_index",
                severity=Severity.WARN,
                message="No vector index; similarity search falls back to a scan.",
                remediation="haiku-rag create-index",
            )
        return CheckResult(
            name="vector_index",
            severity=Severity.OK,
            message=f"No vector index yet (need {256 - num_chunks} more chunks).",
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


async def run_doctor(
    config: AppConfig, db_path: Path, environ: dict[str, str]
) -> DoctorReport:
    """Open the database read-only and run every diagnostic check.

    Opens with validation and migration checks skipped so a drifted or
    pre-migration database can still be diagnosed rather than refusing to open.
    """
    db = await connect_lancedb(config, db_path)
    stats = await get_database_stats(db)

    if not any(entry["exists"] for entry in stats.values()):
        return DoctorReport(
            results=[
                CheckResult(
                    name="tables_present",
                    severity=Severity.FAIL,
                    message="Database is empty.",
                    remediation="haiku-rag init",
                )
            ]
        )

    results = [_check_tables_present(stats)]
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
    return DoctorReport(results=results)
