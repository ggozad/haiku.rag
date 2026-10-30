import json
from importlib import metadata

import httpx
import lancedb
import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from haiku.rag.cli import _cli as cli
from haiku.rag.config.models import (
    AppConfig,
    ConversionOptions,
    DoclingServeConfig,
    DoctorConfig,
    DuplicateDetectionConfig,
    EmbeddingModelConfig,
    EmbeddingsConfig,
    ModelConfig,
    PictureDescriptionConfig,
    ProcessingConfig,
    ProvidersConfig,
    QAConfig,
)
from haiku.rag.doctor import (
    CheckResult,
    DoctorReport,
    Severity,
    _active_models,
    _check_api_keys,
    _check_duplicate_documents,
    _check_embedding_drift,
    _check_vector_index,
    _model_present,
    _probe_endpoint,
    _provider_targets,
    _resolve_endpoint,
    _sample,
    duplicate_families,
    run_doctor,
    run_provider_checks,
)
from haiku.rag.store.schema import (
    DocumentItemRecord,
    DocumentMetaRecord,
    DocumentRecord,
    SettingsRecord,
    create_chunk_model,
)
from haiku.rag.store.upgrades import latest_schema_version

runner = CliRunner()

CURRENT_VERSION = metadata.version("haiku.rag-slim")
VECTOR_DIM = 4
ChunkRecord = create_chunk_model(VECTOR_DIM)


def _config(
    provider: str = "ollama",
    name: str = "test",
    vector_dim: int = VECTOR_DIM,
    multimodal: bool = False,
    query_prefix: str = "",
):
    return AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider=provider,
                name=name,
                vector_dim=vector_dim,
                multimodal=multimodal,
                query_prefix=query_prefix,
            )
        )
    )


async def _build_db(
    path,
    *,
    version: str = CURRENT_VERSION,
    provider: str = "ollama",
    name: str = "test",
    vector_dim: int = VECTOR_DIM,
    stored_vector_dim: int | None = None,
    fts_index: str | None = "covering",
    schema_version: str | None = None,
):
    """Create a consistent single-document database without touching an embedder.

    ``stored_vector_dim`` records a different dimension in settings than the
    chunks table actually uses, to exercise the vector-dimension check.
    ``fts_index`` places the chunks FTS index: "covering" builds it over the
    rows, "empty" builds it before them, None never builds it.
    ``schema_version`` is recorded only when given.
    """
    from lancedb.index import FTS

    db = await lancedb.connect_async(path)
    settings_tbl = await db.create_table("settings", schema=SettingsRecord)
    docs_tbl = await db.create_table("documents", schema=DocumentRecord)
    meta_tbl = await db.create_table("document_meta", schema=DocumentMetaRecord)
    chunks_tbl = await db.create_table("chunks", schema=create_chunk_model(vector_dim))
    items_tbl = await db.create_table("document_items", schema=DocumentItemRecord)
    if fts_index == "empty":
        await chunks_tbl.create_index("content_fts", config=FTS())

    await settings_tbl.add(
        [
            SettingsRecord(
                id="settings",
                settings=json.dumps(
                    {
                        "version": version,
                        **(
                            {"schema_version": schema_version} if schema_version else {}
                        ),
                        "embeddings": {
                            "model": {
                                "provider": provider,
                                "name": name,
                                "vector_dim": stored_vector_dim or vector_dim,
                            }
                        },
                    }
                ),
            )
        ]
    )
    await docs_tbl.add([DocumentRecord(id="d1", content="hello")])
    await meta_tbl.add([DocumentMetaRecord(id="d1", uri="test://d1")])
    await items_tbl.add(
        [
            DocumentItemRecord(
                document_id="d1", position=0, self_ref="#/texts/0", text="x"
            )
        ]
    )
    chunk_model = create_chunk_model(vector_dim)
    await chunks_tbl.add(
        [
            chunk_model(
                id="c1",
                document_id="d1",
                content="hello",
                metadata=json.dumps({"doc_item_refs": ["#/texts/0"]}),
                vector=[0.1] * vector_dim,
            )
        ]
    )
    if fts_index == "covering":
        await chunks_tbl.create_index("content_fts", config=FTS())
    return db


def _result(report: DoctorReport, name: str) -> CheckResult:
    return next(r for r in report.results if r.name == name)


@pytest.fixture(autouse=True)
def _stub_provider_probe(monkeypatch):
    """Default every provider probe to reachable with the test models present,
    so database-integrity tests don't depend on a live Ollama. Provider tests
    re-patch this with their own behavior."""

    async def probe(_client, _url, _headers):
        return (
            True,
            None,
            {
                "models": [
                    {"name": "test"},
                    {"name": "qwen3.8:latest"},
                    {"name": "qwen3-embedding:4b"},
                ]
            },
        )

    monkeypatch.setattr("haiku.rag.doctor._probe_endpoint", probe)


async def test_healthy_db_all_ok(temp_db_path):
    await _build_db(temp_db_path)
    report = await run_doctor(_config(), temp_db_path, {})
    assert not report.failed
    assert report.count(Severity.WARN) == 0
    assert all(r.severity is Severity.OK for r in report.results)


async def test_doctor_reports_progress(temp_db_path):
    await _build_db(temp_db_path)
    labels: list[str] = []
    await run_doctor(_config(), temp_db_path, {}, on_progress=labels.append)
    assert "Inspecting tables" in labels
    assert "Scanning chunk vectors" in labels
    assert "Detecting near-duplicate documents" in labels
    assert "Probing provider endpoints" in labels


async def test_empty_db_fails(temp_db_path):
    report = await run_doctor(_config(), temp_db_path, {})
    assert report.failed
    assert _result(report, "tables_present").message == "Database is empty."


async def test_missing_table_fails_without_opening_store(temp_db_path):
    db = await lancedb.connect_async(temp_db_path)
    await db.create_table("settings", schema=SettingsRecord)
    report = await run_doctor(_config(), temp_db_path, {})
    assert report.failed
    tables = _result(report, "tables_present")
    assert tables.severity is Severity.FAIL
    assert "documents" in tables.details


async def test_missing_fts_index_fails(temp_db_path):
    """optimize indexes the rows of an index that exists; it never creates a
    missing one, so the remediation has to build it."""
    await _build_db(temp_db_path, fts_index=None)
    report = await run_doctor(_config(), temp_db_path, {})
    result = _result(report, "fts_index_coverage")
    assert result.severity is Severity.FAIL
    assert "chunks.content_fts" in result.details[0]
    assert "no index over 1 rows" in result.details[0]
    assert "rebuild --reindex" in (result.remediation or "")
    assert "vacuum" not in (result.remediation or "")
    assert report.failed


async def test_fts_index_covering_no_rows_fails(temp_db_path):
    """The state a bulk write without a closing vacuum leaves behind."""
    await _build_db(temp_db_path, fts_index="empty")
    report = await run_doctor(_config(), temp_db_path, {})
    result = _result(report, "fts_index_coverage")
    assert result.severity is Severity.FAIL
    assert "0 of 1 rows indexed" in result.details[0]
    assert "vacuum" in (result.remediation or "")


async def test_fts_index_missing_rows_warns(temp_db_path):
    """Rows written after the index was built stay outside it until vacuum."""
    db = await _build_db(temp_db_path)
    chunks_tbl = await db.open_table("chunks")
    await chunks_tbl.add(
        [
            ChunkRecord(
                id="c2",
                document_id="d1",
                content="hello again",
                metadata=json.dumps({"doc_item_refs": ["#/texts/0"]}),
                vector=[0.1] * VECTOR_DIM,
            )
        ]
    )
    report = await run_doctor(_config(), temp_db_path, {})
    result = _result(report, "fts_index_coverage")
    assert result.severity is Severity.WARN
    assert result.details == ["chunks.content_fts: 1 of 2 rows indexed"]
    assert "rebuild --reindex" in (result.remediation or "")
    assert not report.failed


async def test_fts_coverage_passes_an_empty_table(temp_db_path):
    db = await _build_db(temp_db_path)
    chunks_tbl = await db.open_table("chunks")
    await chunks_tbl.delete("id = 'c1'")
    report = await run_doctor(_config(), temp_db_path, {})
    assert _result(report, "fts_index_coverage").severity is Severity.OK


async def test_orphaned_chunk_fails(temp_db_path):
    db = await _build_db(temp_db_path)
    chunks_tbl = await db.open_table("chunks")
    await chunks_tbl.add(
        [
            ChunkRecord(
                id="orphan",
                document_id="ghost",
                content="x",
                vector=[0.2] * VECTOR_DIM,
            )
        ]
    )
    report = await run_doctor(_config(), temp_db_path, {})
    result = _result(report, "orphaned_chunks")
    assert result.severity is Severity.FAIL
    assert "ghost" in result.details
    assert report.failed


async def test_orphaned_document_item_fails(temp_db_path):
    db = await _build_db(temp_db_path)
    items_tbl = await db.open_table("document_items")
    await items_tbl.add(
        [DocumentItemRecord(document_id="ghost", position=0, self_ref="#/texts/0")]
    )
    report = await run_doctor(_config(), temp_db_path, {})
    assert _result(report, "orphaned_document_items").severity is Severity.FAIL


async def _add_doc(db, doc_id, *, items, metadata=None, chunks=None):
    docs_tbl = await db.open_table("documents")
    meta_tbl = await db.open_table("document_meta")
    await docs_tbl.add([DocumentRecord(id=doc_id, content="x")])
    await meta_tbl.add(
        [
            DocumentMetaRecord(
                id=doc_id,
                uri=f"test://{doc_id}",
                metadata=json.dumps(metadata or {}),
            )
        ]
    )
    if items:
        items_tbl = await db.open_table("document_items")
        await items_tbl.add(items)
    if chunks:
        chunks_tbl = await db.open_table("chunks")
        await chunks_tbl.add(chunks)


async def test_document_with_text_but_no_chunks_fails(temp_db_path):
    db = await _build_db(temp_db_path)
    await _add_doc(
        db,
        "d2",
        items=[
            DocumentItemRecord(
                document_id="d2",
                position=0,
                self_ref="#/texts/0",
                label="text",
                text="real content",
            )
        ],
    )
    report = await run_doctor(_config(), temp_db_path, {})
    result = _result(report, "documents_text_no_chunks")
    assert result.severity is Severity.FAIL
    assert "d2" in result.details
    assert report.failed


async def test_empty_document_no_chunks_is_ok(temp_db_path):
    db = await _build_db(temp_db_path)
    await _add_doc(db, "d2", items=[])
    report = await run_doctor(_config(), temp_db_path, {})
    assert _result(report, "documents_without_chunks").severity is Severity.OK


async def test_heading_only_document_no_chunks_is_ok(temp_db_path):
    db = await _build_db(temp_db_path)
    await _add_doc(
        db,
        "d2",
        items=[
            DocumentItemRecord(
                document_id="d2",
                position=0,
                self_ref="#/texts/0",
                label="section_header",
                text="title: haiku.rag",
            )
        ],
    )
    report = await run_doctor(_config(), temp_db_path, {})
    assert _result(report, "documents_without_chunks").severity is Severity.OK
    assert all(r.name != "documents_text_no_chunks" for r in report.results)


async def test_image_only_document_text_embedder_warns(temp_db_path):
    db = await _build_db(temp_db_path)
    await _add_doc(
        db,
        "d2",
        items=[
            DocumentItemRecord(
                document_id="d2", position=0, self_ref="#/pictures/0", label="picture"
            )
        ],
    )
    report = await run_doctor(_config(), temp_db_path, {})
    result = _result(report, "documents_images_unsearchable")
    assert result.severity is Severity.WARN
    assert "d2" in result.details


async def test_image_only_document_multimodal_embedder_warns(temp_db_path):
    db = await _build_db(temp_db_path, provider="vllm", name="qwen-vl")
    await _add_doc(
        db,
        "d2",
        items=[
            DocumentItemRecord(
                document_id="d2", position=0, self_ref="#/pictures/0", label="picture"
            )
        ],
    )
    report = await run_doctor(
        _config(provider="vllm", name="qwen-vl", multimodal=True), temp_db_path, {}
    )
    result = _result(report, "documents_pictures_no_chunks")
    assert result.severity is Severity.WARN
    assert "d2" in result.details


async def test_document_meta_parity_fails(temp_db_path):
    db = await _build_db(temp_db_path)
    docs_tbl = await db.open_table("documents")
    await docs_tbl.add([DocumentRecord(id="d2", content="no meta")])
    report = await run_doctor(_config(), temp_db_path, {})
    result = _result(report, "document_meta_parity")
    assert result.severity is Severity.FAIL
    assert any("d2" in d for d in result.details)


async def test_dangling_doc_item_ref_fails(temp_db_path):
    db = await _build_db(temp_db_path)
    chunks_tbl = await db.open_table("chunks")
    await chunks_tbl.add(
        [
            ChunkRecord(
                id="c2",
                document_id="d1",
                content="x",
                metadata=json.dumps({"doc_item_refs": ["#/texts/999"]}),
                vector=[0.3] * VECTOR_DIM,
            )
        ]
    )
    report = await run_doctor(_config(), temp_db_path, {})
    result = _result(report, "dangling_doc_item_refs")
    assert result.severity is Severity.FAIL
    assert "c2" in result.details


async def test_unembedded_chunk_fails(temp_db_path):
    db = await _build_db(temp_db_path)
    chunks_tbl = await db.open_table("chunks")
    await chunks_tbl.add(
        [
            ChunkRecord(
                id="zero",
                document_id="d1",
                content="x",
                metadata=json.dumps({"doc_item_refs": ["#/texts/0"]}),
                vector=[0.0] * VECTOR_DIM,
            )
        ]
    )
    report = await run_doctor(_config(), temp_db_path, {})
    result = _result(report, "unembedded_chunks")
    assert result.severity is Severity.FAIL
    assert "zero" in result.details
    assert report.failed


async def test_chunked_document_without_items_fails(temp_db_path):
    db = await _build_db(temp_db_path)
    await _add_doc(
        db,
        "d2",
        items=[],
        chunks=[
            ChunkRecord(
                id="c2", document_id="d2", content="x", vector=[0.1] * VECTOR_DIM
            )
        ],
    )
    report = await run_doctor(_config(), temp_db_path, {})
    result = _result(report, "documents_without_items")
    assert result.severity is Severity.FAIL
    assert "d2" in result.details
    assert report.failed


async def test_empty_document_without_items_is_ok(temp_db_path):
    db = await _build_db(temp_db_path)
    await _add_doc(db, "d2", items=[])
    report = await run_doctor(_config(), temp_db_path, {})
    assert _result(report, "documents_without_items").severity is Severity.OK


async def test_missing_picture_data_in_text_document_is_ok(temp_db_path):
    db = await _build_db(temp_db_path)
    await _add_doc(
        db,
        "d2",
        metadata={"content_type": "text/markdown"},
        items=[
            DocumentItemRecord(
                document_id="d2",
                position=0,
                self_ref="#/pictures/0",
                label="picture",
                picture_data=None,
            )
        ],
    )
    report = await run_doctor(_config(), temp_db_path, {})
    assert _result(report, "picture_data").severity is Severity.OK


async def test_missing_picture_data_in_pdf_document_warns(temp_db_path):
    db = await _build_db(temp_db_path)
    await _add_doc(
        db,
        "d2",
        metadata={"content_type": "application/pdf"},
        items=[
            DocumentItemRecord(
                document_id="d2",
                position=0,
                self_ref="#/pictures/0",
                label="picture",
                picture_data=None,
            )
        ],
    )
    report = await run_doctor(_config(), temp_db_path, {})
    result = _result(report, "picture_data")
    assert result.severity is Severity.WARN
    assert "d2" in result.details


async def test_missing_picture_data_warns(temp_db_path):
    db = await _build_db(temp_db_path)
    items_tbl = await db.open_table("document_items")
    await items_tbl.add(
        [
            DocumentItemRecord(
                document_id="d1",
                position=1,
                self_ref="#/pictures/0",
                label="picture",
                picture_data=None,
            )
        ]
    )
    report = await run_doctor(_config(), temp_db_path, {})
    assert _result(report, "picture_data").severity is Severity.WARN
    assert not report.failed


async def test_picture_with_data_ok(temp_db_path):
    db = await _build_db(temp_db_path)
    items_tbl = await db.open_table("document_items")
    await items_tbl.add(
        [
            DocumentItemRecord(
                document_id="d1",
                position=1,
                self_ref="#/pictures/0",
                label="picture",
                picture_data=b"\x89PNG",
            )
        ]
    )
    report = await run_doctor(_config(), temp_db_path, {})
    assert _result(report, "picture_data").severity is Severity.OK


@pytest.mark.parametrize(
    "stored, configured, details",
    [
        ({"name": "test"}, {"name": "different"}, ["name: 'test' -> 'different'"]),
        (
            {"provider": "ollama"},
            {"provider": "vllm"},
            ["provider: 'ollama' -> 'vllm'"],
        ),
        ({}, {"query_prefix": "Q: "}, ["query_prefix: '' -> 'Q: '"]),
    ],
)
async def test_embedding_identity_drift_warns(
    temp_db_path, stored, configured, details
):
    await _build_db(temp_db_path, **stored)
    report = await run_doctor(_config(**configured), temp_db_path, {})
    result = _result(report, "embedding_drift")
    assert result.severity is Severity.WARN
    assert result.details == details
    assert not report.failed


async def test_embedding_dim_drift_fails(temp_db_path):
    await _build_db(temp_db_path, vector_dim=VECTOR_DIM)
    report = await run_doctor(_config(vector_dim=VECTOR_DIM + 1), temp_db_path, {})
    assert _result(report, "embedding_drift").severity is Severity.FAIL
    assert report.failed


async def test_vector_dimension_mismatch_fails(temp_db_path):
    await _build_db(
        temp_db_path, vector_dim=VECTOR_DIM, stored_vector_dim=VECTOR_DIM + 1
    )
    report = await run_doctor(_config(vector_dim=VECTOR_DIM + 1), temp_db_path, {})
    result = _result(report, "vector_dimension")
    assert result.severity is Severity.FAIL
    assert report.failed


async def test_pending_migration_fails(temp_db_path):
    await _build_db(temp_db_path, version="0.40.0")
    report = await run_doctor(_config(), temp_db_path, {})
    assert _result(report, "pending_migrations").severity is Severity.FAIL
    assert report.failed


@pytest.mark.parametrize("schema_version", [None, latest_schema_version()])
async def test_a_known_schema_passes(temp_db_path, schema_version):
    await _build_db(temp_db_path, schema_version=schema_version)
    report = await run_doctor(_config(), temp_db_path, {})
    assert _result(report, "schema_version").severity is Severity.OK


async def test_a_newer_schema_fails_alone(temp_db_path):
    """Doctor runs no content check against a schema newer than this release."""
    await _build_db(temp_db_path, schema_version="100.0.0")
    report = await run_doctor(_config(), temp_db_path, {})
    result = _result(report, "schema_version")
    assert result.severity is Severity.FAIL
    assert "100.0.0" in result.message
    assert result.remediation == "Upgrade haiku.rag."
    names = {r.name for r in report.results}
    assert not names & {"pending_migrations", "embedding_drift", "vector_index"}


async def test_missing_api_key_fails(temp_db_path):
    await _build_db(temp_db_path, provider="openai", name="text-embedding-3-small")
    config = _config(provider="openai", name="text-embedding-3-small")
    report = await run_doctor(config, temp_db_path, environ={})
    result = _result(report, "api_keys")
    assert result.severity is Severity.FAIL
    assert any("OPENAI_API_KEY" in d for d in result.details)


async def test_present_api_key_ok(temp_db_path):
    await _build_db(temp_db_path, provider="openai", name="text-embedding-3-small")
    config = _config(provider="openai", name="text-embedding-3-small")
    report = await run_doctor(config, temp_db_path, environ={"OPENAI_API_KEY": "sk-x"})
    assert _result(report, "api_keys").severity is Severity.OK


async def test_settings_row_missing_fails(temp_db_path):
    db = await _build_db(temp_db_path)
    settings_tbl = await db.open_table("settings")
    await settings_tbl.delete("id = 'settings'")
    report = await run_doctor(_config(), temp_db_path, {})
    assert _result(report, "settings_row").severity is Severity.FAIL
    assert report.failed


async def test_many_orphans_are_sampled(temp_db_path):
    db = await _build_db(temp_db_path)
    chunks_tbl = await db.open_table("chunks")
    await chunks_tbl.add(
        [
            ChunkRecord(
                id=f"o{i}",
                document_id=f"ghost{i}",
                content="x",
                vector=[0.2] * VECTOR_DIM,
            )
            for i in range(8)
        ]
    )
    report = await run_doctor(_config(), temp_db_path, {})
    details = _result(report, "orphaned_chunks").details
    assert len(details) == 6
    assert details[-1] == "... (+3 more)"


def test_sample_returns_all_within_limit():
    assert _sample(["a", "b"]) == ["a", "b"]


def test_embedding_drift_ok_without_stored_identity():
    assert _check_embedding_drift({}, _config()).severity is Severity.OK


@pytest.mark.parametrize(
    "chunks, severity, remediation",
    [
        ({"num_rows": 10, "has_vector_index": False}, Severity.OK, None),
        ({"num_rows": 300, "has_vector_index": False}, Severity.OK, None),
        (
            {"num_rows": 100_000, "has_vector_index": False},
            Severity.WARN,
            "haiku-rag create-index",
        ),
        (
            {"num_rows": 300, "has_vector_index": True, "num_unindexed_rows": 5},
            Severity.WARN,
            "haiku-rag create-index",
        ),
        (
            {"num_rows": 300, "has_vector_index": True, "num_unindexed_rows": 0},
            Severity.OK,
            None,
        ),
    ],
)
def test_vector_index_check(chunks, severity, remediation):
    result = _check_vector_index({"chunks": chunks})
    assert result.severity is severity
    assert result.remediation == remediation


def test_cli_doctor_nonexistent_db_exits_1(tmp_path):
    result = runner.invoke(cli, ["doctor", "--db", str(tmp_path / "nope.lancedb")])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def _doctor_cli_args(db_path, tmp_path, monkeypatch, **build) -> list[str]:
    """`doctor` arguments over a database built by `_build_db`, its config, and an
    unreachable endpoint."""
    import asyncio

    import haiku.rag.config as config_module

    monkeypatch.setattr(config_module, "_config", None)
    asyncio.run(_build_db(db_path, **build))
    config_path = tmp_path / "haiku.rag.yaml"
    config_path.write_text(
        yaml.safe_dump(_config().model_dump(mode="json")), encoding="utf-8"
    )
    monkeypatch.setattr(
        "haiku.rag.doctor._probe_endpoint",
        _fake_probe((False, "Connection refused", None)),
    )
    return ["--config", str(config_path), "doctor", "--db", str(db_path)]


def test_cli_doctor_unreachable_endpoint_fails_unless_skipped(
    temp_db_path, tmp_path, monkeypatch
):
    args = _doctor_cli_args(temp_db_path, tmp_path, monkeypatch)

    checked = runner.invoke(cli, args)
    assert checked.exit_code == 1
    assert "is unreachable" in checked.output

    skipped = runner.invoke(cli, [*args, "--skip-providers"])
    assert skipped.exit_code == 0
    assert "is unreachable" not in skipped.output


def test_cli_doctor_json_prints_only_the_report(temp_db_path, tmp_path, monkeypatch):
    args = _doctor_cli_args(temp_db_path, tmp_path, monkeypatch)

    result = runner.invoke(cli, [*args, "--json"])

    assert result.exit_code == 1
    report = DoctorReport.model_validate_json(result.stdout)
    assert _result(report, "tables_present").severity is Severity.OK
    assert any(
        r.name.startswith("provider:") and r.severity is Severity.FAIL
        for r in report.results
    )


def test_cli_doctor_json_reports_a_pending_migration(
    temp_db_path, tmp_path, monkeypatch
):
    args = _doctor_cli_args(temp_db_path, tmp_path, monkeypatch, version="0.40.0")

    result = runner.invoke(cli, [*args, "--json", "--skip-providers"])

    assert result.exit_code == 1
    report = DoctorReport.model_validate_json(result.stdout)
    assert _result(report, "pending_migrations").severity is Severity.FAIL


def test_cli_doctor_json_still_writes_duplicates_out(
    temp_db_path, tmp_path, monkeypatch
):
    args = _doctor_cli_args(temp_db_path, tmp_path, monkeypatch)
    target = tmp_path / "dupes.yaml"

    result = runner.invoke(
        cli, [*args, "--json", "--skip-providers", "--duplicates-out", str(target)]
    )

    assert result.exit_code == 0
    assert not DoctorReport.model_validate_json(result.stdout).failed
    assert target.exists()


def test_cli_doctor_json_reports_a_missing_database(tmp_path):
    result = runner.invoke(
        cli, ["doctor", "--db", str(tmp_path / "nope.lancedb"), "--json"]
    )

    assert result.exit_code == 1
    report = DoctorReport.model_validate_json(result.stdout)
    assert [(r.name, r.severity) for r in report.results] == [
        ("database_missing", Severity.FAIL)
    ]


# --- Active models / API keys ---


def test_api_key_not_required_for_custom_openai_base_url():
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="openai",
                name="x",
                vector_dim=4,
                base_url="http://localhost:1234/v1",
            )
        )
    )
    assert _check_api_keys(config, {}).severity is Severity.OK


def test_api_key_required_for_openai_without_base_url():
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(provider="openai", name="x", vector_dim=4)
        )
    )
    result = _check_api_keys(config, {})
    assert result.severity is Severity.FAIL
    assert any("OPENAI_API_KEY" in d for d in result.details)


def test_api_key_not_required_when_config_supplies_it():
    """A key in the config is the point of the field; doctor must not demand
    the environment variable as well."""
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="openai", name="x", vector_dim=4, api_key="sk-inline"
            )
        )
    )
    assert _check_api_keys(config, {}).severity is Severity.OK


def test_active_models_includes_picture_description_when_enabled():
    base = _active_models(AppConfig())
    with_pictures = _active_models(
        AppConfig(processing=ProcessingConfig(pictures="description"))
    )
    assert len(with_pictures) == len(base) + 1


def test_active_models_includes_title_model_when_auto_title():
    base = _active_models(AppConfig())
    with_title = _active_models(AppConfig(processing=ProcessingConfig(auto_title=True)))
    assert len(with_title) == len(base) + 1


def test_picture_description_model_checked_for_api_key():
    config = AppConfig(
        processing=ProcessingConfig(
            pictures="description",
            conversion_options=ConversionOptions(
                picture_description=PictureDescriptionConfig(
                    model=ModelConfig(provider="openai", name="gpt-4o")
                )
            ),
        )
    )
    result = _check_api_keys(config, {})
    assert result.severity is Severity.FAIL
    assert any("OPENAI_API_KEY" in d for d in result.details)


# --- Provider connectivity ---


@pytest.mark.parametrize(
    "provider, base_url, fallback, expected",
    [
        (
            "ollama",
            "http://h:1/v1",
            "http://fallback",
            ("http://h:1/api/tags", "ollama", "http://h:1"),
        ),
        (
            "ollama",
            None,
            "http://fallback:11434",
            ("http://fallback:11434/api/tags", "ollama", "http://fallback:11434"),
        ),
        (
            "vllm",
            None,
            "http://o",
            ("http://localhost:8000/v1/models", "openai", "http://localhost:8000/v1"),
        ),
        (
            "vllm",
            "http://vllm:8000",
            "http://o",
            ("http://vllm:8000/v1/models", "openai", "http://vllm:8000/v1"),
        ),
        ("openai", None, "http://o", None),
        (
            "openai",
            "http://lmstudio:1234/v1",
            "http://o",
            ("http://lmstudio:1234/v1/models", "openai", "http://lmstudio:1234/v1"),
        ),
        ("sentence-transformers", None, "http://o", "local"),
    ],
)
def test_resolve_endpoint(provider, base_url, fallback, expected):
    assert _resolve_endpoint(provider, base_url, fallback) == expected


def test_model_present_tag_insensitive():
    assert _model_present("gpt-oss", {"gpt-oss:latest"})
    assert _model_present("qwen:4b", {"qwen:4b"})
    assert not _model_present("qwen:4b", {"qwen:8b"})


def test_provider_targets_default_groups_ollama_models():
    targets, local = _provider_targets(AppConfig())
    assert not local
    assert len(targets) == 1
    entry = next(iter(targets.values()))
    assert entry["kind"] == "ollama"
    assert {"qwen3-embedding:4b", "qwen3.8"} <= entry["models"]


def test_provider_targets_includes_docling_serve():
    config = AppConfig(
        processing=ProcessingConfig(converter="docling-serve"),
        providers=ProvidersConfig(
            docling_serve=DoclingServeConfig(base_url="http://docling:5001")
        ),
    )
    targets, _ = _provider_targets(config)
    assert "http://docling:5001/health" in targets
    assert targets["http://docling:5001/health"]["kind"] == "docling-serve"


def test_provider_targets_carries_model_api_key():
    """A secured endpoint answers the probe only with its key."""
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="openai",
                name="x",
                vector_dim=4,
                base_url="http://vllm:8000/v1",
                api_key="sk-probe",
            )
        )
    )
    targets, _ = _provider_targets(config)
    entry = targets["http://vllm:8000/v1/models"]
    assert entry["headers"] == {"Authorization": "Bearer sk-probe"}


def test_provider_targets_carries_docling_serve_api_key():
    config = AppConfig(
        processing=ProcessingConfig(converter="docling-serve"),
        providers=ProvidersConfig(
            docling_serve=DoclingServeConfig(
                base_url="http://docling:5001", api_key="ds-key"
            )
        ),
    )
    targets, _ = _provider_targets(config)
    assert targets["http://docling:5001/health"]["headers"] == {"X-Api-Key": "ds-key"}


def test_provider_targets_probe_an_unnamed_model_without_listing_it():
    config = AppConfig(qa=QAConfig(model=ModelConfig(provider="ollama", name="")))
    targets, _ = _provider_targets(config)
    entry = next(iter(targets.values()))
    assert "" not in entry["models"]
    assert "qwen3-embedding:4b" in entry["models"]


def test_provider_targets_collects_local_providers():
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="sentence-transformers", name="x", vector_dim=4
            )
        )
    )
    _, local = _provider_targets(config)
    assert "sentence-transformers" in local


def _fake_probe(result):
    async def probe(_client, _url, _headers):
        return result

    return probe


async def test_provider_check_ok_when_models_present(monkeypatch):
    monkeypatch.setattr(
        "haiku.rag.doctor._probe_endpoint",
        _fake_probe(
            (
                True,
                None,
                {
                    "models": [
                        {"name": "qwen3-embedding:4b"},
                        {"name": "qwen3.8:latest"},
                    ]
                },
            )
        ),
    )
    results = await run_provider_checks(AppConfig())
    assert all(r.severity is Severity.OK for r in results)


async def test_provider_check_warns_on_missing_model(monkeypatch):
    monkeypatch.setattr(
        "haiku.rag.doctor._probe_endpoint",
        _fake_probe((True, None, {"models": [{"name": "something-else"}]})),
    )
    results = await run_provider_checks(AppConfig())
    result = next(r for r in results if r.name.startswith("provider:"))
    assert result.severity is Severity.WARN
    assert result.details


async def test_provider_check_fails_when_unreachable(monkeypatch):
    monkeypatch.setattr(
        "haiku.rag.doctor._probe_endpoint",
        _fake_probe((False, "Connection refused", None)),
    )
    results = await run_provider_checks(AppConfig())
    result = next(r for r in results if r.name.startswith("provider:"))
    assert result.severity is Severity.FAIL
    assert "Connection refused" in result.details


async def test_provider_check_reports_local_provider(monkeypatch):
    monkeypatch.setattr(
        "haiku.rag.doctor._probe_endpoint",
        _fake_probe((True, None, {"models": [{"name": "qwen3.8:latest"}]})),
    )
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="sentence-transformers", name="x", vector_dim=4
            )
        )
    )
    results = await run_provider_checks(config)
    local = next(r for r in results if r.name == "provider:sentence-transformers")
    assert local.severity is Severity.OK
    assert "local" in local.message


async def test_provider_check_probes_nothing_without_endpoints(monkeypatch):
    async def probe(_client, _url, _headers):
        raise AssertionError("no endpoint to probe")

    monkeypatch.setattr("haiku.rag.doctor._probe_endpoint", probe)
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="sentence-transformers", name="x", vector_dim=4
            )
        ),
        qa=QAConfig(model=ModelConfig(provider="anthropic", name="claude")),
    )
    results = await run_provider_checks(config)
    assert [(r.name, r.severity) for r in results] == [
        ("provider:sentence-transformers", Severity.OK)
    ]


async def test_run_doctor_includes_provider_results(temp_db_path, monkeypatch):
    await _build_db(temp_db_path)
    monkeypatch.setattr(
        "haiku.rag.doctor._probe_endpoint",
        _fake_probe(
            (True, None, {"models": [{"name": "test"}, {"name": "qwen3.8:latest"}]})
        ),
    )
    report = await run_doctor(_config(), temp_db_path, {})
    assert any(r.name.startswith("provider:") for r in report.results)
    assert not report.failed


async def test_run_doctor_without_providers_skips_keys_and_probes(
    temp_db_path, monkeypatch
):
    await _build_db(temp_db_path, provider="openai", name="text-embedding-3-small")
    monkeypatch.setattr(
        "haiku.rag.doctor._probe_endpoint",
        _fake_probe((False, "Connection refused", None)),
    )
    config = _config(provider="openai", name="text-embedding-3-small")
    report = await run_doctor(config, temp_db_path, environ={}, providers=False)
    assert not any(
        r.name == "api_keys" or r.name.startswith("provider:") for r in report.results
    )
    assert not report.failed


async def _probe_with_handler(handler, headers: dict[str, str] | None = None):
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        return await _probe_endpoint(client, "http://x", headers or {})


@pytest.mark.parametrize(
    "outcome, reachable, error, payload",
    [
        (httpx.Response(200, json={"models": []}), True, None, {"models": []}),
        (httpx.Response(200, content=b"not json"), True, None, None),
        (httpx.Response(503), False, "503", None),
        (httpx.ConnectError("refused"), False, "refused", None),
    ],
)
async def test_probe_endpoint_outcome(outcome, reachable, error, payload):
    def handler(_request):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    reached, message, body = await _probe_with_handler(handler)

    assert reached is reachable
    assert body == payload
    if error is None:
        assert message is None
    else:
        assert message is not None and error in message


async def test_probe_endpoint_sends_headers():
    """A secured endpoint needs its key on the probe request too."""
    seen: dict[str, str] = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, json={})

    await _probe_with_handler(handler, {"Authorization": "Bearer sk-probe"})

    assert seen["authorization"] == "Bearer sk-probe"


# --- Duplicate-document detection ----------------------------------------


def _centroids(
    spec: dict[str, list[int]], dim: int = 8
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Summed one-hot centroids + chunk counts per document, as
    ``duplicate_families`` consumes them.

    Orthogonal one-hot chunks make the centroid cosine of two documents equal to
    ``shared / sqrt(len(a) * len(b))``: identical documents score 1.0, fully
    distinct documents score 0.0.
    """
    eye = np.eye(dim)
    doc_ids = list(spec)
    centroids = np.array(
        [eye[idxs].sum(axis=0) for idxs in spec.values()], dtype=np.float32
    )
    counts = np.array([len(idxs) for idxs in spec.values()], dtype=np.int64)
    return doc_ids, centroids, counts


def test_duplicate_families_identical_docs_flagged():
    families = duplicate_families(
        *_centroids({"a": [0, 1, 2, 3], "b": [0, 1, 2, 3]}), DuplicateDetectionConfig()
    )
    assert len(families) == 1
    assert set(families[0].members) == {"a", "b"}
    assert families[0].similarity == {"a": pytest.approx(1.0), "b": pytest.approx(1.0)}


def test_duplicate_families_append_only_not_flagged():
    # A is fully contained in the larger B, but their centroids diverge
    # (cosine sqrt(3/6) ~= 0.71), so it stays below the similarity cutoff.
    families = duplicate_families(
        *_centroids({"a": [0, 1, 2], "b": [0, 1, 2, 3, 4, 5]}),
        DuplicateDetectionConfig(),
    )
    assert families == []


def test_duplicate_families_distinct_docs_none():
    families = duplicate_families(
        *_centroids({"a": [0, 1, 2], "b": [3, 4, 5]}), DuplicateDetectionConfig()
    )
    assert families == []


def test_duplicate_families_three_way_one_family():
    families = duplicate_families(
        *_centroids({"a": [0, 1, 2, 3], "b": [0, 1, 2, 3], "c": [0, 1, 2, 3]}),
        DuplicateDetectionConfig(),
    )
    assert len(families) == 1
    assert set(families[0].members) == {"a", "b", "c"}
    # Equal sizes -> smallest id is kept.
    assert families[0].keep == "a"


def test_duplicate_families_clique_single_family():
    # A self-similar corpus (all identical) is one clique. Union-find collapses
    # it to a single family without materializing every pair.
    spec = {chr(ord("a") + k): [0, 1, 2, 3] for k in range(8)}
    families = duplicate_families(*_centroids(spec), DuplicateDetectionConfig())
    assert len(families) == 1
    assert set(families[0].members) == set(spec)
    assert all(s == pytest.approx(1.0) for s in families[0].similarity.values())


def test_duplicate_families_tiny_docs_ignored():
    # min_chunks = 3 excludes the one-chunk documents.
    families = duplicate_families(
        *_centroids({"a": [0], "b": [0]}), DuplicateDetectionConfig()
    )
    assert families == []


def test_duplicate_families_threshold_is_configurable():
    # Share 3 of 4 chunks each -> centroid cosine 0.75.
    spec = {"a": [0, 1, 2, 3], "b": [0, 1, 2, 4]}
    assert duplicate_families(*_centroids(spec), DuplicateDetectionConfig()) == []
    flagged = duplicate_families(
        *_centroids(spec), DuplicateDetectionConfig(similarity_threshold=0.7)
    )
    assert len(flagged) == 1
    assert set(flagged[0].members) == {"a", "b"}


def test_duplicate_documents_report_truncates_summary():
    pairs = 7  # more than the terminal detail cap of 5
    spec: dict[str, list[int]] = {}
    for k in range(pairs):
        idx = [3 * k, 3 * k + 1, 3 * k + 2]
        spec[f"a{k}"] = idx
        spec[f"b{k}"] = list(idx)
    uris = {d: f"file:///srv/shared/library/docs/{d}.pdf" for d in spec}
    result = _check_duplicate_documents(
        *_centroids(spec, dim=3 * pairs), uris, {}, DuplicateDetectionConfig()
    )
    assert result.severity is Severity.WARN
    # The summary message still reports the full total.
    assert f"{pairs} group(s)" in result.message
    # The terminal detail shows only the first few groups and points at export.
    assert sum(1 for d in result.details if d.startswith("group ")) == 5
    assert any("more groups" in d and "--duplicates-out" in d for d in result.details)
    assert any("keep #" in d for d in result.details)


def test_duplicate_documents_report_factors_common_path():
    base = "file:///srv/shared/library/docs/"
    uris = {"a": base + "alpha.pdf", "b": base + "beta.pdf"}
    result = _check_duplicate_documents(
        *_centroids({"a": [0, 1, 2], "b": [0, 1, 2]}, dim=3),
        uris,
        {},
        DuplicateDetectionConfig(),
    )
    assert f"common path: {base}" in result.details
    member_lines = [d for d in result.details if d.lstrip().startswith("#")]
    assert {d.strip() for d in member_lines} == {"#1 alpha.pdf", "#2 beta.pdf"}
    assert not any(base in d for d in member_lines)


def test_duplicate_documents_writes_yaml(tmp_path):
    # a,b identical (a 4-chunk duplicate); c distinct and excluded.
    spec = {"a": [0, 1, 2, 3], "b": [0, 1, 2, 3], "c": [4, 5, 6]}
    uris = {"a": "file:///x/a.pdf", "b": "file:///x/b.pdf", "c": "file:///x/c.pdf"}
    out = tmp_path / "dups.yaml"
    _check_duplicate_documents(
        *_centroids(spec, dim=8), uris, {}, DuplicateDetectionConfig(), yaml_path=out
    )
    data = yaml.safe_load(out.read_text())
    assert len(data["groups"]) == 1
    group = data["groups"][0]
    assert group["group"] == 1 and group["keep"] == "a"
    docs_out = group["documents"]
    assert [d["document_id"] for d in docs_out] == ["a", "b"]
    assert [d["document"] for d in docs_out] == ["file:///x/a.pdf", "file:///x/b.pdf"]
    assert all(d["chunks"] == 4 for d in docs_out)
    assert all(d["similarity"] == pytest.approx(1.0) for d in docs_out)
    assert {d["document_id"]: d["keep_suggested"] for d in docs_out} == {
        "a": True,
        "b": False,
    }


def test_duplicate_documents_writes_empty_yaml_when_none(tmp_path):
    spec = {"a": [0, 1, 2], "b": [3, 4, 5]}  # distinct
    out = tmp_path / "dups.yaml"
    _check_duplicate_documents(
        *_centroids(spec, dim=6),
        {"a": "u", "b": "v"},
        {},
        DuplicateDetectionConfig(),
        yaml_path=out,
    )
    assert yaml.safe_load(out.read_text()) == {"groups": []}


async def _build_dup_db(path, docs: dict[str, list[int]], *, vector_dim: int = 8):
    """Build a multi-document database with one-hot chunk vectors."""
    eye = np.eye(vector_dim)
    db = await lancedb.connect_async(path)
    settings_tbl = await db.create_table("settings", schema=SettingsRecord)
    docs_tbl = await db.create_table("documents", schema=DocumentRecord)
    meta_tbl = await db.create_table("document_meta", schema=DocumentMetaRecord)
    chunk_model = create_chunk_model(vector_dim)
    chunks_tbl = await db.create_table("chunks", schema=chunk_model)
    items_tbl = await db.create_table("document_items", schema=DocumentItemRecord)

    await settings_tbl.add(
        [
            SettingsRecord(
                id="settings",
                settings=json.dumps(
                    {
                        "version": CURRENT_VERSION,
                        "embeddings": {
                            "model": {
                                "provider": "ollama",
                                "name": "test",
                                "vector_dim": vector_dim,
                            }
                        },
                    }
                ),
            )
        ]
    )
    for doc_id, idxs in docs.items():
        await docs_tbl.add([DocumentRecord(id=doc_id, content="x")])
        await meta_tbl.add([DocumentMetaRecord(id=doc_id, uri=f"test://{doc_id}")])
        await items_tbl.add(
            [
                DocumentItemRecord(
                    document_id=doc_id, position=0, self_ref="#/texts/0", text="x"
                )
            ]
        )
        await chunks_tbl.add(
            [
                chunk_model(
                    id=f"{doc_id}-c{n}",
                    document_id=doc_id,
                    content="x",
                    metadata=json.dumps({"doc_item_refs": ["#/texts/0"]}),
                    vector=eye[i].tolist(),
                )
                for n, i in enumerate(idxs)
            ]
        )
    return db


async def test_scan_vectors_reduces_chunks_to_document_centroids(temp_db_path):
    from haiku.rag.doctor import scan_vectors
    from haiku.rag.store.engine import Store

    db = await _build_dup_db(temp_db_path, {"a": [0, 1], "b": [2]})
    chunk_model = create_chunk_model(8)
    chunks_tbl = await db.open_table("chunks")
    await chunks_tbl.add(
        [
            chunk_model(
                id=f"zero-{n}",
                document_id="b" if n < 4 else "c",
                content="x",
                metadata=json.dumps({"doc_item_refs": ["#/texts/0"]}),
                vector=[0.0] * 8,
            )
            for n in range(7)
        ]
    )
    async with Store(
        temp_db_path,
        config=_config(vector_dim=8),
        skip_validation=True,
        read_only=True,
        skip_migration_check=True,
    ) as store:
        scan = await scan_vectors(store)

    assert scan.stored_dimension == 8
    assert scan.dimension == 8
    by_id = dict(zip(scan.document_ids, range(len(scan.document_ids))))
    assert sorted(by_id) == ["a", "b", "c"]
    assert scan.counts[by_id["a"]] == 2
    assert scan.counts[by_id["b"]] == 1
    assert scan.counts[by_id["c"]] == 0
    assert scan.centroids[by_id["a"]].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    assert scan.centroids[by_id["b"]].tolist() == [0, 0, 1, 0, 0, 0, 0, 0]
    assert scan.centroids[by_id["c"]].tolist() == [0] * 8
    assert scan.unembedded_count == 7
    assert scan.unembedded_sample == [f"zero-{n}" for n in range(5)]


def test_duplicates_yaml_is_what_the_export_writes(tmp_path):
    from haiku.rag.doctor import duplicates_yaml

    spec = {"a": [0, 1, 2, 3], "b": [0, 1, 2, 3]}
    out = tmp_path / "dups.yaml"
    _check_duplicate_documents(
        *_centroids(spec), {}, {}, DuplicateDetectionConfig(), yaml_path=out
    )
    families = duplicate_families(*_centroids(spec), DuplicateDetectionConfig())
    assert duplicates_yaml(families, lambda doc_id: doc_id) == out.read_text()


async def test_duplicate_documents_check_warns_end_to_end(temp_db_path):
    await _build_dup_db(temp_db_path, {"a": [0, 1, 2, 3], "b": [0, 1, 2, 3]})
    report = await run_doctor(_config(vector_dim=8), temp_db_path, {})
    result = _result(report, "duplicate_documents")
    assert result.severity is Severity.WARN
    blob = " ".join(result.details)
    assert "test://a" in blob and "test://b" in blob


async def test_duplicate_documents_check_ok_when_distinct(temp_db_path):
    await _build_dup_db(temp_db_path, {"a": [0, 1, 2], "b": [3, 4, 5]})
    report = await run_doctor(_config(vector_dim=8), temp_db_path, {})
    assert _result(report, "duplicate_documents").severity is Severity.OK


async def test_duplicate_documents_check_reads_config(temp_db_path):
    # Share 3 of 5 -> centroid cosine 0.6, below the default 0.97 cutoff.
    await _build_dup_db(temp_db_path, {"a": [0, 1, 2, 3, 4], "b": [0, 1, 2, 5, 6]})

    base = _config(vector_dim=8)
    assert (
        _result(
            await run_doctor(base, temp_db_path, {}), "duplicate_documents"
        ).severity
        is Severity.OK
    )

    tuned = _config(vector_dim=8)
    tuned.doctor = DoctorConfig(
        duplicates=DuplicateDetectionConfig(similarity_threshold=0.5)
    )
    assert (
        _result(
            await run_doctor(tuned, temp_db_path, {}), "duplicate_documents"
        ).severity
        is Severity.WARN
    )


async def test_many_unembedded_chunks_are_sampled(temp_db_path):
    """Beyond the sample limit the detail list ends with a count of the rest."""
    db = await _build_db(temp_db_path)
    chunks_tbl = await db.open_table("chunks")
    await chunks_tbl.add(
        [
            ChunkRecord(
                id=f"z{i}",
                document_id="d1",
                content="x",
                metadata=json.dumps({"doc_item_refs": ["#/texts/0"]}),
                vector=[0.0] * VECTOR_DIM,
            )
            for i in range(8)
        ]
    )
    report = await run_doctor(_config(), temp_db_path, {})
    details = _result(report, "unembedded_chunks").details
    assert len(details) == 6
    assert details[-1] == "... (+3 more)"


def test_api_key_required_for_openrouter():
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="openrouter",
                name="nvidia/llama-nemotron-embed-vl-1b-v2:free",
                vector_dim=2048,
            )
        )
    )
    result = _check_api_keys(config, {})
    assert result.severity is Severity.FAIL
    assert any("OPENROUTER_API_KEY" in d for d in result.details)
    assert (
        _check_api_keys(config, {"OPENROUTER_API_KEY": "sk-or-x"}).severity
        is Severity.OK
    )


async def test_embedding_drift_advises_re_embedding_not_adopting(temp_db_path):
    """Drift can be an interrupted rebuild, whose table holds vectors of both
    embedders, so adopting the configured one is advice only for an embedder
    that produces the same vectors."""
    await _build_db(temp_db_path, name="test")
    report = await run_doctor(_config(name="different"), temp_db_path, {})
    remediation = _result(report, "embedding_drift").remediation
    assert remediation is not None
    assert "'haiku-rag rebuild --embed-only'" in remediation
    assert "interrupted rebuild again" in remediation
    assert (
        "'haiku-rag rebuild --set-embedder' only for an embedder that produces "
        "the same vectors" in remediation
    )
