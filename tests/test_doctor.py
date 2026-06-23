import json
from importlib import metadata
from unittest.mock import AsyncMock, MagicMock

import lancedb
import pytest
from typer.testing import CliRunner

from haiku.rag.cli import _cli as cli
from haiku.rag.config.models import (
    AppConfig,
    ConversionOptions,
    DoclingServeConfig,
    EmbeddingModelConfig,
    EmbeddingsConfig,
    ModelConfig,
    PictureDescriptionConfig,
    ProcessingConfig,
    ProvidersConfig,
)
from haiku.rag.doctor import (
    CheckResult,
    DoctorReport,
    Severity,
    _active_models,
    _check_api_keys,
    _check_embedding_drift,
    _check_vector_index,
    _model_present,
    _probe_endpoint,
    _provider_targets,
    _resolve_endpoint,
    _sample,
    run_doctor,
    run_provider_checks,
)
from haiku.rag.store.engine import (
    DocumentItemRecord,
    DocumentMetaRecord,
    DocumentRecord,
    SettingsRecord,
    create_chunk_model,
)

runner = CliRunner()

CURRENT_VERSION = metadata.version("haiku.rag-slim")
VECTOR_DIM = 4
ChunkRecord = create_chunk_model(VECTOR_DIM)


def _config(
    provider: str = "ollama",
    name: str = "test",
    vector_dim: int = VECTOR_DIM,
    multimodal: bool = False,
):
    return AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider=provider,
                name=name,
                vector_dim=vector_dim,
                multimodal=multimodal,
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
):
    """Create a consistent single-document database without touching an embedder.

    ``stored_vector_dim`` records a different dimension in settings than the
    chunks table actually uses, to exercise the vector-dimension check.
    """
    db = await lancedb.connect_async(path)
    settings_tbl = await db.create_table("settings", schema=SettingsRecord)
    docs_tbl = await db.create_table("documents", schema=DocumentRecord)
    meta_tbl = await db.create_table("document_meta", schema=DocumentMetaRecord)
    chunks_tbl = await db.create_table("chunks", schema=create_chunk_model(vector_dim))
    items_tbl = await db.create_table("document_items", schema=DocumentItemRecord)

    await settings_tbl.add(
        [
            SettingsRecord(
                id="settings",
                settings=json.dumps(
                    {
                        "version": version,
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
    await meta_tbl.add([DocumentMetaRecord(document_id="d1", uri="test://d1")])
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
    return db


def _result(report: DoctorReport, name: str) -> CheckResult:
    return next(r for r in report.results if r.name == name)


@pytest.fixture(autouse=True)
def _stub_provider_probe(monkeypatch):
    """Default every provider probe to reachable with the test models present,
    so database-integrity tests don't depend on a live Ollama. Provider tests
    re-patch this with their own behavior."""

    async def probe(_client, _url):
        return (
            True,
            None,
            {
                "models": [
                    {"name": "test"},
                    {"name": "gpt-oss:latest"},
                    {"name": "qwen3-embedding:4b"},
                ]
            },
        )

    monkeypatch.setattr("haiku.rag.doctor._probe_endpoint", probe)


@pytest.mark.asyncio
async def test_healthy_db_all_ok(temp_db_path):
    await _build_db(temp_db_path)
    report = await run_doctor(_config(), temp_db_path, {})
    assert not report.failed
    assert report.count(Severity.WARN) == 0
    assert all(r.severity is Severity.OK for r in report.results)


@pytest.mark.asyncio
async def test_empty_db_fails(temp_db_path):
    report = await run_doctor(_config(), temp_db_path, {})
    assert report.failed
    assert _result(report, "tables_present").message == "Database is empty."


@pytest.mark.asyncio
async def test_missing_table_fails_without_opening_store(temp_db_path):
    db = await lancedb.connect_async(temp_db_path)
    await db.create_table("settings", schema=SettingsRecord)
    report = await run_doctor(_config(), temp_db_path, {})
    assert report.failed
    tables = _result(report, "tables_present")
    assert tables.severity is Severity.FAIL
    assert "documents" in tables.details


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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
                document_id=doc_id,
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


@pytest.mark.asyncio
async def test_document_with_text_but_no_chunks_warns(temp_db_path):
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
    assert result.severity is Severity.WARN
    assert "d2" in result.details
    assert report.count(Severity.FAIL) == 0


@pytest.mark.asyncio
async def test_empty_document_no_chunks_is_ok(temp_db_path):
    db = await _build_db(temp_db_path)
    await _add_doc(db, "d2", items=[])
    report = await run_doctor(_config(), temp_db_path, {})
    assert _result(report, "documents_without_chunks").severity is Severity.OK


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_document_meta_parity_fails(temp_db_path):
    db = await _build_db(temp_db_path)
    docs_tbl = await db.open_table("documents")
    await docs_tbl.add([DocumentRecord(id="d2", content="no meta")])
    report = await run_doctor(_config(), temp_db_path, {})
    result = _result(report, "document_meta_parity")
    assert result.severity is Severity.FAIL
    assert any("d2" in d for d in result.details)


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_unembedded_chunk_warns(temp_db_path):
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
    assert result.severity is Severity.WARN
    assert "zero" in result.details
    assert not report.failed


@pytest.mark.asyncio
async def test_chunked_document_without_items_warns(temp_db_path):
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
    assert result.severity is Severity.WARN
    assert "d2" in result.details


@pytest.mark.asyncio
async def test_empty_document_without_items_is_ok(temp_db_path):
    db = await _build_db(temp_db_path)
    await _add_doc(db, "d2", items=[])
    report = await run_doctor(_config(), temp_db_path, {})
    assert _result(report, "documents_without_items").severity is Severity.OK


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_embedding_name_drift_warns(temp_db_path):
    await _build_db(temp_db_path, name="test")
    report = await run_doctor(_config(name="different"), temp_db_path, {})
    result = _result(report, "embedding_drift")
    assert result.severity is Severity.WARN
    assert not report.failed


@pytest.mark.asyncio
async def test_embedding_dim_drift_fails(temp_db_path):
    await _build_db(temp_db_path, vector_dim=VECTOR_DIM)
    report = await run_doctor(_config(vector_dim=VECTOR_DIM + 1), temp_db_path, {})
    assert _result(report, "embedding_drift").severity is Severity.FAIL
    assert report.failed


@pytest.mark.asyncio
async def test_embedding_provider_drift_warns(temp_db_path):
    await _build_db(temp_db_path, provider="ollama")
    report = await run_doctor(_config(provider="vllm"), temp_db_path, {})
    result = _result(report, "embedding_drift")
    assert result.severity is Severity.WARN
    assert any("provider" in d for d in result.details)


@pytest.mark.asyncio
async def test_vector_dimension_mismatch_fails(temp_db_path):
    await _build_db(
        temp_db_path, vector_dim=VECTOR_DIM, stored_vector_dim=VECTOR_DIM + 1
    )
    report = await run_doctor(_config(vector_dim=VECTOR_DIM + 1), temp_db_path, {})
    result = _result(report, "vector_dimension")
    assert result.severity is Severity.FAIL
    assert report.failed


@pytest.mark.asyncio
async def test_pending_migration_warns(temp_db_path):
    await _build_db(temp_db_path, version="0.40.0")
    report = await run_doctor(_config(), temp_db_path, {})
    assert _result(report, "pending_migrations").severity is Severity.WARN
    assert not report.failed


@pytest.mark.asyncio
async def test_missing_api_key_fails(temp_db_path):
    await _build_db(temp_db_path, provider="openai", name="text-embedding-3-small")
    config = _config(provider="openai", name="text-embedding-3-small")
    report = await run_doctor(config, temp_db_path, environ={})
    result = _result(report, "api_keys")
    assert result.severity is Severity.FAIL
    assert any("OPENAI_API_KEY" in d for d in result.details)


@pytest.mark.asyncio
async def test_present_api_key_ok(temp_db_path):
    await _build_db(temp_db_path, provider="openai", name="text-embedding-3-small")
    config = _config(provider="openai", name="text-embedding-3-small")
    report = await run_doctor(config, temp_db_path, environ={"OPENAI_API_KEY": "sk-x"})
    assert _result(report, "api_keys").severity is Severity.OK


@pytest.mark.asyncio
async def test_settings_row_missing_fails(temp_db_path):
    db = await _build_db(temp_db_path)
    settings_tbl = await db.open_table("settings")
    await settings_tbl.delete("id = 'settings'")
    report = await run_doctor(_config(), temp_db_path, {})
    assert _result(report, "settings_row").severity is Severity.FAIL
    assert report.failed


@pytest.mark.asyncio
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


def test_vector_index_ok_below_threshold():
    stats = {"chunks": {"num_rows": 10, "has_vector_index": False}}
    assert _check_vector_index(stats).severity is Severity.OK


def test_vector_index_warns_when_missing_above_threshold():
    stats = {"chunks": {"num_rows": 300, "has_vector_index": False}}
    result = _check_vector_index(stats)
    assert result.severity is Severity.WARN
    assert result.remediation == "haiku-rag create-index"


def test_vector_index_warns_on_unindexed_backlog():
    stats = {
        "chunks": {"num_rows": 300, "has_vector_index": True, "num_unindexed_rows": 5}
    }
    assert _check_vector_index(stats).severity is Severity.WARN


def test_vector_index_ok_when_fully_indexed():
    stats = {
        "chunks": {"num_rows": 300, "has_vector_index": True, "num_unindexed_rows": 0}
    }
    assert _check_vector_index(stats).severity is Severity.OK


def test_cli_doctor_nonexistent_db_exits_1(tmp_path):
    result = runner.invoke(cli, ["doctor", "--db", str(tmp_path / "nope.lancedb")])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_cli_doctor_exits_0_when_healthy(monkeypatch):
    app = MagicMock()
    app.doctor = AsyncMock(return_value=False)
    monkeypatch.setattr("haiku.rag.cli.create_app", lambda *_a, **_k: app)
    result = runner.invoke(cli, ["doctor", "--db", "/tmp/whatever.lancedb"])
    assert result.exit_code == 0


def test_cli_doctor_exits_1_on_failure(monkeypatch):
    app = MagicMock()
    app.doctor = AsyncMock(return_value=True)
    monkeypatch.setattr("haiku.rag.cli.create_app", lambda *_a, **_k: app)
    result = runner.invoke(cli, ["doctor", "--db", "/tmp/whatever.lancedb"])
    assert result.exit_code == 1


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


def test_active_models_includes_picture_description_when_enabled():
    config = AppConfig(processing=ProcessingConfig(pictures="description"))
    names = [name for _p, name, _b in _active_models(config)]
    assert "ministral-3" in names


def test_active_models_excludes_picture_description_by_default():
    names = [name for _p, name, _b in _active_models(AppConfig())]
    assert "ministral-3" not in names


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


def test_resolve_endpoint_ollama_strips_v1():
    assert _resolve_endpoint("ollama", "http://h:1/v1", "http://fallback") == (
        "http://h:1/api/tags",
        "ollama",
        "http://h:1",
    )


def test_resolve_endpoint_ollama_uses_provider_fallback():
    assert _resolve_endpoint("ollama", None, "http://fallback:11434") == (
        "http://fallback:11434/api/tags",
        "ollama",
        "http://fallback:11434",
    )


def test_resolve_endpoint_vllm_default_and_models_path():
    assert _resolve_endpoint("vllm", None, "http://o") == (
        "http://localhost:8000/v1/models",
        "openai",
        "http://localhost:8000/v1",
    )


def test_resolve_endpoint_vllm_appends_v1():
    assert _resolve_endpoint("vllm", "http://vllm:8000", "http://o") == (
        "http://vllm:8000/v1/models",
        "openai",
        "http://vllm:8000/v1",
    )


def test_resolve_endpoint_openai_saas_is_skipped():
    assert _resolve_endpoint("openai", None, "http://o") is None


def test_resolve_endpoint_openai_with_base_url():
    assert _resolve_endpoint("openai", "http://lmstudio:1234/v1", "http://o") == (
        "http://lmstudio:1234/v1/models",
        "openai",
        "http://lmstudio:1234/v1",
    )


def test_resolve_endpoint_local_provider():
    assert _resolve_endpoint("sentence-transformers", None, "http://o") == "local"


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
    assert {"qwen3-embedding:4b", "gpt-oss"} <= entry["models"]


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
    async def probe(_client, _url):
        return result

    return probe


@pytest.mark.asyncio
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
                        {"name": "gpt-oss:latest"},
                    ]
                },
            )
        ),
    )
    results = await run_provider_checks(AppConfig())
    assert all(r.severity is Severity.OK for r in results)


@pytest.mark.asyncio
async def test_provider_check_warns_on_missing_model(monkeypatch):
    monkeypatch.setattr(
        "haiku.rag.doctor._probe_endpoint",
        _fake_probe((True, None, {"models": [{"name": "something-else"}]})),
    )
    results = await run_provider_checks(AppConfig())
    result = next(r for r in results if r.name.startswith("provider:"))
    assert result.severity is Severity.WARN
    assert result.details


@pytest.mark.asyncio
async def test_provider_check_fails_when_unreachable(monkeypatch):
    monkeypatch.setattr(
        "haiku.rag.doctor._probe_endpoint",
        _fake_probe((False, "Connection refused", None)),
    )
    results = await run_provider_checks(AppConfig())
    result = next(r for r in results if r.name.startswith("provider:"))
    assert result.severity is Severity.FAIL
    assert "Connection refused" in result.details


@pytest.mark.asyncio
async def test_provider_check_reports_local_provider(monkeypatch):
    monkeypatch.setattr(
        "haiku.rag.doctor._probe_endpoint",
        _fake_probe((True, None, {"models": [{"name": "gpt-oss:latest"}]})),
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


@pytest.mark.asyncio
async def test_run_doctor_includes_provider_results(temp_db_path, monkeypatch):
    await _build_db(temp_db_path)
    monkeypatch.setattr(
        "haiku.rag.doctor._probe_endpoint",
        _fake_probe(
            (True, None, {"models": [{"name": "test"}, {"name": "gpt-oss:latest"}]})
        ),
    )
    report = await run_doctor(_config(), temp_db_path, {})
    assert any(r.name.startswith("provider:") for r in report.results)
    assert not report.failed


async def _probe_with_handler(handler):
    import httpx

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        return await _probe_endpoint(client, "http://x")


@pytest.mark.asyncio
async def test_probe_endpoint_success_with_json():
    import httpx

    reachable, error, payload = await _probe_with_handler(
        lambda _request: httpx.Response(200, json={"models": []})
    )
    assert reachable and error is None and payload == {"models": []}


@pytest.mark.asyncio
async def test_probe_endpoint_success_non_json():
    import httpx

    reachable, _, payload = await _probe_with_handler(
        lambda _request: httpx.Response(200, content=b"not json")
    )
    assert reachable and payload is None


@pytest.mark.asyncio
async def test_probe_endpoint_http_error_status():
    import httpx

    reachable, error, _ = await _probe_with_handler(
        lambda _request: httpx.Response(503)
    )
    assert not reachable
    assert error is not None and "503" in error


@pytest.mark.asyncio
async def test_probe_endpoint_connection_error():
    import httpx

    def handler(_request):
        raise httpx.ConnectError("refused")

    reachable, error, _ = await _probe_with_handler(handler)
    assert not reachable
    assert error is not None and "refused" in error
