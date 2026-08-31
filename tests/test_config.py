import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from haiku.rag.config import AppConfig, set_config
from haiku.rag.config.loader import (
    find_config_file,
    generate_default_config,
    load_yaml_config,
)


def test_load_yaml_config(tmp_path):
    """Test loading a YAML config file."""
    config_file = tmp_path / "test.yaml"
    config_file.write_text("""
environment: production
embeddings:
  model:
    provider: ollama
    name: test-model
    vector_dim: 1024
""")

    config = load_yaml_config(config_file)
    assert config["environment"] == "production"
    assert config["embeddings"]["model"]["provider"] == "ollama"
    assert config["embeddings"]["model"]["name"] == "test-model"
    assert config["embeddings"]["model"]["vector_dim"] == 1024


def test_find_config_file_cwd(tmp_path, monkeypatch):
    """Test finding config in current directory."""
    monkeypatch.delenv("HAIKU_RAG_CONFIG_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "haiku.rag.yaml"
    config_file.write_text("environment: production")

    found = find_config_file()
    assert found == config_file


def test_find_config_file_user_config(tmp_path, monkeypatch):
    """Test finding config in user config directory."""
    monkeypatch.delenv("HAIKU_RAG_CONFIG_PATH", raising=False)

    # The data dir must differ from the cwd, or the cwd branch answers first
    # and this never reaches the user-directory lookup.
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr("haiku.rag.utils.get_default_data_dir", lambda: data_dir)

    config_file = data_dir / "haiku.rag.yaml"
    config_file.write_text("environment: production")

    found = find_config_file()
    assert found == config_file


def test_find_config_file_cli_path(tmp_path):
    """Test finding config via CLI path parameter."""
    config_file = tmp_path / "custom.yaml"
    config_file.write_text("environment: production")

    found = find_config_file(config_file)
    assert found == config_file


def test_find_config_file_env_var(tmp_path, monkeypatch):
    """Test finding config via HAIKU_RAG_CONFIG_PATH env var."""
    config_file = tmp_path / "from-env.yaml"
    config_file.write_text("environment: production")

    monkeypatch.setenv("HAIKU_RAG_CONFIG_PATH", str(config_file))

    found = find_config_file()
    assert found == config_file


def test_find_config_file_env_var_tilde_expansion(tmp_path, monkeypatch):
    """Test that ~ in HAIKU_RAG_CONFIG_PATH is expanded."""
    config_file = tmp_path / "from-env.yaml"
    config_file.write_text("environment: production")

    # Point HOME to tmp_path so ~ expands there
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HAIKU_RAG_CONFIG_PATH", "~/from-env.yaml")

    found = find_config_file()
    assert found == config_file


def test_find_config_file_not_found(tmp_path, monkeypatch):
    """Test returning None when no config found."""
    monkeypatch.delenv("HAIKU_RAG_CONFIG_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    # Mock get_default_data_dir to return tmp_path
    def mock_get_default_data_dir():
        return tmp_path

    monkeypatch.setattr(
        "haiku.rag.utils.get_default_data_dir", mock_get_default_data_dir
    )

    found = find_config_file()
    assert found is None


def test_find_config_file_cli_path_not_exists(tmp_path):
    """Test error when CLI path doesn't exist."""
    config_file = tmp_path / "nonexistent.yaml"

    with pytest.raises(FileNotFoundError):
        find_config_file(config_file)


def test_config_precedence_cwd_over_user(tmp_path, monkeypatch):
    """Test that cwd config takes precedence over user config."""
    monkeypatch.delenv("HAIKU_RAG_CONFIG_PATH", raising=False)

    # Create separate directories for cwd and user config
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    user_dir = tmp_path / "user"
    user_dir.mkdir()

    monkeypatch.chdir(cwd_dir)

    # Mock get_default_data_dir to return user_dir
    def mock_get_default_data_dir():
        return user_dir

    monkeypatch.setattr(
        "haiku.rag.utils.get_default_data_dir", mock_get_default_data_dir
    )

    # Create both configs
    cwd_config = cwd_dir / "haiku.rag.yaml"
    cwd_config.write_text("environment: from-cwd")

    user_config = user_dir / "haiku.rag.yaml"
    user_config.write_text("environment: from-user")

    found = find_config_file()
    assert found == cwd_config
    assert found is not None

    config_data = load_yaml_config(found)
    assert config_data["environment"] == "from-cwd"


def test_config_precedence_env_var_over_cwd(tmp_path, monkeypatch):
    """Test that HAIKU_RAG_CONFIG_PATH env var takes precedence."""
    monkeypatch.chdir(tmp_path)

    # Create cwd config
    cwd_config = tmp_path / "haiku.rag.yaml"
    cwd_config.write_text("environment: from-cwd")

    # Create env var config
    env_config = tmp_path / "from-env.yaml"
    env_config.write_text("environment: from-env-var")
    monkeypatch.setenv("HAIKU_RAG_CONFIG_PATH", str(env_config))

    found = find_config_file()
    assert found == env_config
    assert found is not None

    config = load_yaml_config(found)
    assert config["environment"] == "from-env-var"


def test_generate_default_config_completeness():
    """Test that generated config has all fields from AppConfig and validates."""
    config_data = generate_default_config()

    # Validate against AppConfig model
    config = AppConfig.model_validate(config_data)

    # Get all fields from AppConfig
    expected_fields = set(AppConfig.model_fields.keys())
    actual_fields = set(config_data.keys())

    # Verify all expected fields are present
    assert expected_fields == actual_fields, (
        f"Missing fields: {expected_fields - actual_fields}, "
        f"Extra fields: {actual_fields - expected_fields}"
    )

    # Verify nested structures are dicts/lists, not model instances
    for field_name, field_value in config_data.items():
        assert not hasattr(field_value, "model_dump"), (
            f"Field {field_name} should be dict/primitive, not Pydantic model"
        )

    # Verify config validates successfully
    assert config.environment == "production"
    assert config.embeddings.model.provider == "ollama"
    assert config.qa.model.provider == "ollama"
    assert config.reranking.model is None


def test_init_config_writes_a_loadable_config(tmp_path):
    """`haiku-rag init-config` output has to load back as an AppConfig."""
    from typer.testing import CliRunner

    from haiku.rag.cli import _cli as cli

    config_file = tmp_path / "test-config.yaml"
    result = CliRunner().invoke(cli, ["init-config", str(config_file)])

    assert result.exit_code == 0, result.output
    config = AppConfig.model_validate(yaml.safe_load(config_file.read_text()))
    assert config.environment == "production"


def test_init_config_refuses_to_overwrite(tmp_path):
    """Overwriting a config in place would lose an operator's settings."""
    from typer.testing import CliRunner

    from haiku.rag.cli import _cli as cli

    config_file = tmp_path / "test-config.yaml"
    config_file.write_text("environment: development\n")

    result = CliRunner().invoke(cli, ["init-config", str(config_file)])

    assert result.exit_code == 1
    assert "already exists" in result.output
    assert config_file.read_text() == "environment: development\n"


def _write(tmp_path, body: str):
    p = tmp_path / "haiku.rag.yaml"
    p.write_text(body)
    return p


def test_pictures_field_defaults_to_image():
    """`processing.pictures` defaults to `'image'` — current ingest behavior
    (store picture bytes, no VLM) preserved on fresh configs."""
    cfg = AppConfig()
    assert cfg.processing.pictures == "image"


def test_picture_description_has_no_enabled_field():
    """`enabled` is gone from PictureDescriptionConfig — activation is now
    controlled by `processing.pictures == 'description'`."""
    from haiku.rag.config.models import PictureDescriptionConfig

    assert "enabled" not in PictureDescriptionConfig.model_fields


def test_fetch_remote_images_default_true():
    """`fetch_remote_images` defaults to True and round-trips through YAML."""
    from haiku.rag.config.models import ConversionOptions

    assert ConversionOptions().fetch_remote_images is True

    cfg = AppConfig()
    assert cfg.processing.conversion_options.fetch_remote_images is True

    data = generate_default_config()
    assert data["processing"]["conversion_options"]["fetch_remote_images"] is True


def test_fetch_remote_images_override_via_yaml(tmp_path):
    """User can disable image fetching via YAML."""
    config_file = _write(
        tmp_path,
        """
processing:
  conversion_options:
    fetch_remote_images: false
""",
    )
    data = load_yaml_config(config_file)
    cfg = AppConfig.model_validate(data)
    assert cfg.processing.conversion_options.fetch_remote_images is False


def test_analysis_model_defaults_to_none():
    """``AnalysisConfig.model`` is ``None`` by default; consumers resolve via
    ``config.analysis.model or config.qa.model``. Keeps the field semantics
    simple: ``None`` means "no override, inherit from QA"."""
    cfg = AppConfig()
    assert cfg.analysis.model is None


def test_analysis_model_unset_resolves_to_qa(tmp_path):
    """When YAML configures ``qa.model`` and omits ``analysis.model``, the
    resolve idiom yields qa.model."""
    cfg = AppConfig.model_validate(
        load_yaml_config(
            _write(
                tmp_path,
                """
qa:
  model:
    provider: openai
    name: my/qwen
    base_url: http://example/v1
    vision: true
""",
            )
        )
    )
    assert cfg.qa.model.name == "my/qwen"
    assert cfg.analysis.model is None
    resolved = cfg.analysis.model or cfg.qa.model
    assert resolved.name == "my/qwen"
    assert resolved.vision is True


def test_analysis_other_fields_keep_defaults_with_unset_model(tmp_path):
    """``analysis`` may contain non-model overrides (e.g. ``code_timeout``)
    without a ``model`` key; model stays None, other fields take user values."""
    cfg = AppConfig.model_validate(
        load_yaml_config(
            _write(
                tmp_path,
                """
analysis:
  code_timeout: 120
""",
            )
        )
    )
    assert cfg.analysis.model is None
    assert cfg.analysis.code_timeout == 120.0


def test_analysis_model_explicit_overrides_qa(tmp_path):
    """An explicit ``analysis.model`` in YAML wins over the qa fallback."""
    cfg = AppConfig.model_validate(
        load_yaml_config(
            _write(
                tmp_path,
                """
qa:
  model:
    name: qa-model
    provider: openai
analysis:
  model:
    name: analysis-model
    provider: ollama
""",
            )
        )
    )
    assert cfg.qa.model.name == "qa-model"
    assert cfg.analysis.model is not None
    assert cfg.analysis.model.name == "analysis-model"
    resolved = cfg.analysis.model or cfg.qa.model
    assert resolved.name == "analysis-model"


def test_redact_secrets_masks_nested_secret_keys():
    from haiku.rag.config.loader import redact_secrets

    data = {
        "api_key": "sk-123",
        "name": "ollama",
        "ingester": {"api": {"auth_token": "secret", "host": "0.0.0.0"}},
        "missing_token": None,
        "sources": [{"password": "pw", "url": "http://x"}],
        "storage_options": {"aws_secret_access_key": "abc"},
    }

    redacted = redact_secrets(data)

    assert redacted["api_key"] == "***"
    assert redacted["name"] == "ollama"
    assert redacted["ingester"]["api"]["auth_token"] == "***"
    assert redacted["ingester"]["api"]["host"] == "0.0.0.0"
    assert redacted["missing_token"] is None
    assert redacted["sources"][0]["password"] == "***"
    assert redacted["sources"][0]["url"] == "http://x"
    assert redacted["storage_options"]["aws_secret_access_key"] == "***"


def test_expand_env_var_set(tmp_path, monkeypatch):
    """A ${VAR} referencing a set variable is substituted."""
    monkeypatch.setenv("HAIKU_TEST_MODEL", "my-model")
    config_file = tmp_path / "test.yaml"
    config_file.write_text("""
embeddings:
  model:
    name: ${HAIKU_TEST_MODEL}
""")

    config = load_yaml_config(config_file)
    assert config["embeddings"]["model"]["name"] == "my-model"


def test_expand_env_var_dburi_preserves_special_chars(tmp_path, monkeypatch):
    """A password containing : and @ fills the string without breaking the URL."""
    monkeypatch.setenv("HAIKU_TEST_PGPW", "p@ss:word")
    config_file = tmp_path / "test.yaml"
    config_file.write_text("""
ingester:
  queue:
    dburi: "postgresql+asyncpg://user:${HAIKU_TEST_PGPW}@host/db"
""")

    config = load_yaml_config(config_file)
    assert (
        config["ingester"]["queue"]["dburi"]
        == "postgresql+asyncpg://user:p@ss:word@host/db"
    )


def test_expand_env_var_unset_raises(tmp_path, monkeypatch):
    """An unset ${VAR} without a default raises, naming the variable."""
    from haiku.rag.config.loader import MissingEnvVarError

    monkeypatch.delenv("HAIKU_TEST_MISSING", raising=False)
    config_file = tmp_path / "test.yaml"
    config_file.write_text("environment: ${HAIKU_TEST_MISSING}")

    with pytest.raises(MissingEnvVarError, match="HAIKU_TEST_MISSING"):
        load_yaml_config(config_file)


def test_expand_env_var_empty_raises(tmp_path, monkeypatch):
    """A bare ${VAR} set to an empty string is treated as unset and raises."""
    from haiku.rag.config.loader import MissingEnvVarError

    monkeypatch.setenv("HAIKU_TEST_EMPTY", "")
    config_file = tmp_path / "test.yaml"
    config_file.write_text("api_key: ${HAIKU_TEST_EMPTY}")

    with pytest.raises(MissingEnvVarError, match="HAIKU_TEST_EMPTY"):
        load_yaml_config(config_file)


def test_expand_env_var_default_when_unset(tmp_path, monkeypatch):
    """${VAR:-default} falls back to the default when VAR is unset."""
    monkeypatch.delenv("HAIKU_TEST_MISSING", raising=False)
    config_file = tmp_path / "test.yaml"
    config_file.write_text("environment: ${HAIKU_TEST_MISSING:-production}")

    config = load_yaml_config(config_file)
    assert config["environment"] == "production"


def test_expand_env_var_default_when_empty(tmp_path, monkeypatch):
    """${VAR:-default} falls back to the default when VAR is set but empty."""
    monkeypatch.setenv("HAIKU_TEST_EMPTY", "")
    config_file = tmp_path / "test.yaml"
    config_file.write_text("environment: ${HAIKU_TEST_EMPTY:-production}")

    config = load_yaml_config(config_file)
    assert config["environment"] == "production"


def test_expand_env_var_default_overridden_when_set(tmp_path, monkeypatch):
    """${VAR:-default} uses the variable when it is set and non-empty."""
    monkeypatch.setenv("HAIKU_TEST_ENV", "staging")
    config_file = tmp_path / "test.yaml"
    config_file.write_text("environment: ${HAIKU_TEST_ENV:-production}")

    config = load_yaml_config(config_file)
    assert config["environment"] == "staging"


def test_expand_env_var_dollar_escape(tmp_path):
    """$$ collapses to a literal $, leaving ${...} text intact."""
    config_file = tmp_path / "test.yaml"
    config_file.write_text("environment: $${NOT_A_VAR}")

    config = load_yaml_config(config_file)
    assert config["environment"] == "${NOT_A_VAR}"


def test_expand_env_var_nested_in_list_and_dict(tmp_path, monkeypatch):
    """Expansion recurses through lists and nested dicts."""
    monkeypatch.setenv("HAIKU_TEST_TOKEN", "abc123")
    monkeypatch.setenv("HAIKU_TEST_KEY", "AKIA")
    config_file = tmp_path / "test.yaml"
    config_file.write_text("""
ingester:
  sources:
    - type: http
      id: arxiv
      urls:
        - https://example.com/${HAIKU_TEST_TOKEN}.pdf
      storage_options:
        aws_access_key_id: ${HAIKU_TEST_KEY}
""")

    config = load_yaml_config(config_file)
    source = config["ingester"]["sources"][0]
    assert source["urls"][0] == "https://example.com/abc123.pdf"
    assert source["storage_options"]["aws_access_key_id"] == "AKIA"


def test_expand_env_var_leaves_non_strings_untouched(tmp_path):
    """Non-string scalars pass through unchanged."""
    config_file = tmp_path / "test.yaml"
    config_file.write_text("""
embeddings:
  model:
    vector_dim: 1024
ingester:
  sources:
    - type: s3
      storage_options:
        allow_http: true
""")

    config = load_yaml_config(config_file)
    assert config["embeddings"]["model"]["vector_dim"] == 1024
    assert config["ingester"]["sources"][0]["storage_options"]["allow_http"] is True


def test_expand_env_var_plain_string_unchanged(tmp_path):
    """A string with no ${...} reference is returned as-is."""
    config_file = tmp_path / "test.yaml"
    config_file.write_text("environment: production")

    config = load_yaml_config(config_file)
    assert config["environment"] == "production"


def test_find_config_file_returns_none_when_nothing_exists(tmp_path, monkeypatch):
    """No env var, no file in cwd, none in the data dir."""
    monkeypatch.delenv("HAIKU_RAG_CONFIG_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    empty_data_dir = tmp_path / "data"
    empty_data_dir.mkdir()
    monkeypatch.setattr("haiku.rag.utils.get_default_data_dir", lambda: empty_data_dir)

    assert find_config_file() is None


def test_load_default_config_falls_back_to_builtin_defaults(monkeypatch):
    """With no config file discoverable, the packaged defaults are used."""
    from haiku.rag.config import _load_default_config

    monkeypatch.setattr("haiku.rag.config.find_config_file", lambda _=None: None)

    config = _load_default_config()

    assert config.model_dump() == AppConfig().model_dump()


def test_get_config_initialises_lazily_then_reuses(monkeypatch):
    """get_config() builds the instance on first use and caches it.

    Asserted explicitly rather than relying on some test happening to be the
    first caller in its worker process: under xdist that depends on how cases
    shard across workers, which varies with the core count.
    """
    from haiku.rag import config as config_module

    monkeypatch.setattr(config_module, "_config", None)

    first = config_module.get_config()
    assert isinstance(first, AppConfig)
    assert config_module.get_config() is first


# Every YAML config shipped in the repo has to load. These are what users copy.
_EXAMPLE_CONFIGS = sorted(
    (Path(__file__).resolve().parent.parent).glob("**/*.yaml.example")
)


def test_example_configs_are_present():
    """Guard against the glob silently matching nothing."""
    assert _EXAMPLE_CONFIGS


@pytest.mark.parametrize(
    "path", _EXAMPLE_CONFIGS, ids=lambda p: p.parent.name + "/" + p.name
)
def test_example_config_validates(path: Path):
    AppConfig.model_validate(yaml.safe_load(path.read_text()) or {})


def _use_config(monkeypatch, cfg):
    """Install cfg as the global config, restored by monkeypatch teardown."""
    import haiku.rag.config as config_module

    monkeypatch.setattr(config_module, "_config", None)
    set_config(cfg)


def test_set_config_reaches_the_factories(monkeypatch, tmp_path):
    """A config installed after import must reach every factory and
    constructor that resolves the global config itself."""
    from haiku.rag.chunkers import get_chunker
    from haiku.rag.chunkers.docling_serve import DoclingServeChunker
    from haiku.rag.client import HaikuRAG
    from haiku.rag.config.models import ModelConfig, RerankingConfig
    from haiku.rag.converters import get_converter
    from haiku.rag.converters.docling_serve import DoclingServeConverter
    from haiku.rag.embeddings import get_embedder
    from haiku.rag.reranking import get_reranker
    from haiku.rag.reranking.vllm import VLLMReranker
    from haiku.rag.store.engine import Store

    cfg = AppConfig()
    cfg.processing.converter = "docling-serve"
    cfg.processing.chunker = "docling-serve"
    cfg.embeddings.model.vector_dim = 7
    cfg.reranking.model = ModelConfig(
        provider="vllm", name="reranker-x", base_url="http://localhost:9/v1"
    )
    assert isinstance(cfg.reranking, RerankingConfig)

    _use_config(monkeypatch, cfg)

    assert isinstance(get_converter(), DoclingServeConverter)
    assert isinstance(get_chunker(), DoclingServeChunker)
    assert get_embedder().vector_dim == 7

    reranker = get_reranker()
    assert isinstance(reranker, VLLMReranker)
    assert reranker._model == "reranker-x"

    assert HaikuRAG(tmp_path / "db")._config is cfg
    assert Store(tmp_path / "db", create=True)._config is cfg


@pytest.mark.parametrize(
    "data, bad_key",
    [
        ({"bogus": 1}, "bogus"),
        ({"search": {"bogus": 1}}, "search.bogus"),
        (
            {"processing": {"conversion_options": {"bogus": 1}}},
            "processing.conversion_options.bogus",
        ),
        (
            {"providers": {"docling_serve": {"bogus": 1}}},
            "providers.docling_serve.bogus",
        ),
        ({"qa": {"model": {"bogus": 1}}}, "qa.model.bogus"),
        ({"ingester": {"queue": {"bogus": 1}}}, "ingester.queue.bogus"),
        (
            {"ingester": {"sources": [{"type": "fs", "root": "/tmp", "bogus": 1}]}},
            "ingester.sources.0.fs.bogus",
        ),
    ],
)
def test_unknown_keys_are_rejected(data, bad_key):
    with pytest.raises(ValidationError) as excinfo:
        AppConfig.model_validate(data)

    errors = excinfo.value.errors()
    assert any(err["type"] == "extra_forbidden" for err in errors)
    locations = {".".join(str(part) for part in err["loc"]) for err in errors}
    assert bad_key in locations


@pytest.mark.parametrize(
    "data",
    [
        {"processing": {"converter": "docling-loca"}},
        {"processing": {"chunker": "docling-remote"}},
        {"processing": {"chunker_type": "semantic"}},
        {"search": {"vector_index_metric": "dot"}},
    ],
)
def test_finite_switches_reject_unknown_values(data):
    with pytest.raises(ValidationError):
        AppConfig.model_validate(data)


@pytest.mark.parametrize(
    "data",
    [
        {"search": {"limit": 0}},
        {"search": {"max_context_chars": 0}},
        {"embeddings": {"batch_size": 0}},
        {"embeddings": {"model": {"vector_dim": 0}}},
        {"processing": {"chunk_size": 0}},
        {"storage": {"vacuum_retention_seconds": -1}},
        {"analysis": {"code_timeout": 0}},
        {"doctor": {"duplicates": {"similarity_threshold": 1.5}}},
        {"ingester": {"workers": {"worker_count": -1}}},
        {"ingester": {"api": {"port": 70000}}},
        {"ingester": {"queue": {"retention_days": -1}}},
        {"ingester": {"sources": [{"type": "fs", "root": "/tmp", "max_file_size": 0}]}},
        {"qa": {"model": {"max_tokens": 0}}},
        {"providers": {"docling_serve": {"max_attempts": 0}}},
        {"providers": {"docling_serve": {"circuit_breaker": {"failure_threshold": 0}}}},
        {"providers": {"docling_serve": {"circuit_breaker": {"cooldown_s": -1}}}},
    ],
)
def test_out_of_range_numbers_are_rejected(data):
    with pytest.raises(ValidationError):
        AppConfig.model_validate(data)


_DOCS_ROOT = Path(__file__).resolve().parent.parent


def _documented_config_blocks() -> list[tuple[str, int, str]]:
    """Every fenced yaml block in the docs that looks like an AppConfig fragment."""
    known = set(AppConfig.model_fields)
    blocks = []
    sources = sorted(_DOCS_ROOT.glob("docs/**/*.md")) + [
        _DOCS_ROOT / "README.md",
        _DOCS_ROOT / "haiku_rag_slim" / "README.md",
    ]
    for path in sources:
        if not path.exists():
            continue
        text = path.read_text()
        for match in re.finditer(r"```yaml\n(.*?)```", text, re.S):
            data = yaml.safe_load(match.group(1))
            if not isinstance(data, dict) or not (set(data) & known):
                continue
            line = text[: match.start()].count("\n") + 1
            blocks.append((str(path.relative_to(_DOCS_ROOT)), line, match.group(1)))
    return blocks


def test_documented_config_blocks_found():
    """Guard against the regex silently matching nothing."""
    assert len(_documented_config_blocks()) > 20


@pytest.mark.parametrize(
    "rel_path, line, block",
    _documented_config_blocks(),
    ids=[f"{rel}:{line}" for rel, line, _ in _documented_config_blocks()],
)
def test_documented_config_block_validates(rel_path, line, block):
    """A config example a reader can copy has to load. Unknown keys, wrong types
    and removed settings all fail here rather than on their first run."""
    AppConfig.model_validate(yaml.safe_load(block))


def test_documented_search_limit_matches_the_default():
    """Prose that states a default drifts silently; pin the ones that are stated."""
    qa_doc = (_DOCS_ROOT / "docs" / "configuration" / "qa.md").read_text()
    assert f"Default: {AppConfig().search.limit}" in qa_doc


@pytest.mark.parametrize("value", ["", " "])
def test_empty_data_dir_means_the_platform_default(value):
    """Both doc pages promise this, and soliplex's example config relies on it.
    Without it the value coerces to Path("") and the database lands in whatever
    directory the process started from."""
    from haiku.rag.utils import get_default_data_dir

    config = AppConfig.model_validate({"storage": {"data_dir": value}})

    assert config.storage.data_dir == get_default_data_dir()


def test_explicit_relative_data_dir_is_kept():
    """`.` is a deliberate choice and must not be rewritten."""
    config = AppConfig.model_validate({"storage": {"data_dir": "."}})

    assert config.storage.data_dir == Path(".")


# Values the complete example shows deliberately rather than as defaults.
_EXAMPLE_DEVIATIONS = {"storage.data_dir", "ingester.sources"}


def _flatten(data: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in (data or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, path + "."))
        else:
            flat[path] = value
    return flat


def test_complete_example_matches_the_defaults():
    """The complete configuration example doubles as the default reference, so
    every value in it either is the default or is listed as a deliberate
    deviation. This is what catches `limit: 10` when the default is 5."""
    text = (_DOCS_ROOT / "docs" / "configuration" / "index.md").read_text()
    blocks = re.findall(r"```yaml\n(.*?)```", text, re.S)
    documented = _flatten(yaml.safe_load(max(blocks, key=len)))
    defaults = _flatten(AppConfig().model_dump(mode="json"))

    drifted = {
        key: (value, defaults[key])
        for key, value in documented.items()
        if key in defaults and defaults[key] != value and key not in _EXAMPLE_DEVIATIONS
    }

    assert not drifted, f"documented value != default: {drifted}"
