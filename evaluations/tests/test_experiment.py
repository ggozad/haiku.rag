import subprocess
from pathlib import Path

from docling_core.types.doc import DocItemLabel, DoclingDocument

from evaluations.experiment import (
    build_experiment_metadata,
    code_revision,
    config_hash,
    corpus_fingerprint,
)
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig, ModelConfig
from haiku.rag.store.models.chunk import Chunk
from tests.specs import stub_spec


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


class TestCodeRevision:
    def test_reports_the_checked_out_commit(self) -> None:
        here = Path(__file__).parent
        revision = code_revision(here)
        assert revision["git_sha"] == _git(here, "rev-parse", "HEAD")
        assert isinstance(revision["git_dirty"], bool)

    def test_defaults_to_the_package_checkout(self) -> None:
        assert code_revision() == code_revision(Path(__file__).parent)

    def test_flags_uncommitted_changes(self, tmp_path: Path) -> None:
        _git(tmp_path, "init", "-q")
        (tmp_path / "a.txt").write_text("a")
        _git(tmp_path, "add", "a.txt")
        _git(
            tmp_path,
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "init",
        )
        assert code_revision(tmp_path)["git_dirty"] is False
        (tmp_path / "untracked.log").write_text("noise")
        assert code_revision(tmp_path)["git_dirty"] is False
        (tmp_path / "a.txt").write_text("b")
        assert code_revision(tmp_path)["git_dirty"] is True

    def test_outside_a_repository_reports_nothing(self, tmp_path: Path) -> None:
        assert code_revision(tmp_path) == {"git_sha": None, "git_dirty": None}


class TestConfigHash:
    def test_equal_configs_hash_equal(self) -> None:
        digest = config_hash(AppConfig())
        assert digest == config_hash(AppConfig())
        assert len(digest) == 64

    def test_a_changed_key_changes_the_hash(self) -> None:
        changed = AppConfig()
        changed.search.limit = 7
        assert config_hash(changed) != config_hash(AppConfig())


class TestBuildExperimentMetadata:
    def test_a_retrieval_run_records_config_and_run_identity(self) -> None:
        config = AppConfig()
        result = build_experiment_metadata(
            dataset_key="t", test_cases=42, config=config
        )

        assert result["dataset"] == "t"
        assert result["test_cases"] == 42
        assert result["embedder_provider"] == config.embeddings.model.provider
        assert result["embedder_model"] == config.embeddings.model.name
        assert result["embedder_dim"] == config.embeddings.model.vector_dim
        assert result["chunk_size"] == config.processing.chunk_size
        assert result["search_limit"] == config.search.limit
        assert result["qa_max_searches"] == config.qa.max_searches
        assert result["qa_max_executions"] == config.qa.max_executions
        assert result["sandbox_code_timeout"] == config.sandbox.code_timeout
        assert result["sandbox_max_output_chars"] == config.sandbox.max_output_chars
        assert result["rerank_provider"] is None
        assert result["rerank_model"] is None
        assert result["document_filter"] is None
        for absent in (
            "qa_provider",
            "qa_model",
            "judge_provider",
            "target",
            "capability_provider",
            "capability_model",
        ):
            assert absent not in result
        revision = code_revision()
        assert result["git_sha"] == revision["git_sha"]
        assert result["git_dirty"] == revision["git_dirty"]
        assert result["config_hash"] == config_hash(config)

    def test_a_qa_run_records_judge_capability_reranker_and_filter(self) -> None:
        config = AppConfig()
        config.reranking.model = ModelConfig(
            provider="vllm", name="Qwen/Qwen3-Reranker-4B"
        )
        judge = ModelConfig(
            provider="ollama",
            name="gpt-oss",
            thinking=False,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        capability = ModelConfig(
            provider="ollama",
            name="gpt-oss-large",
            temperature=0.2,
            thinking="low",
            extra_body={"min_p": 0},
        )
        result = build_experiment_metadata(
            dataset_key="t",
            test_cases=1,
            config=config,
            judge_config=judge,
            capability_config=capability,
            document_filter="uri LIKE '%arxiv%'",
        )

        assert result["judge_provider"] == "ollama"
        assert result["judge_model"] == "gpt-oss"
        assert result["judge_temperature"] == 0.0
        assert result["judge_thinking"] is False
        assert result["judge_extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": True}
        }
        assert "judge_enable_thinking" not in result
        assert result["capability_provider"] == "ollama"
        assert result["capability_model"] == "gpt-oss-large"
        assert result["capability_temperature"] == 0.2
        assert result["capability_thinking"] == "low"
        assert result["capability_extra_body"] == {"min_p": 0}
        assert "capability_enable_thinking" not in result
        assert "qa_extra_body" not in result
        assert result["rerank_provider"] == "vllm"
        assert result["rerank_model"] == "Qwen/Qwen3-Reranker-4B"
        assert result["document_filter"] == "uri LIKE '%arxiv%'"


class TestCorpusFingerprint:
    async def test_no_database_placed_reports_nothing(self) -> None:
        fingerprint = await corpus_fingerprint(None, AppConfig())
        assert fingerprint == {
            "db_path": None,
            "db_documents": None,
            "db_chunks": None,
            "db_embedder_provider": None,
            "db_embedder_model": None,
            "db_embedder_dim": None,
            "db_version": None,
            "db_written_at": None,
        }

    async def test_missing_database_reports_only_its_path(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.lancedb"
        fingerprint = await corpus_fingerprint(path, AppConfig())
        assert fingerprint["db_path"] == str(path)
        assert fingerprint["db_documents"] is None
        assert fingerprint["db_embedder_model"] is None
        assert not path.exists()

    async def test_a_directory_without_tables_reports_only_its_path(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "empty.lancedb"
        path.mkdir()
        fingerprint = await corpus_fingerprint(path, AppConfig())
        assert fingerprint["db_path"] == str(path)
        assert fingerprint["db_documents"] is None

    async def test_reads_counts_and_the_stored_embedder(self, tmp_path: Path) -> None:
        config = AppConfig()
        dim = config.embeddings.model.vector_dim
        path = tmp_path / "corpus.lancedb"
        document = DoclingDocument(name="doc")
        document.add_text(label=DocItemLabel.TEXT, text="alpha")
        document.add_text(label=DocItemLabel.TEXT, text="beta")
        chunks = [
            Chunk(content="alpha", embedding=[0.1] * dim),
            Chunk(content="beta", embedding=[0.2] * dim),
        ]
        async with HaikuRAG(path, config=config, create=True) as rag:
            await rag.import_document(document, chunks, uri="test://doc")

        fingerprint = await corpus_fingerprint(path, config)

        assert fingerprint["db_path"] == str(path)
        assert fingerprint["db_documents"] == 1
        assert fingerprint["db_chunks"] == 2
        assert fingerprint["db_embedder_provider"] == config.embeddings.model.provider
        assert fingerprint["db_embedder_model"] == config.embeddings.model.name
        assert fingerprint["db_embedder_dim"] == dim
        assert fingerprint["db_version"] not in (None, "unknown")
        assert fingerprint["db_written_at"] is not None

    async def test_a_rewrite_moves_the_write_time_at_equal_counts(
        self, tmp_path: Path
    ) -> None:
        """Counts read a rebuilt corpus as the one it replaced. The write does not."""
        config = AppConfig()
        dim = config.embeddings.model.vector_dim
        path = tmp_path / "corpus.lancedb"
        document = DoclingDocument(name="doc")
        document.add_text(label=DocItemLabel.TEXT, text="alpha")
        async with HaikuRAG(path, config=config, create=True) as rag:
            doc = await rag.import_document(
                document, [Chunk(content="alpha", embedding=[0.1] * dim)], uri="d://1"
            )
            before = await corpus_fingerprint(path, config)
            assert doc.id is not None
            doc.content = "rewritten"
            await rag.document_repository.update_meta(doc)
            after = await corpus_fingerprint(path, config)

        assert after["db_documents"] == before["db_documents"]
        assert after["db_chunks"] == before["db_chunks"]
        assert after["db_written_at"] > before["db_written_at"]


class TestPopulationKeepsTheConfigHash:
    async def test_populating_leaves_the_callers_config_alone(
        self, tmp_path: Path
    ) -> None:
        from evaluations.population import populate_db

        spec = stub_spec(
            key="empty",
            db_filename="empty.lancedb",
            document_loader=lambda: [],
        )
        config = AppConfig()
        before = config_hash(config)

        await populate_db(spec, config, db_path=tmp_path / "empty.lancedb")

        assert config_hash(config) == before
