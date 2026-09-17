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
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models.chunk import Chunk


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


class TestMetadataRecordsRunIdentity:
    def test_records_revision_and_config_hash(self) -> None:
        config = AppConfig()
        result = build_experiment_metadata(dataset_key="t", test_cases=1, config=config)
        revision = code_revision()
        assert result["git_sha"] == revision["git_sha"]
        assert result["git_dirty"] == revision["git_dirty"]
        assert result["config_hash"] == config_hash(config)


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
        }

    async def test_missing_database_reports_only_its_path(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.lancedb"
        fingerprint = await corpus_fingerprint(path, AppConfig())
        assert fingerprint["db_path"] == str(path)
        assert fingerprint["db_documents"] is None
        assert fingerprint["db_embedder_model"] is None
        assert not path.exists()

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
