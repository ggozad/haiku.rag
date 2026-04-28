import random
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.models.test import TestModel

from evaluations.skill_runner import SkillRunResult, run_skill_question
from haiku.rag.agents.research.models import Citation
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.embeddings import EmbedderWrapper
from haiku.rag.skills.analysis import (
    AnalysisState,
    create_skill as create_analysis_skill,
)
from haiku.rag.skills.rag import RAGState, create_skill as create_rag_skill
from haiku.rag.store.models.chunk import SearchResult

VECTOR_DIM = 2560


@pytest.fixture(autouse=True)
def mock_embedder(monkeypatch: pytest.MonkeyPatch):
    """Deterministic embeddings so search is reproducible."""

    async def fake_embed_query(self, text):
        random.seed(hash(text) % (2**32))
        return [random.random() for _ in range(VECTOR_DIM)]

    async def fake_embed_documents(self, texts):
        result = []
        for t in texts:
            random.seed(hash(t) % (2**32))
            result.append([random.random() for _ in range(VECTOR_DIM)])
        return result

    monkeypatch.setattr(EmbedderWrapper, "embed_query", fake_embed_query)
    monkeypatch.setattr(EmbedderWrapper, "embed_documents", fake_embed_documents)


@pytest.fixture
def app_config():
    return AppConfig(environment="skill-runner-test")


@pytest.fixture
async def rag_db(tmp_path: Path):
    """A small two-document database."""
    db_path = tmp_path / "test.lancedb"
    async with HaikuRAG(db_path, create=True) as rag:
        await rag.create_document(
            "Artificial intelligence is transforming healthcare and finance.",
            title="AI Overview",
            uri="test://ai",
        )
        await rag.create_document(
            "Machine learning includes supervised, unsupervised, and reinforcement.",
            title="ML Basics",
            uri="test://ml",
        )
    return db_path


class TestRunSkillQuestionMocked:
    """Verify the runner reads state correctly without going through a real skill loop."""

    async def test_extracts_cited_and_searched_uris(
        self, monkeypatch: pytest.MonkeyPatch, app_config: AppConfig, rag_db: Path
    ) -> None:
        async def fake_run_skill(
            model: Any,
            skill: Any,
            request: str,
            state: Any = None,
            event_sink: Any = None,
        ) -> tuple[str, list[Any], list[Any]]:
            state.citation_index["c1"] = Citation(
                chunk_id="c1",
                document_id="d1",
                document_uri="test://doc-a",
                document_title="A",
                content="alpha",
            )
            state.citation_index["c2"] = Citation(
                chunk_id="c2",
                document_id="d2",
                document_uri="test://doc-b",
                document_title="B",
                content="beta",
            )
            state.citations = ["c1", "c2"]
            state.searches["q1"] = [
                SearchResult(content="x", score=0.9, document_uri="test://doc-a"),
                SearchResult(content="y", score=0.8, document_uri="test://doc-c"),
            ]
            state.searches["q2"] = [
                SearchResult(content="z", score=0.7, document_uri="test://doc-a"),
            ]
            return "answer", [], []

        monkeypatch.setattr("evaluations.skill_runner.run_skill", fake_run_skill)

        result = await run_skill_question(
            skill_factory=create_rag_skill,
            db_path=rag_db,
            config=app_config,
            question="anything?",
            skill_model=TestModel(),
        )

        assert isinstance(result, SkillRunResult)
        assert result.answer == "answer"
        assert result.cited_chunk_ids == ["c1", "c2"]
        assert result.cited_uris == ["test://doc-a", "test://doc-b"]
        assert result.searched_uris == ["test://doc-a", "test://doc-c"]
        assert result.n_searches == 2

    async def test_skips_chunks_missing_from_index(
        self, monkeypatch: pytest.MonkeyPatch, app_config: AppConfig, rag_db: Path
    ) -> None:
        async def fake_run_skill(
            model: Any,
            skill: Any,
            request: str,
            state: Any = None,
            event_sink: Any = None,
        ) -> tuple[str, list[Any], list[Any]]:
            state.citation_index["c1"] = Citation(
                chunk_id="c1",
                document_id="d1",
                document_uri="test://doc-a",
                content="a",
            )
            state.citations = ["c1", "missing"]
            return "ok", [], []

        monkeypatch.setattr("evaluations.skill_runner.run_skill", fake_run_skill)

        result = await run_skill_question(
            skill_factory=create_rag_skill,
            db_path=rag_db,
            config=app_config,
            question="?",
            skill_model=TestModel(),
        )

        assert result.cited_chunk_ids == ["c1", "missing"]
        assert result.cited_uris == ["test://doc-a"]

    async def test_document_filter_is_set_on_state(
        self, monkeypatch: pytest.MonkeyPatch, app_config: AppConfig, rag_db: Path
    ) -> None:
        captured: dict = {}

        async def fake_run_skill(
            model: Any,
            skill: Any,
            request: str,
            state: Any = None,
            event_sink: Any = None,
        ) -> tuple[str, list[Any], list[Any]]:
            captured["filter"] = state.document_filter
            captured["state_type"] = type(state)
            return "ok", [], []

        monkeypatch.setattr("evaluations.skill_runner.run_skill", fake_run_skill)

        await run_skill_question(
            skill_factory=create_rag_skill,
            db_path=rag_db,
            config=app_config,
            question="?",
            skill_model=TestModel(),
            document_filter="uri = 'test://ai'",
        )

        assert captured["filter"] == "uri = 'test://ai'"
        assert captured["state_type"] is RAGState

    async def test_request_limit_override(
        self, monkeypatch: pytest.MonkeyPatch, app_config: AppConfig, rag_db: Path
    ) -> None:
        captured: dict = {}

        async def fake_run_skill(
            model: Any,
            skill: Any,
            request: str,
            state: Any = None,
            event_sink: Any = None,
        ) -> tuple[str, list[Any], list[Any]]:
            captured["request_limit"] = skill.request_limit
            return "ok", [], []

        monkeypatch.setattr("evaluations.skill_runner.run_skill", fake_run_skill)

        await run_skill_question(
            skill_factory=create_rag_skill,
            db_path=rag_db,
            config=app_config,
            question="?",
            skill_model=TestModel(),
            request_limit=42,
        )

        assert captured["request_limit"] == 42

    async def test_request_limit_unset_leaves_skill_default(
        self, monkeypatch: pytest.MonkeyPatch, app_config: AppConfig, rag_db: Path
    ) -> None:
        captured: dict = {}

        async def fake_run_skill(
            model: Any,
            skill: Any,
            request: str,
            state: Any = None,
            event_sink: Any = None,
        ) -> tuple[str, list[Any], list[Any]]:
            captured["request_limit"] = skill.request_limit
            return "ok", [], []

        monkeypatch.setattr("evaluations.skill_runner.run_skill", fake_run_skill)

        await run_skill_question(
            skill_factory=create_rag_skill,
            db_path=rag_db,
            config=app_config,
            question="?",
            skill_model=TestModel(),
        )

        assert captured["request_limit"] is None

    async def test_analysis_skill_uses_analysis_state(
        self, monkeypatch: pytest.MonkeyPatch, app_config: AppConfig, rag_db: Path
    ) -> None:
        captured: dict = {}

        async def fake_run_skill(
            model: Any,
            skill: Any,
            request: str,
            state: Any = None,
            event_sink: Any = None,
        ) -> tuple[str, list[Any], list[Any]]:
            captured["state_type"] = type(state)
            return "ok", [], []

        monkeypatch.setattr("evaluations.skill_runner.run_skill", fake_run_skill)

        await run_skill_question(
            skill_factory=create_analysis_skill,
            db_path=rag_db,
            config=app_config,
            question="?",
            skill_model=TestModel(),
        )

        assert captured["state_type"] is AnalysisState

    async def test_raises_when_skill_has_no_state_type(
        self, app_config: AppConfig, rag_db: Path
    ) -> None:
        from haiku.skills.models import Skill, SkillMetadata, SkillSource

        def factory(*, db_path, config) -> Skill:
            return Skill(
                metadata=SkillMetadata(name="bare", description="No state."),
                source=SkillSource.ENTRYPOINT,
                instructions="Do nothing.",
            )

        with pytest.raises(ValueError, match="no state_type"):
            await run_skill_question(
                skill_factory=factory,
                db_path=rag_db,
                config=app_config,
                question="?",
                skill_model=TestModel(),
            )


class TestRunSkillQuestionEndToEnd:
    """Real skill loop against a real LanceDB. Verifies the wiring beyond mocks."""

    async def test_rag_skill_runs_against_real_db(
        self,
        allow_model_requests: None,
        app_config: AppConfig,
        rag_db: Path,
    ) -> None:
        result = await run_skill_question(
            skill_factory=create_rag_skill,
            db_path=rag_db,
            config=app_config,
            question="What is machine learning?",
            skill_model=TestModel(),
        )

        assert isinstance(result, SkillRunResult)
        assert result.answer
        assert result.n_searches >= 1
        assert all(uri.startswith("test://") for uri in result.searched_uris)


@pytest.fixture
def allow_model_requests():
    import pydantic_ai.models

    with pydantic_ai.models.override_allow_model_requests(True):
        yield
