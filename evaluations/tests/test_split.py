from pathlib import Path
from unittest.mock import patch

import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from typer.testing import CliRunner

from evaluations.datasets.collection_routing import shard_of
from evaluations.split import split_database
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models import Chunk


@pytest.fixture(autouse=True)
def _logfire_unconfigured_is_fine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGFIRE_IGNORE_NO_CONFIG", "1")


def _uris_per_shard(shards: int, each: int) -> list[list[str]]:
    picked: list[list[str]] = [[] for _ in range(shards)]
    i = 0
    while any(len(p) < each for p in picked):
        uri = f"test://doc{i}"
        if len(picked[shard_of(uri, shards)]) < each:
            picked[shard_of(uri, shards)].append(uri)
        i += 1
    return picked


async def _seed(
    path: Path, config: AppConfig, uris: list[str]
) -> dict[str, list[float]]:
    dim = config.embeddings.model.vector_dim
    vectors: dict[str, list[float]] = {}
    async with HaikuRAG(path, config=config, create=True) as rag:
        for n, uri in enumerate(uris):
            doc = DoclingDocument(name=uri)
            doc.add_text(label=DocItemLabel.TEXT, text=f"text of {uri}")
            vectors[uri] = [0.1 * (n + 1)] * dim
            await rag.import_document(
                doc,
                [Chunk(content=f"text of {uri}", embedding=vectors[uri], order=0)],
                uri=uri,
                title=f"title {n}",
                metadata={"n": n},
            )
    return vectors


async def _read(
    path: Path, config: AppConfig
) -> tuple[dict[str, dict], dict[str, list[float]]]:
    async with HaikuRAG(path, config=config, read_only=True) as rag:
        documents = {d.uri: d for d in await rag.list_documents()}
        rows = await rag.store.chunks_table.query().to_list()
        by_doc = {row["document_id"]: list(row["vector"]) for row in rows}
        return (
            {
                uri: {"title": d.title, "metadata": d.metadata}
                for uri, d in documents.items()
            },
            {uri: by_doc[d.id] for uri, d in documents.items()},
        )


async def test_every_document_lands_in_the_shard_its_uri_hashes_to(
    tmp_path: Path,
) -> None:
    config = AppConfig()
    per_shard = _uris_per_shard(2, 2)
    source = tmp_path / "src.lancedb"
    vectors = await _seed(source, config, per_shard[0] + per_shard[1])
    destinations = [tmp_path / "a.lancedb", tmp_path / "b.lancedb"]

    with patch(
        "haiku.rag.embeddings.embed_chunks", side_effect=AssertionError("embedded")
    ):
        counts = await split_database(source, destinations, config)

    assert counts == [2, 2]
    for shard, destination in enumerate(destinations):
        documents, embeddings = await _read(destination, config)
        assert sorted(documents) == sorted(per_shard[shard])
        for uri in per_shard[shard]:
            assert embeddings[uri] == pytest.approx(vectors[uri])
        assert all(d["title"].startswith("title ") for d in documents.values())
        assert all("n" in d["metadata"] for d in documents.values())


async def test_a_document_without_a_uri_refuses(tmp_path: Path) -> None:
    config = AppConfig()
    source = tmp_path / "src.lancedb"
    dim = config.embeddings.model.vector_dim
    async with HaikuRAG(source, config=config, create=True) as rag:
        doc = DoclingDocument(name="anonymous")
        doc.add_text(label=DocItemLabel.TEXT, text="no uri")
        await rag.import_document(
            doc, [Chunk(content="no uri", embedding=[0.1] * dim, order=0)]
        )
    with pytest.raises(ValueError, match="uri"):
        await split_database(source, [tmp_path / "a.lancedb"], config)


def test_the_command_reports_the_counts(tmp_path: Path) -> None:
    from evaluations.benchmark import app

    with patch("evaluations.benchmark.split_database", return_value=[3, 4]) as split:
        result = CliRunner().invoke(
            app,
            [
                "split",
                str(tmp_path / "s.lancedb"),
                str(tmp_path / "a.lancedb"),
                str(tmp_path / "b.lancedb"),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "a.lancedb: 3" in result.output and "b.lancedb: 4" in result.output
    source, destinations, _config = split.call_args[0]
    assert source == tmp_path / "s.lancedb"
    assert destinations == [tmp_path / "a.lancedb", tmp_path / "b.lancedb"]
