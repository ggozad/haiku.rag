from pathlib import Path
from unittest.mock import patch

import pytest
from docling_core.types.doc.base import BoundingBox, Size
from docling_core.types.doc.document import DoclingDocument, ImageRef, ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel
from PIL import Image as PILImage
from typer.testing import CliRunner

from evaluations.datasets.collection_routing import shard_of
from evaluations.split import DOCUMENT_COLUMN, split_database
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


def _document(uri: str, color: str) -> DoclingDocument:
    """A page with an image and a picture with bytes, so both blobs exist."""
    doc = DoclingDocument(name=uri)
    image = ImageRef.from_pil(PILImage.new("RGB", (8, 8), color), dpi=72)
    doc.add_page(page_no=1, size=Size(width=100, height=100), image=image)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text=f"text of {uri}",
        prov=ProvenanceItem(
            page_no=1, bbox=BoundingBox(l=0, t=0, r=50, b=10), charspan=(0, 0)
        ),
    )
    doc.add_picture(
        image=image,
        prov=ProvenanceItem(
            page_no=1, bbox=BoundingBox(l=0, t=20, r=50, b=60), charspan=(0, 0)
        ),
    )
    return doc


async def _seed(path: Path, config: AppConfig, uris: list[str]) -> None:
    dim = config.embeddings.model.vector_dim
    async with HaikuRAG(path, config=config, create=True) as rag:
        for n, uri in enumerate(uris):
            await rag.import_document(
                _document(uri, ["red", "green", "blue", "white"][n % 4]),
                [Chunk(content=f"text of {uri}", embedding=[0.1 * (n + 1)] * dim)],
                uri=uri,
                title=f"title {n}",
                metadata={"n": n},
            )


async def _rows(rag: HaikuRAG, table: str, ids: list[str]) -> list[dict]:
    column = DOCUMENT_COLUMN[table]
    listed = ", ".join(f"'{i}'" for i in ids)
    arrow = await (
        getattr(rag.store, f"{table}_table")
        .query()
        .where(f"{column} IN ({listed})")
        .to_arrow()
    )
    rows = arrow.to_pylist()
    return sorted(
        rows,
        key=lambda row: tuple(
            str(row.get(key, "")) for key in ("document_id", "id", "self_ref", "order")
        ),
    )


async def test_every_row_of_a_document_lands_verbatim_in_its_shard(
    tmp_path: Path,
) -> None:
    config = AppConfig()
    per_shard = _uris_per_shard(2, 2)
    source = tmp_path / "src.lancedb"
    await _seed(source, config, per_shard[0] + per_shard[1])
    destinations = [tmp_path / "a.lancedb", tmp_path / "b.lancedb"]

    with patch(
        "haiku.rag.embeddings.embed_chunks", side_effect=AssertionError("embedded")
    ):
        counts = await split_database(source, destinations, config)

    assert counts == [2, 2]
    async with HaikuRAG(source, config=config, read_only=True) as src:
        ids_by_uri = {d.uri: d.id for d in await src.list_documents()}
        for shard, destination in enumerate(destinations):
            ids = [ids_by_uri[uri] for uri in per_shard[shard]]
            async with HaikuRAG(destination, config=config, read_only=True) as dst:
                assert sorted(d.uri for d in await dst.list_documents()) == sorted(
                    per_shard[shard]
                )
                for table in DOCUMENT_COLUMN:
                    expected = await _rows(src, table, ids)
                    assert expected, table
                    assert await _rows(dst, table, ids) == expected, table
                documents = await _rows(dst, "documents", ids)
                assert all(row["docling_pages"] for row in documents)
                items = await _rows(dst, "document_items", ids)
                assert any(row["picture_data"] for row in items)
                stats = await dst.store.chunks_table.index_stats("content_fts_idx")
                assert stats is not None and stats.num_indexed_rows == 2


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
