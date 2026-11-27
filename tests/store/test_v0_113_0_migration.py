import base64
import json
import logging
import os
import subprocess
import sys
import textwrap
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

import lance
import pyarrow as pa
import pytest
from docling_core.types.doc.base import Size
from docling_core.types.doc.document import DoclingDocument, ImageRef
from PIL import Image

from haiku.rag.client import HaikuRAG
from haiku.rag.store.compression import compress_json
from haiku.rag.store.engine import ConnectionMode, Store
from haiku.rag.store.exceptions import MigrationRequiredError
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.repositories.settings import SettingsRepository
from haiku.rag.store.upgrades import get_pending_upgrades
from haiku.rag.store.upgrades.v0_113_0 import _apply_original_columns
from tests.conftest import capture_logs

SIZES = {1: (612.0, 792.0), 2: (792.0, 612.0)}


def _docling_with_rasters(name: str) -> DoclingDocument:
    doc = DoclingDocument(name=name)
    noise = Image.frombytes("L", (400, 400), os.urandom(400 * 400)).convert("RGB")
    for page_no, (width, height) in SIZES.items():
        doc.add_page(
            page_no=page_no,
            size=Size(width=width, height=height),
            image=ImageRef.from_pil(noise, dpi=72),
        )
    doc.add_picture(image=ImageRef.from_pil(noise, dpi=72))
    return doc


def _previous_release_row(doc: DoclingDocument) -> dict[str, bytes]:
    """A docling document stored as the release before 0.113.0 stored it:
    the structure without pages, and the pages with their rasters."""
    data = doc.model_dump(mode="json")
    pages = data.pop("pages")
    for picture in data["pictures"]:
        picture["image"] = None
    return {
        "docling_document": compress_json(json.dumps(data)),
        "docling_pages": compress_json(json.dumps(pages)),
    }


async def _at_previous_release(store: Store, docs: dict[str, DoclingDocument]):
    """Rewrite the documents table to the 0.112 schema with stored rasters,
    and record the release before 0.113.0."""
    table = store.documents_table
    await table.drop_columns(["original", "original_sha256"])
    await table.add_columns([pa.field("docling_pages", pa.large_binary())])
    schema = await table.schema()
    columns = ["id", "docling_document", "docling_pages"]
    rows = [{"id": i, **_previous_release_row(doc)} for i, doc in docs.items()]
    await (
        table.merge_insert("id")
        .when_matched_update_all()
        .execute(
            pa.Table.from_pylist(
                rows, schema=pa.schema([schema.field(c) for c in columns])
            )
        )
    )
    await store._write_settings(version="0.112.3", schema_version="0.91.0")


async def _seed(path: Path, count: int = 3) -> list[str]:
    async with Store(path, create=True) as store:
        repo = DocumentRepository(store)
        ids = []
        for n in range(count):
            created = await repo.create(Document(content=f"doc {n}", uri=f"t://{n}"))
            assert created.id is not None
            ids.append(created.id)
        await _at_previous_release(
            store, {i: _docling_with_rasters(f"doc {n}") for n, i in enumerate(ids)}
        )
    return ids


def _table_bytes(path: Path) -> int:
    root = path / "documents.lance"
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())


async def test_a_new_database_needs_no_migration(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        recorded = await SettingsRepository(store).get_current_settings()

    assert recorded["version"] == metadata.version("haiku.rag-slim")
    assert get_pending_upgrades(recorded["version"]) == []


async def _structures(path: Path) -> dict[str, bytes]:
    async with Store(path, skip_migration_check=True) as store:
        rows = (
            await store.documents_table.query()
            .select(["id", "docling_document"])
            .to_list()
        )
    return {r["id"]: r["docling_document"] for r in rows}


async def test_migrate_deletes_page_images(temp_db_path):
    ids = await _seed(temp_db_path)
    structures = await _structures(temp_db_path)
    async with Store(temp_db_path, skip_migration_check=True) as store:
        indexes = sorted(i.name for i in await store.documents_table.list_indices())
    before = _table_bytes(temp_db_path)

    with pytest.raises(MigrationRequiredError):
        async with Store(temp_db_path, read_only=True):
            pass
    async with Store(temp_db_path, skip_migration_check=True) as store:
        await store.migrate()

    async with Store(temp_db_path, read_only=True) as store:
        table = store.documents_table
        fields = {f.name: str(f.type) for f in await table.schema()}
        rows = await table.query().to_list()
        assert sorted(i.name for i in await table.list_indices()) == indexes
    assert "docling_pages" not in fields
    assert (fields["original"], fields["original_sha256"]) == (
        "extension<lance.blob.v2<BlobType>>",
        "string",
    )
    assert {r["id"]: r["docling_document"] for r in rows} == structures
    for row in rows:
        assert row["content"] == f"doc {ids.index(row['id'])}"
        assert (row["original"], row["original_sha256"]) == (None, None)
    assert _table_bytes(temp_db_path) < before / 10


async def test_a_tag_keeps_its_page_images_and_needs_migrating(temp_db_path):
    await _seed(temp_db_path)
    async with Store(temp_db_path, skip_migration_check=True) as store:
        await store.create_tag("before")

    with capture_logs(logging.getLogger("haiku.rag"), logging.WARNING) as records:
        async with Store(temp_db_path, skip_migration_check=True) as store:
            await store.migrate()
            tagged = (await store.list_tags())["before"].tables["documents"]

    dataset = lance.dataset(
        str(temp_db_path / "documents.lance"), version=tagged
    ).to_table(columns=["docling_pages"])
    assert all(len(pages) > 0 for pages in dataset.column(0).to_pylist())
    assert any("before" in r.getMessage() for r in records)
    with pytest.raises(MigrationRequiredError):
        async with HaikuRAG(temp_db_path, tag="before"):
            pass


async def test_a_cloud_migration_skips_maintenance(temp_db_path):
    await _seed(temp_db_path)

    with patch.object(Store, "_connection_mode", ConnectionMode.CLOUD):
        async with Store(temp_db_path, skip_migration_check=True) as store:
            versions = len(await store.documents_table.list_versions())
            await store.migrate()
            after = await store.documents_table.list_versions()

    assert len(after) > versions


@pytest.mark.parametrize("failing", ["_write_settings", "_compact_to_target"])
async def test_a_rerun_after_an_interruption_completes(temp_db_path, failing):
    await _seed(temp_db_path)
    structures = await _structures(temp_db_path)
    before = _table_bytes(temp_db_path)

    async with Store(temp_db_path, skip_migration_check=True) as store:
        with (
            patch.object(Store, failing, side_effect=RuntimeError("interrupted")),
            pytest.raises(RuntimeError, match="interrupted"),
        ):
            await store.migrate()

    async with Store(temp_db_path, skip_migration_check=True) as store:
        await store.migrate()

    async with Store(temp_db_path, read_only=True) as store:
        fields = [f.name for f in await store.documents_table.schema()]
    assert "docling_pages" not in fields
    assert await _structures(temp_db_path) == structures
    assert _table_bytes(temp_db_path) < before / 10


async def test_a_skipped_vacuum_fails_the_migration(temp_db_path):
    await _seed(temp_db_path)
    before = _table_bytes(temp_db_path)

    async with Store(temp_db_path, skip_migration_check=True) as store:
        with (
            patch.object(Store, "vacuum", return_value=False),
            pytest.raises(RuntimeError, match="haiku-rag migrate"),
        ):
            await store.migrate()
    async with Store(temp_db_path, skip_migration_check=True) as store:
        await store.migrate()

    assert _table_bytes(temp_db_path) < before / 10


_MIGRATE_RSS = textwrap.dedent(
    """
    import asyncio, sys, threading, time
    from pathlib import Path
    import psutil
    from haiku.rag.store.engine import Store

    proc = psutil.Process()
    peak = {"v": 0}

    def watch():
        while True:
            peak["v"] = max(peak["v"], proc.memory_info().rss)
            time.sleep(0.01)

    async def main():
        async with Store(Path(sys.argv[1]), skip_migration_check=True) as store:
            base = peak["v"] = proc.memory_info().rss
            threading.Thread(target=watch, daemon=True).start()
            await store.migrate()
        print((peak["v"] - base) / 1024**2)

    asyncio.run(main())
    """
)


async def _migration_peak(path: Path, rows: int) -> float:
    """Peak RSS growth in MB while migrating `rows` rows of 8 MB page images."""
    ids = await _seed(path, rows)
    image = base64.b64encode(os.urandom(8 * 1024**2)).decode()
    pages = {
        "1": {
            "size": {"width": 612.0, "height": 792.0},
            "page_no": 1,
            "image": {"uri": f"data:image/png;base64,{image}"},
        }
    }
    stored = compress_json(json.dumps(pages))
    async with Store(path, skip_migration_check=True) as store:
        table = store.documents_table
        schema = await table.schema()
        await (
            table.merge_insert("id")
            .when_matched_update_all()
            .execute(
                pa.Table.from_pylist(
                    [{"id": i, "docling_pages": stored} for i in ids],
                    schema=pa.schema(
                        [schema.field(c) for c in ("id", "docling_pages")]
                    ),
                )
            )
        )
    out = subprocess.run(
        [sys.executable, "-c", _MIGRATE_RSS, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip().splitlines()[-1])


@pytest.mark.integration
async def test_migrating_never_reads_the_page_images(tmp_path):
    """Peak memory does not grow with the stored page images."""
    small = await _migration_peak(tmp_path / "small.lancedb", 16)
    large = await _migration_peak(tmp_path / "large.lancedb", 48)

    assert large < small * 1.5


async def test_rerunning_the_step_changes_nothing(temp_db_path):
    await _seed(temp_db_path)
    async with Store(temp_db_path, skip_migration_check=True) as store:
        await _apply_original_columns(store)
        rows = await store.documents_table.query().to_list()

        await _apply_original_columns(store)

        assert await store.documents_table.query().to_list() == rows
