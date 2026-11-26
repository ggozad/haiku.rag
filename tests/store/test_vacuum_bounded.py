import asyncio
import logging
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from typing import cast

import lance
import pyarrow as pa
import pytest
from pydantic import ValidationError

from haiku.rag.config.models import AppConfig, StorageConfig
from haiku.rag.store import engine
from haiku.rag.store.engine import (
    COMPACTION_TARGET_KEY,
    CompactionSizing,
    Store,
    kept_target,
    size_compaction,
)
from tests.conftest import capture_logs

MB = 1024**2


def _write(path, rows, payload_bytes, per_fragment=10):
    schema = pa.schema(
        [pa.field("id", pa.string()), pa.field("blob", pa.large_binary())]
    )
    dataset = lance.dataset(path) if os.path.exists(path) else None
    for start in range(0, rows, per_fragment):
        batch = pa.Table.from_arrays(
            [
                pa.array(
                    [f"d{i}" for i in range(start, start + per_fragment)], pa.string()
                ),
                pa.array(
                    [os.urandom(payload_bytes) for _ in range(per_fragment)],
                    pa.large_binary(),
                ),
            ],
            schema=schema,
        )
        dataset = lance.write_dataset(
            batch, path, mode="append" if dataset else "create"
        )
    return dataset


def _target(dataset, target_bytes):
    sizing = size_compaction(dataset, target_bytes)
    assert sizing is not None
    return sizing.target_rows


def test_target_scales_inversely_with_row_size(tmp_path):
    dataset = _write(tmp_path / "wide.lance", 20, 1 * MB)

    assert _target(dataset, 8 * MB) == pytest.approx(8, abs=1)
    # halving the budget halves the rows a task may take
    assert _target(dataset, 4 * MB) == pytest.approx(4, abs=1)


def test_target_uses_widest_fragment_not_the_average(tmp_path):
    """A few outsized rows must shrink the target for the whole table."""
    path = tmp_path / "skewed.lance"
    _write(path, 10, 4 * MB)
    _write(path, 10, 64 * 1024)  # a second, much narrower fragment
    dataset = lance.dataset(path)

    target = _target(dataset, 16 * MB)

    # the average row is ~2 MB, which would allow ~8 rows; the widest is 4 MB
    assert target is not None and target <= 5


def test_a_table_of_oversized_rows_has_nothing_to_merge(tmp_path):
    """Rows wider than half the target cannot share a fragment within it."""
    dataset = _write(tmp_path / "huge.lance", 10, 2 * MB)

    sizing = size_compaction(dataset, 1024)

    assert sizing is not None
    assert sizing.target_rows is None
    assert sorted(sizing.oversized) == [f.fragment_id for f in dataset.get_fragments()]


def test_an_empty_table_has_nothing_to_merge(tmp_path):
    schema = pa.schema(
        [pa.field("id", pa.string()), pa.field("blob", pa.large_binary())]
    )
    dataset = lance.write_dataset(schema.empty_table(), tmp_path / "empty.lance")

    assert size_compaction(dataset, 1024) == CompactionSizing(
        target_rows=None, oversized=[]
    )


class _FakeDataFile:
    def __init__(self, size):
        self.file_size_bytes = size


class _FakeFragment:
    def __init__(self, rows, sizes, fragment_id=0):
        self.fragment_id = fragment_id
        self.physical_rows = rows
        self._files = [_FakeDataFile(s) for s in sizes]

    def data_files(self):
        return self._files


class _FakeDataset:
    """Stands in for a manifest lance 10 will not produce.

    Every file lance writes today records its size, so a mixed manifest can only
    come from a database written by a much older version.
    """

    def __init__(self, fragments):
        self._fragments = fragments

    def get_fragments(self):
        return self._fragments


def test_one_unmeasurable_fragment_disables_the_whole_table():
    """Sizing from the measurable fragments would still hand the other one over
    to compaction whole."""
    dataset = _FakeDataset(
        [
            _FakeFragment(rows=10, sizes=[1 * MB]),  # measurable and narrow
            _FakeFragment(rows=10, sizes=[None]),  # size missing
        ]
    )

    assert size_compaction(cast("lance.LanceDataset", dataset), 8 * MB) is None


def test_empty_fragments_do_not_block_sizing():
    dataset = _FakeDataset(
        [
            _FakeFragment(rows=0, sizes=[]),
            _FakeFragment(rows=10, sizes=[10 * MB]),
        ]
    )

    assert _target(cast("lance.LanceDataset", dataset), 8 * MB) == 8


def test_a_row_of_half_the_target_still_sets_it():
    """At exactly half the target two rows fit, so the target is two."""
    dataset = _FakeDataset([_FakeFragment(rows=1, sizes=[4 * MB], fragment_id=7)])

    sizing = size_compaction(cast("lance.LanceDataset", dataset), 8 * MB)

    assert sizing == CompactionSizing(target_rows=2, oversized=[])


def test_a_row_wider_than_half_the_target_is_oversized():
    dataset = _FakeDataset(
        [
            _FakeFragment(rows=8, sizes=[1 * MB], fragment_id=1),
            _FakeFragment(rows=1, sizes=[4 * MB + 1], fragment_id=2),
        ]
    )

    sizing = size_compaction(cast("lance.LanceDataset", dataset), 8 * MB)

    # the oversized row does not pull the target down to one
    assert sizing == CompactionSizing(target_rows=64, oversized=[2])


@pytest.mark.parametrize(
    ("stored", "computed", "kept"),
    [
        (None, 40, 40),  # nothing stored: the computed target
        (40, 4, 4),  # a lower target replaces the stored one
        (4, 12, 4),  # a higher one never does
        (4, None, 4),  # every fragment oversized: the stored target stays
        (None, None, None),
    ],
)
def test_the_kept_target_only_falls(stored, computed, kept):
    assert kept_target(stored, computed) == kept


def test_budget_must_be_positive():
    with pytest.raises(ValidationError):
        StorageConfig(compaction_target_bytes=0)


async def test_vacuum_leaves_the_handle_usable(temp_db_path):
    """Reads and writes must work through the same handle after a vacuum.

    The bounded path mutates the dataset behind the open AsyncTable, so a
    missing checkout_latest leaves the handle on a stale version.
    """
    config = AppConfig()
    config.storage.auto_vacuum = False

    async with Store(temp_db_path, config=config, create=True) as store:
        await store.documents_table.add(
            [{"id": "a", "content": "x", "docling_document": b"0" * MB}]
        )
        await store.vacuum(retention_seconds=0)

        assert await store.documents_table.count_rows() == 1
        await store.documents_table.add(
            [{"id": "b", "content": "y", "docling_document": b"1" * MB}]
        )
        assert await store.documents_table.count_rows() == 2


async def test_handle_is_refreshed_when_a_later_step_fails(temp_db_path, monkeypatch):
    """A committed compaction followed by a failure must not strand the handle."""
    config = AppConfig()
    config.storage.auto_vacuum = False

    async with Store(temp_db_path, config=config, create=True) as store:
        await store.documents_table.add(
            [{"id": "a", "content": "x", "docling_document": b"0" * MB}]
        )
        await store.documents_table.add(
            [{"id": "b", "content": "y", "docling_document": b"1" * MB}]
        )

        def explode(dataset):
            raise RuntimeError("index optimize failed")

        real = Store._run_lance_maintenance
        calls = {"n": 0}

        async def fail_on_second_step(self, table, step):
            calls["n"] += 1
            # 1 store the target, 2 compact, 3 optimize_indices, 4 prune --
            # fail after the compaction commit
            return await real(self, table, explode if calls["n"] == 3 else step)

        monkeypatch.setattr(Store, "_run_lance_maintenance", fail_on_second_step)
        with pytest.raises(RuntimeError, match="index optimize failed"):
            await store.vacuum(retention_seconds=0)

        # the compaction committed before the failure
        assert len((await store.documents_table.to_lance()).get_fragments()) == 1
        # the handle must still serve reads and writes despite the failure
        assert await store.documents_table.count_rows() == 2
        await store.documents_table.add(
            [{"id": "c", "content": "z", "docling_document": b"2" * MB}]
        )
        assert await store.documents_table.count_rows() == 3


# 1 store the target, 2 compact, 3 optimize_indices, 4 prune -- all four on
# the first payload table
@pytest.mark.parametrize("cancel_at", [1, 2, 3, 4])
async def test_cancelling_vacuum_waits_for_the_running_step(
    temp_db_path, cancel_at, monkeypatch
):
    """A worker thread cannot be cancelled, so the step must be awaited.

    Releasing the write lock while lance is still mutating the dataset would let
    the next write race a commit that is still in flight, so a cancellation has
    to wait for the thread rather than abandon it.
    """
    config = AppConfig()
    config.storage.auto_vacuum = False

    async with Store(temp_db_path, config=config, create=True) as store:
        await store.documents_table.add(
            [{"id": "a", "content": "x", "docling_document": b"0" * MB}]
        )
        in_thread = threading.Event()
        release = threading.Event()
        completed: list[int] = []
        real = Store._run_lance_maintenance
        calls = {"n": 0}

        def blocking(dataset):
            in_thread.set()
            release.wait(30)
            completed.append(1)

        async def block_on_nth(self, table, step):
            calls["n"] += 1
            return await real(
                self, table, blocking if calls["n"] == cancel_at else step
            )

        monkeypatch.setattr(Store, "_run_lance_maintenance", block_on_nth)
        try:
            task = asyncio.create_task(store.vacuum(retention_seconds=0))
            while not in_thread.is_set():
                await asyncio.sleep(0.01)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()

        # the step ran to completion despite the cancellation
        assert completed == [1]
        assert not store._write_lock.locked()
        assert await store.documents_table.count_rows() == 1


async def test_only_payload_tables_take_the_sized_path(temp_db_path, monkeypatch):
    """Only payload-bearing tables take the sized path, and none uses optimize()."""
    config = AppConfig()
    config.storage.auto_vacuum = False
    optimized: list[str] = []
    compacted: list[str] = []

    async with Store(temp_db_path, config=config, create=True) as store:
        real_optimize = type(store.chunks_table).optimize

        async def record_optimize(self, **kwargs):
            optimized.append(self.name)
            return await real_optimize(self, **kwargs)

        async def record_compact(self, table):
            compacted.append(table.name)

        monkeypatch.setattr(type(store.chunks_table), "optimize", record_optimize)
        monkeypatch.setattr(Store, "_compact_to_target", record_compact)
        await store.vacuum(retention_seconds=0)

    assert sorted(compacted) == ["document_items", "documents"]
    assert optimized == []


async def test_unmeasurable_table_still_prunes(temp_db_path, monkeypatch, caplog):
    config = AppConfig()
    config.storage.auto_vacuum = False
    steps: list[str] = []

    async with Store(temp_db_path, config=config, create=True) as store:
        await store.documents_table.add(
            [{"id": "a", "content": "x", "docling_document": b"0" * MB}]
        )
        monkeypatch.setattr("haiku.rag.store.engine.size_compaction", lambda *_: None)
        real = Store._run_lance_maintenance

        async def record(self, table, step):
            steps.append(table.name)
            return await real(self, table, step)

        monkeypatch.setattr(Store, "_run_lance_maintenance", record)
        with caplog.at_level("WARNING"):
            await store.vacuum(retention_seconds=0)

    # compaction skipped, pruning still ran, and the skip is visible
    assert steps.count("documents") == 1
    assert steps.count("document_items") == 1
    assert "sizes are absent" in caplog.text


KB = 1024


def _doc(i, size):
    return {"id": f"d{i}", "content": "x", "docling_document": os.urandom(size)}


async def _add_each(store, docs):
    """One write per document, as the ingester writes."""
    for doc in docs:
        await store.documents_table.add([doc])


async def _fragments(store):
    dataset = await store.documents_table.to_lance()
    return [(f.fragment_id, f.physical_rows) for f in dataset.get_fragments()]


async def _stored_target(store):
    dataset = await store.documents_table.to_lance()
    return int(dataset.config()[COMPACTION_TARGET_KEY])


def _small_target_config():
    config = AppConfig()
    config.storage.auto_vacuum = False
    config.storage.compaction_target_bytes = 8 * MB
    return config


async def test_a_rising_target_does_not_rewrite_compacted_fragments(temp_db_path):
    """Merging a wide row with narrow ones lowers the widest fragment's bytes
    per row, so the computed target rises after it fell. Fragments written at
    the fallen target must survive that rise."""
    async with Store(temp_db_path, config=_small_target_config(), create=True) as store:
        await _add_each(store, [_doc(i, 200 * KB) for i in range(20)])
        await store.vacuum(retention_seconds=0)
        first = await _stored_target(store)
        [(settled, _)] = await _fragments(store)

        wide = 21
        await _add_each(
            store, [_doc(i, 2 * MB if i == wide else 200 * KB) for i in range(20, 28)]
        )
        await store.vacuum(retention_seconds=0)
        fallen = await _stored_target(store)
        assert fallen < first
        merged = {fid for fid, _ in await _fragments(store)} - {settled}
        sizing = size_compaction(await store.documents_table.to_lance(), 8 * MB)
        assert sizing is not None and sizing.target_rows is not None
        assert sizing.target_rows > fallen

        await _add_each(store, [_doc(28, 200 * KB)])
        await store.vacuum(retention_seconds=0)

        assert merged <= {fid for fid, _ in await _fragments(store)}
        assert await _stored_target(store) == fallen


async def test_oversized_documents_are_never_merged(temp_db_path):
    """Consecutive rows wider than half the target each keep their fragment,
    and the narrow rows on either side merge without them."""
    async with Store(temp_db_path, config=_small_target_config(), create=True) as store:
        await _add_each(store, [_doc(i, 200 * KB) for i in range(5)])
        await _add_each(store, [_doc(i, 5 * MB) for i in range(5, 8)])
        oversized = {fid for fid, _ in await _fragments(store)}
        await _add_each(store, [_doc(i, 200 * KB) for i in range(8, 13)])
        oversized -= {fid for fid, _ in (await _fragments(store))[:5]}

        await store.vacuum(retention_seconds=0)

        after = await _fragments(store)
        assert oversized <= {fid for fid, _ in after}
        # the two narrow runs merged on their own, not across the wide rows
        assert sorted(rows for _, rows in after) == [1, 1, 1, 5, 5]


async def test_deletions_in_an_oversized_fragment_are_compacted_alone(temp_db_path):
    async with Store(temp_db_path, config=_small_target_config(), create=True) as store:
        await _add_each(store, [_doc(i, 200 * KB) for i in range(5)])
        await store.documents_table.add([_doc(i, 5 * MB) for i in range(5, 8)])
        await _add_each(store, [_doc(i, 200 * KB) for i in range(8, 13)])
        # compaction appends what it writes, so the second pass finds the two
        # narrow runs adjacent and merges them
        await store.vacuum(retention_seconds=0)
        await store.vacuum(retention_seconds=0)
        before = dict(await _fragments(store))
        assert sorted(before.values()) == [3, 10]
        [wide] = [fid for fid, rows in before.items() if rows == 3]

        await store.documents_table.delete("id = 'd6'")
        await store.vacuum(retention_seconds=0)

        after = dict(await _fragments(store))
        assert sorted(after.values()) == [2, 10]
        # the oversized fragment was rewritten, its neighbours were not
        narrow = {fid for fid in before if fid != wide}
        assert narrow <= set(after)
        assert wide not in after
        assert await store.documents_table.count_rows() == 12


async def test_a_table_of_oversized_rows_still_optimizes_indices(
    temp_db_path, monkeypatch
):
    optimized: list[str] = []
    real = engine._optimize_indices

    def record(dataset):
        optimized.append(Path(dataset.uri).name)
        real(dataset)

    monkeypatch.setattr(engine, "_optimize_indices", record)
    async with Store(temp_db_path, config=_small_target_config(), create=True) as store:
        await _add_each(store, [_doc(i, 5 * MB) for i in range(3)])
        with capture_logs(engine.logger, logging.WARNING) as records:
            await store.vacuum(retention_seconds=0)

        assert [rows for _, rows in await _fragments(store)] == [1, 1, 1]
    assert "documents.lance" in optimized
    assert records == []


async def test_restoring_a_tag_restores_its_target(temp_db_path):
    async with Store(temp_db_path, config=_small_target_config(), create=True) as store:
        await _add_each(store, [_doc(i, 200 * KB) for i in range(10)])
        await store.vacuum(retention_seconds=0)
        tagged = await _stored_target(store)
        await store.create_tag("before-wide")

        await _add_each(store, [_doc(10, 2 * MB)])
        await store.vacuum(retention_seconds=0)
        assert await _stored_target(store) < tagged

        await store.restore_tag("before-wide")
        await store.vacuum(retention_seconds=0)

        assert await _stored_target(store) == tagged
        assert await store.documents_table.count_rows() == 10


# The failure is peak RSS while rewriting, and a bounded and an unbounded run
# produce identical files, so only a memory measurement can tell them apart.
_RSS_PROBE = textwrap.dedent(
    """
    import asyncio, os, sys, threading, time
    from pathlib import Path
    import psutil

    db, budget, repo = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    sys.path.insert(0, repo)
    from haiku.rag.config.models import AppConfig
    from haiku.rag.store.engine import Store

    proc = psutil.Process()
    peak = {"v": 0}

    def watch():
        while True:
            peak["v"] = max(peak["v"], proc.memory_info().rss)
            time.sleep(0.01)

    threading.Thread(target=watch, daemon=True).start()

    async def main():
        config = AppConfig()
        config.storage.auto_vacuum = False
        if budget:
            config.storage.compaction_target_bytes = budget
        peaks = []
        async with Store(Path(db), config=config, create=True) as store:
            row = 0
            for round_no in range(4):
                await store.documents_table.add([
                    {"id": f"d{row + i}", "content": "x",
                     "docling_document": os.urandom(4 * 1024 * 1024)}
                    for i in range(8)])
                row += 8
                if round_no:
                    # churn: rewrite an old row so deletions accumulate
                    await store.documents_table.delete(f"id = 'd{round_no}'")
                base = peak["v"] = proc.memory_info().rss
                await store.vacuum(retention_seconds=0)
                peaks.append((peak["v"] - base) / (1024 * 1024))
        print(",".join(f"{p:.1f}" for p in peaks))

    asyncio.run(main())
    """
)


def _probe(tmp_path, name, budget):
    """Peak RSS in MB per append+churn+vacuum round, in a fresh process."""
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            _RSS_PROBE,
            str(tmp_path / name),
            str(budget),
            os.getcwd(),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [float(p) for p in out.stdout.strip().splitlines()[-1].split(",")]


@pytest.mark.integration
def test_sized_vacuum_does_not_grow_with_the_table(tmp_path):
    """Drives Store.vacuum end to end, with deletions, not lance directly."""
    unbounded = _probe(tmp_path, "unbounded.lancedb", 8 * 1024**3)
    bounded = _probe(tmp_path, "bounded.lancedb", 24 * MB)

    # the unsized pass re-reads everything written so far, so its cost rises
    assert unbounded[-1] > unbounded[1]
    assert max(bounded) < max(unbounded)
    # and the sized one does not trend upward as the table grows
    assert max(bounded[2:]) <= max(bounded[1], 1.0) * 1.5


@pytest.mark.integration
def test_a_fragment_larger_than_the_target_is_rewritten_whole(tmp_path):
    """The target sizes what compaction writes, not what it may read.

    A fragment written larger than the target -- by a big ingest batch, or by an
    older unsized vacuum -- is rewritten in one piece the first time deletions
    make it a candidate, costing roughly its own size regardless of the target.
    Recorded because it is the ceiling the config knob cannot lower.
    """
    path = tmp_path / "batched.lance"
    _write(path, 40, 2 * MB, per_fragment=40)  # one 80 MB fragment
    dataset = lance.dataset(path)
    target = _target(dataset, 8 * MB)
    assert target is not None and target <= 4

    # deletions must pass lance's 10% threshold for the fragment to be a candidate
    dataset.delete("id in ('d0','d1','d2','d3','d4','d5')")
    dataset = lance.dataset(path)
    metrics = dataset.optimize.compact_files(target_rows_per_fragment=target)

    # the oversized fragment was read whole and split into target-sized pieces
    assert metrics.fragments_removed == 1
    assert metrics.fragments_added > 1
