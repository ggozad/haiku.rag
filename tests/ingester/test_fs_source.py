import hashlib
from pathlib import Path

import pytest

from haiku.rag.ingester.sources.base import SourceEventKind
from haiku.rag.ingester.sources.fs import FSSource


@pytest.fixture
def fs_root(tmp_path: Path) -> Path:
    (tmp_path / "a.md").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.md").write_text("gamma")
    (tmp_path / "skip.log").write_text("noise")
    return tmp_path


def test_fs_source_supports_file_uri(fs_root: Path):
    src = FSSource(root=fs_root)
    assert src.supports((fs_root / "a.md").as_uri())
    assert src.supports(str(fs_root / "a.md"))


def test_fs_source_rejects_other_schemes(fs_root: Path):
    src = FSSource(root=fs_root)
    assert not src.supports("http://example.com/a.md")
    assert not src.supports("s3://bucket/a.md")


def test_fs_source_source_id_is_canonical(fs_root: Path):
    src = FSSource(root=fs_root)
    assert src.source_id == f"fs:{fs_root.resolve()}"


@pytest.mark.asyncio
async def test_fs_source_fetch_returns_bytes_and_md5(fs_root: Path):
    src = FSSource(root=fs_root)
    target = fs_root / "a.md"
    result = await src.fetch(target.as_uri())
    assert result.uri == target.as_uri()
    assert result.body == b"alpha"
    assert (
        result.content_hash == hashlib.md5(b"alpha", usedforsecurity=False).hexdigest()
    )
    assert result.content_type == "text/markdown"
    assert result.revision == str(target.stat().st_mtime_ns)


@pytest.mark.asyncio
async def test_fs_source_fetch_accepts_bare_path(fs_root: Path):
    src = FSSource(root=fs_root)
    target = fs_root / "a.md"
    result = await src.fetch(str(target))
    assert result.uri == target.as_uri()


@pytest.mark.asyncio
async def test_fs_source_fetch_missing_file_raises(fs_root: Path):
    src = FSSource(root=fs_root)
    with pytest.raises(FileNotFoundError):
        await src.fetch((fs_root / "missing.md").as_uri())


@pytest.mark.asyncio
async def test_fs_source_discover_initial_scan_yields_upsert(fs_root: Path):
    src = FSSource(root=fs_root, supported_extensions=[".md", ".txt"])
    events = [e async for e in src.discover(since=None)]
    uris = {e.uri for e in events}
    assert uris == {
        (fs_root / "a.md").as_uri(),
        (fs_root / "b.txt").as_uri(),
        (fs_root / "sub" / "c.md").as_uri(),
    }
    assert all(e.kind is SourceEventKind.UPSERT for e in events)
    assert all(e.source_id == src.source_id for e in events)
    assert all(e.revision is not None for e in events)


@pytest.mark.asyncio
async def test_fs_source_discover_unchanged_against_snapshot(fs_root: Path):
    src = FSSource(root=fs_root, supported_extensions=[".md", ".txt"])
    initial = {e.uri: e.revision or "" async for e in src.discover(since=None)}
    again = [e async for e in src.discover(since=initial)]
    assert again
    assert all(e.kind is SourceEventKind.UNCHANGED for e in again)


@pytest.mark.asyncio
async def test_fs_source_discover_changed_yields_upsert(fs_root: Path):
    src = FSSource(root=fs_root, supported_extensions=[".md", ".txt"])
    initial = {e.uri: e.revision or "" async for e in src.discover(since=None)}
    stale = {uri: "0" for uri in initial}
    events = [e async for e in src.discover(since=stale)]
    assert {e.kind for e in events} == {SourceEventKind.UPSERT}


@pytest.mark.asyncio
async def test_fs_source_discover_emits_delete_for_missing(fs_root: Path):
    src = FSSource(root=fs_root, supported_extensions=[".md", ".txt"])
    snapshot = {(fs_root / "ghost.md").as_uri(): "999"}
    events = [e async for e in src.discover(since=snapshot)]
    deletes = [e for e in events if e.kind is SourceEventKind.DELETE]
    assert len(deletes) == 1
    assert deletes[0].uri == (fs_root / "ghost.md").as_uri()
    assert deletes[0].revision is None


@pytest.mark.asyncio
async def test_fs_source_discover_respects_extension_filter(fs_root: Path):
    src = FSSource(root=fs_root, supported_extensions=[".md"])
    uris = {e.uri async for e in src.discover(since=None)}
    assert (fs_root / "a.md").as_uri() in uris
    assert (fs_root / "b.txt").as_uri() not in uris


@pytest.mark.asyncio
async def test_fs_source_discover_respects_ignore_patterns(fs_root: Path):
    src = FSSource(
        root=fs_root,
        supported_extensions=[".md", ".txt"],
        ignore_patterns=["**/sub/**"],
    )
    uris = {e.uri async for e in src.discover(since=None)}
    assert (fs_root / "sub" / "c.md").as_uri() not in uris
    assert (fs_root / "a.md").as_uri() in uris


@pytest.mark.asyncio
async def test_fs_source_discover_respects_include_patterns(fs_root: Path):
    src = FSSource(
        root=fs_root,
        supported_extensions=[".md", ".txt"],
        include_patterns=["**/*.md"],
    )
    uris = {e.uri async for e in src.discover(since=None)}
    assert (fs_root / "b.txt").as_uri() not in uris
    assert (fs_root / "a.md").as_uri() in uris


def test_filefilter_backward_compatible_reexport():
    from haiku.rag.ingester.sources.filter import FileFilter as IngesterFileFilter
    from haiku.rag.monitor import FileFilter as MonitorFileFilter

    assert MonitorFileFilter is IngesterFileFilter
