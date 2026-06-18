import hashlib
from pathlib import Path

import pytest

from haiku.rag.client.exceptions import UnsupportedSourceError
from haiku.rag.ingester.sources.base import FileTooLargeError, SourceEventKind
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
    assert result.disk_path == target


@pytest.mark.asyncio
async def test_fs_source_head_returns_mtime(fs_root: Path):
    src = FSSource(root=fs_root)
    target = fs_root / "a.md"
    assert await src.head(target.as_uri()) == str(target.stat().st_mtime_ns)


@pytest.mark.asyncio
async def test_fs_source_head_returns_none_for_missing_file(fs_root: Path):
    src = FSSource(root=fs_root)
    assert await src.head((fs_root / "missing.md").as_uri()) is None


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
    known = {(fs_root / "ghost.md").as_uri()}
    events = [e async for e in src.discover(known_uris=known)]
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
async def test_fs_source_discover_skips_file_deleted_during_stat(
    fs_root: Path, monkeypatch
):
    """A file deleted between is_file() and stat() should be silently
    skipped instead of crashing the entire discover() sweep."""
    victim = fs_root / "b.txt"
    original_stat = Path.stat
    victim_calls = 0

    def _stat_that_fails_on_second_call(self, *args, **kwargs):
        nonlocal victim_calls
        if self == victim:
            victim_calls += 1
            # First calls are from is_symlink/is_file; the later call
            # is the explicit stat().st_mtime_ns we want to fail.
            if victim_calls > 2:
                raise FileNotFoundError(f"[Errno 2] No such file: '{self}'")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _stat_that_fails_on_second_call)

    src = FSSource(root=fs_root, supported_extensions=[".md", ".txt"])
    events = [e async for e in src.discover(since=None)]
    uris = {e.uri for e in events}
    # b.txt was skipped due to the simulated race; a.md and sub/c.md are fine.
    assert (fs_root / "a.md").as_uri() in uris
    assert (fs_root / "sub" / "c.md").as_uri() in uris
    assert victim.as_uri() not in uris


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


# --- symlink escape defenses ---


def test_fs_source_supports_rejects_paths_outside_root(fs_root: Path, tmp_path: Path):
    """A URI for a file outside the configured root must not match — even
    if it's a valid file:// URI. Otherwise the resolve_fetcher chain could
    end up handing /etc/passwd to FSSource.fetch()."""
    outside = tmp_path.parent / "outside.md"
    outside.write_text("not yours")
    src = FSSource(root=fs_root)
    assert src.supports(outside.as_uri()) is False


@pytest.mark.asyncio
async def test_fs_source_fetch_rejects_paths_outside_root(
    fs_root: Path, tmp_path: Path
):
    outside = tmp_path.parent / "outside.md"
    outside.write_text("not yours")
    src = FSSource(root=fs_root)
    with pytest.raises(UnsupportedSourceError, match="escapes FS root"):
        await src.fetch(outside.as_uri())


@pytest.mark.asyncio
async def test_fs_source_fetch_rejects_symlink_to_outside_file(
    fs_root: Path, tmp_path: Path
):
    """Symlink under root that points outside — the classic FS escape.
    resolve() chases the link to the real path, which fails the root check."""
    secret = tmp_path.parent / "secret.md"
    secret.write_text("sensitive")
    link = fs_root / "looks_local.md"
    link.symlink_to(secret)
    src = FSSource(root=fs_root)
    with pytest.raises(UnsupportedSourceError, match="escapes FS root"):
        await src.fetch(link.as_uri())


@pytest.mark.asyncio
async def test_fs_source_discover_skips_symlinks_pointing_outside_root(
    fs_root: Path, tmp_path: Path
):
    """A symlink under root whose target resolves outside root must not be
    discovered — otherwise a stray link could exfiltrate files the operator
    didn't intend to expose. Mirrors the resolve-then-check guard that
    supports/head/fetch use."""
    secret = tmp_path.parent / "secret.md"
    secret.write_text("sensitive")
    link = fs_root / "evil.md"
    link.symlink_to(secret)
    src = FSSource(root=fs_root, supported_extensions=[".md"])
    uris = {e.uri async for e in src.discover(since=None)}
    assert link.as_uri() not in uris
    assert secret.as_uri() not in uris
    # And the legitimate files in fs_root still come through.
    assert (fs_root / "a.md").as_uri() in uris


@pytest.mark.asyncio
async def test_fs_source_discover_follows_within_root_symlinks(fs_root: Path):
    """A symlink whose target lives inside root is legitimate — supports/
    head/fetch all accept it (resolve-then-check), so discover() must too,
    otherwise an ad-hoc add-src on a link works but the poller never picks
    it up. We emit the resolved target's URI, never the alias's; redundant
    yields from walking both alias and target are absorbed by the queue's
    unique index on (source_id, uri, op)."""
    target = fs_root / "real.md"
    target.write_text("real content")
    (fs_root / "alias.md").symlink_to(target)
    src = FSSource(root=fs_root, supported_extensions=[".md"])
    events = [e async for e in src.discover(since=None)]
    uris = [e.uri for e in events]
    assert target.as_uri() in uris
    # The alias's own URI is not emitted — we normalise to the target.
    assert (fs_root / "alias.md").as_uri() not in uris


@pytest.mark.asyncio
async def test_fs_source_discover_skips_symlinked_directories(
    fs_root: Path, tmp_path: Path
):
    """os.walk(followlinks=False) must NOT descend into directory symlinks —
    otherwise a `ln -s /etc /docs/escape` would walk into /etc and try to
    yield its contents."""
    outside_dir = tmp_path.parent / "outside_dir"
    outside_dir.mkdir()
    (outside_dir / "stolen.md").write_text("not yours")
    (fs_root / "escape").symlink_to(outside_dir)
    src = FSSource(root=fs_root, supported_extensions=[".md"])
    uris = {e.uri async for e in src.discover(since=None)}
    # No URI under /escape/* should appear.
    assert not any("escape" in u for u in uris)


@pytest.mark.asyncio
async def test_fs_source_fetch_rejects_file_exceeding_max_size(fs_root: Path):
    src = FSSource(root=fs_root, max_file_size=3)
    with pytest.raises(FileTooLargeError):
        await src.fetch((fs_root / "a.md").as_uri())  # "alpha" = 5 bytes


@pytest.mark.asyncio
async def test_fs_source_fetch_allows_file_within_max_size(fs_root: Path):
    src = FSSource(root=fs_root, max_file_size=100)
    result = await src.fetch((fs_root / "a.md").as_uri())
    assert result.body == b"alpha"


@pytest.mark.asyncio
async def test_fs_source_fetch_no_limit_when_max_size_is_none(fs_root: Path):
    src = FSSource(root=fs_root, max_file_size=None)
    result = await src.fetch((fs_root / "a.md").as_uri())
    assert result.body == b"alpha"


@pytest.mark.asyncio
async def test_fs_source_fetch_reads_off_event_loop_thread(fs_root: Path):
    """The file read and md5 are both proportional to file size and must run
    off the event-loop thread, or a large file would freeze every other
    worker's coroutine for the duration of the read. Capture the thread the
    read+hash runs on and assert it is not the main thread."""
    import threading

    src = FSSource(root=fs_root)
    target = fs_root / "a.md"

    called_from: list[threading.Thread] = []
    original = src._read_body

    def spy(path, uri):
        called_from.append(threading.current_thread())
        return original(path, uri)

    src._read_body = spy  # type: ignore[method-assign]

    result = await src.fetch(target.as_uri())
    assert result.body == b"alpha"
    assert called_from, "_read_body was never called"
    assert called_from[0] is not threading.main_thread(), (
        "FSSource._read_body ran on the event-loop thread; the read+hash must "
        "be dispatched via asyncio.to_thread"
    )
