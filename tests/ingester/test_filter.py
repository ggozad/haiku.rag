from watchfiles import Change

from haiku.rag.ingester.sources.filter import FileFilter, _default_supported_extensions


def test_default_supported_extensions_returns_nonempty_list():
    exts = _default_supported_extensions()
    assert isinstance(exts, list)
    assert len(exts) > 0
    assert all(ext.startswith(".") for ext in exts)


def test_filter_uses_default_extensions_when_none():
    f = FileFilter(supported_extensions=None)
    assert len(f.extensions) > 0


def test_call_delegates_to_include_file_then_default_filter():
    """__call__ is the watchfiles callback entry point. It should reject
    files that don't pass include_file and accept ones that do."""
    f = FileFilter(supported_extensions=[".md"])
    # DefaultFilter rejects dotfiles and common noise; a normal .md path passes.
    assert f(Change.added, "/tmp/docs/readme.md") is True
    # Wrong extension — include_file returns False before DefaultFilter runs.
    assert f(Change.added, "/tmp/docs/readme.log") is False
