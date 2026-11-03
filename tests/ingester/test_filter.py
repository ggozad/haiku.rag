import re
from collections import Counter
from pathlib import Path

import pytest
from watchfiles import Change

from haiku.rag.converters.docling_local import DoclingLocalConverter
from haiku.rag.converters.docling_serve import DoclingServeConverter
from haiku.rag.converters.text_utils import TextFileHandler
from haiku.rag.sources.filter import FileFilter, _default_supported_extensions


def test_default_supported_extensions_returns_nonempty_list():
    exts = _default_supported_extensions()
    assert isinstance(exts, list)
    assert len(exts) > 0
    assert all(ext.startswith(".") for ext in exts)


def test_filter_uses_default_extensions_when_none():
    f = FileFilter(supported_extensions=None)
    assert len(f.extensions) > 0


@pytest.mark.parametrize("path", ["Report.PDF", "notes.TXT", "scan.JPG", "report.PdF"])
def test_extensions_ignore_case(path: str):
    f = FileFilter()
    assert f.include_file(path) is True
    assert f(Change.added, path) is True


def test_patterns_preserve_path_case():
    f = FileFilter(
        supported_extensions=[".pdf"],
        include_patterns=["Reports/*"],
        ignore_patterns=["Reports/Draft*"],
    )
    assert f.include_file("Reports/Final.PDF") is True
    assert f.include_file("reports/Final.PDF") is False
    assert f.include_file("Reports/Draft.PDF") is False
    assert f.include_file("Reports/draft.PDF") is True
    assert f.include_file("Reports/Final.EXE") is False


def test_call_delegates_to_include_file_then_default_filter():
    """__call__ is the watchfiles callback entry point. It should reject
    files that don't pass include_file and accept ones that do."""
    f = FileFilter(supported_extensions=[".md"])
    # DefaultFilter rejects dotfiles and common noise; a normal .md path passes.
    assert f(Change.added, "/tmp/docs/readme.md") is True
    # Wrong extension — include_file returns False before DefaultFilter runs.
    assert f(Change.added, "/tmp/docs/readme.log") is False


def test_unsupported_extensions_are_counted_by_lowercased_suffix():
    f = FileFilter(supported_extensions=[".md"])
    unsupported: Counter[str] = Counter()
    for path in ["docs/a.rst", "docs/B.RST", "docs/c.md", "Makefile", "x.tar.gz"]:
        f.include_file(path, unsupported)
    assert unsupported == {".rst": 2, "": 1, ".gz": 1}


def test_files_excluded_by_patterns_are_not_counted():
    f = FileFilter(
        supported_extensions=[".md"],
        include_patterns=["docs/**"],
        ignore_patterns=["docs/drafts/**"],
    )
    unsupported: Counter[str] = Counter()
    for path in ["other/a.rst", "docs/drafts/b.rst", "docs/c.rst"]:
        assert f.include_file(path, unsupported) is False
    assert unsupported == {".rst": 1}


def _documented_extensions(label: str) -> list[str]:
    docs = Path(__file__).resolve().parents[2] / "docs" / "ingester.md"
    [line] = [
        line
        for line in docs.read_text(encoding="utf-8").splitlines()
        if line.startswith(f"- {label}: ")
    ]
    return re.findall(r"`([^`]+)`", line)


def test_documented_extensions_match_the_converters():
    docling = _documented_extensions("Converted by docling, with either converter")
    assert docling == DoclingLocalConverter.docling_extensions
    assert docling == DoclingServeConverter.docling_serve_extensions
    assert _documented_extensions("Read as text") == TextFileHandler.text_extensions
