import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, MonitorConfig
from haiku.rag.monitor import FileWatcher
from haiku.rag.store.models.document import Document


@pytest.mark.asyncio
async def test_file_watcher_upsert_document():
    """Test FileWatcher._upsert_document method."""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test.txt"
        temp_path.write_text("Test content for file watcher")

        mock_client = AsyncMock(spec=HaikuRAG)
        mock_doc = Document(id="1", content="Test content", uri=temp_path.as_uri())
        mock_client.create_document_from_source.return_value = mock_doc
        mock_client.get_document_by_uri.return_value = None  # No existing document

        test_config = AppConfig(monitor=MonitorConfig(directories=[temp_path.parent]))
        watcher = FileWatcher(client=mock_client, config=test_config)

        result = await watcher._upsert_document(temp_path)

        assert result is not None
        assert result.id == "1"
        mock_client.get_document_by_uri.assert_called_once_with(temp_path.as_uri())
        mock_client.create_document_from_source.assert_called_once_with(str(temp_path))


@pytest.mark.asyncio
async def test_file_watcher_upsert_existing_document():
    """Test FileWatcher._upsert_document with existing document."""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test.txt"
        temp_path.write_text("Test content for file watcher")

        mock_client = AsyncMock(spec=HaikuRAG)
        existing_doc = Document(id="1", content="Old content", uri=temp_path.as_uri())
        updated_doc = Document(
            id="1", content="Updated content", uri=temp_path.as_uri()
        )

        mock_client.get_document_by_uri.return_value = existing_doc
        mock_client.create_document_from_source.return_value = updated_doc

        test_config = AppConfig(monitor=MonitorConfig(directories=[temp_path.parent]))
        watcher = FileWatcher(client=mock_client, config=test_config)

        result = await watcher._upsert_document(temp_path)

        assert result is not None
        assert result.content == "Updated content"
        mock_client.get_document_by_uri.assert_called_once_with(temp_path.as_uri())
        mock_client.create_document_from_source.assert_called_once_with(str(temp_path))


@pytest.mark.asyncio
async def test_file_watcher_delete_document():
    """Test FileWatcher._delete_document method."""
    temp_path = Path("/tmp/test_file.txt")

    mock_client = AsyncMock(spec=HaikuRAG)
    existing_doc = Document(id="1", content="Content to delete", uri=temp_path.as_uri())
    mock_client.get_document_by_uri.return_value = existing_doc
    mock_client.delete_document.return_value = True

    test_config = AppConfig(monitor=MonitorConfig(directories=[temp_path.parent]))
    watcher = FileWatcher(client=mock_client, config=test_config)

    await watcher._delete_document(temp_path)

    mock_client.get_document_by_uri.assert_called_once_with(temp_path.as_uri())
    mock_client.delete_document.assert_called_once_with("1")


@pytest.mark.asyncio
async def test_file_watcher_delete_nonexistent_document():
    """Test FileWatcher._delete_document with non-existent document."""
    temp_path = Path("/tmp/nonexistent_file.txt")

    mock_client = AsyncMock(spec=HaikuRAG)
    mock_client.get_document_by_uri.return_value = None

    test_config = AppConfig(monitor=MonitorConfig(directories=[temp_path.parent]))
    watcher = FileWatcher(client=mock_client, config=test_config)

    await watcher._delete_document(temp_path)

    mock_client.get_document_by_uri.assert_called_once_with(temp_path.as_uri())
    mock_client.delete_document.assert_not_called()


@pytest.mark.asyncio
async def test_file_filter_ignore_patterns():
    """Test FileFilter with ignore patterns."""
    from watchfiles import Change

    from haiku.rag.monitor import FileFilter

    filter = FileFilter(ignore_patterns=["*draft*.md", "temp/", "**/archive/**"])

    # Should ignore draft markdown files
    assert not filter(Change.added, "/path/to/draft-post.md")

    # Should ignore files in temp/ directory
    assert not filter(Change.added, "/path/temp/notes.txt")

    # Should ignore files in archive directories
    assert not filter(Change.added, "/path/to/archive/old.pdf")

    # Should NOT ignore regular markdown files
    assert filter(Change.added, "/path/to/readme.md")

    # Should NOT ignore files outside temp/
    assert filter(Change.added, "/path/to/notes.txt")


@pytest.mark.asyncio
async def test_file_filter_include_patterns():
    """Test FileFilter with include patterns (whitelist mode)."""
    from watchfiles import Change

    from haiku.rag.monitor import FileFilter

    filter = FileFilter(include_patterns=["*.md", "**/docs/**"])

    # Should include .md files
    assert filter(Change.added, "/path/to/file.md")

    # Should include files in docs/ directory
    assert filter(Change.added, "/path/to/docs/guide.txt")

    # Should NOT include .txt files outside docs/
    assert not filter(Change.added, "/path/to/file.txt")


@pytest.mark.asyncio
async def test_file_filter_combined_patterns():
    """Test FileFilter with both include and ignore patterns."""
    from watchfiles import Change

    from haiku.rag.monitor import FileFilter

    # Include all markdown files, but ignore drafts
    filter = FileFilter(
        include_patterns=["*.md"], ignore_patterns=["*draft*.md", "archive/"]
    )

    # Should include regular .md files
    assert filter(Change.added, "/path/to/readme.md")

    # Should ignore draft .md files (ignore takes precedence after include)
    assert not filter(Change.added, "/path/to/draft-post.md")

    # Should ignore .md files in archive/ directory
    assert not filter(Change.added, "/path/archive/old.md")

    # Should NOT include .txt files (not in include patterns)
    assert not filter(Change.added, "/path/to/file.txt")


@pytest.mark.asyncio
async def test_file_filter_extension_check():
    """Test that FileFilter still respects extension filtering."""
    from watchfiles import Change

    from haiku.rag.monitor import FileFilter

    filter = FileFilter()

    # Should include files with supported extensions
    assert filter(Change.added, "/path/to/document.pdf")
    assert filter(Change.added, "/path/to/notes.md")

    # Should not include files with unsupported extensions
    assert not filter(Change.added, "/path/to/file.xyz")
    assert not filter(Change.added, "/path/to/binary.bin")


@pytest.mark.asyncio
async def test_file_watcher_with_ignore_patterns():
    """Test FileWatcher respects ignore patterns from config."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        draft_file = temp_path / "draft.md"
        readme_file = temp_path / "readme.md"

        draft_file.write_text("Draft content")
        readme_file.write_text("Readme content")

        mock_client = AsyncMock(spec=HaikuRAG)
        mock_doc = Document(id="1", content="Readme", uri=readme_file.as_uri())
        mock_client.create_document_from_source.return_value = mock_doc
        mock_client.get_document_by_uri.return_value = None

        test_config = AppConfig(
            monitor=MonitorConfig(directories=[temp_path], ignore_patterns=["draft*"])
        )
        watcher = FileWatcher(client=mock_client, config=test_config)

        # Run refresh which should only process readme.md, not draft.md
        await watcher.refresh()

        # Should have only called for the readme file
        assert mock_client.create_document_from_source.call_count == 1
        mock_client.create_document_from_source.assert_called_with(str(readme_file))


@pytest.mark.asyncio
async def test_file_watcher_with_include_patterns():
    """Test FileWatcher respects include patterns from config."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        md_file = temp_path / "readme.md"
        pdf_file = temp_path / "document.pdf"
        py_file = temp_path / "script.py"

        md_file.write_text("Markdown content")
        pdf_file.write_text("PDF content")
        py_file.write_text("Python content")

        mock_client = AsyncMock(spec=HaikuRAG)
        mock_doc = Document(id="1", content="Markdown", uri=md_file.as_uri())
        mock_client.create_document_from_source.return_value = mock_doc
        mock_client.get_document_by_uri.return_value = None

        test_config = AppConfig(
            monitor=MonitorConfig(directories=[temp_path], include_patterns=["*.md"])
        )
        watcher = FileWatcher(client=mock_client, config=test_config)

        # Run refresh which should only process .md file, not .pdf or .py
        await watcher.refresh()

        # Should have only called for the .md file
        assert mock_client.create_document_from_source.call_count == 1
        mock_client.create_document_from_source.assert_called_with(str(md_file))


@pytest.mark.asyncio
async def test_file_watcher_skips_unchanged_document(caplog):
    """Test FileWatcher skips document when content hasn't changed."""
    from datetime import datetime

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.txt"
        test_content = "Test content"
        test_file.write_text(test_content)

        mock_client = AsyncMock(spec=HaikuRAG)
        # Existing document with a timestamp
        now = datetime.now()
        existing_doc = Document(
            id="1",
            content=test_content,
            uri=test_file.as_uri(),
            created_at=now,
            updated_at=now,
        )
        mock_client.get_document_by_uri.return_value = existing_doc
        # Client returns same document with same timestamp (unchanged)
        mock_client.create_document_from_source.return_value = existing_doc

        test_config = AppConfig(monitor=MonitorConfig(directories=[temp_path]))
        watcher = FileWatcher(client=mock_client, config=test_config)

        with caplog.at_level("INFO"):
            result = await watcher._upsert_document(test_file)

        assert result is not None
        assert result.id == "1"
        # Should log that document was skipped, not updated
        assert any("Skipped" in record.message for record in caplog.records)
        assert not any(
            "Updated document" in record.message for record in caplog.records
        )
