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
async def test_file_watcher_skips_unchanged_document():
    """Test FileWatcher returns existing document when content hasn't changed."""
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

        result = await watcher._upsert_document(test_file)

        assert result is not None
        assert result.id == "1"
        # Verify timestamp hasn't changed (document wasn't updated)
        assert result.updated_at == now


@pytest.mark.asyncio
async def test_file_watcher_deletes_orphans():
    """Test FileWatcher deletes documents whose files no longer exist."""
    import asyncio

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        existing_file = temp_path / "exists.txt"
        existing_file.write_text("Existing file")

        # Create a document for a file that doesn't exist
        orphan_uri = (temp_path / "deleted.txt").as_uri()

        mock_client = AsyncMock(spec=HaikuRAG)
        orphan_doc = Document(id="orphan-1", content="Orphaned content", uri=orphan_uri)
        existing_doc = Document(
            id="existing-1", content="Existing content", uri=existing_file.as_uri()
        )

        # Mock list_documents to return both documents
        mock_client.list_documents.return_value = [orphan_doc, existing_doc]
        mock_client.get_document_by_uri.return_value = None
        mock_client.create_document_from_source.return_value = existing_doc

        test_config = AppConfig(
            monitor=MonitorConfig(directories=[temp_path], delete_orphans=True)
        )
        watcher = FileWatcher(client=mock_client, config=test_config)

        # Run refresh which should delete orphan and process existing file
        await watcher.refresh()

        # Give background task time to complete
        await asyncio.sleep(0.1)

        # Should have deleted the orphan document
        mock_client.delete_document.assert_called_once_with("orphan-1")
        # Should have processed the existing file
        mock_client.create_document_from_source.assert_called_once()


@pytest.mark.asyncio
async def test_file_watcher_skips_orphan_deletion_when_disabled():
    """Test FileWatcher does not delete orphans when delete_orphans is False."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a document for a file that doesn't exist
        orphan_uri = (temp_path / "deleted.txt").as_uri()

        mock_client = AsyncMock(spec=HaikuRAG)
        orphan_doc = Document(id="orphan-1", content="Orphaned content", uri=orphan_uri)

        # Mock list_documents to return orphan document
        mock_client.list_documents.return_value = [orphan_doc]

        test_config = AppConfig(
            monitor=MonitorConfig(directories=[temp_path], delete_orphans=False)
        )
        watcher = FileWatcher(client=mock_client, config=test_config)

        # Run refresh
        await watcher.refresh()

        # Should NOT have deleted the orphan document
        mock_client.delete_document.assert_not_called()


@pytest.mark.asyncio
async def test_file_watcher_orphan_deletion_respects_patterns():
    """Test orphan deletion respects include/ignore patterns."""
    import asyncio

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create documents for files that don't exist
        ignored_orphan_uri = (temp_path / "draft.md").as_uri()
        excluded_orphan_uri = (temp_path / "file.pdf").as_uri()
        included_orphan_uri = (temp_path / "readme.md").as_uri()

        mock_client = AsyncMock(spec=HaikuRAG)

        ignored_doc = Document(
            id="ignored-1", content="Ignored", uri=ignored_orphan_uri
        )
        excluded_doc = Document(
            id="excluded-1", content="Excluded", uri=excluded_orphan_uri
        )
        included_doc = Document(
            id="included-1", content="Included", uri=included_orphan_uri
        )

        # Mock list_documents to return all orphan documents
        mock_client.list_documents.return_value = [
            ignored_doc,
            excluded_doc,
            included_doc,
        ]

        # Config with patterns: only .md files, but exclude draft*
        test_config = AppConfig(
            monitor=MonitorConfig(
                directories=[temp_path],
                delete_orphans=True,
                include_patterns=["*.md"],
                ignore_patterns=["draft*"],
            )
        )
        watcher = FileWatcher(client=mock_client, config=test_config)

        # Run refresh
        await watcher.refresh()

        # Give background task time to complete
        await asyncio.sleep(0.1)

        # Should only delete the included orphan (readme.md)
        # - draft.md matches ignore pattern -> NOT deleted
        # - file.pdf doesn't match include pattern -> NOT deleted
        # - readme.md matches include and not ignored -> DELETED
        mock_client.delete_document.assert_called_once_with("included-1")


@pytest.mark.asyncio
async def test_file_watcher_orphan_handles_spaces_in_filenames():
    """Test orphan deletion correctly handles files with spaces in names."""
    import asyncio

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Create a file with spaces that exists
        existing_file = temp_path / "my file with spaces.txt"
        existing_file.write_text("Existing file")

        mock_client = AsyncMock(spec=HaikuRAG)
        # Document with URI that has URL-encoded spaces (%20)
        existing_doc = Document(
            id="existing-1", content="Existing", uri=existing_file.as_uri()
        )

        # Mock list_documents to return document with encoded spaces
        mock_client.list_documents.return_value = [existing_doc]
        mock_client.get_document_by_uri.return_value = None
        mock_client.create_document_from_source.return_value = existing_doc

        test_config = AppConfig(
            monitor=MonitorConfig(directories=[temp_path], delete_orphans=True)
        )
        watcher = FileWatcher(client=mock_client, config=test_config)

        # Run refresh
        await watcher.refresh()

        # Give background task time to complete
        await asyncio.sleep(0.1)

        # Should NOT delete the document since file exists
        mock_client.delete_document.assert_not_called()


@pytest.mark.asyncio
async def test_file_watcher_observe_raises_on_missing_paths():
    """Test observe() raises FileNotFoundError when directories don't exist."""
    mock_client = AsyncMock(spec=HaikuRAG)

    test_config = AppConfig(
        monitor=MonitorConfig(
            directories=[Path("/nonexistent/path/that/does/not/exist")]
        )
    )
    watcher = FileWatcher(client=mock_client, config=test_config)

    with pytest.raises(FileNotFoundError) as exc_info:
        await watcher.observe()

    assert "Monitor directories do not exist" in str(exc_info.value)
    assert "haiku.rag.yaml" in str(exc_info.value)


@pytest.mark.asyncio
async def test_file_watcher_observe_returns_early_when_no_directories():
    """Test observe() returns early when no directories are configured."""
    mock_client = AsyncMock(spec=HaikuRAG)

    test_config = AppConfig(monitor=MonitorConfig(directories=[]))
    watcher = FileWatcher(client=mock_client, config=test_config)

    # Should return without error when no directories configured
    await watcher.observe()

    # No documents should have been processed
    mock_client.create_document_from_source.assert_not_called()
