import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.monitor import FileWatcher
from haiku.rag.store.models.document import Document


@pytest.mark.asyncio
async def test_file_watcher_upsert_document():
    """Test FileWatcher._upsert_document method."""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test.txt"
        temp_path.write_text("Test content for file watcher")

        mock_client = AsyncMock(spec=HaikuRAG)
        mock_doc = Document(id=1, content="Test content", uri=temp_path.as_uri())
        mock_client.create_document_from_source.return_value = mock_doc
        mock_client.get_document_by_uri.return_value = None  # No existing document

        watcher = FileWatcher(paths=[temp_path.parent], client=mock_client)

        result = await watcher._upsert_document(temp_path)

        assert result is not None
        assert result.id == 1
        mock_client.get_document_by_uri.assert_called_once_with(temp_path.as_uri())
        mock_client.create_document_from_source.assert_called_once_with(str(temp_path))


@pytest.mark.asyncio
async def test_file_watcher_upsert_existing_document():
    """Test FileWatcher._upsert_document with existing document."""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test.txt"
        temp_path.write_text("Test content for file watcher")

        mock_client = AsyncMock(spec=HaikuRAG)
        existing_doc = Document(id=1, content="Old content", uri=temp_path.as_uri())
        updated_doc = Document(id=1, content="Updated content", uri=temp_path.as_uri())

        mock_client.get_document_by_uri.return_value = existing_doc
        mock_client.create_document_from_source.return_value = updated_doc

        watcher = FileWatcher(paths=[temp_path.parent], client=mock_client)

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
    existing_doc = Document(id=1, content="Content to delete", uri=temp_path.as_uri())
    mock_client.get_document_by_uri.return_value = existing_doc
    mock_client.delete_document.return_value = True

    watcher = FileWatcher(paths=[temp_path.parent], client=mock_client)

    await watcher._delete_document(temp_path)

    mock_client.get_document_by_uri.assert_called_once_with(temp_path.as_uri())
    mock_client.delete_document.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_file_watcher_delete_nonexistent_document():
    """Test FileWatcher._delete_document with non-existent document."""
    temp_path = Path("/tmp/nonexistent_file.txt")

    mock_client = AsyncMock(spec=HaikuRAG)
    mock_client.get_document_by_uri.return_value = None

    watcher = FileWatcher(paths=[temp_path.parent], client=mock_client)

    await watcher._delete_document(temp_path)

    mock_client.get_document_by_uri.assert_called_once_with(temp_path.as_uri())
    mock_client.delete_document.assert_not_called()
