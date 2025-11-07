import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pathspec
from pathspec.patterns.gitwildmatch import GitWildMatchPattern
from watchfiles import Change, DefaultFilter, awatch

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, Config
from haiku.rag.store.models.document import Document

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FileFilter(DefaultFilter):
    def __init__(
        self,
        *,
        ignore_patterns: list[str] | None = None,
        include_patterns: list[str] | None = None,
        supported_extensions: list[str] | None = None,
    ) -> None:
        if supported_extensions is None:
            # Default to docling-local extensions if not provided
            from haiku.rag.converters.docling_local import DoclingLocalConverter
            from haiku.rag.converters.text_utils import TextFileHandler

            supported_extensions = (
                DoclingLocalConverter.docling_extensions
                + TextFileHandler.text_extensions
            )

        self.extensions = tuple(supported_extensions)
        self.ignore_spec = (
            pathspec.PathSpec.from_lines(GitWildMatchPattern, ignore_patterns)
            if ignore_patterns
            else None
        )
        self.include_spec = (
            pathspec.PathSpec.from_lines(GitWildMatchPattern, include_patterns)
            if include_patterns
            else None
        )
        super().__init__()

    def __call__(self, change: Change, path: str) -> bool:
        if not self.include_file(path):
            return False

        # Apply default watchfiles filter
        return super().__call__(change, path)

    def include_file(self, path: str) -> bool:
        """Check if a file should be included based on filters."""
        # Check extension filter
        if not path.endswith(self.extensions):
            return False

        # Apply include patterns if specified (whitelist mode)
        if self.include_spec:
            if not self.include_spec.match_file(path):
                return False

        # Apply ignore patterns (blacklist mode)
        if self.ignore_spec:
            if self.ignore_spec.match_file(path):
                return False

        return True


class FileWatcher:
    def __init__(
        self,
        client: HaikuRAG,
        config: AppConfig = Config,
    ):
        from haiku.rag.converters import get_converter

        self.paths = config.monitor.directories
        self.client = client
        self.ignore_patterns = config.monitor.ignore_patterns or None
        self.include_patterns = config.monitor.include_patterns or None
        self.delete_orphans = config.monitor.delete_orphans
        self.supported_extensions = get_converter(config).supported_extensions

    async def observe(self):
        logger.info(f"Watching files in {self.paths}")
        filter = FileFilter(
            ignore_patterns=self.ignore_patterns,
            include_patterns=self.include_patterns,
            supported_extensions=self.supported_extensions,
        )
        await self.refresh()

        async for changes in awatch(*self.paths, watch_filter=filter):
            await self.handler(changes)

    async def handler(self, changes: set[tuple[Change, str]]):
        for change, path in changes:
            if change == Change.added or change == Change.modified:
                await self._upsert_document(Path(path))
            elif change == Change.deleted:
                await self._delete_document(Path(path))

    async def refresh(self):
        # Delete orphaned documents in background if enabled
        if self.delete_orphans:
            logger.info("Starting orphan cleanup in background")
            asyncio.create_task(self._delete_orphans())

        # Create filter to apply same logic as observe()
        filter = FileFilter(
            ignore_patterns=self.ignore_patterns,
            include_patterns=self.include_patterns,
            supported_extensions=self.supported_extensions,
        )

        for path in self.paths:
            for f in Path(path).rglob("**/*"):
                if f.is_file() and f.suffix in self.supported_extensions:
                    # Apply pattern filters
                    if filter(Change.added, str(f)):
                        await self._upsert_document(f)

    async def _upsert_document(self, file: Path) -> Document | None:
        try:
            uri = file.as_uri()
            existing_doc = await self.client.get_document_by_uri(uri)

            result = await self.client.create_document_from_source(str(file))
            doc = result if isinstance(result, Document) else result[0]

            if existing_doc:
                # Check if document was actually updated by comparing updated_at timestamps
                if doc.updated_at > existing_doc.updated_at:
                    logger.info(f"Updated document {existing_doc.id} from {file}")
                else:
                    logger.info(
                        f"Skipped unchanged document {existing_doc.id} from {file}"
                    )
            else:
                logger.info(f"Created new document {doc.id} from {file}")

            return doc
        except Exception as e:
            logger.error(f"Failed to upsert document from {file}: {e}")
            return None

    async def _delete_orphans(self):
        """Delete documents whose source files no longer exist."""
        try:
            from urllib.parse import unquote, urlparse

            # Create filter to apply same include/exclude logic
            filter = FileFilter(
                ignore_patterns=self.ignore_patterns,
                include_patterns=self.include_patterns,
            )

            all_docs = await self.client.list_documents()

            for doc in all_docs:
                if not doc.uri or not doc.id:
                    continue

                # Only check file:// URIs
                parsed = urlparse(doc.uri)
                if parsed.scheme != "file":
                    continue

                # Convert URI to Path, decoding URL-encoded characters (like %20 for spaces)
                file_path = Path(unquote(parsed.path))

                # Check if file exists
                if not file_path.exists():
                    # Check if file is within monitored directories
                    is_monitored = any(
                        file_path.is_relative_to(monitored_path)
                        for monitored_path in self.paths
                    )

                    # Check if file would have been included by filters
                    if is_monitored and filter.include_file(str(file_path)):
                        await self.client.delete_document(doc.id)
                        logger.info(
                            f"Deleted orphaned document {doc.id} for {file_path}"
                        )
        except Exception as e:
            logger.error(f"Failed to delete orphaned documents: {e}")

    async def _delete_document(self, file: Path):
        try:
            uri = file.as_uri()
            existing_doc = await self.client.get_document_by_uri(uri)

            if existing_doc and existing_doc.id:
                await self.client.delete_document(existing_doc.id)
                logger.info(f"Deleted document {existing_doc.id} for {file}")
        except Exception as e:
            logger.error(f"Failed to delete document for {file}: {e}")
