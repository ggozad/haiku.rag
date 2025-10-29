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
    ) -> None:
        # Lazy import to avoid loading docling
        from haiku.rag.reader import FileReader

        self.extensions = tuple(FileReader.extensions)
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

        # Apply default watchfiles filter
        return super().__call__(change, path)


class FileWatcher:
    def __init__(
        self,
        client: HaikuRAG,
        config: AppConfig = Config,
    ):
        self.paths = config.monitor.directories
        self.client = client
        self.ignore_patterns = config.monitor.ignore_patterns or None
        self.include_patterns = config.monitor.include_patterns or None

    async def observe(self):
        logger.info(f"Watching files in {self.paths}")
        filter = FileFilter(
            ignore_patterns=self.ignore_patterns, include_patterns=self.include_patterns
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
        # Lazy import to avoid loading docling
        from haiku.rag.reader import FileReader

        # Create filter to apply same logic as observe()
        filter = FileFilter(
            ignore_patterns=self.ignore_patterns, include_patterns=self.include_patterns
        )

        for path in self.paths:
            for f in Path(path).rglob("**/*"):
                if f.is_file() and f.suffix in FileReader.extensions:
                    # Apply pattern filters
                    if filter(Change.added, str(f)):
                        await self._upsert_document(f)

    async def _upsert_document(self, file: Path) -> Document | None:
        try:
            uri = file.as_uri()
            existing_doc = await self.client.get_document_by_uri(uri)
            if existing_doc:
                result = await self.client.create_document_from_source(str(file))
                # Since we're passing a file (not directory), result should be a single Document
                doc = result if isinstance(result, Document) else result[0]
                logger.info(f"Updated document {existing_doc.id} from {file}")
                return doc
            else:
                result = await self.client.create_document_from_source(str(file))
                # Since we're passing a file (not directory), result should be a single Document
                doc = result if isinstance(result, Document) else result[0]
                logger.info(f"Created new document {doc.id} from {file}")
                return doc
        except Exception as e:
            logger.error(f"Failed to upsert document from {file}: {e}")
            return None

    async def _delete_document(self, file: Path):
        try:
            uri = file.as_uri()
            existing_doc = await self.client.get_document_by_uri(uri)

            if existing_doc and existing_doc.id:
                await self.client.delete_document(existing_doc.id)
                logger.info(f"Deleted document {existing_doc.id} for {file}")
        except Exception as e:
            logger.error(f"Failed to delete document for {file}: {e}")
