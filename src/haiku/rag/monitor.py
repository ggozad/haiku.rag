import logging
from pathlib import Path
from typing import TYPE_CHECKING

from watchfiles import Change, DefaultFilter, awatch

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models.document import Document

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FileFilter(DefaultFilter):
    def __init__(self, *, ignore_paths: list[Path] | None = None) -> None:
        # Lazy import to avoid loading docling
        from haiku.rag.reader import FileReader

        self.extensions = tuple(FileReader.extensions)
        super().__init__(ignore_paths=ignore_paths)

    def __call__(self, change: Change, path: str) -> bool:
        return path.endswith(self.extensions) and super().__call__(change, path)


class FileWatcher:
    def __init__(self, paths: list[Path], client: HaikuRAG):
        self.paths = paths
        self.client = client

    async def observe(self):
        logger.info(f"Watching files in {self.paths}")
        filter = FileFilter()
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

        for path in self.paths:
            for f in Path(path).rglob("**/*"):
                if f.is_file() and f.suffix in FileReader.extensions:
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
