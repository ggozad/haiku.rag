import asyncio
import hashlib

import lancedb
import pyarrow as pa

from haiku.rag.store.engine import ConnectionMode, Store
from haiku.rag.store.exceptions import UnsupportedRecordingStorageError
from haiku.rag.store.models.recording import RecordingFile, StoredRecording
from haiku.rag.store.schema import get_recordings_arrow_schema
from haiku.rag.utils.sql import escape_sql_string


class RecordingRepository:
    """The stored audio and video files, one per recording document."""

    def __init__(self, store: Store) -> None:
        self.store = store

    @property
    def _table(self) -> lancedb.AsyncTable:
        """The table a writable store always has."""
        assert self.store.recordings_table is not None
        return self.store.recordings_table

    async def put(
        self, document_id: str, recording: RecordingFile, duration: float
    ) -> None:
        """Store `recording` for the document, replacing any stored before."""
        self.store._assert_writable()
        if self.store._connection_mode == ConnectionMode.CLOUD:
            raise UnsupportedRecordingStorageError(
                "Recordings cannot be stored on LanceDB Cloud."
            )
        row = {
            "document_id": document_id,
            "suffix": recording.suffix.lower(),
            "sha256": hashlib.sha256(recording.content).hexdigest(),
            "duration": duration,
            "subtitles": recording.subtitles,
            "content": {
                "data": recording.content,
                "uri": None,
                "position": None,
                "size": None,
            },
        }
        await self.delete(document_id)
        await self._table.add(
            pa.Table.from_pylist([row], schema=get_recordings_arrow_schema())
        )

    async def open(self, document_id: str) -> StoredRecording | None:
        """The document's stored recording, or None when it has none."""
        # LanceDB Cloud stores no recording, and lance cannot open its tables.
        if (
            self.store.recordings_table is None
            or self.store._connection_mode == ConnectionMode.CLOUD
        ):
            return None
        dataset = await self.store.recordings_table.to_lance()

        def read() -> StoredRecording | None:
            rows = dataset.to_table(
                columns=["document_id", "suffix", "sha256", "duration", "subtitles"],
                filter=f"document_id = '{escape_sql_string(document_id)}'",
                with_row_id=True,
            ).to_pylist()
            if not rows:
                return None
            row = rows[0]
            [blob] = dataset.take_blobs("content", ids=[row.pop("_rowid")])
            return StoredRecording(**row, blob=blob)

        return await asyncio.to_thread(read)

    async def digest(self, document_id: str) -> str | None:
        """The sha256 of the document's stored recording, or None when it has none."""
        if self.store.recordings_table is None:
            return None
        rows = (
            await self.store.recordings_table.query()
            .where(f"document_id = '{escape_sql_string(document_id)}'")
            .select(["sha256"])
            .limit(1)
            .to_list()
        )
        return rows[0]["sha256"] if rows else None

    async def delete(self, document_id: str) -> None:
        self.store._assert_writable()
        await self._table.delete(f"document_id = '{escape_sql_string(document_id)}'")

    async def delete_all(self) -> None:
        self.store._assert_writable()
        await self._table.delete("true")
