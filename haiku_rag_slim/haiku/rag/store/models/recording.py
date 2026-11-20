from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


@dataclass(frozen=True)
class RecordingFile:
    """An audio or video file as ingested: its bytes, the suffix naming its
    format, and the subtitles read beside it, if any."""

    content: bytes
    suffix: str
    subtitles: str | None = None


@dataclass(frozen=True)
class StoredRecording:
    """A stored recording, its bytes read by range on demand."""

    document_id: str
    suffix: str
    sha256: str
    duration: float
    subtitles: str | None
    blob: Any

    @property
    def size(self) -> int:
        return self.blob.size()

    def read_range(self, offset: int, length: int) -> bytes:
        return self.blob.read_range(offset, length)


def names_file(docling_document: "DoclingDocument", sha256: str) -> bool:
    """Whether the document was converted from the file with this sha256.

    docling keeps only the low 64 bits of the hash it is given.
    """
    origin = docling_document.origin
    return origin is not None and origin.binary_hash == int(sha256, 16) & (2**64 - 1)
