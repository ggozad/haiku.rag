from typing import TYPE_CHECKING

# The ref of a frame decoded from a stored recording on demand, then its time.
DECODED_FRAME_PREFIX = "#/frames/"

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


def names_file(docling_document: "DoclingDocument", sha256: str) -> bool:
    """Whether the document was converted from the file with this sha256.

    docling keeps only the low 64 bits of the hash it is given.
    """
    origin = docling_document.origin
    return origin is not None and origin.binary_hash == int(sha256, 16) & (2**64 - 1)
