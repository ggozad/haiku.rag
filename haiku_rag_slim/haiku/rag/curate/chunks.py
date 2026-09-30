import hashlib
from collections.abc import Sequence

import numpy as np


def chunk_stats(
    lengths: Sequence[int], short_chunk_chars: int
) -> dict[str, float | None]:
    """Chunk length percentiles in characters and the share of short chunks."""
    if not lengths:
        return {"p10": None, "p50": None, "p90": None, "short_share": None}
    values = np.asarray(lengths)
    p10, p50, p90 = np.percentile(values, [10, 50, 90]).tolist()
    return {
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "short_share": float((values < short_chunk_chars).mean()),
    }


def normalized_text(text: str) -> str:
    """A chunk's text with whitespace collapsed and case folded."""
    return " ".join(text.split()).casefold()


def chunk_text_hash(text: str) -> str:
    """Hash of a chunk's normalized text."""
    return hashlib.sha256(normalized_text(text).encode()).hexdigest()
