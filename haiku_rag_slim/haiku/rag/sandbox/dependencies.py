from dataclasses import dataclass


@dataclass
class AnalysisContext:
    """Mutable context accumulating data during code execution."""

    filter: str | None = None
    sources: list[str] | None = None
    separate_recordings: bool = False
    """`search()` and `list_documents()` leave recordings out, which the media
    functions search and list."""
