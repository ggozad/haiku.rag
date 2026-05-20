from dataclasses import dataclass


@dataclass
class AnalysisContext:
    """Mutable context accumulating data during analysis execution."""

    filter: str | None = None
