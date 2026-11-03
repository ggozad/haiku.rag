from logfire.testing import TestExporter


class Tree:
    """The exported spans, indexed for parent/child questions."""

    def __init__(self, exporter: TestExporter):
        # logfire exports a pending span when one opens and the real span when
        # it closes; only the latter carries final attributes.
        self.spans = [
            span
            for span in exporter.exported_spans
            if (span.attributes or {}).get("logfire.span_type") != "pending_span"
        ]
        self._names = {
            span.context.span_id: span.name for span in self.spans if span.context
        }

    def names(self) -> list[str]:
        return [span.name for span in self.spans]

    def one(self, name: str):
        matches = [span for span in self.spans if span.name == name]
        assert len(matches) == 1, f"expected exactly one {name!r}, got {len(matches)}"
        return matches[0]

    def parent_of(self, name: str) -> str | None:
        span = self.one(name)
        if span.parent is None:
            return None
        return self._names.get(span.parent.span_id)

    def attrs(self, name: str) -> dict:
        return dict(self.one(name).attributes or {})

    def exceptions(self, name: str) -> list[str]:
        """The types of the exceptions recorded on the span."""
        return [
            (event.attributes or {})["exception.type"]
            for event in self.one(name).events
            if event.name == "exception"
        ]

    def every(self, name: str) -> list[dict]:
        """The attributes of every span with that name, in export order."""
        return [dict(span.attributes or {}) for span in self.spans if span.name == name]
