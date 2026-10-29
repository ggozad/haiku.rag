from haiku.rag.sources.base import SourceEventKind


def test_source_event_kind_values():
    assert SourceEventKind.UPSERT.value == "upsert"
    assert SourceEventKind.DELETE.value == "delete"
    assert SourceEventKind.UNCHANGED.value == "unchanged"
