from haiku.rag.utils.sql import build_document_id_filter


def test_build_document_id_filter_empty():
    """No selection means no filter."""
    assert build_document_id_filter([]) is None


def test_build_document_id_filter_matches_exactly():
    """An id filter never widens: names repeat, ids do not."""
    result = build_document_id_filter(["id-one", "id-two"])

    assert result == "id IN ('id-one', 'id-two')"


def test_build_document_id_filter_escapes_quotes():
    assert build_document_id_filter(["O'Reilly"]) == "id IN ('O''Reilly')"
