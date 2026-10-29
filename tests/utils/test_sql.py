from haiku.rag.utils.sql import build_document_id_filter, title_or_uri_filter


def test_build_document_id_filter_empty():
    """No selection means no filter."""
    assert build_document_id_filter([]) is None


def test_build_document_id_filter_matches_exactly():
    """An id filter never widens: names repeat, ids do not."""
    result = build_document_id_filter(["id-one", "id-two"])

    assert result == "id IN ('id-one', 'id-two')"


def test_build_document_id_filter_escapes_quotes():
    assert build_document_id_filter(["O'Reilly"]) == "id IN ('O''Reilly')"


def test_escape_like_pattern_makes_wildcards_literal():
    from haiku.rag.utils.sql import escape_like_pattern

    assert escape_like_pattern("a_b%c\\d") == "a\\_b\\%c\\\\d"


class TestDocumentSearchFilter:
    """A typed term reaches SQL as a title or URI match."""

    def test_no_term_means_no_filter(self):
        assert title_or_uri_filter("   ") is None

    def test_a_term_matches_titles_and_uris(self):
        built = title_or_uri_filter("NoBeL")

        assert built == (
            "LOWER(title) LIKE '%nobel%' ESCAPE '\\' "
            "OR LOWER(uri) LIKE '%nobel%' ESCAPE '\\'"
        )

    def test_like_wildcards_in_the_term_are_literal(self):
        """A term is text to find, not a pattern: `_` matches any character."""
        built = title_or_uri_filter("100%_raw")

        assert built is not None
        assert "100\\%\\_raw" in built
        assert "ESCAPE '\\'" in built

    def test_a_quote_in_the_term_is_escaped(self):
        """Whatever was typed is data, not syntax."""
        built = title_or_uri_filter("O'Brien")

        assert built is not None
        assert "o''brien" in built
