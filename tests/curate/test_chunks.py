import pytest

from haiku.rag.curate.chunks import chunk_stats, chunk_text_hash


def test_chunk_stats_percentiles_and_short_share():
    stats = chunk_stats([10, 20, 30, 40, 100], short_chunk_chars=25)
    assert stats["p50"] == 30
    assert stats["p10"] == pytest.approx(14.0)
    assert stats["p90"] == pytest.approx(76.0)
    assert stats["short_share"] == pytest.approx(0.4)


def test_chunk_stats_without_chunks():
    assert chunk_stats([], short_chunk_chars=25) == {
        "p10": None,
        "p50": None,
        "p90": None,
        "short_share": None,
    }


def test_chunk_text_hash_ignores_whitespace_and_case():
    assert chunk_text_hash("Page  1 of\n10") == chunk_text_hash("page 1 of 10")
    assert chunk_text_hash("page 1 of 10") != chunk_text_hash("page 2 of 10")
