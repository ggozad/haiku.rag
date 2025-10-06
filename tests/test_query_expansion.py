import pytest

from haiku.rag.query_expansion import QueryExpander


@pytest.mark.asyncio
async def test_query_expander_generates_variants():
    """Test that QueryExpander generates multiple query variants."""
    expander = QueryExpander(provider="ollama", model="gpt-oss", num_queries=2)

    original_query = "What is machine learning?"
    variants = await expander.expand(original_query)

    # Should return the requested number of variants
    assert len(variants) == 2

    # All variants should be non-empty strings
    for variant in variants:
        assert isinstance(variant, str)
        assert len(variant.strip()) > 0

    # Original query should not be in variants (we want alternatives)
    assert original_query not in variants
