from haiku.rag.tools.analysis import AnalysisResult


def test_analysis_result_defaults():
    """Test AnalysisResult has sensible defaults."""
    result = AnalysisResult(answer="The result is 42")
    assert result.code_executed is True


def test_analysis_result_with_values():
    """Test AnalysisResult with explicit values."""
    result = AnalysisResult(
        answer="The result is 42",
        code_executed=True,
    )
    assert result.answer == "The result is 42"
    assert result.code_executed is True
