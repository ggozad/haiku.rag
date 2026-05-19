from haiku.rag.sandbox import AnalysisResult


class TestAnalysisResult:
    def test_create_result(self):
        result = AnalysisResult(answer="The answer is 42")
        assert result.answer == "The answer is 42"
        assert result.citations == []
