from haiku.rag.agents.analysis.models import AnalysisResult, CodeExecution


class TestCodeExecution:
    def test_create_successful_execution(self):
        execution = CodeExecution(
            code="print('hello')",
            stdout="hello\n",
            stderr="",
            success=True,
        )
        assert execution.code == "print('hello')"
        assert execution.stdout == "hello\n"
        assert execution.stderr == ""
        assert execution.success is True

    def test_create_failed_execution(self):
        execution = CodeExecution(
            code="1/0",
            stdout="",
            stderr="ZeroDivisionError: division by zero",
            success=False,
        )
        assert execution.success is False
        assert "ZeroDivisionError" in execution.stderr


class TestAnalysisResult:
    def test_create_result(self):
        result = AnalysisResult(answer="The answer is 42", program="print(42)")
        assert result.answer == "The answer is 42"
        assert result.program == "print(42)"
