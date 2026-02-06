from haiku.rag.agents.rlm.models import CodeExecution, RLMResult


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


class TestRLMResult:
    def test_create_result(self):
        result = RLMResult(answer="The answer is 42", program="print(42)")
        assert result.answer == "The answer is 42"
        assert result.program == "print(42)"
