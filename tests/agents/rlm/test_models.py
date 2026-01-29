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
    def test_create_result_with_answer_only(self):
        result = RLMResult(answer="The answer is 42")
        assert result.answer == "The answer is 42"
        assert result.citations == []
        assert result.code_executions == []

    def test_create_result_with_code_executions(self):
        executions = [
            CodeExecution(
                code="x = 1 + 1",
                stdout="",
                stderr="",
                success=True,
            ),
            CodeExecution(
                code="print(x)",
                stdout="2\n",
                stderr="",
                success=True,
            ),
        ]
        result = RLMResult(
            answer="x equals 2",
            code_executions=executions,
        )
        assert len(result.code_executions) == 2
        assert result.code_executions[1].stdout == "2\n"

    def test_create_result_with_citations(self):
        from haiku.rag.agents.research.models import Citation

        citations = [
            Citation(
                document_id="doc1",
                chunk_id="chunk1",
                document_uri="file://test.pdf",
                document_title="Test Doc",
                content="Some content",
            )
        ]
        result = RLMResult(
            answer="Found in Test Doc",
            citations=citations,
        )
        assert len(result.citations) == 1
        assert result.citations[0].document_title == "Test Doc"
