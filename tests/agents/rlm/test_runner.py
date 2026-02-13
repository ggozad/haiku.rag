import json
from io import StringIO

from haiku.rag.agents.rlm.runner import execute_code, send_response


def test_execute_code_success():
    namespace: dict = {}
    result = execute_code("x = 1 + 1", namespace, max_output_chars=1000)
    assert result["success"] is True
    assert result["stderr"] == ""


def test_execute_code_stdout_capture():
    namespace: dict = {}
    result = execute_code("print('hello')", namespace, max_output_chars=1000)
    assert result["success"] is True
    assert "hello" in result["stdout"]


def test_execute_code_exception():
    namespace: dict = {}
    result = execute_code("raise ValueError('boom')", namespace, max_output_chars=1000)
    assert result["success"] is False
    assert "ValueError" in result["stderr"]
    assert "boom" in result["stderr"]


def test_execute_code_output_truncation():
    namespace: dict = {}
    code = "print('x' * 100)"
    result = execute_code(code, namespace, max_output_chars=10)
    assert result["success"] is True
    assert "truncated" in result["stdout"]
    assert len(result["stdout"]) < 100


def test_send_response(monkeypatch):
    buf = StringIO()
    monkeypatch.setattr("sys.stdout", buf)

    payload = {"success": True, "stdout": "hi", "stderr": ""}
    send_response(payload)

    output = buf.getvalue()
    lines = output.split("\n", 1)
    length = int(lines[0])
    body = lines[1]
    assert json.loads(body) == payload
    assert length == len(json.dumps(payload))
