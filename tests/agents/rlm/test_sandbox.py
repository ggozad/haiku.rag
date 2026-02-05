from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent.parent / "cassettes" / "test_sandbox")


class TestSafeBuiltins:
    """Test that safe builtins are available."""

    @pytest.mark.asyncio
    async def test_print_available(self, repl_env_empty):
        result = await repl_env_empty.execute_async("print('hello')")
        assert result.success
        assert "hello" in result.stdout

    @pytest.mark.asyncio
    async def test_len_available(self, repl_env_empty):
        result = await repl_env_empty.execute_async("print(len([1, 2, 3]))")
        assert result.success
        assert "3" in result.stdout

    @pytest.mark.asyncio
    async def test_range_available(self, repl_env_empty):
        result = await repl_env_empty.execute_async("print(list(range(3)))")
        assert result.success
        assert "[0, 1, 2]" in result.stdout

    @pytest.mark.asyncio
    async def test_enumerate_available(self, repl_env_empty):
        result = await repl_env_empty.execute_async(
            "print(list(enumerate(['a', 'b'])))"
        )
        assert result.success
        assert "[(0, 'a'), (1, 'b')]" in result.stdout

    @pytest.mark.asyncio
    async def test_sorted_available(self, repl_env_empty):
        result = await repl_env_empty.execute_async("print(sorted([3, 1, 2]))")
        assert result.success
        assert "[1, 2, 3]" in result.stdout

    @pytest.mark.asyncio
    async def test_sum_available(self, repl_env_empty):
        result = await repl_env_empty.execute_async("print(sum([1, 2, 3]))")
        assert result.success
        assert "6" in result.stdout

    @pytest.mark.asyncio
    async def test_min_max_available(self, repl_env_empty):
        result = await repl_env_empty.execute_async(
            "print(min([3, 1, 2]), max([3, 1, 2]))"
        )
        assert result.success
        assert "1 3" in result.stdout

    @pytest.mark.asyncio
    async def test_all_any_available(self, repl_env_empty):
        result = await repl_env_empty.execute_async(
            "print(all([True, True]), any([False, True]))"
        )
        assert result.success
        assert "True True" in result.stdout

    @pytest.mark.asyncio
    async def test_dict_list_set_tuple_available(self, repl_env_empty):
        result = await repl_env_empty.execute_async(
            "print(dict(a=1), list((1,2)), set([1,2,1]), tuple([1,2]))"
        )
        assert result.success
        assert "{'a': 1}" in result.stdout

    @pytest.mark.asyncio
    async def test_str_int_float_bool_available(self, repl_env_empty):
        result = await repl_env_empty.execute_async(
            "print(str(1), int('2'), float('3.0'), bool(1))"
        )
        assert result.success
        assert "1 2 3.0 True" in result.stdout

    @pytest.mark.asyncio
    async def test_zip_map_filter_available(self, repl_env_empty):
        result = await repl_env_empty.execute_async(
            "print(list(zip([1,2], ['a','b'])), "
            "list(map(str, [1,2])), "
            "list(filter(lambda x: x > 1, [1,2,3])))"
        )
        assert result.success
        assert "[(1, 'a'), (2, 'b')]" in result.stdout

    @pytest.mark.asyncio
    async def test_isinstance_type_available(self, repl_env_empty):
        result = await repl_env_empty.execute_async(
            "print(isinstance(1, int), type([]))"
        )
        assert result.success
        assert "True" in result.stdout


class TestDangerousBuiltinsBlocked:
    """Test that dangerous builtins are blocked."""

    @pytest.mark.asyncio
    async def test_eval_blocked(self, repl_env_empty):
        result = await repl_env_empty.execute_async("eval('1+1')")
        assert not result.success
        assert "eval" in result.stderr.lower() or "not defined" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_exec_blocked(self, repl_env_empty):
        result = await repl_env_empty.execute_async("exec('x = 1')")
        assert not result.success
        assert "exec" in result.stderr.lower() or "not defined" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_compile_blocked(self, repl_env_empty):
        result = await repl_env_empty.execute_async(
            "compile('1+1', '<string>', 'eval')"
        )
        assert not result.success
        assert (
            "compile" in result.stderr.lower() or "not defined" in result.stderr.lower()
        )

    @pytest.mark.asyncio
    async def test_open_blocked(self, repl_env_empty):
        result = await repl_env_empty.execute_async("open('/etc/passwd')")
        assert not result.success
        assert "open" in result.stderr.lower() or "not defined" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_input_blocked(self, repl_env_empty):
        result = await repl_env_empty.execute_async("input('Enter: ')")
        assert not result.success
        assert (
            "input" in result.stderr.lower() or "not defined" in result.stderr.lower()
        )

    @pytest.mark.asyncio
    async def test___import___blocked(self, repl_env_empty):
        result = await repl_env_empty.execute_async("__import__('os')")
        assert not result.success

    @pytest.mark.asyncio
    async def test_globals_blocked(self, repl_env_empty):
        result = await repl_env_empty.execute_async("globals()")
        assert not result.success

    @pytest.mark.asyncio
    async def test_locals_blocked(self, repl_env_empty):
        result = await repl_env_empty.execute_async("locals()")
        assert not result.success

    @pytest.mark.asyncio
    async def test_breakpoint_blocked(self, repl_env_empty):
        result = await repl_env_empty.execute_async("breakpoint()")
        assert not result.success

    @pytest.mark.asyncio
    async def test_getattr_setattr_blocked(self, repl_env_empty):
        result = await repl_env_empty.execute_async("getattr(object, '__class__')")
        assert not result.success

    @pytest.mark.asyncio
    async def test_delattr_blocked(self, repl_env_empty):
        result = await repl_env_empty.execute_async("delattr(object, 'x')")
        assert not result.success


class TestAllowedImports:
    """Test that allowed imports work."""

    @pytest.mark.asyncio
    async def test_json_import(self, repl_env_empty):
        result = await repl_env_empty.execute_async(
            "import json\nprint(json.dumps({'a': 1}))"
        )
        assert result.success
        assert '{"a": 1}' in result.stdout

    @pytest.mark.asyncio
    async def test_re_import(self, repl_env_empty):
        result = await repl_env_empty.execute_async(
            "import re\nprint(re.match(r'\\d+', '123').group())"
        )
        assert result.success
        assert "123" in result.stdout

    @pytest.mark.asyncio
    async def test_math_import(self, repl_env_empty):
        result = await repl_env_empty.execute_async("import math\nprint(math.sqrt(4))")
        assert result.success
        assert "2.0" in result.stdout

    @pytest.mark.asyncio
    async def test_statistics_import(self, repl_env_empty):
        result = await repl_env_empty.execute_async(
            "import statistics\nprint(statistics.mean([1, 2, 3]))"
        )
        assert result.success
        assert "2" in result.stdout

    @pytest.mark.asyncio
    async def test_collections_import(self, repl_env_empty):
        result = await repl_env_empty.execute_async(
            "from collections import Counter\nprint(Counter(['a', 'b', 'a']))"
        )
        assert result.success
        assert "'a': 2" in result.stdout

    @pytest.mark.asyncio
    async def test_itertools_import(self, repl_env_empty):
        result = await repl_env_empty.execute_async(
            "from itertools import chain\nprint(list(chain([1], [2])))"
        )
        assert result.success
        assert "[1, 2]" in result.stdout

    @pytest.mark.asyncio
    async def test_functools_import(self, repl_env_empty):
        result = await repl_env_empty.execute_async(
            "from functools import reduce\nprint(reduce(lambda a, b: a+b, [1,2,3]))"
        )
        assert result.success
        assert "6" in result.stdout

    @pytest.mark.asyncio
    async def test_datetime_import(self, repl_env_empty):
        result = await repl_env_empty.execute_async(
            "from datetime import date\nprint(date(2025, 1, 1))"
        )
        assert result.success
        assert "2025-01-01" in result.stdout


class TestDangerousImportsBlocked:
    """Test that dangerous imports are blocked."""

    @pytest.mark.asyncio
    async def test_os_blocked(self, repl_env_empty):
        result = await repl_env_empty.execute_async("import os")
        assert not result.success
        assert (
            "not allowed" in result.stderr.lower() or "error" in result.stderr.lower()
        )

    @pytest.mark.asyncio
    async def test_sys_blocked(self, repl_env_empty):
        result = await repl_env_empty.execute_async("import sys")
        assert not result.success

    @pytest.mark.asyncio
    async def test_subprocess_blocked(self, repl_env_empty):
        result = await repl_env_empty.execute_async("import subprocess")
        assert not result.success

    @pytest.mark.asyncio
    async def test_shutil_blocked(self, repl_env_empty):
        result = await repl_env_empty.execute_async("import shutil")
        assert not result.success

    @pytest.mark.asyncio
    async def test_socket_blocked(self, repl_env_empty):
        result = await repl_env_empty.execute_async("import socket")
        assert not result.success

    @pytest.mark.asyncio
    async def test_requests_blocked(self, repl_env_empty):
        result = await repl_env_empty.execute_async("import requests")
        assert not result.success

    @pytest.mark.asyncio
    async def test_builtins_module_blocked(self, repl_env_empty):
        result = await repl_env_empty.execute_async("import builtins")
        assert not result.success


class TestHaikuRAGBridgeFunctions:
    """Test haiku.rag bridge functions in sandbox."""

    @pytest.mark.asyncio
    async def test_search(self, repl_env_empty):
        """Test search function calls client with correct args."""
        from unittest.mock import AsyncMock

        from haiku.rag.store.models import SearchResult

        mock_results = [
            SearchResult(
                chunk_id="chunk-1",
                document_id="doc-1",
                document_title="Test Doc",
                document_uri="test://doc",
                content="Test content about foxes",
                score=0.9,
                page_numbers=[1],
                headings=["Heading"],
            )
        ]
        repl_env_empty.client.search = AsyncMock(return_value=mock_results)

        result = await repl_env_empty.execute_async(
            "results = search('fox', limit=5)\n"
            "print(len(results), results[0]['chunk_id'], 'fox' in results[0]['content'].lower())"
        )
        assert result.success
        assert "1 chunk-1 True" in result.stdout
        repl_env_empty.client.search.assert_called_once_with(
            "fox", limit=5, filter=None
        )

    @pytest.mark.asyncio
    async def test_list_documents(self, repl_env_empty):
        """Test list_documents returns list structure."""
        result = await repl_env_empty.execute_async(
            "docs = list_documents()\nprint(type(docs).__name__, len(docs))"
        )
        assert result.success
        assert "list 0" in result.stdout

    @pytest.mark.asyncio
    async def test_get_document(self, repl_env_empty):
        """Test get_document calls client correctly."""
        from unittest.mock import AsyncMock

        from haiku.rag.store.models import Document

        mock_doc = Document(
            id="doc-1",
            uri="test://doc",
            title="Test Doc",
            content="The quick brown fox",
        )
        repl_env_empty.client.get_document_by_id = AsyncMock(return_value=mock_doc)

        result = await repl_env_empty.execute_async(
            "doc = get_document('doc-1')\nprint('fox' in doc.lower())"
        )
        assert result.success
        assert "True" in result.stdout

    @pytest.mark.asyncio
    async def test_get_document_missing(self, repl_env_empty):
        """Test get_document returns None for missing document."""
        result = await repl_env_empty.execute_async(
            "doc = get_document('Nonexistent')\nprint(doc is None)"
        )
        assert result.success
        assert "True" in result.stdout

    @pytest.mark.asyncio
    async def test_llm(self, repl_env_empty):
        """Test llm function is available in sandbox."""
        result = await repl_env_empty.execute_async("print(callable(llm))")
        assert result.success
        assert "True" in result.stdout


class TestSandboxExecution:
    """Test general sandbox execution behavior."""

    @pytest.mark.asyncio
    async def test_variable_persistence(self, repl_env_empty):
        """Variables persist across executions."""
        await repl_env_empty.execute_async("x = 42")
        result = await repl_env_empty.execute_async("print(x)")
        assert result.success
        assert "42" in result.stdout

    @pytest.mark.asyncio
    async def test_function_definition(self, repl_env_empty):
        """Can define and call functions."""
        result = await repl_env_empty.execute_async(
            "def add(a, b):\n    return a + b\nprint(add(1, 2))"
        )
        assert result.success
        assert "3" in result.stdout

    @pytest.mark.asyncio
    async def test_class_definition(self, repl_env_empty):
        """Can define and use classes."""
        result = await repl_env_empty.execute_async(
            "class Point:\n"
            "    def __init__(self, x, y):\n"
            "        self.x = x\n"
            "        self.y = y\n"
            "p = Point(1, 2)\n"
            "print(p.x, p.y)"
        )
        assert result.success
        assert "1 2" in result.stdout

    @pytest.mark.asyncio
    async def test_list_comprehension(self, repl_env_empty):
        """List comprehensions work."""
        result = await repl_env_empty.execute_async("print([x**2 for x in range(5)])")
        assert result.success
        assert "[0, 1, 4, 9, 16]" in result.stdout

    @pytest.mark.asyncio
    async def test_dict_comprehension(self, repl_env_empty):
        """Dict comprehensions work."""
        result = await repl_env_empty.execute_async(
            "print({x: x**2 for x in range(3)})"
        )
        assert result.success
        assert "{0: 0, 1: 1, 2: 4}" in result.stdout

    @pytest.mark.asyncio
    async def test_exception_handling(self, repl_env_empty):
        """Can catch and handle exceptions."""
        result = await repl_env_empty.execute_async(
            "try:\n    x = 1/0\nexcept ZeroDivisionError:\n    print('caught')"
        )
        assert result.success
        assert "caught" in result.stdout

    @pytest.mark.asyncio
    async def test_uncaught_exception_reports_error(self, repl_env_empty):
        """Uncaught exceptions are reported."""
        result = await repl_env_empty.execute_async("x = 1/0")
        assert not result.success
        assert "ZeroDivisionError" in result.stderr

    @pytest.mark.asyncio
    async def test_syntax_error_reports_error(self, repl_env_empty):
        """Syntax errors are reported."""
        result = await repl_env_empty.execute_async("def foo(")
        assert not result.success
        assert "SyntaxError" in result.stderr

    @pytest.mark.asyncio
    async def test_output_truncation(self, repl_env_empty):
        """Output is truncated if too long."""
        repl_env_empty.config.max_output_chars = 100
        result = await repl_env_empty.execute_async("print('x' * 1000)")
        assert result.success
        assert (
            len(result.stdout) <= 100 + 50
        )  # Allow some margin for truncation message


class TestContextFilter:
    """Test that context filter is applied to all searches."""

    @pytest.mark.asyncio
    async def test_context_filter_applied_to_search(self, temp_db_path):
        """Search applies context filter automatically."""
        from unittest.mock import AsyncMock

        from haiku.rag.agents.rlm.dependencies import RLMContext
        from haiku.rag.agents.rlm.sandbox import REPLEnvironment
        from haiku.rag.client import HaikuRAG
        from haiku.rag.config.models import RLMConfig

        async with HaikuRAG(temp_db_path, create=True) as client:
            context = RLMContext(filter="uri LIKE '%medical%'")
            repl = REPLEnvironment(client=client, config=RLMConfig(), context=context)
            client.search = AsyncMock(return_value=[])

            await repl.execute_async("search('test query')")

            client.search.assert_called_once_with(
                "test query", limit=10, filter="uri LIKE '%medical%'"
            )

    @pytest.mark.asyncio
    async def test_context_filter_applied_to_list_documents(self, temp_db_path):
        """list_documents applies context filter automatically."""
        from unittest.mock import AsyncMock

        from haiku.rag.agents.rlm.dependencies import RLMContext
        from haiku.rag.agents.rlm.sandbox import REPLEnvironment
        from haiku.rag.client import HaikuRAG
        from haiku.rag.config.models import RLMConfig

        async with HaikuRAG(temp_db_path, create=True) as client:
            context = RLMContext(filter="title = 'Report'")
            repl = REPLEnvironment(client=client, config=RLMConfig(), context=context)
            client.list_documents = AsyncMock(return_value=[])

            await repl.execute_async("list_documents()")

            client.list_documents.assert_called_once_with(
                limit=10, offset=0, filter="title = 'Report'"
            )


class TestPreloadedDocuments:
    """Test pre-loaded documents context variable."""

    @pytest.mark.asyncio
    async def test_documents_variable_available_when_preloaded(self, temp_db_path):
        """documents variable is available when context.documents is set."""
        from haiku.rag.agents.rlm.dependencies import RLMContext
        from haiku.rag.agents.rlm.sandbox import REPLEnvironment
        from haiku.rag.client import HaikuRAG
        from haiku.rag.config.models import RLMConfig
        from haiku.rag.store.models import Document

        async with HaikuRAG(temp_db_path, create=True) as client:
            preloaded = [
                Document(
                    id="doc-1",
                    title="First Doc",
                    uri="test://first",
                    content="Content of first document about cats.",
                ),
                Document(
                    id="doc-2",
                    title="Second Doc",
                    uri="test://second",
                    content="Content of second document about dogs.",
                ),
            ]
            context = RLMContext(documents=preloaded)
            repl = REPLEnvironment(client=client, config=RLMConfig(), context=context)

            result = await repl.execute_async(
                "print(len(documents))\n"
                "print([d['title'] for d in documents])\n"
                "print('cats' in documents[0]['content'])"
            )
            assert result.success
            assert "2" in result.stdout
            assert "First Doc" in result.stdout
            assert "Second Doc" in result.stdout
            assert "True" in result.stdout

    @pytest.mark.asyncio
    async def test_documents_variable_not_available_without_preload(self, temp_db_path):
        """documents variable is not available when context.documents is None."""
        from haiku.rag.agents.rlm.dependencies import RLMContext
        from haiku.rag.agents.rlm.sandbox import REPLEnvironment
        from haiku.rag.client import HaikuRAG
        from haiku.rag.config.models import RLMConfig

        async with HaikuRAG(temp_db_path, create=True) as client:
            context = RLMContext()
            repl = REPLEnvironment(client=client, config=RLMConfig(), context=context)

            result = await repl.execute_async("print(documents)")
            assert not result.success
            assert "NameError" in result.stderr

    @pytest.mark.asyncio
    async def test_documents_has_expected_fields(self, temp_db_path):
        """documents variable contains expected dict fields."""
        from haiku.rag.agents.rlm.dependencies import RLMContext
        from haiku.rag.agents.rlm.sandbox import REPLEnvironment
        from haiku.rag.client import HaikuRAG
        from haiku.rag.config.models import RLMConfig
        from haiku.rag.store.models import Document

        async with HaikuRAG(temp_db_path, create=True) as client:
            preloaded = [
                Document(
                    id="doc-1",
                    title="Test Doc",
                    uri="test://doc",
                    content="Test content",
                ),
            ]
            context = RLMContext(documents=preloaded)
            repl = REPLEnvironment(client=client, config=RLMConfig(), context=context)

            result = await repl.execute_async(
                "d = documents[0]\n"
                "print(sorted(d.keys()))\n"
                "print(d['id'], d['title'], d['uri'])"
            )
            assert result.success
            assert "['content', 'id', 'title', 'uri']" in result.stdout
            assert "doc-1" in result.stdout
            assert "Test Doc" in result.stdout
            assert "test://doc" in result.stdout


class TestSandboxEscapeVectors:
    """Test that known sandbox escape techniques are blocked.

    Each test contains actual exploit code that would work without the fix.
    """

    @pytest.mark.asyncio
    async def test_type_dict_subclasses_escape_blocked(self, repl_env_empty):
        """Cannot escape via type.__dict__['__subclasses__'].

        Without fix: This would enumerate all loaded classes and find
        subprocess.Popen to execute arbitrary shell commands.
        """
        result = await repl_env_empty.execute_async("""
# EXPLOIT: Access __subclasses__ via dict to bypass AST check
subclasses_method = type.__dict__['__subclasses__']
all_classes = subclasses_method(object)
print(f"Found {len(all_classes)} classes")
""")
        assert not result.success
        assert "not allowed" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_popen_shell_execution_blocked(self, repl_env_empty):
        """Cannot execute shell commands via Popen.

        Without fix: This would execute 'whoami' and return the username.
        """
        result = await repl_env_empty.execute_async("""
# EXPLOIT: Find subprocess.Popen and execute shell commands
subclasses_method = type.__dict__['__subclasses__']
all_classes = subclasses_method(object)
popen = [c for c in all_classes if c.__name__ == 'Popen'][0]
proc = popen('whoami', shell=True, stdout=-1)
print(proc.stdout.read())
""")
        assert not result.success

    @pytest.mark.asyncio
    async def test_socket_creation_blocked(self, repl_env_empty):
        """Cannot create network sockets for data exfiltration.

        Without fix: This would create a socket that could connect to external servers.
        """
        result = await repl_env_empty.execute_async("""
# EXPLOIT: Find socket class and create network connection
subclasses_method = type.__dict__['__subclasses__']
all_classes = subclasses_method(object)
socket_cls = [c for c in all_classes if c.__name__ == 'socket'][0]
s = socket_cls(2, 1)  # AF_INET, SOCK_STREAM
print(f"Created socket: {s}")
""")
        assert not result.success

    @pytest.mark.asyncio
    async def test_type_three_arg_class_creation_blocked(self, repl_env_empty):
        """Cannot use type() with 3 arguments to create classes dynamically."""
        result = await repl_env_empty.execute_async(
            "EvilClass = type('EvilClass', (object,), {'x': 1})"
        )
        assert not result.success

    @pytest.mark.asyncio
    async def test_dict_key_dunder_access_blocked(self, repl_env_empty):
        """Cannot access dunder methods via dictionary key access."""
        result = await repl_env_empty.execute_async(
            "method = str.__dict__['__add__']\nprint(method)"
        )
        assert not result.success
        assert "not allowed" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_dict_key_private_access_blocked(self, repl_env_empty):
        """Cannot access private attributes via dictionary key access."""
        result = await repl_env_empty.execute_async(
            "method = object.__dict__['_private']\nprint(method)"
        )
        assert not result.success
        assert "not allowed" in result.stderr.lower()

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_sql_injection_in_get_document_blocked(self, temp_db_path):
        """SQL injection in get_document cannot bypass context filter.

        Without fix: Injecting quotes would leak documents that should be
        protected by the context filter.
        """
        from haiku.rag.agents.rlm.dependencies import RLMContext
        from haiku.rag.agents.rlm.sandbox import REPLEnvironment
        from haiku.rag.client import HaikuRAG
        from haiku.rag.config.models import RLMConfig

        async with HaikuRAG(temp_db_path, create=True) as client:
            # Create documents: one secret, one public
            await client.create_document(
                content="TOP SECRET: Launch codes 1234",
                uri="secret://classified",
                title="Classified Intel",
            )
            await client.create_document(
                content="Public weather report",
                uri="public://weather",
                title="Weather",
            )

            # Sandbox restricted to public:// only
            context = RLMContext(filter="uri LIKE 'public://%'")
            repl = REPLEnvironment(client=client, config=RLMConfig(), context=context)

            # EXPLOIT: SQL injection to access secret document
            result = await repl.execute_async("""
# Injection payload breaks out of quotes and adds OR clause
content = get_document("x' OR uri LIKE 'secret://%")
if content:
    print(f"LEAKED: {content}")
else:
    print("NO LEAK")
""")
            assert result.success
            assert "TOP SECRET" not in result.stdout
            assert "Launch codes" not in result.stdout


class TestSecurityEscapes:
    """Test that common security escape attempts are blocked."""

    @pytest.mark.asyncio
    async def test_eval_via_builtins_dict(self, repl_env_empty):
        """Cannot access eval through __builtins__."""
        result = await repl_env_empty.execute_async("__builtins__['eval']('1+1')")
        assert not result.success

    @pytest.mark.asyncio
    async def test_import_via_builtins(self, repl_env_empty):
        """Cannot import os through builtins trickery."""
        result = await repl_env_empty.execute_async("__builtins__.__import__('os')")
        assert not result.success

    @pytest.mark.asyncio
    async def test_class_bases_escape(self, repl_env_empty):
        """Cannot escape through __class__.__bases__."""
        result = await repl_env_empty.execute_async(
            "().__class__.__bases__[0].__subclasses__()"
        )
        assert not result.success

    @pytest.mark.asyncio
    async def test_code_object_escape(self, repl_env_empty):
        """Cannot create code objects."""
        result = await repl_env_empty.execute_async(
            "def f(): pass\n"
            "type(f.__code__)(0, 0, 0, 0, 0, 0, b'', (), (), (), '', '', 0, b'')"
        )
        assert not result.success

    @pytest.mark.asyncio
    async def test_import_system_escape(self, repl_env_empty):
        """Cannot escape through importlib."""
        result = await repl_env_empty.execute_async("import importlib")
        assert not result.success

    @pytest.mark.asyncio
    async def test_pickle_escape(self, repl_env_empty):
        """Cannot use pickle for code execution."""
        result = await repl_env_empty.execute_async("import pickle")
        assert not result.success
