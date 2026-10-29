import asyncio
from pathlib import Path

import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.sandbox import AnalysisContext, Sandbox, SandboxResult
from haiku.rag.sandbox import sandbox as sandbox_module
from haiku.rag.sandbox.sandbox import CappedOutput
from haiku.rag.store.models.chunk import Chunk


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent / "cassettes" / "test_sandbox")


async def _import_text(
    client: HaikuRAG,
    config: AppConfig,
    uri: str,
    text: str = "Foxes and dogs.",
    chunk_metadata: dict | None = None,
    **document_fields,
):
    docling = DoclingDocument(name="d")
    docling.add_text(label=DocItemLabel.TEXT, text=text)
    chunk = Chunk(
        content=text,
        embedding=[0.1] * config.embeddings.model.vector_dim,
        order=0,
        metadata=chunk_metadata or {},
    )
    return await client.import_document(docling, [chunk], uri=uri, **document_fields)


class TestSandboxBasics:
    """Test basic sandbox functionality."""

    async def test_the_documented_modules_import(self, sandbox):
        """The modules the instructions and the MCP description promise."""
        result = await sandbox.execute(
            "import json, re, math, pathlib, datetime\n"
            "import collections, itertools, functools, dataclasses, decimal\n"
            "print(collections.Counter('aab').most_common(1),"
            " list(itertools.islice(itertools.count(), 2)),"
            " decimal.Decimal('0.1') + decimal.Decimal('0.2'))"
        )
        assert result.success, result.stderr
        assert "[('a', 2)] [0, 1] 0.3" in result.stdout

    async def test_sleep_returns_at_once(self, sandbox):
        """A program's `time.sleep` does not hold the worker."""
        loop = asyncio.get_running_loop()
        started = loop.time()
        result = await sandbox.execute("import time\ntime.sleep(5)\nprint('woke')")
        await sandbox.close()

        assert result.success, result.stderr
        assert loop.time() - started < 3

    async def test_execute_print_and_expression(self, sandbox):
        """Printed output and the value of the last expression both reach stdout."""
        result = await sandbox.execute("print('hello')\n42")
        assert isinstance(result, SandboxResult)
        assert result.success
        assert "hello" in result.stdout
        assert "42" in result.stdout
        assert result.stderr == ""


class TestSandboxErrors:
    """Test error handling in sandbox."""

    @pytest.mark.parametrize(
        ("code", "error"),
        [
            ("def foo(", ""),
            ("x = 1/0", "ZeroDivisionError"),
            ("print(undefined_variable)", "NameError"),
            ("import json, string", ""),
        ],
        ids=["syntax", "runtime", "name", "unsupported-module"],
    )
    async def test_errors_are_reported(self, sandbox, code, error):
        """A failing program reports failure and an error message in stderr."""
        result = await sandbox.execute(code)
        assert not result.success
        assert result.stderr != ""
        assert error in result.stderr


class TestSandboxListDocuments:
    """Test list_documents function in sandbox."""

    async def test_list_documents_empty(self, sandbox):
        """An empty database lists no documents, and its VFS reads without error."""
        result = await sandbox.execute(
            "from pathlib import Path\n"
            "Path('/documents').exists()\n"
            "docs = await list_documents()\n"
            "print(type(docs).__name__, len(docs))"
        )
        assert result.success, result.stderr
        assert "list 0" in result.stdout

    @pytest.mark.vcr()
    async def test_list_documents_with_data(self, temp_db_path):
        """list_documents returns documents when populated."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document(
                content="Test content",
                uri="test://doc1",
                title="Test Document",
            )

            context = AnalysisContext()
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            result = await sb.execute(
                "docs = await list_documents()\n"
                "print(len(docs))\n"
                "print(docs[0]['title'])\n"
                "print(docs[0]['source'])"
            )
            assert result.success
            assert "1" in result.stdout
            assert "Test Document" in result.stdout
            assert temp_db_path.stem in result.stdout

    async def test_list_documents_carries_metadata(self, temp_db_path):
        """Rows carry the document's metadata, so a corpus-wide pass over it is
        one call rather than a file read per document."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            await _import_text(
                client,
                config,
                "test://doc1",
                text="Test content",
                title="Test Document",
                metadata={"author": "Ada"},
            )

        sb = Sandbox(db_path=temp_db_path, config=config, context=AnalysisContext())
        try:
            result = await sb.execute(
                "docs = await list_documents()\nprint(docs[0]['metadata']['author'])"
            )
        finally:
            await sb.close()
        assert result.success, result.stderr
        assert "Ada" in result.stdout


class TestSandboxSearch:
    """Test search function in sandbox."""

    @pytest.mark.vcr()
    async def test_search_with_data(self, temp_db_path):
        """Test search function works."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document(
                content="The quick brown fox jumps over the lazy dog.",
                uri="test://animals",
                title="Animals",
            )

            context = AnalysisContext()
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            result = await sb.execute(
                "results = await search('fox', limit=5)\n"
                "print(len(results))\n"
                "r = results[0]\n"
                "print('fox' in r['content'].lower())\n"
                "print(type(r['content']).__name__)\n"
                "print(type(r['doc_item_refs']).__name__)\n"
                "print(type(r['labels']).__name__)"
            )
            assert result.success, result.stderr
            assert result.stdout.splitlines() == ["1", "True", "str", "list", "list"]
            found = sb.search_results
            assert isinstance(found, tuple)
            assert found and all(item.chunk_id for item in found)
            assert "fox" in found[0].content.lower()
            with pytest.raises(AttributeError):
                setattr(sb, "search_results", ())

    async def test_search_returns_the_matched_chunks_metadata(
        self, temp_db_path, monkeypatch
    ):
        """Results carry the stored metadata of the chunk that matched, custom
        keys included."""
        from haiku.rag.embeddings import EmbedderWrapper

        config = AppConfig()
        dim = config.embeddings.model.vector_dim

        async def embed_query(self, text):
            return [0.1] * dim

        monkeypatch.setattr(EmbedderWrapper, "embed_query", embed_query)
        async with HaikuRAG(temp_db_path, create=True) as client:
            await _import_text(
                client,
                config,
                "test://paras",
                text="Paragraph fourteen.",
                chunk_metadata={"para_no": "14"},
            )

        sb = Sandbox(db_path=temp_db_path, config=config, context=AnalysisContext())
        try:
            result = await sb.execute(
                "results = await search('fourteen', limit=1)\n"
                "print(results[0]['chunk_meta']['para_no'])"
            )
        finally:
            await sb.close()
        assert result.success, result.stderr
        assert "14" in result.stdout


class TestSandboxExternalFunctionEdgeCases:
    """Test edge cases in external function dispatch."""

    async def test_unknown_external_function(self, sandbox):
        """Test that calling an unregistered external function resumes with KeyError."""
        original_build = sandbox._build_external_functions

        def patched_build():
            fns = original_build()
            fns["search"] = None
            return fns

        sandbox._build_external_functions = patched_build

        result = await sandbox.execute(
            "try:\n"
            "    await search('hello')\n"
            "except:\n"
            "    print('caught')\n"
            "print('done')"
        )
        assert result.success
        assert "caught" in result.stdout
        assert "done" in result.stdout

    async def test_external_function_raises_exception(self, sandbox):
        """Test that exceptions from async external functions surface as errors.

        With run_monty_async, exceptions from async external functions
        propagate as MontyRuntimeError rather than being catchable inside
        Monty's try/except.
        """
        original_build = sandbox._build_external_functions

        def patched_build():
            fns = original_build()

            async def failing_search(*args, **kwargs):
                raise ValueError("external error")

            fns["search"] = failing_search
            return fns

        sandbox._build_external_functions = patched_build

        result = await sandbox.execute("await search('hello')")
        assert not result.success
        assert "external error" in result.stderr

    async def test_a_failing_search_keeps_its_message_for_the_program(
        self, sandbox, monkeypatch
    ):
        """A host-side failure inside search() reaches the program with its
        message, which the agent reads to repair its code."""

        async def boom(self, *args, **kwargs):
            raise ValueError("failed at /secret/path")

        monkeypatch.setattr(HaikuRAG, "search", boom)

        result = await sandbox.execute("await search('hello')")

        assert not result.success
        assert "ValueError: failed at /secret/path" in result.stderr


class TestSandboxOutputTruncation:
    """Test output truncation behavior."""

    async def test_truncate_stdout_on_runtime_error(self, temp_db_path):
        """Test stdout is truncated when a runtime error occurs after large output."""
        async with HaikuRAG(temp_db_path, create=True):
            config = AppConfig()
            config.sandbox.max_output_chars = 20
            context = AnalysisContext()
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            result = await sb.execute("print('a' * 100)\nx = 1/0")
            assert not result.success
            assert "ZeroDivisionError" in result.stderr
            assert result.stdout.endswith("... (output truncated)")
            assert len(result.stdout) < 100

    async def test_many_small_prints_past_the_cap_still_report_truncation(
        self, temp_db_path
    ):
        """No single print exceeds the cap, their sum does."""
        async with HaikuRAG(temp_db_path, create=True):
            config = AppConfig()
            config.sandbox.max_output_chars = 20
            context = AnalysisContext()
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            result = await sb.execute("for i in range(100):\n    print('c' * 5)")
            assert result.success
            assert result.stdout.startswith("c" * 5)
            assert result.stdout.endswith("... (output truncated)")
            assert len(result.stdout) == 20 + len("\n... (output truncated)")


class TestCappedOutput:
    def test_holds_at_most_the_cap(self):
        out = CappedOutput(10)
        for _ in range(1000):
            out.write("stdout", "abcdef")
        assert sum(len(p) for p in out.parts) == 10
        assert len(out.parts) == 2
        assert out.text() == "abcdefabcd\n... (output truncated)"

    def test_short_output_is_returned_whole(self):
        out = CappedOutput(10)
        out.write("stdout", "abc")
        out.write("stdout", "def")
        assert out.text() == "abcdef"


class TestSandboxVFS:
    """Test virtual filesystem for document access."""

    async def test_chunks_jsonl(self, temp_db_path):
        """chunks.jsonl lists a document's chunks in order with their stored
        metadata; a chunk found by its metadata leads to its items through
        their chunk_ids."""
        config = AppConfig()
        dim = config.embeddings.model.vector_dim
        docling = DoclingDocument(name="d")
        docling.add_text(label=DocItemLabel.TEXT, text="Paragraph thirteen.")
        docling.add_text(label=DocItemLabel.TEXT, text="Paragraph fourteen.")
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.import_document(
                docling,
                [
                    Chunk(
                        content="Paragraph thirteen.",
                        embedding=[0.1] * dim,
                        order=0,
                        metadata={"para_no": "13", "doc_item_refs": ["#/texts/0"]},
                    ),
                    Chunk(
                        content="Paragraph fourteen.",
                        embedding=[0.1] * dim,
                        order=1,
                        metadata={"para_no": "14", "doc_item_refs": ["#/texts/1"]},
                    ),
                ],
                uri="test://paras",
            )

        sb = Sandbox(db_path=temp_db_path, config=config, context=AnalysisContext())
        try:
            result = await sb.execute(
                "from pathlib import Path\n"
                "import json\n"
                f"root = Path('/documents/{doc.id}')\n"
                "def rows(name):\n"
                "    return [json.loads(l) for l in (root / name).read_text().strip().split('\\n')]\n"
                "chunks = rows('chunks.jsonl')\n"
                "print(len(chunks))\n"
                "hit = [c for c in chunks if c['metadata'].get('para_no') == '14']\n"
                "print(len(hit))\n"
                "items = rows('items.jsonl')\n"
                "print([i['text'] for i in items if hit[0]['chunk_id'] in i['chunk_ids']])"
            )
        finally:
            await sb.close()
        assert result.success, result.stderr
        assert result.stdout.splitlines() == ["2", "1", "['Paragraph fourteen.']"]

    async def test_relative_paths_resolve_under_documents(self, temp_db_path):
        """The working directory is `/documents`, which holds one directory per
        document."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await _import_text(client, config, "test://relative")

        sb = Sandbox(db_path=temp_db_path, config=config, context=AnalysisContext())
        try:
            result = await sb.execute(
                "from pathlib import Path\n"
                "print([(p.name, p.is_dir()) for p in Path('.').iterdir()])\n"
                f"print(open('{doc.id}/content.txt').read())"
            )
        finally:
            await sb.close()

        assert result.success, result.stderr
        assert result.stdout.splitlines() == [
            f"[('{doc.id}', True)]",
            "Foxes and dogs.",
        ]

    @pytest.mark.parametrize(
        ("filename", "write"),
        [
            ("content.txt", "Path(path).write_text('nope')"),
            ("items.jsonl", "Path(path).write_text('nope')"),
            ("chunks.jsonl", "Path(path).write_text('nope')"),
            ("toc.json", "Path(path).write_text('nope')"),
            ("metadata.json", "Path(path).write_text('nope')"),
            ("content.txt", "with open(path, 'w') as f: f.write('nope')"),
        ],
    )
    async def test_write_denied_for_every_document_file(
        self, temp_db_path, filename, write
    ):
        """Every file in the document VFS is read-only to `Path.write_text` and to
        `open()` in write mode."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await _import_text(client, config, "test://readonly")

        sb = Sandbox(db_path=temp_db_path, config=config, context=AnalysisContext())
        try:
            result = await sb.execute(
                "from pathlib import Path\n"
                f"path = '/documents/{doc.id}/{filename}'\n"
                "try:\n"
                f"    {write}\n"
                "    print('WROTE')\n"
                "except PermissionError:\n"
                "    print('DENIED')"
            )
            assert result.success, result.stderr
            assert "DENIED" in result.stdout
            assert "WROTE" not in result.stdout
        finally:
            await sb.close()

    async def test_open_file_objects_iterate_by_line(self, temp_db_path):
        """`for line in open(...)` reads a mounted document line by line."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await _import_text(client, config, "test://lines", text="one\ntwo")

        sb = Sandbox(db_path=temp_db_path, config=config, context=AnalysisContext())
        try:
            path = f"/documents/{doc.id}/content.txt"
            result = await sb.execute(
                f"lines = [line for line in open('{path}')]\n"
                f"print(len(lines), lines == open('{path}').readlines())"
            )
            assert result.success, result.stderr
            assert result.stdout.strip() == "2 True"
        finally:
            await sb.close()

    @pytest.mark.vcr()
    async def test_context_filter_limits_vfs(self, temp_db_path):
        """Context filter restricts which documents appear in VFS."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document(
                content="Public content",
                uri="public://doc1",
                title="Public Doc",
            )
            await client.create_document(
                content="Private content",
                uri="private://doc2",
                title="Private Doc",
            )

            context = AnalysisContext(filter="uri LIKE 'public://%'")
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            result = await sb.execute(
                "from pathlib import Path\n"
                "import json\n"
                "dirs = list(Path('/documents').iterdir())\n"
                "print(len(dirs))\n"
                "meta = json.loads((dirs[0] / 'metadata.json').read_text())\n"
                "print(meta['title'])"
            )
            assert result.success
            assert "1" in result.stdout
            assert "Public Doc" in result.stdout
            assert "Private Doc" not in result.stdout


class TestSandboxHeldConnection:
    """VFS reads must work while another connection to the same DB stays open.

    Mirrors the analysis-capability lifespan, which keeps a read-only connection open
    for the whole turn while sandboxed code reads the document VFS.
    """

    @pytest.mark.vcr()
    async def test_vfs_reads_while_connection_held(self, temp_db_path):
        """All three VFS readers work while another connection to the DB is open."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.create_document(
                content="Foxes and dogs roam the quiet hills.",
                uri="test://doc",
                title="Doc",
            )
        assert doc.id

        async with HaikuRAG(temp_db_path, config=config, read_only=True):
            context = AnalysisContext()
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            try:
                result = await sb.execute(
                    "from pathlib import Path\n"
                    f"d = Path('/documents/{doc.id}')\n"
                    "print('foxes' in (d / 'content.txt').read_text().lower())\n"
                    "print(len((d / 'items.jsonl').read_text().strip().split('\\n')))\n"
                    "print('tree' in (d / 'toc.json').read_text())"
                )
                assert result.success, result.stderr
                lines = result.stdout.strip().split("\n")
                assert lines[0] == "True"
                assert int(lines[1]) > 0
                assert lines[2] == "True"
            finally:
                await sb.close()

    @pytest.mark.vcr()
    async def test_vfs_read_concurrent_with_connection_read(self, temp_db_path):
        """A VFS read and a direct read on the shared connection run together.

        Both serialize through the shared lock, so concurrent tasks never have
        two operations in flight on the one connection at once.
        """
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.create_document(
                content="Foxes and dogs roam the quiet hills.",
                uri="test://doc",
                title="Doc",
            )
        assert doc.id

        async with HaikuRAG(temp_db_path, config=config, read_only=True) as rag:
            lock = asyncio.Lock()
            sb = Sandbox(
                db_path=temp_db_path,
                config=config,
                context=AnalysisContext(),
                rag=rag,
                lock=lock,
            )
            read_code = (
                "from pathlib import Path\n"
                f"print(Path('/documents/{doc.id}/content.txt').read_text())"
            )

            async def direct_read() -> str | None:
                async with lock:
                    return await rag.document_repository.get_content(doc.id)

            exec_result, content = await asyncio.gather(
                sb.execute(read_code),
                direct_read(),
            )
            assert exec_result.success, exec_result.stderr
            assert "Foxes" in exec_result.stdout
            assert content and "Foxes" in content

    @pytest.mark.vcr()
    async def test_vfs_reads_repeatable_across_executes(self, temp_db_path):
        """Repeated VFS reads across execute() calls return consistent content."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.create_document(
                content="Foxes and dogs.",
                uri="test://doc",
                title="Doc",
            )
        assert doc.id

        context = AnalysisContext()
        sb = Sandbox(db_path=temp_db_path, config=config, context=context)
        try:
            read = (
                "from pathlib import Path\n"
                f"print(Path('/documents/{doc.id}/content.txt').read_text())"
            )
            first = await sb.execute(read)
            second = await sb.execute(read)
            assert first.success and second.success
            assert "Foxes" in first.stdout
            assert first.stdout == second.stdout
        finally:
            await sb.close()

    @pytest.mark.vcr()
    async def test_variables_persist_across_executes_after_vfs_read(self, temp_db_path):
        """REPL state persists across execute() calls, including after a VFS read."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.create_document(
                content="Foxes and dogs.",
                uri="test://doc",
                title="Doc",
            )
        assert doc.id

        context = AnalysisContext()
        sb = Sandbox(db_path=temp_db_path, config=config, context=context)
        try:
            first = await sb.execute(
                "from pathlib import Path\n"
                f"x = len(Path('/documents/{doc.id}/content.txt').read_text())"
            )
            assert first.success, first.stderr
            second = await sb.execute("print(x)")
            assert second.success, second.stderr
            assert int(second.stdout.strip()) > 0
        finally:
            await sb.close()

    @pytest.mark.vcr()
    async def test_close_is_idempotent_after_vfs_read(self, temp_db_path):
        """close() is idempotent before any code has run and after a VFS read."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.create_document(
                content="Foxes and dogs.",
                uri="test://doc",
                title="Doc",
            )
        assert doc.id

        context = AnalysisContext()
        sb = Sandbox(db_path=temp_db_path, config=config, context=context)
        await sb.close()
        await sb.close()
        result = await sb.execute(
            "from pathlib import Path\n"
            f"print(Path('/documents/{doc.id}/content.txt').read_text())"
        )
        assert result.success
        await sb.close()
        await sb.close()


class TestSandboxReadDeadline:
    """The VFS bridge suspends the worker for the length of a read, and Monty's
    feed limit does not count a read in flight. The sandbox enforces the
    deadline itself, before each read."""

    async def test_the_deadline_covers_reads_from_memory_and_in_code_calls(
        self, temp_db_path, monkeypatch
    ):
        """Once a call's time is up, a file served from memory and an in-code
        listing are refused like a database read. A slow first read spends the
        budget, since the feed limit does not count time spent waiting on the
        host. A program that catches both refusals and ends still reports
        failure."""
        config = AppConfig()
        config.sandbox.code_timeout = 1.0
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await _import_text(client, config, "test://deadline-paths")
            repository = type(client.document_repository)

        async def slow_content(self, *args, **kwargs):
            await asyncio.sleep(1.3)
            return "body"

        monkeypatch.setattr(repository, "get_content", slow_content)
        sb = Sandbox(db_path=temp_db_path, config=config, context=AnalysisContext())
        try:
            result = await sb.execute(
                "from pathlib import Path\n"
                f"root = Path('/documents/{doc.id}')\n"
                "print(len((root / 'content.txt').read_text()))\n"
                "try:\n"
                "    (root / 'metadata.json').read_text()\n"
                "    print('static: read')\n"
                "except Exception as e:\n"
                "    print('static:', type(e).__name__)\n"
                "try:\n"
                "    await list_documents()\n"
                "    print('call: listed')\n"
                "except Exception as e:\n"
                "    print('call:', type(e).__name__)\n"
                "print('done')"
            )
        finally:
            await sb.close()

        assert "static: TimeoutError" in result.stdout
        assert "call: TimeoutError" in result.stdout
        assert "done" in result.stdout
        assert not result.success
        assert "time limit exceeded" in result.stderr

    async def test_a_program_swallowing_the_deadline_is_stopped(
        self, temp_db_path, monkeypatch
    ):
        """Reads refused past the deadline cost the worker no compute, so a loop
        that catches the refusal outlives the feed limit. The sandbox stops it."""
        monkeypatch.setattr(sandbox_module, "_DEADLINE_GRACE_SECONDS", 0.5)
        config = AppConfig()
        config.sandbox.code_timeout = 1.0
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await _import_text(client, config, "test://loop")
        sb = Sandbox(db_path=temp_db_path, config=config, context=AnalysisContext())
        try:
            result = await asyncio.wait_for(
                sb.execute(
                    "from pathlib import Path\n"
                    f"p = Path('/documents/{doc.id}/metadata.json')\n"
                    "print('start')\n"
                    "while True:\n"
                    "    try:\n"
                    "        p.read_text()\n"
                    "    except Exception:\n"
                    "        pass\n"
                ),
                timeout=10,
            )
            after = await sb.execute("print('fresh')")
        finally:
            await sb.close()

        assert not result.success
        assert "start" in result.stdout
        assert "time limit exceeded" in result.stderr
        assert "The interpreter restarted" in result.stderr
        assert after.success and after.stdout.strip() == "fresh"

    async def test_read_after_deadline_raises_without_scheduling(self, sandbox):
        """A read attempted past the deadline fails instead of querying."""
        scheduled = False

        async def _never_runs():
            nonlocal scheduled
            scheduled = True

        sandbox._loop = asyncio.get_running_loop()
        sandbox._deadline = sandbox._loop.time() - 1.0

        coro = _never_runs()
        with pytest.raises(TimeoutError, match="time limit exceeded"):
            sandbox._run_on_loop(coro)

        coro.close()
        assert scheduled is False

    async def test_each_call_gets_the_whole_time_limit(self, temp_db_path):
        """A call that spends most of `code_timeout` leaves the next call its own."""
        config = AppConfig()
        config.sandbox.code_timeout = 1.0
        async with HaikuRAG(temp_db_path, create=True):
            pass
        burn = "import time\nt0 = time.monotonic()\nwhile time.monotonic() - t0 < 0.7:\n    pass\nprint('done')"
        sb = Sandbox(db_path=temp_db_path, config=config, context=AnalysisContext())
        try:
            results = [await sb.execute(burn) for _ in range(3)]
        finally:
            await sb.close()

        assert all(r.success for r in results), [r.stderr for r in results]

    def test_host_calls_are_not_capped(self, temp_db_path):
        cap = Sandbox(
            db_path=temp_db_path, config=AppConfig(), context=AnalysisContext()
        )._session_limits()["max_suspensions"]
        assert cap is not None
        assert cap >= 1_000_000

    async def test_a_sandbox_built_for_one_call_ignores_the_execution_budget(
        self, temp_db_path
    ):
        """The MCP tool builds one sandbox per call; a capability budget of zero
        executions is not its concern."""
        config = AppConfig()
        config.qa.max_executions = 0
        async with HaikuRAG(temp_db_path, create=True):
            pass
        sb = Sandbox(db_path=temp_db_path, config=config, context=AnalysisContext())
        try:
            result = await sb.execute("print(6 * 7)")
        finally:
            await sb.close()

        assert result.success, result.stderr
        assert "42" in result.stdout

    @pytest.mark.slow
    async def test_a_program_may_read_more_than_a_thousand_times(self, temp_db_path):
        """Monty caps host callbacks per checkout at 1000 unless told otherwise;
        a corpus-wide pass over documents reads far more than that."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await _import_text(client, config, "test://many-reads")

        sb = Sandbox(db_path=temp_db_path, config=config, context=AnalysisContext())
        try:
            result = await sb.execute(
                "from pathlib import Path\n"
                f"p = Path('/documents/{doc.id}/content.txt')\n"
                "n = 0\n"
                "for i in range(1100):\n"
                "    n += len(p.read_text())\n"
                "print(n)"
            )
        finally:
            await sb.close()

        assert result.success, result.stderr
        assert result.stdout.strip() == str(1100 * len("Foxes and dogs."))


class TestSandboxWorkerCrash:
    """A dead worker must not poison every later call in the run."""

    async def test_crashed_worker_is_replaced(self, temp_db_path):
        """The crash fails one call. The next call gets a fresh session."""
        import os
        import signal
        import sys

        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True):
            pass

        sb = Sandbox(db_path=temp_db_path, config=config, context=AnalysisContext())
        try:
            first = await sb.execute("x = 1\nprint(x)")
            assert first.success, first.stderr
            assert sb._session is not None
            pid = sb._session.worker_pid
            assert pid is not None

            # On Windows os.kill terminates the process for any signal.
            os.kill(pid, signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL)

            crashed = await sb.execute("print(2)")
            assert crashed.success is False
            assert "restarted" in crashed.stderr

            # Without the discard every later call fails on the dead session.
            recovered = await sb.execute("print(3)")
            assert recovered.success, recovered.stderr
            assert "3" in recovered.stdout

            # The replacement worker starts empty, which the failure said.
            lost = await sb.execute("print(x)")
            assert lost.success is False
        finally:
            await sb.close()


class TestSandboxTimeLimit:
    """Monty's feed limit bounds a call that never reads."""

    async def test_runaway_compute_is_stopped_and_the_next_call_recovers(
        self, temp_db_path
    ):
        """Code that never reads escapes the read deadline. The feed limit stops it,
        and the session it leaves behind is replaced."""
        config = AppConfig()
        config.sandbox.code_timeout = 1.0
        async with HaikuRAG(temp_db_path, create=True):
            pass

        sb = Sandbox(db_path=temp_db_path, config=config, context=AnalysisContext())
        try:
            await sb.execute("x = 1")
            runaway = await sb.execute(
                "x = 0\nfor i in range(500000000):\n    x += i\nprint(x)"
            )
            assert runaway.success is False
            assert "restarted" in runaway.stderr

            recovered = await sb.execute("print('alive')")
            assert recovered.success, recovered.stderr
            assert "alive" in recovered.stdout

            lost = await sb.execute("print(x)")
            assert lost.success is False
            assert "NameError" in lost.stderr
        finally:
            await sb.close()

    async def test_a_refused_read_keeps_the_session(self, temp_db_path, monkeypatch):
        """The read deadline's TimeoutError fails the call and is answered by the
        host, so the session's variables survive it."""

        def _past_deadline(self, coro):
            coro.close()
            raise self._time_limit()

        monkeypatch.setattr(Sandbox, "_run_on_loop", _past_deadline)
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await _import_text(client, config, "test://refused-keeps-session")

        sb = Sandbox(db_path=temp_db_path, config=config, context=AnalysisContext())
        try:
            await sb.execute("x = 1")
            refused = await sb.execute(
                f"open('/documents/{doc.id}/content.txt').read()"
            )
            assert refused.success is False
            assert "time limit exceeded" in refused.stderr
            assert "no further document reads" in refused.stderr
            assert "restarted" not in refused.stderr

            kept = await sb.execute("print(x)")
            assert kept.success, kept.stderr
        finally:
            await sb.close()


class TestSandboxClose:
    async def test_a_failing_teardown_still_releases_the_rest(self, tmp_path):
        """Each of the session, the pool and the held connection is released,
        whichever of them fails."""
        from unittest.mock import AsyncMock

        sb = Sandbox(
            db_path=tmp_path / "x.lancedb",
            config=AppConfig(),
            context=AnalysisContext(),
        )
        session, pool, opened = AsyncMock(), AsyncMock(), AsyncMock()
        session.__aexit__.side_effect = RuntimeError("worker already gone")
        sb._session, sb._pool = session, pool
        sb._opened = opened

        await sb.close()

        pool.__aexit__.assert_awaited_once()
        opened.__aexit__.assert_awaited_once()
        assert sb._session is None
        assert sb._pool is None
        assert sb._opened is None
