import logging
from contextlib import asynccontextmanager

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config import get_config
from haiku.rag.hooks import DocumentRejected, Hook, HookError
from haiku.rag.store.models.chunk import Chunk, SearchResult
from haiku.rag.telemetry import logfire
from tests.conftest import capture_logs, fake_ingest_fetch_result
from tests.documents import docling_doc
from tests.spans import Tree

ERROR_LEVEL = 17


class ObservingHook(Hook):
    async def after_ingest(self, client, event):
        pass

    async def after_delete(self, client, event):
        pass


class FailingObserverHook(Hook):
    async def after_ingest(self, client, event):
        raise RuntimeError("ingest hook boom")

    async def after_delete(self, client, event):
        raise RuntimeError("delete hook boom")


class FailingExitHook(Hook):
    @asynccontextmanager
    async def lifespan(self, client):
        yield
        raise RuntimeError("cannot stop")


class FailingEntryHook(Hook):
    @asynccontextmanager
    async def lifespan(self, client):
        raise RuntimeError("cannot start")
        yield


class RejectingHook(Hook):
    async def before_ingest(self, client, request):
        raise DocumentRejected("draft")


class RejectingAttachmentHook(Hook):
    async def before_ingest(self, client, request):
        if request.uri.endswith("#attachment=a.txt"):
            raise DocumentRejected("draft")
        return request


class RaisingIngestHook(Hook):
    async def before_ingest(self, client, request):
        raise RuntimeError("hook refused")


class PassThroughSearchHook(Hook):
    async def before_search(self, client, request):
        return request

    async def after_search(self, client, request, results):
        return results[1:]


class FailingAfterSearchHook(Hook):
    async def after_search(self, client, request, results):
        raise RuntimeError("search hook boom")


class SeedResultsHook(Hook):
    async def after_search(self, client, request, results):
        return [SearchResult(content=c, score=0.5) for c in ("a", "b", "c")]


def _hook_names(tree: Tree) -> list[str]:
    return [name for name in tree.names() if name.startswith("hook.")]


async def _import(client: HaikuRAG, uri: str = "mem://a"):
    dim = get_config().embeddings.model.vector_dim
    return await client.import_document(
        docling_doc("a", "Alpha body"),
        [Chunk(content="Alpha body", embedding=[0.1] * dim, order=0)],
        uri=uri,
        title="Alpha",
    )


async def _empty_repo_search(*args, **kwargs):
    return []


async def test_no_hooks_emit_no_hook_spans(temp_db_path, exporter):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await _import(client)
        assert doc.id is not None
        await client.delete_document(doc.id)
        client.chunk_repository.search = _empty_repo_search
        await client.search("alpha")

    assert _hook_names(Tree(exporter)) == []


async def test_lifespan_entry_and_exit_each_get_a_span(temp_db_path, exporter):
    client = HaikuRAG(temp_db_path, create=True)
    client._hooks = [Hook()]
    with logfire.span("caller"):
        async with client:
            pass

    tree = Tree(exporter)
    assert _hook_names(tree) == ["hook.lifespan.enter", "hook.lifespan.exit"]
    assert tree.parent_of("hook.lifespan.enter") == "caller"
    assert tree.parent_of("hook.lifespan.exit") == "caller"
    assert tree.attrs("hook.lifespan.enter")["hook"] == "haiku.rag.hooks.Hook"
    assert tree.attrs("hook.lifespan.exit")["hook"] == "haiku.rag.hooks.Hook"


async def test_a_lifespan_exit_failure_is_recorded_and_swallowed(
    temp_db_path, exporter
):
    client = HaikuRAG(temp_db_path, create=True)
    client._hooks = [FailingExitHook()]
    with capture_logs(logging.getLogger("haiku.rag.hooks"), logging.ERROR) as records:
        async with client:
            pass

    tree = Tree(exporter)
    assert tree.exceptions("hook.lifespan.exit") == ["RuntimeError"]
    assert tree.attrs("hook.lifespan.exit")["logfire.level_num"] == ERROR_LEVEL
    assert [r.getMessage() for r in records] == [
        "tests.test_hook_spans.FailingExitHook lifespan teardown failed"
    ]


async def test_a_lifespan_entry_failure_is_recorded_and_raised(temp_db_path, exporter):
    client = HaikuRAG(temp_db_path, create=True)
    client._hooks = [FailingEntryHook()]
    with pytest.raises(RuntimeError, match="cannot start"):
        async with client:
            pass

    assert Tree(exporter).exceptions("hook.lifespan.enter") == ["RuntimeError"]


async def test_observer_spans_nest_under_the_operation(temp_db_path, exporter):
    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [ObservingHook()]
        with logfire.span("caller"):
            doc = await _import(client)
            assert doc.id is not None
            await client.delete_document(doc.id)

    tree = Tree(exporter)
    assert tree.parent_of("hook.after_ingest") == "caller"
    assert tree.parent_of("hook.after_delete") == "caller"
    ingest = tree.attrs("hook.after_ingest")
    assert ingest["hook"] == "tests.test_hook_spans.ObservingHook"
    assert ingest["operation"] == "create"
    assert ingest["documents"] == 1
    delete = tree.attrs("hook.after_delete")
    assert delete["hook"] == "tests.test_hook_spans.ObservingHook"
    assert delete["documents"] == 1
    assert tree.exceptions("hook.after_ingest") == []


async def test_observer_failures_are_recorded_logged_and_swallowed(
    temp_db_path, exporter
):
    with capture_logs(logging.getLogger("haiku.rag.hooks"), logging.ERROR) as records:
        async with HaikuRAG(temp_db_path, create=True) as client:
            client._hooks = [FailingObserverHook()]
            doc = await _import(client)
            assert doc.id is not None
            assert await client.delete_document(doc.id) is True

    tree = Tree(exporter)
    for name in ("hook.after_ingest", "hook.after_delete"):
        assert tree.exceptions(name) == ["RuntimeError"]
        assert tree.attrs(name)["logfire.level_num"] == ERROR_LEVEL
    assert len(records) == 2


async def test_before_ingest_span_names_the_document(
    temp_db_path, tmp_path, exporter, monkeypatch
):
    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result", fake_ingest_fetch_result
    )
    path = tmp_path / "doc.md"
    path.write_bytes(b"# Notes\n\nBody.\n")

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [Hook()]
        await client.create_document_from_source(path)

    tree = Tree(exporter)
    attrs = tree.attrs("hook.before_ingest")
    assert attrs["hook"] == "haiku.rag.hooks.Hook"
    assert attrs["uri"] == path.as_uri()
    assert attrs["content_type"] == "text/markdown"
    assert "rejected" not in attrs
    assert tree.exceptions("hook.before_ingest") == []


async def test_a_rejection_is_an_outcome_on_the_span(temp_db_path, tmp_path, exporter):
    path = tmp_path / "doc.md"
    path.write_bytes(b"# Notes\n\nBody.\n")

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [RejectingHook()]
        with pytest.raises(DocumentRejected):
            await client.create_document_from_source(path)

    tree = Tree(exporter)
    attrs = tree.attrs("hook.before_ingest")
    assert attrs["rejected"] is True
    assert attrs["reason"] == "draft"
    assert tree.exceptions("hook.before_ingest") == []


async def test_a_before_ingest_failure_is_recorded_as_a_hook_error(
    temp_db_path, tmp_path, exporter
):
    path = tmp_path / "doc.md"
    path.write_bytes(b"# Notes\n\nBody.\n")

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [RaisingIngestHook()]
        with pytest.raises(HookError):
            await client.create_document_from_source(path)

    assert Tree(exporter).exceptions("hook.before_ingest") == [
        "haiku.rag.hooks.HookError"
    ]


async def test_search_hook_spans_count_results(temp_db_path, exporter):
    async with HaikuRAG(temp_db_path, create=True) as client:
        client.chunk_repository.search = _empty_repo_search
        client._hooks = [SeedResultsHook(), PassThroughSearchHook()]
        with logfire.span("caller"):
            results = await client.search("alpha")

    assert len(results) == 2
    tree = Tree(exporter)
    caller = tree.one("caller").context
    assert caller is not None
    hook_spans = [s for s in tree.spans if s.name.startswith("hook.")]
    assert len(hook_spans) == 4
    assert all(
        s.parent is not None and s.parent.span_id == caller.span_id for s in hook_spans
    )
    assert [
        (a["hook"], a["results_in"], a["results"])
        for a in tree.every("hook.after_search")
    ] == [
        ("tests.test_hook_spans.SeedResultsHook", 0, 3),
        ("tests.test_hook_spans.PassThroughSearchHook", 3, 2),
    ]
    assert [a["hook"] for a in tree.every("hook.before_search")] == [
        "tests.test_hook_spans.SeedResultsHook",
        "tests.test_hook_spans.PassThroughSearchHook",
    ]


async def test_an_after_search_failure_is_recorded_and_raised(temp_db_path, exporter):
    async with HaikuRAG(temp_db_path, create=True) as client:
        client.chunk_repository.search = _empty_repo_search
        client._hooks = [FailingAfterSearchHook()]
        with pytest.raises(RuntimeError, match="search hook boom"):
            await client.search("alpha")

    tree = Tree(exporter)
    attrs = tree.attrs("hook.after_search")
    assert attrs["results_in"] == 0
    assert "results" not in attrs
    assert tree.exceptions("hook.after_search") == ["RuntimeError"]


async def test_a_rejected_attachment_is_an_outcome_on_its_span(
    temp_db_path, exporter, monkeypatch
):
    from haiku.rag.client.documents import _reconcile_pdf_attachments
    from haiku.rag.store.models.document import Document
    from tests.conftest import build_pdf, writing

    monkeypatch.setattr(
        "haiku.rag.client.documents._ingest_fetch_result", fake_ingest_fetch_result
    )
    parent_uri = "file:///fixtures/parent.pdf"

    async with HaikuRAG(temp_db_path, create=True) as client:
        parent = await client.document_repository.create(
            Document(
                content="", uri=parent_uri, metadata={"content_type": "application/pdf"}
            )
        )
        client._hooks = [RejectingAttachmentHook()]
        with capture_logs(
            logging.getLogger("haiku.rag.client.documents"), logging.WARNING
        ) as records:
            await _reconcile_pdf_attachments(
                writing(client),
                client,
                parent,
                build_pdf([("a.txt", b"A"), ("b.txt", b"B")]),
                depth=0,
            )
        stored = await client.get_document_by_uri(f"{parent_uri}#attachment=b.txt")

    assert stored is not None
    assert [r.getMessage() for r in records] == [
        f"{parent_uri}#attachment=a.txt rejected: draft"
    ]
    tree = Tree(exporter)
    spans = {a["uri"]: a for a in tree.every("hook.before_ingest")}
    rejected = spans[f"{parent_uri}#attachment=a.txt"]
    assert rejected["rejected"] is True
    assert rejected["reason"] == "draft"
    assert "rejected" not in spans[f"{parent_uri}#attachment=b.txt"]
    assert not any(
        event.name == "exception"
        for span in tree.spans
        if span.name == "hook.before_ingest"
        for event in span.events
    )
