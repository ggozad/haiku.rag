import logging
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload, runtime_checkable

from haiku.rag.config.models import HookConfig
from haiku.rag.store.models.chunk import SearchResult, SearchType
from haiku.rag.store.models.document import Document
from haiku.rag.telemetry import logfire

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument
    from PIL import Image as PILImage

    from haiku.rag.client import HaikuRAG

ENTRY_POINT_GROUP = "haiku.rag.hooks"

logger = logging.getLogger(__name__)

IngestOperation = Literal["create", "update"]
LifespanPoint = Literal["lifespan", "ingester_lifespan"]


class HookError(Exception):
    """A ``before_ingest`` hook raised. The hook's exception is the cause."""


class DocumentRejected(Exception):
    """Raised by a ``before_ingest`` hook to refuse a document. Ingestion sets
    ``uri`` to the refused document's."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason
        self.uri: str | None = None

    def __str__(self) -> str:
        if self.uri is None:
            return self.reason
        return f"{self.uri} rejected: {self.reason}"


@dataclass
class IngestEvent:
    """Documents whose content was written in one operation. Batch imports
    carry the whole batch in a single event."""

    documents: list[Document]
    operation: IngestOperation


@dataclass
class DeleteEvent:
    """Documents removed in one operation. A cascade delete carries the root
    and all its children in a single event. The documents no longer exist in
    the database; the models are the last-known state."""

    documents: list[Document]


@dataclass
class SearchRequest:
    """The parameters a search will run with. ``before_search`` hooks may
    modify ``query``, ``filter``, ``search_type``, and ``limit``."""

    query: "str | bytes | PILImage.Image"
    filter: str | None
    search_type: SearchType | None
    limit: int


@dataclass
class IngestRequest:
    """A fetched document about to be converted. ``before_ingest`` hooks may
    replace ``body`` and add keys to ``metadata``, which starts empty. The
    other fields describe the document. ``uri`` is the URI the document is
    stored under. ``source_id`` is the ingesting source, a PDF attachment's
    parent's included. An ad-hoc ingest passes the source recorded on the
    stored document, and ``None`` when there is none. "Metadata lifetime"
    in docs/hooks.md says how long the added keys last."""

    uri: str
    source_id: str | None
    body: bytes
    content_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConvertedDocument:
    """A document haiku.rag converted, about to be chunked. ``after_convert``
    hooks may change ``docling_document`` in place or replace it. ``uri`` and
    ``source_id`` are those ``before_ingest`` saw. ``content_type`` is ``None``
    for text given to ``create_document`` or ``update_document``."""

    uri: str | None
    source_id: str | None
    content_type: str | None
    docling_document: "DoclingDocument"


class Hook:
    """Base class for client lifecycle hooks. Subclasses override any subset.

    A package registers a factory under the ``haiku.rag.hooks`` entry-point
    group; ``config.hooks`` lists the hooks to activate, each with the keyword
    arguments its factory receives, and they run in the listed order at every
    hook point. Hooks receive the ``HaikuRAG`` client, so they may search, read repositories, or keep their own state in
    the database via ``client.store`` (table names must use the ``hook_``
    prefix to stay clear of core tables and migrations).
    """

    @asynccontextmanager
    async def lifespan(self, client: "HaikuRAG") -> AsyncGenerator[None]:
        """Hold whatever resources the hook needs for as long as the client is
        open. Factories stay resource-free; acquire here instead.

        Entered once the store is open, in configured order, and exited in
        reverse order while the store, embedder and reranker are still usable.
        Raising on entry fails client entry and unwinds the hooks already
        started. Raising on exit is logged and swallowed, and a hook can never
        suppress an exception raised by the client's caller.

        A hook running background work owns stopping it. Cancel or signal the
        tasks before whatever awaits them: a clean ``asyncio.TaskGroup`` exit
        waits for its children without cancelling them, so an endless task
        parked in one hangs shutdown instead of ending it.

        Every process that opens a client enters it, readers included. Start
        background work in ``ingester_lifespan``.
        """
        yield

    @asynccontextmanager
    async def ingester_lifespan(self, client: "HaikuRAG") -> AsyncGenerator[None]:
        """Run background work while haiku-ingester ingests into the client's
        database.

        Only haiku-ingester enters it, once per database, on that database's
        writable client: inside ``lifespan``, after reconciliation and before
        the first job. It exits after the last job, so every ``after_ingest``
        and ``after_delete`` from the ingester's jobs runs inside it. Failures
        follow ``lifespan``'s rules, and a failure on entry fails the
        database's start.
        """
        yield

    async def before_ingest(
        self, client: "HaikuRAG", request: IngestRequest
    ) -> IngestRequest:
        """Transform a fetched document before conversion. Runs on every
        fetch, unchanged bytes included, and once per PDF attachment, before
        the unchanged-bytes check, so the stored ``md5`` is that of the
        returned body. Added metadata keys win over a metadata provider's and a
        source's, and the keys
        ingestion owns (``content_type``, ``md5``, ``source_revision``,
        ``source_id``, ``parent_uri``) are dropped. Raise ``DocumentRejected``
        to refuse the document. Any other exception fails the ingest as a
        ``HookError``."""
        return request

    async def after_convert(
        self, client: "HaikuRAG", document: ConvertedDocument
    ) -> ConvertedDocument:
        """Transform a document haiku.rag converted, before it is chunked and
        embedded. Runs for sources, PDF attachments, a FULL rebuild (its
        fallback to stored content included), and text given to
        ``create_document`` or ``update_document``. Documents the
        caller converted (``import_document(s)``, ``update_document`` with a
        ``docling_document``) and ``rebuild --rechunk`` skip it. Any exception
        fails the ingest as a ``HookError``."""
        return document

    async def after_ingest(self, client: "HaikuRAG", event: IngestEvent) -> None:
        """Content was written for ``event.documents``. ``event.operation``
        is ``"create"`` for new documents and ``"update"`` when an existing
        document's content was rewritten (including creation against an
        already-stored URI). Replace any state derived from the documents
        regardless of the operation: even a creation may be a retry.
        Metadata/title-only updates do not fire, so a hook may store derived
        keys with a metadata-only ``update_document``. "Metadata lifetime" in
        docs/hooks.md says how long they last. A PDF's event fires after its
        attachments', once its ``md5`` is stored.

        Best-effort observer: the operation has already committed, so
        exceptions are logged and never raised, and subsequent hooks still
        run. Correctness-critical derived state needs its own
        reconciliation. Post-commit hooks are not a supported transformation
        point: mutating event models does not alter the committed record,
        and explicit client writes are separate operations, not atomic with
        the original write."""

    async def after_delete(self, client: "HaikuRAG", event: DeleteEvent) -> None:
        """``event.documents`` were deleted; cascades arrive as one event.
        Best-effort observer with the same contract as ``after_ingest``."""

    async def before_search(
        self, client: "HaikuRAG", request: SearchRequest
    ) -> SearchRequest:
        """Transform the search parameters before retrieval. Text queries
        only; the returned request's query feeds both the vector and FTS
        sides."""
        return request

    async def after_search(
        self,
        client: "HaikuRAG",
        request: SearchRequest,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Transform or annotate search results before they are returned.
        ``request`` reflects any ``before_search`` transformations, and its
        ``search_type`` is the one retrieval ran with: ``"hybrid"`` where a
        text search was left unset, ``"vector"`` for an image query."""
        return results


def hook_name(hook: Hook) -> str:
    cls = type(hook)
    return f"{cls.__module__}.{cls.__qualname__}"


def overrides(hook: Hook, point: str) -> bool:
    """Whether ``hook``'s class replaces Hook's ``point``."""
    return getattr(type(hook), point, None) is not getattr(Hook, point)


def hook_span(point: str, hook: Hook, **attributes: Any):
    # logfire names a span after an f-string argument's template.
    name = f"hook.{point}"
    return logfire.span(name, hook=hook_name(hook), **attributes)


def _lifespan_exit(
    hook: Hook, lifespan: AbstractAsyncContextManager[None], point: LifespanPoint
):
    """Wrap a started lifespan's exit so a teardown failure is logged rather
    than raised, and so the hook cannot suppress the caller's exception."""

    async def _exit(
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        with hook_span(f"{point}.exit", hook) as span:
            try:
                await lifespan.__aexit__(exc_type, exc, tb)
            except Exception as e:
                span.record_exception(e)
                span.set_level("error")
                logger.exception("%s %s teardown failed", hook_name(hook), point)
        return False

    return _exit


async def enter_lifespans(
    stack: AsyncExitStack,
    hooks: Sequence[Hook],
    client: "HaikuRAG",
    point: LifespanPoint = "lifespan",
) -> None:
    """Start each hook's ``point`` on ``stack`` in order. A hook that fails to
    start propagates, leaving the earlier hooks registered on the stack for the
    caller to unwind."""
    for hook in hooks:
        if not overrides(hook, point):
            continue
        lifespan = getattr(hook, point)(client)
        with hook_span(f"{point}.enter", hook):
            await lifespan.__aenter__()
        stack.push_async_exit(_lifespan_exit(hook, lifespan, point))


@overload
async def notify(
    hooks: Sequence[Hook],
    method: Literal["after_ingest"],
    client: "HaikuRAG",
    event: IngestEvent,
) -> None: ...


@overload
async def notify(
    hooks: Sequence[Hook],
    method: Literal["after_delete"],
    client: "HaikuRAG",
    event: DeleteEvent,
) -> None: ...


async def notify(
    hooks: Sequence[Hook],
    method: Literal["after_ingest", "after_delete"],
    client: "HaikuRAG",
    event: IngestEvent | DeleteEvent,
) -> None:
    """Fire post-commit observer hooks best-effort: a hook failure is logged
    and never raised (the operation already committed), and subsequent hooks
    still run."""
    attributes: dict[str, Any] = {"documents": len(event.documents)}
    if isinstance(event, IngestEvent):
        attributes["operation"] = event.operation
    for hook in hooks:
        if not overrides(hook, method):
            continue
        with hook_span(method, hook, **attributes) as span:
            try:
                await getattr(hook, method)(client, event)
            except Exception as e:
                span.record_exception(e)
                span.set_level("error")
                logger.exception(
                    "%s.%s failed for documents %s",
                    hook_name(hook),
                    method,
                    [d.id for d in event.documents],
                )


HookFactory = Callable[..., Hook]


@runtime_checkable
class LoadableEntryPoint(Protocol):
    """The slice of ``importlib.metadata.EntryPoint`` ``build_hooks`` needs:
    a deferred ``load()`` returning the hook factory."""

    def load(self) -> HookFactory: ...


def load_hooks() -> dict[str, LoadableEntryPoint]:
    """Discover registered hook entry points, keyed by name. The entry points
    are not imported here; ``build_hooks`` loads only the ones the config
    references, so an unused hook with a broken import does not fail client
    construction."""
    return {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}


def build_hooks(
    configured: Sequence[str | HookConfig],
    discovered: Mapping[str, LoadableEntryPoint],
) -> list[Hook]:
    """Load and instantiate the configured hooks in order, passing each its
    options. Raises ValueError naming the hook for a name with no registered
    entry point, a module that fails to import, or a factory that fails."""
    hooks: list[Hook] = []
    for entry in configured:
        if isinstance(entry, str):
            entry = HookConfig(name=entry)
        name = entry.name
        try:
            entry_point = discovered[name]
        except KeyError:
            raise ValueError(
                f"Config references unknown hook {name!r}; no entry point "
                f"registered under {ENTRY_POINT_GROUP!r}."
            ) from None
        try:
            factory: HookFactory = entry_point.load()
            hooks.append(factory(**entry.options))
        except Exception as exc:
            raise ValueError(f"hook {name!r}: {exc}") from exc
    return hooks
