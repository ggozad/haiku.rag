import base64
from dataclasses import dataclass, field, replace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import Agent, DeferredToolResults, RunContext
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from haiku.rag.capabilities.compaction import (
    RECEIPT,
    Capsule,
    compact_history,
    picture_label,
)
from haiku.rag.capabilities.compaction import create_capability as create_compaction
from haiku.rag.capabilities.ledger import CapabilityEvidenceRecord
from haiku.rag.capabilities.rag import RAGCapability, RAGState
from haiku.rag.capabilities.rag import create_capability as create_rag
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models.chunk import SearchResult

OWNED = frozenset({"rag_search"})
PNG = BinaryContent(data=b"fake-image-bytes", media_type="image/png")


def retrieved_image(chunk_id: str = "chunk-1", self_ref: str = "#/pictures/0"):
    """A page image on the wire, labelled the way a search result attaches it."""
    return UserPromptPart(content=[picture_label(chunk_id, self_ref), PNG])


def answered_question(question: str, *, evidence: str, images: bool = False):
    """One settled question: prompt, search, result, answer."""
    returned: list[Any] = [ToolReturnPart("rag_search", evidence, "call-1")]
    if images:
        returned.append(retrieved_image())
    return [
        ModelRequest(parts=[UserPromptPart(question)]),
        ModelResponse(parts=[ToolCallPart("rag_search", {"query": "q"}, "call-1")]),
        ModelRequest(parts=returned),
        ModelResponse(parts=[TextPart("an answer")]),
    ]


def returns_of(messages) -> list[str]:
    return [
        str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def images_of(messages) -> list[BinaryContent]:
    return [
        item
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
        for item in part.content
        if isinstance(item, BinaryContent)
    ]


def texts_of(messages) -> list[str]:
    return [
        item
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
        for item in part.content
        if isinstance(item, str)
    ]


def test_nothing_before_the_first_question_is_compacted():
    messages = answered_question("first", evidence="EVIDENCE")

    compacted = compact_history(
        messages, boundary=0, owned_tools=OWNED, capsule_text="CAPSULE"
    )

    assert compacted == messages


def test_the_newest_earlier_return_carries_the_capsule():
    messages = [
        *answered_question("first", evidence="OLD EVIDENCE"),
        ModelRequest(parts=[UserPromptPart("second")]),
    ]

    compacted = compact_history(
        messages, boundary=4, owned_tools=OWNED, capsule_text="CAPSULE"
    )

    assert returns_of(compacted) == ["CAPSULE"]
    assert "OLD EVIDENCE" not in returns_of(compacted)


def test_older_returns_become_receipts_and_only_the_newest_carries_the_capsule():
    messages = [
        *answered_question("first", evidence="OLDEST"),
        *answered_question("second", evidence="NEWER"),
        ModelRequest(parts=[UserPromptPart("third")]),
    ]

    compacted = compact_history(
        messages, boundary=8, owned_tools=OWNED, capsule_text="CAPSULE"
    )

    assert returns_of(compacted) == [RECEIPT, "CAPSULE"]


def test_the_current_question_keeps_its_own_evidence():
    messages = [
        *answered_question("first", evidence="OLD EVIDENCE"),
        ModelRequest(parts=[UserPromptPart("second")]),
        ModelResponse(parts=[ToolCallPart("rag_search", {"query": "q"}, "call-2")]),
        ModelRequest(parts=[ToolReturnPart("rag_search", "LIVE EVIDENCE", "call-2")]),
    ]

    compacted = compact_history(
        messages, boundary=4, owned_tools=OWNED, capsule_text="CAPSULE"
    )

    assert returns_of(compacted) == ["CAPSULE", "LIVE EVIDENCE"]


def test_another_capabilitys_return_is_left_alone():
    messages = [
        ModelRequest(parts=[UserPromptPart("first")]),
        ModelResponse(parts=[ToolCallPart("other_tool", {}, "call-1")]),
        ModelRequest(parts=[ToolReturnPart("other_tool", "NOT OURS", "call-1")]),
        ModelResponse(parts=[TextPart("an answer")]),
        ModelRequest(parts=[UserPromptPart("second")]),
    ]

    compacted = compact_history(
        messages, boundary=4, owned_tools=OWNED, capsule_text="CAPSULE"
    )

    assert returns_of(compacted) == ["NOT OURS"]


def test_a_cite_acknowledgement_survives():
    """A receipt of the model's own action, not evidence."""
    messages = [
        ModelRequest(parts=[UserPromptPart("first")]),
        ModelResponse(
            parts=[ToolCallPart("rag_cite", {"chunk_ids": ["c1"]}, "call-1")]
        ),
        ModelRequest(
            parts=[ToolReturnPart("rag_cite", "Registered 1 citation.", "c1")]
        ),
        ModelResponse(parts=[TextPart("an answer")]),
        ModelRequest(parts=[UserPromptPart("second")]),
    ]

    compacted = compact_history(
        messages, boundary=4, owned_tools=OWNED, capsule_text="CAPSULE"
    )

    assert returns_of(compacted) == ["Registered 1 citation."]


def test_an_uncited_earlier_image_is_dropped_with_its_label():
    messages = [
        *answered_question("first", evidence="OLD", images=True),
        ModelRequest(parts=[UserPromptPart("second")]),
    ]

    compacted = compact_history(
        messages, boundary=4, owned_tools=OWNED, capsule_text="CAPSULE"
    )

    assert images_of(compacted) == []
    assert texts_of(compacted) == []


def test_cited_pictures_are_attached_beside_the_capsule():
    messages = [
        *answered_question("first", evidence="OLD", images=True),
        ModelRequest(parts=[UserPromptPart("second")]),
    ]
    fresh = BinaryContent(data=b"cited-bytes", media_type="image/png")

    compacted = compact_history(
        messages,
        boundary=4,
        owned_tools=OWNED,
        capsule_text="CAPSULE",
        capsule_images=[picture_label("cited-chunk", "#/pictures/3"), fresh],
    )

    assert images_of(compacted) == [fresh]
    assert texts_of(compacted) == [picture_label("cited-chunk", "#/pictures/3")]
    carrier = [
        index
        for index, message in enumerate(compacted)
        if isinstance(message, ModelRequest)
        and any(
            isinstance(part, ToolReturnPart) and part.content == "CAPSULE"
            for part in message.parts
        )
    ]
    attached = [
        index
        for index, message in enumerate(compacted)
        if isinstance(message, ModelRequest)
        and any(
            isinstance(part, UserPromptPart) and not isinstance(part.content, str)
            for part in message.parts
        )
    ]
    assert carrier == attached


def test_a_user_attached_image_is_never_dropped():
    """The user's own picture is not ours to remove, even in an earlier question."""
    mine = UserPromptPart(content=["look at this", PNG])
    messages = [
        ModelRequest(parts=[mine]),
        ModelResponse(parts=[ToolCallPart("rag_search", {"query": "q"}, "call-1")]),
        ModelRequest(parts=[ToolReturnPart("rag_search", "OLD", "call-1")]),
        ModelResponse(parts=[TextPart("an answer")]),
        ModelRequest(parts=[UserPromptPart("second")]),
    ]

    compacted = compact_history(
        messages, boundary=4, owned_tools=OWNED, capsule_text="CAPSULE"
    )

    assert images_of(compacted) == [PNG]
    assert texts_of(compacted) == ["look at this"]


def test_the_stored_messages_are_never_mutated():
    messages = [
        *answered_question("first", evidence="OLD EVIDENCE", images=True),
        ModelRequest(parts=[UserPromptPart("second")]),
    ]
    before = [list(message.parts) for message in messages]

    compact_history(messages, boundary=4, owned_tools=OWNED, capsule_text="CAPSULE")

    assert [list(message.parts) for message in messages] == before
    assert "OLD EVIDENCE" in returns_of(messages)


def test_nothing_cited_leaves_only_receipts():
    messages = [
        *answered_question("first", evidence="OLD"),
        ModelRequest(parts=[UserPromptPart("second")]),
    ]

    compacted = compact_history(
        messages, boundary=4, owned_tools=OWNED, capsule_text=""
    )

    assert returns_of(compacted) == [RECEIPT]


@dataclass
class Deps:
    state: dict[str, Any] = field(default_factory=dict)


def rag_and_compactor(temp_db_path):
    return (
        create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False),
        create_compaction(),
    )


def in_flight_history() -> list[Any]:
    """A question already asked and searched, still awaiting its answer."""
    return [
        ModelRequest(parts=[UserPromptPart("what does the supervisor do?")]),
        ModelResponse(parts=[ToolCallPart("rag_search", {"query": "s"}, "call-1")]),
        ModelRequest(
            parts=[ToolReturnPart("rag_search", "EVIDENCE FOR THE LIVE TURN", "call-1")]
        ),
    ]


def resuming_deps(question: int = 0) -> Deps:
    """State as a resumption always finds it: the question identified and unfinished."""
    return Deps(
        state={
            "rag": RAGState(
                evidence=CapabilityEvidenceRecord(question=question, in_progress=True)
            ).model_dump(mode="json")
        }
    )


@pytest.mark.asyncio
async def test_without_the_compactor_the_history_is_untouched(temp_db_path):
    """Omission is the switch: there is no flag to test, only absence."""
    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)
    wire: list[list[Any]] = []

    async def model(messages, _info):
        wire.append(list(messages))
        return ModelResponse(parts=[TextPart("answer")])

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag])
    settled = [*in_flight_history(), ModelResponse(parts=[TextPart("first answer")])]

    await agent.run("a different question", deps=Deps(), message_history=settled)

    assert returns_of(wire[-1]) == ["EVIDENCE FOR THE LIVE TURN"]


@pytest.mark.asyncio
async def test_with_the_compactor_a_new_question_compacts_the_previous_one(
    temp_db_path,
):
    rag, compactor = rag_and_compactor(temp_db_path)
    wire: list[list[Any]] = []

    async def model(messages, _info):
        wire.append(list(messages))
        return ModelResponse(parts=[TextPart("answer")])

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag, compactor])
    settled = [*in_flight_history(), ModelResponse(parts=[TextPart("first answer")])]

    await agent.run("a different question", deps=Deps(), message_history=settled)

    assert returns_of(wire[-1]) == [RECEIPT]


@pytest.mark.parametrize(
    "resume_kwargs",
    [
        pytest.param({}, id="no prompt"),
        pytest.param(
            {"deferred_tool_results": DeferredToolResults()}, id="deferred results"
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_resumed_question_keeps_the_evidence_it_is_answering_from(
    temp_db_path, resume_kwargs
):
    """The boundary is the stored identity of the question in progress.

    An earlier question below it is compacted; the evidence the model is still
    answering from sits above it and survives. Deriving the boundary from message
    shape instead would put the live evidence below it and answer with a receipt
    where the search result should be.
    """
    rag, compactor = rag_and_compactor(temp_db_path)
    wire: list[list[Any]] = []

    async def model(messages, _info):
        wire.append(list(messages))
        return ModelResponse(parts=[TextPart("answer")])

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag, compactor])
    history = [
        *answered_question("an earlier question", evidence="EVIDENCE FOR THE OLD TURN"),
        *in_flight_history(),
    ]

    await agent.run(
        deps=resuming_deps(question=4), message_history=history, **resume_kwargs
    )

    assert returns_of(wire[-1]) == [RECEIPT, "EVIDENCE FOR THE LIVE TURN"]


@pytest.mark.asyncio
async def test_compaction_never_reaches_the_stored_message_history(temp_db_path):
    """Rewriting is for the wire; hosts keep the evidence they gathered."""
    rag, compactor = rag_and_compactor(temp_db_path)
    turns = iter(
        [
            ModelResponse(parts=[ToolCallPart("rag_search", {"query": "q"}, "call-1")]),
            ModelResponse(parts=[TextPart("first answer")]),
            ModelResponse(parts=[TextPart("second answer")]),
        ]
    )

    async def model(_messages, _info):
        return next(turns)

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag, compactor])
    deps = Deps()

    with patch.object(
        RAGCapability, "_search", AsyncMock(return_value="REAL EVIDENCE")
    ):
        first = await agent.run("old question", deps=deps)
        second = await agent.run(
            "current question", deps=deps, message_history=first.all_messages()
        )

    assert "REAL EVIDENCE" in returns_of(second.all_messages())
    assert RECEIPT not in returns_of(second.all_messages())


REAL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)


async def _search_with_a_picture(self, query: str, _limit: int | None) -> str:
    """Record a result carrying a page image, the way a real search does."""
    cast(Any, self.state).searches[query] = [
        SearchResult(
            content="evidence",
            score=1.0,
            chunk_id="chunk-1",
            document_id="doc-1",
            doc_item_refs=["#/pictures/0"],
        )
    ]
    self._note_evidence()
    return "EVIDENCE"


async def _cite_a_picture_chunk(temp_db_path, fetched: bytes | None):
    """Two questions: cite a picture chunk, then ask something else."""
    rag, compactor = rag_and_compactor(temp_db_path)
    calls = iter(
        [
            [ToolCallPart("rag_search", {"query": "supervisor"}, "call-1")],
            [ToolCallPart("rag_cite", {"chunk_ids": ["chunk-1"]}, "call-2")],
            [TextPart("first answer")],
            [TextPart("second answer")],
        ]
    )
    wire: list[list[Any]] = []

    async def model(messages, _info):
        wire.append(list(messages))
        return ModelResponse(parts=next(calls))

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag, compactor])
    deps = Deps()

    with (
        patch.object(RAGCapability, "_search", _search_with_a_picture),
        patch.object(
            RAGCapability, "get_picture_bytes", AsyncMock(return_value=fetched)
        ),
    ):
        first = await agent.run("what does the supervisor do?", deps=deps)
        await agent.run(
            "and what else?", deps=deps, message_history=first.all_messages()
        )
    return wire


@pytest.mark.asyncio
async def test_a_cited_picture_is_fetched_and_attached_with_its_label(temp_db_path):
    wire = await _cite_a_picture_chunk(temp_db_path, REAL_PNG)

    assert [picture.data for picture in images_of(wire[-1])] == [REAL_PNG]
    assert texts_of(wire[-1]) == [picture_label("chunk-1", "#/pictures/0")]


@pytest.mark.asyncio
async def test_a_picture_that_cannot_be_fetched_emits_neither_image_nor_label(
    temp_db_path,
):
    """A label without its picture tells the model a figure is there when it is not."""
    wire = await _cite_a_picture_chunk(temp_db_path, None)

    assert images_of(wire[-1]) == []
    assert texts_of(wire[-1]) == []


@pytest.mark.asyncio
async def test_a_picture_that_will_not_decode_emits_neither_image_nor_label(
    temp_db_path,
):
    """One vision placeholder is rendered per attachment, so a corrupt one miscounts."""
    wire = await _cite_a_picture_chunk(temp_db_path, b"not-an-image")

    assert images_of(wire[-1]) == []
    assert texts_of(wire[-1]) == []


@pytest.mark.asyncio
async def test_the_capsule_is_built_once_per_request_and_again_for_the_next(
    temp_db_path,
):
    """Two hook passes for one request must not rebuild; the next request must."""
    rag, compactor = rag_and_compactor(temp_db_path)
    deps = Deps()
    ctx = RunContext(
        deps=deps, model=TestModel(), usage=RunUsage(), run_id="run-1", run_step=1
    )
    run_rag = await rag.for_run(ctx)
    run_compactor = await compactor.for_run(ctx)
    cast(Any, run_rag.state).evidence.begin_question(4)
    ctx = replace(ctx, capabilities={"rag": run_rag, "compaction": run_compactor})
    builds = 0

    def counting_build(evidence):
        nonlocal builds
        builds += 1
        return Capsule(text="CAPSULE")

    async def handler(_request_context):
        return ModelResponse(parts=[TextPart("answer")])

    request = ModelRequestContext(
        messages=[*answered_question("first", evidence="OLD")],
        model=TestModel(),
        model_request_parameters=ModelRequestParameters(),
        model_settings=None,
    )

    with patch("haiku.rag.capabilities.compaction.build_capsule", counting_build):
        await run_compactor.wrap_model_request(
            ctx, request_context=request, handler=handler
        )
        await run_compactor.wrap_model_request(
            ctx, request_context=request, handler=handler
        )
        assert builds == 1

        await run_compactor.wrap_model_request(
            replace(ctx, run_step=2), request_context=request, handler=handler
        )

    assert builds == 2


def test_a_user_quoting_our_wording_keeps_their_picture_and_their_text():
    """Prose is not proof of ownership: a user can write any phrase.

    Recognising our own pictures by a natural-language substring removed a user's
    image, its text, and with it the whole message part.
    """
    quoted = UserPromptPart(
        content=[
            "Here is a page image retrieved from the knowledge base for my report",
            PNG,
        ]
    )
    messages = [
        ModelRequest(parts=[quoted]),
        ModelResponse(parts=[ToolCallPart("rag_search", {"query": "q"}, "call-1")]),
        ModelRequest(parts=[ToolReturnPart("rag_search", "OLD", "call-1")]),
        ModelResponse(parts=[TextPart("an answer")]),
        ModelRequest(parts=[UserPromptPart("second")]),
    ]

    compacted = compact_history(
        messages, boundary=4, owned_tools=OWNED, capsule_text="CAPSULE"
    )

    assert images_of(compacted) == [PNG]
    assert texts_of(compacted) == [
        "Here is a page image retrieved from the knowledge base for my report"
    ]
    assert all(message.parts for message in compacted)


def test_a_label_of_ours_with_no_picture_after_it_is_kept():
    """Only a genuine pair is ours to remove; a lone label is someone else's text."""
    lonely = UserPromptPart(
        content=[picture_label("chunk-1", "#/pictures/0"), "and more"]
    )
    messages = [
        ModelRequest(parts=[lonely]),
        ModelResponse(parts=[ToolCallPart("rag_search", {"query": "q"}, "call-1")]),
        ModelRequest(parts=[ToolReturnPart("rag_search", "OLD", "call-1")]),
        ModelResponse(parts=[TextPart("an answer")]),
        ModelRequest(parts=[UserPromptPart("second")]),
    ]

    compacted = compact_history(
        messages, boundary=4, owned_tools=OWNED, capsule_text="CAPSULE"
    )

    assert texts_of(compacted) == [
        picture_label("chunk-1", "#/pictures/0"),
        "and more",
    ]


@pytest.mark.asyncio
async def test_a_picture_whose_fetch_raises_costs_the_picture_not_the_answer(
    temp_db_path,
):
    rag, compactor = rag_and_compactor(temp_db_path)
    calls = iter(
        [
            [ToolCallPart("rag_search", {"query": "supervisor"}, "call-1")],
            [ToolCallPart("rag_cite", {"chunk_ids": ["chunk-1"]}, "call-2")],
            [TextPart("first answer")],
            [TextPart("second answer")],
        ]
    )
    wire: list[list[Any]] = []

    async def model(messages, _info):
        wire.append(list(messages))
        return ModelResponse(parts=next(calls))

    agent = Agent(FunctionModel(model), deps_type=Deps, capabilities=[rag, compactor])
    deps = Deps()

    with (
        patch.object(RAGCapability, "_search", _search_with_a_picture),
        patch.object(
            RAGCapability,
            "get_picture_bytes",
            AsyncMock(side_effect=OSError("the read failed")),
        ),
    ):
        first = await agent.run("what does the supervisor do?", deps=deps)
        second = await agent.run(
            "and what else?", deps=deps, message_history=first.all_messages()
        )

    assert second.output == "second answer"
    assert images_of(wire[-1]) == []
    assert texts_of(wire[-1]) == []


def test_a_request_is_never_left_with_no_parts():
    """Emptying a message would leave something that is not a message.

    Our own pictures always travel with the tool return in their request, so this
    shape does not come from us — but a rewritten history can hold it, and a
    partless request is invalid whatever produced it.
    """
    ours_alone = ModelRequest(
        parts=[UserPromptPart(content=[picture_label("chunk-1", "#/pictures/0"), PNG])]
    )
    messages = [
        ours_alone,
        ModelResponse(parts=[ToolCallPart("rag_search", {"query": "q"}, "call-1")]),
        ModelRequest(parts=[ToolReturnPart("rag_search", "OLD", "call-1")]),
        ModelResponse(parts=[TextPart("an answer")]),
        ModelRequest(parts=[UserPromptPart("second")]),
    ]

    compacted = compact_history(
        messages, boundary=4, owned_tools=OWNED, capsule_text="CAPSULE"
    )

    assert all(message.parts for message in compacted)
    assert compacted[0] is ours_alone


def test_two_owned_returns_in_one_request_yield_one_capsule():
    """A model can search twice in one response, so a request can hold two returns.

    Identifying the carrier by message alone gave every return in it the capsule,
    which duplicates the whole thing — and it is unbounded.
    """
    messages = [
        ModelRequest(parts=[UserPromptPart("first")]),
        ModelResponse(
            parts=[
                ToolCallPart("rag_search", {"query": "a"}, "call-1"),
                ToolCallPart("rag_search", {"query": "b"}, "call-2"),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart("rag_search", "FIRST EVIDENCE", "call-1"),
                ToolReturnPart("rag_search", "SECOND EVIDENCE", "call-2"),
            ]
        ),
        ModelResponse(parts=[TextPart("an answer")]),
        ModelRequest(parts=[UserPromptPart("second")]),
    ]

    compacted = compact_history(
        messages, boundary=4, owned_tools=OWNED, capsule_text="CAPSULE"
    )

    assert returns_of(compacted) == [RECEIPT, "CAPSULE"]


def test_the_capsule_is_attached_beside_the_newest_return_of_that_request():
    messages = [
        ModelRequest(parts=[UserPromptPart("first")]),
        ModelResponse(
            parts=[
                ToolCallPart("rag_search", {"query": "a"}, "call-1"),
                ToolCallPart("rag_search", {"query": "b"}, "call-2"),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart("rag_search", "FIRST", "call-1"),
                ToolReturnPart("rag_search", "SECOND", "call-2"),
            ]
        ),
        ModelResponse(parts=[TextPart("an answer")]),
        ModelRequest(parts=[UserPromptPart("second")]),
    ]
    fresh = BinaryContent(data=b"cited-bytes", media_type="image/png")

    compacted = compact_history(
        messages,
        boundary=4,
        owned_tools=OWNED,
        capsule_text="CAPSULE",
        capsule_images=[picture_label("cited", "#/pictures/1"), fresh],
    )

    assert images_of(compacted) == [fresh]
    assert returns_of(compacted) == [RECEIPT, "CAPSULE"]
