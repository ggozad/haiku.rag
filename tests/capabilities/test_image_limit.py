import base64
from dataclasses import dataclass, replace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import Agent, RunContext, ToolReturn
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext

from haiku.rag.capabilities._tools import IMAGES_OMITTED, limit_retrieved_images
from haiku.rag.capabilities.compaction import create_capability as create_compaction
from haiku.rag.capabilities.compaction import picture_label
from haiku.rag.capabilities.rag import RAGCapability
from haiku.rag.capabilities.rag import create_capability as create_rag
from haiku.rag.config.models import AppConfig, QAConfig
from haiku.rag.store.models.chunk import SearchResult
from haiku.rag.utils.images import RETRIEVED_IMAGE_TAG
from tests.capabilities.helpers import Deps, make_result, png, scripted, stub_client

REAL_PNG = base64.b64decode(png())


def picture(name: str) -> BinaryContent:
    return BinaryContent(data=b"bytes", media_type="image/png", identifier=name)


def searched(*names: str) -> UserPromptPart:
    """Pictures on the wire, labelled the way a search result attaches them."""
    content: list[Any] = []
    for name in names:
        content += [f"Page image for [{name}]. {RETRIEVED_IMAGE_TAG}", picture(name)]
    return UserPromptPart(content=content)


def cited(*names: str) -> UserPromptPart:
    """Pictures on the wire, labelled the way the capsule attaches them."""
    content: list[Any] = []
    for name in names:
        content += [picture_label(name, name), picture(name)]
    return UserPromptPart(content=content)


def request(*parts: Any) -> ModelRequest:
    return ModelRequest(parts=list(parts))


def answer() -> ModelResponse:
    return ModelResponse(parts=[TextPart("ok")])


def pictures_in(messages: list[ModelMessage]) -> list[str | None]:
    return [
        item.identifier
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
        for item in part.content
        if isinstance(item, BinaryContent)
    ]


def test_a_request_under_the_limit_is_returned_as_it_is():
    messages: list[ModelMessage] = [request(searched("a", "b"))]

    assert limit_retrieved_images(messages, 2) is messages


def test_the_oldest_message_loses_its_pictures_first():
    messages = [
        request(searched("old-1", "old-2")),
        answer(),
        request(searched("mid-1")),
        answer(),
        request(searched("new-1")),
    ]

    assert pictures_in(limit_retrieved_images(messages, 2)) == ["mid-1", "new-1"]


def test_within_a_part_the_last_picture_goes_first():
    messages = [request(searched("first", "second", "third")), answer(), request()]

    limited = limit_retrieved_images(messages, 1)

    assert pictures_in(limited) == ["first"]
    first_request = cast(ModelRequest, limited[0])
    assert len(cast(UserPromptPart, first_request.parts[0]).content) == 2


def test_cited_pictures_outlive_the_uncited_and_the_newest_round_outlives_both():
    messages = [
        request(cited("cited-new", "cited-old")),
        answer(),
        request(searched("round-1")),
        answer(),
        request(searched("round-2a", "round-2b")),
    ]

    assert pictures_in(limit_retrieved_images(messages, 4)) == [
        "cited-new",
        "cited-old",
        "round-2a",
        "round-2b",
    ]
    assert pictures_in(limit_retrieved_images(messages, 3)) == [
        "cited-new",
        "round-2a",
        "round-2b",
    ]
    assert pictures_in(limit_retrieved_images(messages, 1)) == ["round-2a"]


@pytest.mark.parametrize(
    "other",
    [
        pytest.param(
            UserPromptPart(content=["mine", picture("user")]), id="user-binary"
        ),
        pytest.param(
            UserPromptPart(content=[ImageUrl(url="https://example.com/a.png")]),
            id="user-url",
        ),
        pytest.param(
            ToolReturnPart("camera", ["a photo", picture("tool")], "call-9"),
            id="tool-return",
        ),
    ],
)
def test_an_image_from_elsewhere_counts_and_is_kept(other):
    messages = [request(other), answer(), request(searched("a", "b"))]

    limited = limit_retrieved_images(messages, 2)

    assert limited[0] is messages[0]
    assert pictures_in(limited[1:]) == ["a"]


def test_a_document_in_a_tool_return_is_not_an_image():
    pdf = BinaryContent(data=b"%PDF", media_type="application/pdf")
    messages: list[ModelMessage] = [
        request(ToolReturnPart("reader", [pdf], "call-9"), searched("a"))
    ]

    assert limit_retrieved_images(messages, 1) is messages


def test_images_from_elsewhere_over_the_limit_cost_every_retrieved_picture():
    user = UserPromptPart(content=[picture("u1"), picture("u2")])
    messages = [request(user), answer(), request(searched("a"))]

    limited = limit_retrieved_images(messages, 1)

    assert pictures_in(limited) == ["u1", "u2"]


def test_a_limit_of_zero_drops_every_retrieved_picture():
    messages = [request(searched("a")), answer(), request(searched("b"))]

    assert pictures_in(limit_retrieved_images(messages, 0)) == []


def test_a_part_that_loses_every_picture_says_so():
    messages = [request(searched("a", "b")), answer(), request()]

    limited = limit_retrieved_images(messages, 0)

    assert cast(ModelRequest, limited[0]).parts == [
        UserPromptPart(
            content=[IMAGES_OMITTED],
            timestamp=cast(UserPromptPart, messages[0].parts[0]).timestamp,
        )
    ]


def test_the_messages_given_are_not_mutated():
    part = searched("a", "b")
    messages = [request(part), answer(), request()]

    limit_retrieved_images(messages, 0)

    assert pictures_in(messages) == ["a", "b"]
    assert len(part.content) == 4


def _result(name: str) -> SearchResult:
    return make_result(
        chunk_id=name, document_id=name, image_data={f"#/pictures/{name}": png()}
    )


def limited_rag(temp_db_path, max_images: int | None) -> RAGCapability:
    return create_rag(
        db_path=temp_db_path,
        config=AppConfig(qa=QAConfig(max_images=max_images)),
        vision=True,
    )


async def _three_rounds(temp_db_path, max_images: int | None):
    rag = limited_rag(temp_db_path, max_images)
    rag.borrowed_rag = stub_client(
        [_result("a1"), _result("a2")],
        [_result("b1"), _result("b2")],
        [_result("c1"), _result("c2")],
    )
    wire: list[list[ModelMessage]] = []
    responses = [
        [ToolCallPart("search", {"query": "a"}, "call-1")],
        [ToolCallPart("search", {"query": "b"}, "call-2")],
        [ToolCallPart("search", {"query": "c"}, "call-3")],
        [TextPart("an answer")],
    ]
    agent = Agent(scripted(responses, wire), deps_type=Deps, capabilities=[rag])
    result = await agent.run("what do the figures show?", deps=Deps())
    return wire, result


async def test_every_request_of_a_run_leaves_within_the_limit(temp_db_path):
    wire, result = await _three_rounds(temp_db_path, max_images=3)

    assert [len(pictures_in(sent)) for sent in wire] == [0, 2, 3, 3]
    assert pictures_in(wire[-1]) == [
        "#/pictures/b1",
        "#/pictures/c1",
        "#/pictures/c2",
    ]
    assert len(pictures_in(result.all_messages())) == 6


@dataclass
class Camera(AbstractCapability[Any]):
    """Attaches a photo of its own to every request."""

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        current = cast(ModelRequest, request_context.messages[-1])
        photo = UserPromptPart(content=[picture("photo")])
        request_context.messages[-1] = replace(current, parts=[*current.parts, photo])
        return request_context


@pytest.mark.parametrize("rag_first", [True, False], ids=["rag-first", "rag-last"])
async def test_an_image_added_by_another_capability_counts_and_is_kept(
    temp_db_path, rag_first
):
    rag = limited_rag(temp_db_path, max_images=2)
    rag.borrowed_rag = stub_client([_result("a1"), _result("a2")])
    camera = Camera(id="camera")
    wire: list[list[ModelMessage]] = []
    responses = [
        [ToolCallPart("search", {"query": "a"}, "call-1")],
        [TextPart("an answer")],
    ]
    agent = Agent(
        scripted(responses, wire),
        deps_type=Deps,
        capabilities=[rag, camera] if rag_first else [camera, rag],
    )

    result = await agent.run("what do the figures show?", deps=Deps())

    sent = pictures_in(wire[-1])
    assert len(sent) == 2
    assert "photo" in sent
    stored = pictures_in(result.all_messages())
    assert {"#/pictures/a1", "#/pictures/a2"} <= set(stored)


async def test_without_a_limit_every_picture_is_sent(temp_db_path):
    wire, _ = await _three_rounds(temp_db_path, max_images=None)

    assert [len(pictures_in(sent)) for sent in wire] == [0, 2, 4, 6]


async def _search_with_its_picture(
    self, query: str, _limit: int | None, _run_step: int, _sources=None
) -> ToolReturn:
    """A search whose one result carries the picture named by the query."""
    ref = f"#/pictures/{query}"
    cast(Any, self.state).searches[query] = [
        SearchResult(
            content="evidence",
            score=1.0,
            chunk_id=f"chunk-{query}",
            document_id=f"doc-{query}",
            doc_item_refs=[ref],
        )
    ]
    self._note_evidence()
    return ToolReturn(
        return_value="EVIDENCE",
        content=[
            f"Page image for [chunk-{query}]. {RETRIEVED_IMAGE_TAG}",
            BinaryContent(data=REAL_PNG, media_type="image/png", identifier=ref),
        ],
    )


@pytest.mark.parametrize("rag_first", [True, False], ids=["rag-outer", "rag-inner"])
async def test_a_compacted_request_leaves_within_the_limit(temp_db_path, rag_first):
    """The third question holds two cited pictures and searches for a third."""
    rag, compactor = limited_rag(temp_db_path, max_images=2), create_compaction()
    capabilities = [rag, compactor] if rag_first else [compactor, rag]
    responses: list[list[Any]] = []
    for number in ("1", "2"):
        responses += [
            [ToolCallPart("search", {"query": number}, f"search-{number}")],
            [
                ToolCallPart(
                    "cite", {"chunk_ids": [f"chunk-{number}"]}, f"cite-{number}"
                )
            ],
            [TextPart(f"answer {number}")],
        ]
    responses += [
        [ToolCallPart("search", {"query": "3"}, "search-3")],
        [TextPart("answer 3")],
    ]
    wire: list[list[ModelMessage]] = []
    agent = Agent(scripted(responses, wire), deps_type=Deps, capabilities=capabilities)
    deps = Deps()

    with (
        patch.object(RAGCapability, "_search", _search_with_its_picture),
        patch.object(
            RAGCapability, "get_picture_bytes", AsyncMock(return_value=REAL_PNG)
        ),
    ):
        history: list[ModelMessage] = []
        for question in ("first?", "second?", "third?"):
            result = await agent.run(question, deps=deps, message_history=history)
            history = result.all_messages()

    assert all(len(pictures_in(sent)) <= 2 for sent in wire)
    assert pictures_in(wire[-2]) == ["#/pictures/2", "#/pictures/1"]
    assert pictures_in(wire[-1]) == ["#/pictures/2", "#/pictures/3"]
