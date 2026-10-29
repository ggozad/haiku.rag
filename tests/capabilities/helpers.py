import base64
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, cast
from unittest.mock import AsyncMock

from PIL import Image as PILImage
from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from haiku.rag.store.models.chunk import SearchResult


@dataclass
class Deps:
    state: dict[str, Any] = field(default_factory=dict)


def make_context(deps: Deps) -> RunContext[Deps]:
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        run_id="test-run",
    )


async def stub_search(
    self, query: str, _limit: int | None, _run_step: int, _sources=None
) -> str:
    """Record a result the way the real search does, so citing resolves."""
    cast(Any, self.state).searches[query] = [
        SearchResult(content="evidence", score=1.0, chunk_id="chunk-1")
    ]
    self._note_evidence()
    return "EVIDENCE"


def scripted(
    responses: list[list[Any]], sent: list[list[ModelMessage]] | None = None
) -> FunctionModel:
    """A model answering with `responses` in turn, recording what it was sent."""
    turns = iter(responses)

    async def model(messages, _info):
        if sent is not None:
            sent.append(list(messages))
        return ModelResponse(parts=next(turns))

    return FunctionModel(model)


def in_flight_history() -> list[Any]:
    """A question already asked and searched, still awaiting its answer."""
    return [
        ModelRequest(parts=[UserPromptPart("what does the supervisor do?")]),
        ModelResponse(parts=[ToolCallPart("search", {"query": "s"}, "call-1")]),
        ModelRequest(
            parts=[ToolReturnPart("search", "EVIDENCE FOR THE LIVE TURN", "call-1")]
        ),
    ]


def png() -> str:
    buffer = BytesIO()
    PILImage.new("RGB", (4, 4), "red").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def make_result(**overrides: Any) -> SearchResult:
    fields: dict[str, Any] = {
        "content": "body",
        "score": 0.9,
        "source": "main",
        "chunk_id": "c1",
        "document_id": "d1",
        "image_data": {"#/pictures/0": png()},
    }
    fields.update(overrides)
    return SearchResult(**fields)


def stub_client(
    *batches: list[SearchResult], sources: tuple[str, ...] | list[str] = ("main",)
) -> AsyncMock:
    """A client covering `sources` whose successive searches return `batches`."""
    client = AsyncMock()
    client.search.side_effect = list(batches)
    client.expand_context.side_effect = lambda results: results
    client.source_names = sources
    return client
