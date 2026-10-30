"""Tests for the direct VLM client used by ``rebuild --descriptions``."""

import logging
from typing import Any

from haiku.rag.config import AppConfig
from haiku.rag.providers import picture_description
from haiku.rag.providers.picture_description import describe_pictures
from tests.conftest import capture_logs


class _StubAgent:
    """Minimal stand-in for ``pydantic_ai.Agent`` that returns a queue of
    pre-baked responses (or raises) for each ``run`` call."""

    def __init__(self, outputs: list[Any]):
        self._outputs = list(outputs)
        self.calls: list[Any] = []

    async def run(self, prompt: list) -> Any:
        self.calls.append(prompt)
        out = self._outputs.pop(0)
        if isinstance(out, BaseException):
            raise out

        class _Result:
            def __init__(self, output: str):
                self.output = output

        return _Result(out)


def _patch_agent(monkeypatch, outputs: list[Any]) -> _StubAgent:
    """Replace pydantic_ai.Agent in our module with a constructor that
    returns a single shared StubAgent."""
    stub = _StubAgent(outputs)
    monkeypatch.setattr(
        "haiku.rag.providers.picture_description.Agent",
        lambda **kwargs: stub,
    )
    # Skip real model construction — we don't use the returned model anyway.
    monkeypatch.setattr(
        "haiku.rag.providers.picture_description.get_model",
        lambda model_config, app_config: object(),
    )
    return stub


async def test_describe_pictures_returns_text_per_self_ref(monkeypatch):
    """Each picture gets one VLM call carrying its bytes as a PNG BinaryContent
    part, and the response text lands in the result map keyed by self_ref."""
    from pydantic_ai.messages import BinaryContent

    stub = _patch_agent(
        monkeypatch,
        outputs=["A red square.", "A blue triangle."],
    )

    config = AppConfig()
    out = await describe_pictures(
        {"#/pictures/0": b"\x89PNG\r\n\x1a\nred", "#/pictures/1": b"blue-bytes"},
        config=config,
    )

    assert out == {
        "#/pictures/0": "A red square.",
        "#/pictures/1": "A blue triangle.",
    }
    assert len(stub.calls) == 2
    parts = stub.calls[0]
    assert isinstance(parts, list) and len(parts) == 1
    assert isinstance(parts[0], BinaryContent)
    assert parts[0].data == b"\x89PNG\r\n\x1a\nred"
    assert parts[0].media_type == "image/png"


async def test_describe_pictures_drops_empty_output(monkeypatch):
    """Pictures whose VLM response is empty/whitespace are dropped from
    the result map. Caller can decide whether the partial result is
    acceptable."""
    _patch_agent(monkeypatch, outputs=["A real description.", "   ", ""])

    out = await describe_pictures(
        {
            "#/pictures/0": b"a",
            "#/pictures/1": b"b",
            "#/pictures/2": b"c",
        },
        config=AppConfig(),
    )

    assert out == {"#/pictures/0": "A real description."}


async def test_describe_pictures_swallows_exceptions(monkeypatch):
    """A failing VLM call is logged as a warning and the picture is
    skipped — the rest of the batch still gets described."""
    _patch_agent(
        monkeypatch,
        outputs=[
            RuntimeError("boom"),
            "After the failure.",
        ],
    )

    with capture_logs(picture_description.logger, logging.WARNING) as records:
        out = await describe_pictures(
            {"#/pictures/0": b"a", "#/pictures/1": b"b"},
            config=AppConfig(),
        )

    assert out == {"#/pictures/1": "After the failure."}
    assert [r.getMessage() for r in records] == [
        "VLM call failed for #/pictures/0: boom"
    ]


async def test_describe_pictures_empty_input(monkeypatch):
    """No pictures means no VLM calls and an empty result."""
    stub = _patch_agent(monkeypatch, outputs=[])

    out = await describe_pictures({}, config=AppConfig())

    assert out == {}
    assert stub.calls == []
