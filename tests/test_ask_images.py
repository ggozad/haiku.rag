from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image as PILImage
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig
from haiku.rag.utils import image_binary_content


def make_image_bytes(fmt: str) -> bytes:
    buffer = BytesIO()
    PILImage.new("RGB", (4, 4), color="red").save(buffer, format=fmt)
    return buffer.getvalue()


def test_image_binary_content_sniffs_media_type():
    png = image_binary_content(make_image_bytes("PNG"))
    assert isinstance(png, BinaryContent)
    assert png.media_type == "image/png"
    assert image_binary_content(make_image_bytes("JPEG")).media_type == "image/jpeg"


def test_image_binary_content_rejects_non_image_bytes():
    with pytest.raises(ValueError, match="not a recognizable image"):
        image_binary_content(b"definitely not an image")


@pytest.fixture
def captured_run(monkeypatch):
    """Capture the user prompt passed to Agent.run without running a model."""
    captured: dict = {}

    async def fake_run(self, user_prompt, **kwargs):
        captured["user_prompt"] = user_prompt

        class Result:
            output = "answer"

        return Result()

    monkeypatch.setattr(Agent, "run", fake_run)
    return captured


@pytest.mark.asyncio
async def test_ask_without_images_passes_plain_string(temp_db_path: Path, captured_run):
    async with HaikuRAG(temp_db_path, config=AppConfig(), create=True) as client:
        await client.ask("What is this?")
    assert captured_run["user_prompt"] == "What is this?"


@pytest.mark.asyncio
async def test_ask_with_images_passes_binary_content(temp_db_path: Path, captured_run):
    config = AppConfig()
    config.qa.model.vision = True
    png = make_image_bytes("PNG")
    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        await client.ask("What is in this image?", images=[png])
    prompt = captured_run["user_prompt"]
    assert prompt[0] == "What is in this image?"
    assert isinstance(prompt[1], BinaryContent)
    assert prompt[1].data == png
    assert prompt[1].media_type == "image/png"


@pytest.mark.asyncio
async def test_ask_with_images_requires_vision_model(temp_db_path: Path):
    config = AppConfig()
    config.qa.model.vision = False
    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        with pytest.raises(ValueError, match="vision"):
            await client.ask("What is this?", images=[make_image_bytes("PNG")])


@pytest.mark.asyncio
async def test_analyze_with_images_passes_binary_content(
    temp_db_path: Path, captured_run
):
    config = AppConfig()
    config.qa.model.vision = True
    jpeg = make_image_bytes("JPEG")
    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        await client.analyze("Does this image match?", images=[jpeg])
    prompt = captured_run["user_prompt"]
    assert prompt[0] == "Does this image match?"
    assert isinstance(prompt[1], BinaryContent)
    assert prompt[1].media_type == "image/jpeg"


@pytest.mark.asyncio
async def test_analyze_with_images_checks_analysis_model_vision(temp_db_path: Path):
    from haiku.rag.config.models import ModelConfig

    config = AppConfig()
    config.qa.model.vision = True
    config.analysis.model = ModelConfig(provider="openai", name="m", vision=False)
    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        with pytest.raises(ValueError, match="vision"):
            await client.analyze("Does this match?", images=[make_image_bytes("PNG")])
