from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image as PILImage
from pydantic_ai.messages import BinaryContent
from textual.app import App

from haiku.rag.chat.widgets.image_select import (
    ImageDirectoryTree,
    ImageSelect,
    encode_jpeg,
)
from haiku.rag.chat.widgets.prompt import (
    FlexibleInput,
    PostableTextArea,
    build_user_prompt,
)


def make_image_bytes(fmt: str = "PNG") -> bytes:
    buffer = BytesIO()
    PILImage.new("RGB", (4, 4), color="red").save(buffer, format=fmt)
    return buffer.getvalue()


class TestBuildUserPrompt:
    def test_no_images_returns_text(self):
        assert build_user_prompt("hello", []) == "hello"

    def test_images_without_tokens_append_at_end(self):
        img = make_image_bytes()
        prompt = build_user_prompt("hello", [img])
        assert prompt[0] == "hello"
        assert isinstance(prompt[1], BinaryContent)
        assert prompt[1].data == img

    def test_tokens_interleave_images(self):
        first = make_image_bytes("PNG")
        second = make_image_bytes("JPEG")
        prompt = build_user_prompt(
            "compare [Image #1] with [Image #2] please", [first, second]
        )
        assert prompt[0] == "compare "
        assert isinstance(prompt[1], BinaryContent)
        assert prompt[1].data == first
        assert prompt[2] == " with "
        assert isinstance(prompt[3], BinaryContent)
        assert prompt[3].data == second
        assert prompt[4] == " please"

    def test_out_of_range_token_stays_literal(self):
        assert build_user_prompt("see [Image #2]", [make_image_bytes()]) == (
            "see [Image #2]"
        )


class TestImageDirectoryTree:
    def test_filter_paths_keeps_images_and_dirs(self, tmp_path):
        (tmp_path / "photo.png").write_bytes(make_image_bytes())
        (tmp_path / "notes.txt").write_text("nope")
        (tmp_path / "subdir").mkdir()

        tree = ImageDirectoryTree(tmp_path)
        kept = {p.name for p in tree.filter_paths(tmp_path.iterdir())}
        assert kept == {"photo.png", "subdir"}


class TestEncodeJpeg:
    def test_reencodes_to_jpeg(self, tmp_path):
        path = tmp_path / "img.png"
        buffer = BytesIO()
        PILImage.new("RGBA", (4, 4)).save(buffer, format="PNG")
        path.write_bytes(buffer.getvalue())

        data = encode_jpeg(path)
        assert PILImage.open(BytesIO(data)).format == "JPEG"


class PromptApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[str] = []

    def compose(self):
        yield FlexibleInput("", id="chat-input")

    def on_flexible_input_submitted(self, event: FlexibleInput.Submitted) -> None:
        self.submitted.append(event.value)


class TestFlexibleInput:
    @pytest.mark.asyncio
    async def test_enter_submits_text(self):
        app = PromptApp()
        async with app.run_test() as pilot:
            area = app.query_one(PostableTextArea)
            area.focus()
            area.text = "hello"
            await pilot.press("enter")
        assert app.submitted == ["hello"]

    @pytest.mark.asyncio
    async def test_backspace_deletes_whole_image_token(self):
        app = PromptApp()
        async with app.run_test() as pilot:
            area = app.query_one(PostableTextArea)
            area.focus()
            area.text = "look at [Image #1] now"
            area.cursor_location = (0, 18)
            await pilot.press("backspace")
            assert area.text == "look at  now"

    @pytest.mark.asyncio
    async def test_ctrl_i_opens_image_select(self):
        app = PromptApp()
        async with app.run_test() as pilot:
            app.query_one(PostableTextArea).focus()
            await pilot.press("ctrl+i")
            await pilot.pause()
            assert isinstance(app.screen, ImageSelect)


class TestChatAppImageAttach:
    @pytest.mark.asyncio
    async def test_image_added_inserts_token_and_stores_bytes(self, temp_db_path):
        from haiku.rag.chat.app import ChatApp
        from haiku.rag.chat.widgets.image_select import ImageAdded
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True):
            pass

        app = ChatApp(db_path=temp_db_path, capabilities=[])
        async with app.run_test() as pilot:
            data = make_image_bytes()
            app.post_message(ImageAdded(Path("img.png"), data))
            await pilot.pause()
            assert app._images == [data]
            assert "[Image #1]" in app.query_one(PostableTextArea).text


class TestChatAppLayout:
    @pytest.mark.asyncio
    async def test_prompt_stays_compact_and_history_visible(self, temp_db_path):
        from haiku.rag.chat.app import ChatApp
        from haiku.rag.chat.widgets.chat_history import ChatHistory
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True):
            pass

        app = ChatApp(db_path=temp_db_path, capabilities=[])
        async with app.run_test() as pilot:
            await pilot.pause()
            prompt = app.query_one(FlexibleInput)
            history = app.query_one(ChatHistory)
            assert prompt.region.height <= 4
            assert history.region.height > prompt.region.height
            assert history.region.y < prompt.region.y
            area = app.query_one(PostableTextArea)
            assert prompt.region.contains_region(area.region)
