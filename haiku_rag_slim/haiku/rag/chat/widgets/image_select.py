from collections.abc import Iterable
from io import BytesIO
from pathlib import Path

import PIL.Image as PILImage
from PIL import UnidentifiedImageError
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import DirectoryTree, Input, Label
from textual_image.widget import Image

IMAGE_EXTENSIONS = PILImage.registered_extensions()


def encode_jpeg(path: Path) -> bytes:
    """Re-encode an image file as RGB JPEG bytes."""
    image = PILImage.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class ImageAdded(Message):
    """Emitted when the user picks an image to attach to the prompt."""

    def __init__(self, path: Path, data: bytes) -> None:
        self.path = path
        self.data = data
        super().__init__()


class ImageDirectoryTree(DirectoryTree):
    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return [
            path for path in paths if path.suffix in IMAGE_EXTENSIONS or path.is_dir()
        ]


class ImageSelect(ModalScreen[tuple[Path, bytes]]):
    """Modal for picking an image file, with a live preview."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    CSS = """
    ImageSelect {
        align: center middle;
        background: rgba(0, 0, 0, 0.5);
    }

    #image-select-container {
        width: 80%;
        height: 80%;
        background: $surface;
        border: tall $primary;
        padding: 1 2;
    }

    #image-directory-tree {
        width: 40%;
    }

    #image-preview {
        width: 60%;
    }

    #image-preview #image {
        width: auto;
        height: auto;
    }

    #image-select-container Input {
        margin-bottom: 1;
    }
    """

    def action_cancel(self) -> None:
        self.dismiss()

    async def on_mount(self) -> None:
        tree = self.query_one(ImageDirectoryTree)
        tree.show_guides = False
        tree.focus()

    @on(DirectoryTree.FileSelected)
    async def on_image_selected(self, event: DirectoryTree.FileSelected) -> None:
        try:
            self.dismiss((event.path, encode_jpeg(event.path)))
        except UnidentifiedImageError:
            self.dismiss()

    @on(DirectoryTree.NodeHighlighted)
    async def on_image_highlighted(self, event: DirectoryTree.NodeHighlighted) -> None:
        if event.node.data is None:
            return
        path = event.node.data.path
        preview = self.query_one(Image)
        if path.suffix in IMAGE_EXTENSIONS:
            try:
                preview.image = PILImage.open(path.as_posix())
            except UnidentifiedImageError:
                preview.image = None
        else:
            preview.image = None

    @on(Input.Changed)
    async def on_root_changed(self, event: Input.Changed) -> None:
        path = Path(event.value)
        if path.exists() and path.is_dir():
            self.query_one(ImageDirectoryTree).path = path

    def compose(self) -> ComposeResult:
        with Container(id="image-select-container"):
            with Horizontal():
                with Vertical(id="image-directory-tree"):
                    yield Label("Select an image:")
                    yield Label("Root:")
                    yield Input(Path("./").resolve().as_posix())
                    yield ImageDirectoryTree("./")
                with Container(id="image-preview"):
                    yield Image(id="image")
