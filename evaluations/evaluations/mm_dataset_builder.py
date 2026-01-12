import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig


console = Console()


@dataclass(frozen=True)
class BuiltCase:
    id: str
    type: str  # "text" | "image"
    instruction: str | None
    query_text: str | None
    query_image_path: str | None
    relevant: list[dict]

    def to_jsonl(self) -> str:
        obj = {
            "id": self.id,
            "type": self.type,
            "relevant": self.relevant,
        }
        if self.instruction is not None:
            obj["instruction"] = self.instruction
        if self.query_text is not None:
            obj["query_text"] = self.query_text
        if self.query_image_path is not None:
            obj["query_image_path"] = self.query_image_path
        return json.dumps(obj, ensure_ascii=False)


def _crop_from_doc_bbox(
    *,
    pil_image,
    page_width: float,
    page_height: float,
    bbox: dict,
    padding_px: int,
):
    # bbox is in Docling page coordinates with bottom-left origin.
    left = float(bbox["left"])
    top = float(bbox["top"])
    right = float(bbox["right"])
    bottom = float(bbox["bottom"])

    scale_x = pil_image.width / page_width
    scale_y = pil_image.height / page_height

    x0 = left * scale_x
    x1 = right * scale_x
    # invert Y axis: doc bottom-left -> PIL top-left
    y0 = (page_height - top) * scale_y
    y1 = (page_height - bottom) * scale_y
    if y0 > y1:
        y0, y1 = y1, y0

    x0 = max(0, int(x0) - padding_px)
    y0 = max(0, int(y0) - padding_px)
    x1 = min(pil_image.width, int(x1) + padding_px)
    y1 = min(pil_image.height, int(y1) + padding_px)

    if x1 <= x0 or y1 <= y0:
        return None
    return pil_image.crop((x0, y0, x1, y1))


async def build_mm_dataset(
    *,
    config: AppConfig,
    db_path: Path | None,
    out_dir: Path,
    n: int = 50,
    seed: int = 0,
    include_text: bool = True,
    include_image: bool = True,
    instruction: str = "Retrieve images matching this description.",
) -> Path:
    """Auto-generate a JSONL dataset for `evaluations mm` from an existing DB.

    This produces a *sanity* dataset:
    - image→image queries use the exact stored bbox crop (self-retrieval)
    - text→image queries use caption/description when available (self-retrieval)

    You can then hand-edit queries/relevants for a more realistic benchmark.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "query_images"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Optional dependency: Pillow is required to write crops.
    try:
        from PIL import Image  # noqa: F401
    except Exception as e:
        raise ImportError(
            "Building a multimodal eval dataset requires Pillow. "
            "Install it (e.g. `uv pip install pillow`)."
        ) from e

    async with HaikuRAG(db_path, config=config) as rag:
        if not rag._config.multimodal.enabled:
            raise ValueError(
                "Multimodal is disabled in config. Set `multimodal.enabled: true` "
                "and open the DB in writable mode at least once to create mm_assets."
            )
        if rag.store.mm_assets_table is None:
            raise ValueError(
                "mm_assets table is not available. Enable multimodal and open the DB "
                "in writable mode at least once to create it."
            )

        # Recommended way to materialize the *entire table* is via Arrow/Pandas.
        # We prefer Arrow here to avoid any implicit limits in query paths.
        table = rag.store.mm_assets_table
        try:
            rows = table.to_arrow().to_pylist()
        except Exception:
            # Fallback: scan via query builder.
            rows = table.search().limit(100_000).to_list()
        if not rows:
            raise ValueError("mm_assets is empty; index documents with pictures first.")

        rnd = random.Random(seed)
        rnd.shuffle(rows)

        cases: list[BuiltCase] = []
        used = 0

        for row in rows:
            if used >= n:
                break

            asset_id = row.get("id")
            doc_id = row.get("document_id")
            doc_item_ref = row.get("doc_item_ref")
            item_index = row.get("item_index")
            page_no = row.get("page_no")
            bbox_raw = row.get("bbox")
            caption = row.get("caption")
            description = row.get("description")

            if not asset_id or not doc_id or not doc_item_ref:
                continue

            doc = await rag.document_repository.get_by_id(str(doc_id))
            if doc is None or not doc.uri:
                continue

            relevant = [
                {
                    "document_uri": doc.uri,
                    "doc_item_ref": str(doc_item_ref),
                    "item_index": int(item_index) if item_index is not None else None,
                }
            ]

            # text→image case (when we have any text signal)
            if include_text:
                q = None
                if caption:
                    q = str(caption).strip()
                elif description:
                    q = str(description).strip()

                if q:
                    cases.append(
                        BuiltCase(
                            id=f"auto-text-{asset_id}",
                            type="text",
                            instruction=instruction,
                            query_text=q,
                            query_image_path=None,
                            relevant=relevant,
                        )
                    )

            # image→image case (requires bbox + page image)
            if include_image and page_no is not None and bbox_raw:
                try:
                    bbox = json.loads(bbox_raw) if isinstance(bbox_raw, str) else bbox_raw
                except Exception:
                    bbox = None
                if isinstance(bbox, dict) and all(
                    k in bbox for k in ("left", "top", "right", "bottom")
                ):
                    docling_doc = doc.get_docling_document()
                    if docling_doc and int(page_no) in docling_doc.pages:
                        page = docling_doc.pages[int(page_no)]
                        if (
                            page.image is not None
                            and page.image.pil_image is not None
                            and page.size is not None
                        ):
                            crop = _crop_from_doc_bbox(
                                pil_image=page.image.pil_image,
                                page_width=float(page.size.width),
                                page_height=float(page.size.height),
                                bbox=bbox,
                                padding_px=int(rag._config.multimodal.image_crop_padding_px),
                            )
                            if crop is not None:
                                crop_path = crops_dir / f"{asset_id}.png"
                                crop.save(crop_path, format="PNG")
                                cases.append(
                                    BuiltCase(
                                        id=f"auto-image-{asset_id}",
                                        type="image",
                                        instruction=None,
                                        query_text=None,
                                        query_image_path=str(crop_path.resolve()),
                                        relevant=relevant,
                                    )
                                )

            used += 1

    dataset_path = out_dir / "mm_eval.jsonl"
    dataset_path.write_text(
        "\n".join(c.to_jsonl() for c in cases) + "\n", encoding="utf-8"
    )

    console.print(
        f"Wrote {len(cases)} cases to {dataset_path} "
        f"(from {min(n, len(rows))} sampled mm_assets rows).",
        style="green",
    )
    return dataset_path


def build_mm_dataset_sync(**kwargs) -> Path:
    return asyncio.run(build_mm_dataset(**kwargs))

