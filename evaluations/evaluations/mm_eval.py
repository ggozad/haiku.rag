import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig


console = Console()


@dataclass(frozen=True)
class MMRelevantRef:
    """A stable reference to an mm_asset row (avoid UUIDs in datasets)."""

    document_uri: str
    doc_item_ref: str
    item_index: int | None = None


@dataclass(frozen=True)
class MMEvalCase:
    """One multimodal retrieval evaluation case."""

    case_id: str
    query_type: str  # "text" | "image"
    instruction: str | None
    query_text: str | None
    query_image_path: str | None
    relevant: tuple[MMRelevantRef, ...]


@dataclass(frozen=True)
class CaseMetrics:
    recall_at_k: dict[int, float]
    mrr: float


def _quote_lancedb(s: str) -> str:
    # LanceDB uses SQL-ish string quoting with single quotes.
    return s.replace("'", "''")


def load_mm_dataset(dataset_path: Path) -> list[MMEvalCase]:
    cases: list[MMEvalCase] = []
    for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        obj = json.loads(line)

        case_id = str(obj.get("id") or obj.get("case_id") or "")
        if not case_id:
            raise ValueError("Each JSONL row must have a non-empty 'id'.")

        query_type = obj.get("type")
        if query_type not in ("text", "image"):
            raise ValueError(f"Case {case_id}: 'type' must be 'text' or 'image'.")

        instruction = obj.get("instruction")
        query_text = obj.get("query_text")
        query_image_path = obj.get("query_image_path")

        if query_type == "text" and not query_text:
            raise ValueError(f"Case {case_id}: missing 'query_text'.")
        if query_type == "image" and not query_image_path:
            raise ValueError(f"Case {case_id}: missing 'query_image_path'.")

        rel = obj.get("relevant") or []
        if not isinstance(rel, list) or not rel:
            raise ValueError(f"Case {case_id}: 'relevant' must be a non-empty list.")

        relevant: list[MMRelevantRef] = []
        for r in rel:
            if not isinstance(r, dict):
                raise ValueError(f"Case {case_id}: relevant entries must be objects.")
            document_uri = r.get("document_uri")
            doc_item_ref = r.get("doc_item_ref")
            if not document_uri or not doc_item_ref:
                raise ValueError(
                    f"Case {case_id}: relevant entries must include "
                    "'document_uri' and 'doc_item_ref'."
                )
            item_index = r.get("item_index")
            relevant.append(
                MMRelevantRef(
                    document_uri=str(document_uri),
                    doc_item_ref=str(doc_item_ref),
                    item_index=int(item_index) if item_index is not None else None,
                )
            )

        cases.append(
            MMEvalCase(
                case_id=case_id,
                query_type=query_type,
                instruction=str(instruction) if instruction is not None else None,
                query_text=query_text,
                query_image_path=query_image_path,
                relevant=tuple(relevant),
            )
        )
    return cases


async def _resolve_relevant_asset_ids(
    rag: HaikuRAG, cases: list[MMEvalCase]
) -> dict[str, set[str]]:
    """Resolve stable refs (uri + doc_item_ref [+ item_index]) into asset UUIDs."""
    if rag.store.mm_assets_table is None:
        raise ValueError("mm_assets table is not available (multimodal not enabled?).")

    # Resolve document_id per uri.
    uris = sorted({r.document_uri for c in cases for r in c.relevant})
    uri_to_doc_id: dict[str, str] = {}
    for uri in uris:
        doc = await rag.get_document_by_uri(uri)
        if doc is None or not doc.id:
            raise ValueError(f"Dataset references unknown document uri: {uri}")
        uri_to_doc_id[uri] = doc.id

    # Load mm_assets rows once (evaluation datasets are small; this is simplest).
    # Recommended way to materialize the *entire table* is via Arrow/Pandas.
    table = rag.store.mm_assets_table
    try:
        rows = table.to_arrow().to_pylist()
    except Exception:
        rows = table.search().limit(100_000).to_list()

    # Build lookup maps.
    by_doc_ref_idx: dict[tuple[str, str, int], str] = {}
    by_doc_ref: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        asset_id = row.get("id")
        doc_id = row.get("document_id")
        doc_item_ref = row.get("doc_item_ref")
        item_index = row.get("item_index")
        if not asset_id or not doc_id or not doc_item_ref:
            continue
        if item_index is not None:
            try:
                by_doc_ref_idx[(str(doc_id), str(doc_item_ref), int(item_index))] = str(
                    asset_id
                )
            except Exception:
                pass
        by_doc_ref.setdefault((str(doc_id), str(doc_item_ref)), []).append(str(asset_id))

    case_to_relevant_ids: dict[str, set[str]] = {}
    for c in cases:
        ids: set[str] = set()
        for r in c.relevant:
            doc_id = uri_to_doc_id[r.document_uri]
            if r.item_index is not None:
                k = (doc_id, r.doc_item_ref, int(r.item_index))
                asset_id = by_doc_ref_idx.get(k)
                if asset_id:
                    ids.add(asset_id)
                    continue
            # Fallback: any asset(s) matching doc+ref.
            ids.update(by_doc_ref.get((doc_id, r.doc_item_ref), []))
        if not ids:
            raise ValueError(
                f"Case {c.case_id}: could not resolve any relevant asset ids from refs. "
                "Make sure the DB was built with multimodal indexing enabled and the "
                "doc_item_ref/item_index match your Docling conversion."
            )
        case_to_relevant_ids[c.case_id] = ids

    return case_to_relevant_ids


def _compute_case_metrics(
    *, retrieved_ids: list[str], relevant_ids: set[str], ks: list[int]
) -> CaseMetrics:
    recall_at_k: dict[int, float] = {}
    for k in ks:
        topk = retrieved_ids[:k]
        hits = sum(1 for a in topk if a in relevant_ids)
        recall_at_k[k] = hits / max(len(relevant_ids), 1)

    rr = 0.0
    for idx, a in enumerate(retrieved_ids, start=1):
        if a in relevant_ids:
            rr = 1.0 / idx
            break

    return CaseMetrics(recall_at_k=recall_at_k, mrr=rr)


async def run_mm_eval(
    *,
    dataset_path: Path,
    config: AppConfig,
    db_path: Path | None,
    ks: list[int],
    limit: int,
) -> int:
    cases = load_mm_dataset(dataset_path)
    if not cases:
        raise ValueError("No cases found.")

    async with HaikuRAG(db_path, config=config) as rag:
        if not rag._config.multimodal.enabled:
            raise ValueError(
                "Multimodal is disabled in config. Set `multimodal.enabled: true` "
                "and ensure mm_assets is created."
            )

        # Resolve ground-truth.
        relevant_ids = await _resolve_relevant_asset_ids(rag, cases)

        totals_recall = {k: 0.0 for k in ks}
        totals_mrr = 0.0

        for c in cases:
            if c.query_type == "text":
                # The Qwen3-VL embedding examples use an "instruction" field that can
                # materially affect retrieval quality. vLLM /v1/embeddings typically
                # doesn't expose a separate instruction channel, so we fold it into
                # the query text in a stable way for evaluation.
                q = c.query_text or ""
                if c.instruction:
                    q = f"{c.instruction}\n{q}"
                results = await rag.search_images_by_text(
                    q, limit=limit
                )
            else:
                img_path = Path(c.query_image_path or "")
                if not img_path.exists():
                    raise ValueError(
                        f"Case {c.case_id}: query_image_path not found: {img_path}"
                    )
                results = await rag.search_images(img_path, limit=limit)

            retrieved = [r.asset_id for r in results if r.asset_id]
            m = _compute_case_metrics(
                retrieved_ids=retrieved,
                relevant_ids=relevant_ids[c.case_id],
                ks=ks,
            )

            for k in ks:
                totals_recall[k] += m.recall_at_k[k]
            totals_mrr += m.mrr

        n = float(len(cases))
        console.print("\n=== Multimodal Retrieval Benchmark Results ===", style="bold cyan")
        console.print(f"Dataset: {dataset_path}")
        console.print(f"Total queries: {len(cases)}")
        for k in ks:
            console.print(f"recall@{k}: {totals_recall[k] / n:.4f}")
        console.print(f"mrr: {totals_mrr / n:.4f}")

    return 0


def run_mm_eval_sync(**kwargs) -> int:
    return asyncio.run(run_mm_eval(**kwargs))

