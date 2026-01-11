# Multimodal Search (Images)

haiku.rag supports **multimodal embeddings** for:
- **text → image** search (`mm_assets` table)
- **image → image** search (`mm_assets` table)

This feature is **orthogonal** to the existing text chunk search (`chunks` table). Your existing `haiku-rag search` behavior does not change.

---

## What gets indexed (Phase 1)

When enabled, haiku.rag extracts Docling `PictureItem`s and indexes:
- An **image crop** from the Docling page image using the item bbox (plus padding).
- Optional text metadata if available (caption/description) stored alongside the vector.

Each indexed image becomes one row in `mm_assets`, linked to its `document_id` and bbox provenance (page + coordinates), so it can be visualized later.

---

## Requirements

- Your converter must produce **Docling page images** (needed for bbox crops and visualization).
- A multimodal embedding backend must be reachable.

**Current supported backend**
- `provider: vllm` via OpenAI-compatible `POST /v1/embeddings`

---

## Configuration

Add the following to your `haiku.rag.yaml`:

```yaml
multimodal:
  enabled: true
  index_pictures: true

  # bbox → crop settings
  image_crop_padding_px: 8

  # guardrails / throughput knobs
  image_max_side_px: 1024
  embed_batch_size: 8

  model:
    provider: vllm
    name: Qwen/Qwen3-VL-Embedding-2B
    vector_dim: 2048
    base_url: http://localhost:8000   # with or without trailing /v1
    timeout: 60
    # dimensions: 1024               # only if your backend/model supports it
    encoding_format: float           # float | base64
```

### Multimodal toggles

- **`multimodal.enabled`**: master switch. When `false`, haiku.rag will not create/use `mm_assets`.
- **`multimodal.index_pictures`**: when `true`, Docling `PictureItem`s are indexed during `add-src` / update.

### Crop / image settings

- **`image_crop_padding_px`**: extra pixels added around the bbox crop. Helps avoid tight crops clipping labels/axes.
- **`image_max_side_px`**: resize guardrail before upload (keeps requests smaller, avoids backend limits). Set `0` to disable resizing.

### Throughput settings

- **`embed_batch_size`**:
  - Used as the **maximum in-flight requests** for image embedding to avoid overwhelming the backend.
  - Text embeddings may be batched using `input=[...]`, but image embeddings currently run as **one request per image** (because the backend expects multimodal inputs via `messages`).

### Model / backend settings

- **`provider`**: currently only `vllm` is supported (OpenAI-compatible server).
- **`name`**: model identifier sent to the backend (e.g. `Qwen/Qwen3-VL-Embedding-2B`).
- **`vector_dim`**: expected embedding size. This must stay stable for an existing DB (same rule as text embeddings).
- **`base_url`**: server base URL; haiku.rag accepts either `http://host:port` or `http://host:port/v1`.
- **`timeout`**: request timeout in seconds.
- **`dimensions`**: optional request-time output dimension (only for models that support it).
- **`encoding_format`**: `float` is recommended; `base64` is supported by some servers.

!!! note
    Some servers/models reject the presence of `dimensions` unless matryoshka output is supported.
    For vLLM + Qwen3‑VL‑Embedding‑2B, the safe default is to **omit** it and validate the returned vector length (2048).

---

## Using a different model (same backend)

To switch models (still using vLLM), change:
- `multimodal.model.name`
- `multimodal.model.vector_dim`

Best practice: **verify the output dimension** before ingesting a large corpus:

```bash
curl -s http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"<YOUR_MODEL>","input":["hello"],"encoding_format":"float"}' \
| python3 -c 'import sys,json; print(len(json.load(sys.stdin)["data"][0]["embedding"]))'
```

---

## Run vLLM (example)

Serve the embedding model:

```bash
vllm serve Qwen/Qwen3-VL-Embedding-2B --runner=pooling --host 127.0.0.1 --port 8000
```

Quick checks:

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-VL-Embedding-2B","input":["hello"],"encoding_format":"float"}'
```

---

## Index images (Phase 1)

Index documents as usual (multimodal indexing runs during document create/update if enabled):

```bash
haiku-rag add-src /path/to/doc.pdf
```

---

## Search images

### Text → image

```bash
haiku-rag search-image-text "architecture diagram"
```

### Image → image

```bash
haiku-rag search-image /path/to/query.png
```

---

## Visualize results

Search returns `asset_id` values. To see what was actually indexed:

```bash
haiku-rag visualize-asset <asset_id> --mode crop
haiku-rag visualize-asset <asset_id> --mode page
```

!!! note
    Terminals often do not render images inline (you may see gray blocks).
    Prefer opening the saved PNG in your OS viewer via:
    - `haiku-rag visualize-asset ...` (prints the saved image path)
    - `haiku-rag inspect` (`o` key inside the visual modals)

---

## Minimal accuracy validation (recommended)

Generate a quick “sanity” dataset from your DB and compute recall@k + MRR:

```bash
uv run evaluations mm-build --db /path/to/your.lancedb --config /path/to/haiku.rag.yaml --out ./mm_eval_out --n 50
uv run evaluations mm ./mm_eval_out/mm_eval.jsonl --db /path/to/your.lancedb --config /path/to/haiku.rag.yaml --k 1,5,10 --limit 10
```
