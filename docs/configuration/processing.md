# Document Processing

This guide covers how haiku.rag converts and chunks documents. Continuous
ingestion (watching directories, polling HTTP / S3 / WebDAV sources) lives
in the [ingester](../ingester.md) service.

## Document Processing

Configure how documents are converted and chunked:

```yaml
processing:
  # Chunking configuration
  chunk_size: 256                            # Maximum tokens per chunk

  # Converter selection
  converter: docling-local                   # docling-local or docling-serve

  # Chunker selection and configuration
  chunker: docling-local                     # docling-local or docling-serve
  chunker_type: hybrid                       # hybrid or hierarchical
  chunking_tokenizer: "Qwen/Qwen3-Embedding-0.6B"  # HuggingFace model for tokenization
  chunking_merge_peers: true                 # Merge undersized successive chunks
  chunking_use_markdown_tables: false        # Use markdown tables vs narrative format

  # PDF /EmbeddedFiles attachments
  extract_pdf_attachments: true              # Ingest embedded files as separate Documents

  # Automatic title generation
  auto_title: false                          # Auto-generate titles on ingestion
  title_model:                               # LLM for title generation (fallback)
    provider: ollama
    name: gpt-oss
    enable_thinking: false

  # Conversion options (works with both local and remote converters)
  conversion_options:
    # OCR settings
    do_ocr: true                             # Enable OCR for bitmap content
    force_ocr: false                         # Replace existing text with OCR
    ocr_engine: auto                         # OCR engine: auto, easyocr, rapidocr, tesseract, tesserocr, ocrmac
    ocr_lang: []                             # OCR languages (e.g., ["en", "fr", "de"])

    # Table extraction
    do_table_structure: true                 # Extract table structure
    table_mode: accurate                     # fast or accurate
    table_cell_matching: true                # Match table cells back to PDF cells

    # Image settings
    images_scale: 2.0                        # Image scale factor
    generate_page_images: true               # Include rendered page images (for visualize_chunk)

    # VLM settings used when processing.pictures == "description" (see "Picture Handling" below)
    picture_description:
      model:
        provider: ollama
        name: ministral-3
  pictures: image                            # none | description | image
```

### Local vs Remote Processing

**Local processing** (default):

- Uses `docling` library locally
- No external dependencies
- Good for development and small workloads

**Remote processing** (docling-serve):

- Offloads processing to docling-serve API
- Better for heavy workloads and production
- Requires docling-serve instance (see [Remote processing setup](../remote-processing.md))

To use remote processing:

```yaml
processing:
  converter: docling-serve
  chunker: docling-serve

providers:
  docling_serve:
    base_url: http://localhost:5001
    api_key: "your-api-key"  # Optional
```

`base_url` also accepts a list — jobs round-robin across the entries, with
each job's submit / poll / result pinned to one instance (task IDs are
instance-local):

```yaml
providers:
  docling_serve:
    base_url:
      - http://gpu-1:5001
      - http://cpu-1:5001
      - http://cpu-2:5001
```

The round-robin counter is per-process — multiple concurrent ingester or
client processes pick independently, so the distribution evens out over many
jobs without coordination. For tight load balancing, failover, or health
checks, run a real load balancer (nginx `least_conn`, etc.) in front and
configure a single URL here.

**Tuning `ingester.workers.worker_count` for docling-serve users**: convert
is usually the throughput ceiling — a default docling-serve instance
processes one task at a time (configurable via `DOCLING_SERVE_ENG_LOC_NUM_WORKERS`
if you've set it). A reasonable starting point for `worker_count` is **1–2 ×
the number of `docling_serve.base_url` entries**: enough to overlap fetch /
embed / store of one job with the convert of another, without piling jobs
into docling-serve's internal queue beyond what its workers can chew through.
The ingester logs the worker / source / docling-serve counts on startup so
you can eyeball the ratio.

Conversion options work identically for both local and remote processing.

### Large PDFs and docling memory

Docling's parser is memory-hungry and has confirmed leaks in current versions
([docling #2209](https://github.com/docling-project/docling/issues/2209),
[#1343](https://github.com/docling-project/docling/issues/1343),
[#2954](https://github.com/docling-project/docling/issues/2954);
[docling-serve #366](https://github.com/docling-project/docling-serve/issues/366),
[#474](https://github.com/docling-project/docling-serve/issues/474)).
Single-pass conversion of 400-page PDFs can OOM a workstation in local mode,
and long-running docling-serve containers see RSS grow monotonically.

Mitigation in haiku.rag — set `processing.split_pages`:

```yaml
processing:
  split_pages: 10                  # 0 disables (default)
```

When `split_pages > 0`, PDFs are split at the byte level into N-page slices
(using pypdfium2, already bundled), each slice converted independently, then
merged back via `DoclingDocument.concatenate` — preserving page numbers and
re-indexing `self_ref` values across slices. Peak memory per conversion is
bounded by one slice's working set rather than the whole document; in
docling-serve mode each slice is also an independent task that lets the
server release task-local state between requests.

Recommendation: `10` is a sensible starting point for any consistently-large
PDF workload. Smaller slices reduce peak memory but multiply task overhead
(per-slice docling startup + HTTP round-trips for docling-serve). Cross-page
references (named destinations, multi-page link annotations) are dropped at
the split — accepted loss; haiku.rag doesn't surface them downstream.

**Operational note for long-running ingest**: even with `split_pages`,
docling's per-process leak rate is non-zero. For deployments running
continuously:

- *docling-serve mode*: set `mem_limit` on the container in Compose
  (or `resources.limits.memory` in Kubernetes) plus `restart: unless-stopped`
  so the kernel OOM-kills and the runtime restarts. Run multiple
  docling-serve replicas behind the round-robin `base_url` list above so a
  restart of one doesn't stop ingest.
- *docling-local mode*: the leak is inside the `haiku-ingester` process
  itself. Apply the same `mem_limit` + restart policy to the ingester
  container. Restarts are graceful — in-flight jobs land in the queue's
  reaper window and resume on next start.

**Note:** When using `chunker: docling-serve`, OCR options (`do_ocr`, `force_ocr`, `ocr_engine`, `ocr_lang`) from `conversion_options` are passed to the chunking API. This is useful when running docling-serve in a read-only container where OCR model downloads fail. Set `do_ocr: false` to disable OCR entirely.

### Conversion Options

The `conversion_options` section allows fine-grained control over document conversion. These options work with both `docling-local` and `docling-serve` converters.

#### OCR Settings

```yaml
conversion_options:
  do_ocr: true          # Enable OCR for bitmap/scanned content
  force_ocr: false      # Replace all text with OCR output
  ocr_engine: auto      # OCR engine selection
  ocr_lang: []          # List of OCR languages, e.g., ["en", "fr", "de"]
```

- **do_ocr**: When `true`, applies OCR to images and scanned pages. Disable for faster processing if documents contain only native text.
- **force_ocr**: When `true`, replaces existing text layers with OCR output. Useful for documents with poor text extraction.
- **ocr_engine**: Select the OCR engine to use. Options:
  - `auto` (default): Automatically select the best available engine
  - `easyocr`: EasyOCR - supports many languages, good accuracy
  - `rapidocr`: RapidOCR - fast processing
  - `tesseract`: Tesseract OCR
  - `tesserocr`: Tesseract via tesserocr Python binding
  - `ocrmac`: macOS native OCR (macOS only)
- **ocr_lang**: List of language codes for OCR. Empty list uses default language detection. Examples: `["en"]`, `["en", "fr", "de"]`.

#### Table Extraction

```yaml
conversion_options:
  do_table_structure: true    # Extract structured table data
  table_mode: accurate        # fast or accurate
  table_cell_matching: true   # Match cells back to PDF
```

- **do_table_structure**: When `true`, extracts table structure. Disable for faster processing if tables aren't important.
- **table_mode**:
  - `accurate`: Better table structure recognition (slower)
  - `fast`: Faster processing with simpler table detection
- **table_cell_matching**: When `true`, matches detected table cells back to PDF cells. Disable if tables have merged cells across columns.

#### Image Settings

```yaml
conversion_options:
  images_scale: 2.0               # Image resolution scale factor
  generate_page_images: true      # Include rendered page images
  fetch_remote_images: true       # Fetch external <img src> URLs in HTML/MD
```

- **images_scale**: Scale factor for extracted images. Higher values = better quality but larger size. Typical range: 1.0-3.0.
- **generate_page_images**: When `true` (default), rendered images of each PDF page are included in the document. Required for `visualize_chunk()` to show visual grounding. When `false`, page images are excluded to reduce document size.
- **fetch_remote_images**: When `true` (default), HTML and Markdown inputs have their external `<img src="https://...">` URLs fetched and stored as picture bytes. Set `false` for air-gapped ingest. Applies only to `docling-local`. **docling-serve doesn't fetch external `<img>` URLs** (the `ConvertDocumentsOptions` API exposes no equivalent flag, and HTML falls through to docling's `fetch_images=False` default); HTML ingested via docling-serve produces picture items with `picture_data=NULL`. Use `converter: docling-local` if you need image bytes from HTML/Markdown.

#### External image fetching

For HTML and Markdown inputs, docling fetches images referenced by URL when `fetch_remote_images: true`. Pictures end up in `document_items.picture_data` alongside the ones extracted from PDF/DOCX/PPTX. Inherited from docling:

- **SSRF guard**: hostnames must resolve to a global IP. Loopback, private (RFC1918), link-local, reserved, multicast, and unspecified addresses are rejected.
- **Size cap**: 20 MB per image (sent as a `Range` header), enforced again when streaming the response body.
- **Timeouts**: 5 s connect, 30 s read.
- **SVGs are skipped** (PIL cannot rasterize them).
- **`data:` URIs** are decoded inline (no network).
- **`file://` URIs** are *not* fetched. `enable_local_fetch` stays off to keep the SSRF surface narrow for arbitrary HTML/MD content.

Per-image failures (404, timeout, oversized, unreadable) leave that picture as a placeholder with `picture_data=NULL`. The rest of the document still ingests.

**Scope of conversion options across formats:**

| Input | OCR / table options | `images_scale` / `generate_page_images` | `pictures` | `fetch_remote_images` |
|---|---|---|---|---|
| `.pdf` | ✅ | ✅ | ✅ | n/a |
| `.png` / `.jpg` / `.jpeg` / `.bmp` / `.tiff` / `.webp` | ✅ | ✅ | ✅ | n/a |
| `.html` / `.xhtml` | n/a (markup-based) | n/a | ✅ on embedded pictures | ✅ |
| `.md` / `.qmd` / `.rmd` | n/a | n/a | ✅ on embedded pictures | ✅ (only `<img>` HTML blocks; native `![alt](url)` syntax is not fetched by docling) |
| `.docx` / `.pptx` | n/a | n/a | ✅ on embedded pictures | n/a |
| Other (`.csv`, `.xlsx`, `.adoc`, `.tex`, `.xml`) | n/a | n/a | n/a | n/a |

#### Picture Handling

`processing.pictures` picks one of three modes:

| Mode | Picture-image generation in docling | Bytes stored in `document_items.picture_data` | VLM runs at ingest |
|---|---|---|---|
| `none` | off | no | no |
| `description` | on | yes | yes |
| `image` (default) | on | yes | no |

Use `none` when you don't need picture content (e.g. very large reference manuals where RAM is tight). Use `description` to weave VLM-generated text into chunk content and keep bytes for later. Use `image` (default) to keep bytes without paying the VLM cost. The prompt is configurable under `prompts.picture_description`. See [Prompts](prompts.md).

```yaml
processing:
  pictures: description           # none | description | image
  conversion_options:
    picture_description:          # only consulted when pictures == "description"
      model:
        provider: ollama          # any OpenAI-compatible /v1/chat/completions provider
        name: ministral-3
      timeout: 90
      max_tokens: 200
```

!!! warning "Breaking change"
    `processing.conversion_options.picture_description.enabled` is replaced by `processing.pictures`. Map `enabled: true` → `pictures: description`, `enabled: false` → `pictures: image`. The pre-April-30 `generate_picture_images` flag also no longer exists. Use `pictures: none` for the old opt-out.

**Switching modes on an existing database** doesn't require reingesting when the bytes are already stored:

- `image` → `description`: `haiku-rag rebuild --descriptions` runs the VLM over stored bytes and re-chunks. Skips the docling parse entirely.
- `description` → `image`: `haiku-rag rebuild --rechunk` recomposes chunk text from the stripped docling blob without descriptions.
- Switching to/from `none`: a full reingest is needed since the bytes either weren't stored or need to be discarded.

When using `converter: docling-serve`, the VLM is invoked from docling-serve rather than haiku.rag. See [Remote processing](../remote-processing.md#vlm-picture-description-with-docling-serve).

#### Pictures × embedder × QA model: how the pieces compose

Three independent settings drive ingest, retrieval, and QA:

| Setting | Question it answers | Values |
|---|---|---|
| `processing.pictures` | Generate and/or describe pictures at ingest? | `none` / `description` / `image` (default) |
| `embeddings.model.provider` | Can the embedder index image content? | text-only (`ollama`, `openai`, `cohere`, `sentence-transformers`) vs `vllm` (multimodal) |
| `qa.model.vision` | Can the QA model interpret images? | `false` (default) / `true` |

**What gets stored** by `pictures` × embedder:

| `pictures` | Embedder | Text chunks | Synthetic picture chunks |
|---|---|---|---|
| `none` | any | text only (caption/surrounding) | none |
| `image` | text-only | text only (caption/surrounding) | none |
| `image` | multimodal | text only | one per picture, vector = image embedding |
| `description` | text-only | text + descriptions | none |
| `description` | multimodal | text + descriptions | one per picture, vector = image embedding |

**What QA receives** at search time:

- `qa.model.vision: false` — text chunks only (descriptions, when present, answer figure questions in prose).
- `qa.model.vision: true` — text chunks + raw picture bytes via `BinaryContent`. The model reads figures directly. Requires `pictures != none` so the bytes exist.

`qa.model.vision` is independent of ingestion. Flipping it never requires reingesting. Setting `vision: true` against a text-only model causes silent acceptance and confabulation on Ollama and a 400 on OpenAI. Default `false` is the safe choice.

**Recommended combinations:**

| Use case | `processing.pictures` | Embedder | `qa.model.vision` |
|---|---|---|---|
| Pure text RAG, no figures, lowest RAM | `none` | text-only | `false` |
| Text RAG, store figure bytes for later | `image` | text-only | `false` |
| Text RAG, figures answered through descriptions | `description` | text-only | `false` |
| Vision QA on figure-rich docs (no cross-modal search) | `image` or `description` | text-only | `true` |
| Cross-modal search + vision QA | `image` or `description` | multimodal | `true` |
| Cross-modal search, text QA only | `description` | multimodal | `false` |

### Chunking Strategies

**Hybrid chunking** (default):
- Structure-aware chunking
- Respects document boundaries
- Best for most use cases

**Hierarchical chunking**:
- Creates hierarchical chunk structure
- Preserves document hierarchy
- Useful for complex documents

### Chunk Size

```yaml
processing:
  chunk_size: 256  # Maximum tokens per chunk
```

Context expansion settings (for enriching search results with surrounding content) are configured in the `search` section. See [Search Settings](qa.md#search-settings).

### Table Serialization

Control how tables are represented in chunks:

```yaml
processing:
  chunking_use_markdown_tables: false  # Default: narrative format
```

- `false`: Tables as narrative text ("Value A, Column 2 = Value B")
- `true`: Tables as markdown (preserves table structure)

### Automatic Title Generation

Enable automatic title generation during document ingestion:

```yaml
processing:
  auto_title: true
  title_model:
    provider: ollama
    name: gpt-oss
    enable_thinking: false
```

When `auto_title` is enabled, haiku.rag attempts to extract a title for each document during ingestion using a two-tier approach:

1. **Structural extraction** (free, no model calls): Scans the DoclingDocument for semantic labels (HTML `<title>` tags, `<h1>` headings, PDF title blocks, and section headers)
2. **LLM fallback**: When no structural title is found (e.g., plain text), generates a title using the configured `title_model`

Priority order: HTML `<title>` (furniture layer) → h1/PDF title (body layer) → first section header → LLM generation.

Explicit titles passed via `title=` parameter always take precedence and are never overridden. When updating documents, existing titles are preserved. Auto-generation only applies to untitled documents.

To generate titles for existing untitled documents, use [`rebuild --title-only`](../cli.md#rebuild-database).

### PDF Embedded Attachments

A PDF can carry other files inside it via the `/EmbeddedFiles` table (signed memos, appendices, supporting documents). With `extract_pdf_attachments: true` (the default), each embedded file is ingested as a separate Document linked to the wrapper through `metadata.parent_uri`:

```yaml
processing:
  extract_pdf_attachments: true
```

```python
# After ingesting a PDF with two attachments:
parent = await client.create_document_from_source("/path/to/parent.pdf")
children = await client.list_documents(
    filter=f"metadata LIKE '%\"parent_uri\": \"{parent.uri}\"%'"
)
# children: 2 Documents, each with parent.uri in metadata.parent_uri,
# URIs like file:///path/to/parent.pdf#attachment=memo.pdf
```

Behavior:

- Children inherit the standard ingest metadata (`content_type`, `md5`, `source_revision`) plus `parent_uri`.
- Re-ingesting the wrapper reconciles its current attachment set against existing children: new files are added, changed bytes update in place, and dropped names are deleted.
- `delete_document(parent_id)` cascades through `parent_uri` and removes all children.
- Nested attachments (a PDF whose attachment is itself a PDF with attachments) recurse up to 3 levels. Deeper chains log a warning and skip.
- Attachments whose extension or content type the converter does not support log a warning and are skipped without aborting the rest of the set.

Set `extract_pdf_attachments: false` to ingest only the wrapper.

## Continuous ingestion

For automatic ingestion of local directories, S3 buckets, or HTTP
sources (with filtering, retries, and a dead-letter queue), see the
[Ingester](../ingester.md) page.
