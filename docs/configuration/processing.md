# Document processing

This guide covers how haiku.rag converts and chunks documents. Continuous
ingestion (watching directories, polling HTTP / S3 / WebDAV sources) lives
in the [ingester](../ingester.md) service.

## Settings

The processing settings, with their defaults:

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
    name: qwen3.8
    thinking: false

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
        name: qwen3.8
  pictures: image                            # none | description | image
```

### Local and remote conversion

`docling-local` runs docling in-process and needs the `docling` extra. `docling-serve` sends documents to a [docling-serve](../remote-processing.md) service, including a round-robin list of instances with failover. Conversion options apply to both converters, except `fetch_remote_images`, `fetch_headers` and `infer_furniture`, which only `docling-local` reads.

### Large PDFs and docling memory

Docling's parser is memory-hungry and leaks over a long-running process ([docling #2209](https://github.com/docling-project/docling/issues/2209), [#1343](https://github.com/docling-project/docling/issues/1343), [#2954](https://github.com/docling-project/docling/issues/2954)). Single-pass conversion of a 400-page PDF can exhaust a workstation's memory in local mode. `processing.split_pages` bounds it:

```yaml
processing:
  split_pages: 10                  # 0 disables (default)
```

With `split_pages > 0`, PDFs are split into N-page slices, each converted on its own, then merged with `DoclingDocument.concatenate`, which keeps page numbers and re-indexes `self_ref` values. Peak memory is one slice's working set. Under docling-serve each slice is a separate task.

A slice sees only its own pages, so the result differs from single-pass conversion near each boundary: a paragraph spanning one stays two items, and a caption near one can order differently. On a 9-page document that is about one text item and one chunk per boundary. No text is lost or duplicated. Cross-page references (named destinations, multi-page link annotations) are dropped at the split. Changing the setting re-chunks the boundary regions on the next ingest.

`10` is a starting point for a consistently large PDF workload. Smaller slices lower peak memory and add per-slice overhead.

The leak is not bounded by slicing. Under `docling-local` it grows inside the `haiku-ingester` process, so give its container a memory limit and a restart policy: in-flight jobs are reclaimed by the queue's reaper after a restart. For docling-serve, see [Remote processing](../remote-processing.md#operations).

### Conversion options

The `conversion_options` section allows fine-grained control over document conversion. Both converters read these options, except `fetch_remote_images`, `fetch_headers` and `infer_furniture`, which apply to `docling-local` only.

#### PDF parsing

```yaml
processing:
  conversion_options:
    pdf_backend: docling_parse   # docling_parse, threaded_docling_parse, pypdfium2
```

- **pdf_backend**: The parser docling uses to read a PDF. Both converters receive it. The parsers segment a document differently, so a change brings different items, chunk boundaries and chunk ids on the next ingest.
  - `docling_parse` (default): serialized page parsing
  - `threaded_docling_parse`: concurrent page parsing, and docling's own default. On some documents the conversion never returns. Over ten arXiv papers against `docling_parse`: one table undetected, 7% fewer table cells, 9% faster.
  - `pypdfium2`: faster and simpler, less layout detail. Over the same ten papers: 7% fewer words and 28% fewer table cells.

#### OCR settings

```yaml
processing:
  conversion_options:
    do_ocr: true          # Enable OCR for bitmap/scanned content
    force_ocr: false      # Replace all text with OCR output
    ocr_engine: auto      # OCR engine selection
    ocr_lang: []          # List of OCR languages, e.g., ["en", "fr", "de"]
```

- **do_ocr**: When `true`, applies OCR to images and scanned pages. Disable it for documents with only embedded text, which converts faster.
- **force_ocr**: When `true`, replaces existing text layers with OCR output, for documents whose embedded text is poor.
- **ocr_engine**: Select the OCR engine to use. Options:
  - `auto` (default): Automatically select the best available engine
  - `easyocr`: EasyOCR - supports many languages, good accuracy
  - `rapidocr`: RapidOCR - fast processing
  - `tesseract`: the Tesseract command-line binary
  - `tesserocr`: Tesseract through the tesserocr Python binding
  - `ocrmac`: macOS native OCR (macOS only)
- **ocr_lang**: List of language codes for OCR. Empty list uses default language detection. Examples: `["en"]`, `["en", "fr", "de"]`.

#### Table extraction

```yaml
processing:
  conversion_options:
    do_table_structure: true    # Extract structured table data
    table_mode: accurate        # fast or accurate
    table_cell_matching: true   # Match cells back to PDF
```

- **do_table_structure**: When `true`, extracts table structure. Disabling it converts faster, without table structure.
- **table_mode**:
  - `accurate`: Better table structure recognition (slower)
  - `fast`: Faster processing with simpler table detection
- **table_cell_matching**: When `true`, matches detected table cells back to PDF cells. Disable if tables have merged cells across columns.

#### Image settings

```yaml
processing:
  conversion_options:
    images_scale: 2.0               # Image resolution scale factor
    generate_page_images: true      # Include rendered page images
    fetch_remote_images: true       # Fetch external <img src> URLs in HTML/MD
    infer_furniture: false          # Keep HTML content before the first heading
```

- **images_scale**: Scale factor for extracted images. Higher values = better quality but larger size. Typical range: 1.0-3.0.
- **generate_page_images**: When `true` (default), rendered images of each PDF page are included in the document. Required for `visualize_chunk()` to show visual grounding. When `false`, page images are excluded to reduce document size.
- **fetch_remote_images**: When `true` (default), HTML and Markdown inputs have their external `<img src="https://...">` URLs fetched and stored as picture bytes. Set `false` for air-gapped ingest. Applies only to `docling-local`. docling-serve has no such option and does not fetch external images, so HTML it converts has picture items without bytes (`picture_data=NULL`).
- **fetch_headers**: HTTP headers sent with those image fetches. Default: a `User-Agent` naming haiku.rag. `docling-local` only.
- **infer_furniture**: When `false` (default), everything in an HTML page is document content. When `true`, docling files whatever precedes the first heading as page furniture and leaves it out of the document, which removes site banners and navigation on web pages but also removes an article's lead paragraph and infobox. Applies only to `docling-local`; docling-serve keeps docling's rule, so HTML converted there loses the content before its first heading.

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

| Input | OCR / table options | `images_scale` / `generate_page_images` | `pictures` | `fetch_remote_images` | `infer_furniture` |
|---|---|---|---|---|---|
| `.pdf` | ✅ | ✅ | ✅ | n/a | n/a |
| `.png` / `.jpg` / `.jpeg` / `.bmp` / `.tiff` / `.webp` | ✅ | ✅ | ✅ | n/a | n/a |
| `.html` / `.xhtml` | n/a (markup-based) | n/a | ✅ on embedded pictures | ✅ | ✅ |
| `.md` / `.qmd` / `.rmd` | n/a | n/a | ✅ on embedded pictures | ✅ (`<img>` HTML blocks only, not `![alt](url)`) | n/a |
| `.docx` / `.pptx` | n/a | n/a | ✅ on embedded pictures | n/a | n/a |
| Other (`.csv`, `.xlsx`, `.adoc`, `.tex`, `.xml`, `.eml`, `.msg`) | n/a | n/a | n/a | n/a | n/a |

#### Picture handling

`processing.pictures` picks one of three modes:

| Mode | Picture-image generation in docling | Bytes stored in `document_items.picture_data` | VLM runs at ingest |
|---|---|---|---|
| `none` | off | no | no |
| `description` | on | yes | yes |
| `image` (default) | on | yes | no |

Not every picture becomes a picture chunk. Identical picture bytes within a document produce a single chunk, so a watermark or logo repeated on every page embeds once. Pictures smaller than `processing.min_picture_size` pixels on their smaller side (default 64, `0` disables) are skipped entirely. Filtered pictures keep their bytes in `document_items`, so context expansion and vision QA still see them.

Use `none` when you don't need picture content (e.g. very large reference manuals where RAM is tight). Use `description` to include VLM-generated text in chunk content and keep bytes for later. Use `image` (default) to keep bytes without paying the VLM cost. The prompt is configurable under `prompts.picture_description`. See [Prompts](prompts.md).

```yaml
processing:
  pictures: description           # none | description | image
  conversion_options:
    picture_description:          # only consulted when pictures == "description"
      model:
        provider: ollama
        name: qwen3.8
        thinking: false
      timeout: 90
      max_tokens: 200
```

The model is called at `/v1/chat/completions` under its `base_url`, which may be written with or without `/v1`. Without a `base_url`, only `ollama` and `openai` are accepted. Writing a `model` block replaces the default `thinking: false`, so set it explicitly on a model that thinks by default. `timeout` and `max_tokens` bound each call during conversion. `rebuild --descriptions` runs the same model through Pydantic AI and does not apply them.

**Switching modes on an existing database** doesn't require reingesting when the bytes are already stored:

- `image` → `description`: `haiku-rag rebuild --descriptions` runs the VLM over stored bytes and re-chunks. Skips the docling parse entirely.
- `description` → `image`: `haiku-rag rebuild --rechunk` recomposes chunk text from the stripped docling blob without descriptions.
- Switching to/from `none`: a full reingest is needed since the bytes either weren't stored or need to be discarded.

When using `converter: docling-serve`, the VLM is invoked from docling-serve rather than haiku.rag. See [Remote processing](../remote-processing.md#picture-descriptions).

#### Pictures × embedder × QA model: how the pieces compose

Three independent settings drive ingest, retrieval, and QA:

| Setting | Question it answers | Values |
|---|---|---|
| `processing.pictures` | Generate and/or describe pictures at ingest? | `none` / `description` / `image` (default) |
| `embeddings.model.multimodal` | Can the embedder index image content? | `false` (default, text-only) / `true` (supported on `vllm`, `openrouter`, `voyageai`, `cohere`) |
| `qa.model.vision` | Can the QA model interpret images? | `false` / `true` (default) |

The Embedder column below is driven by `embeddings.model.multimodal`, not the provider name. A vision-capable model under a text-only configuration still indexes no images, and an image-only document then produces zero chunks. See [Multimodal embedders](providers.md#multimodal-embedders).

**What gets stored** by `pictures` × embedder:

| `pictures` | Embedder | Text chunks | Synthetic picture chunks |
|---|---|---|---|
| `none` | any | text only (caption/surrounding) | none |
| `image` | text-only | text only (caption/surrounding) | none |
| `image` | multimodal | text only | one per distinct picture, vector = image embedding |
| `description` | text-only | text + descriptions | none |
| `description` | multimodal | text + descriptions | one per distinct picture, vector = image embedding |

**What QA receives** at search time:

- `qa.model.vision: false`: text chunks only (descriptions, when present, answer figure questions in prose).
- `qa.model.vision: true`: text chunks and raw picture bytes via `BinaryContent`. The model reads figures directly. Requires `pictures != none` so the bytes exist.

`qa.model.vision` is independent of ingestion. Flipping it never requires reingesting. It declares what the model can read: the default `qwen3.8` is vision-capable, so the default is `true`. Set it `false` when pointing `qa.model` at a text-only model, where `true` causes silent acceptance and confabulation on Ollama and a 400 on OpenAI.

**Recommended combinations:**

| Use case | `processing.pictures` | Embedder | `qa.model.vision` |
|---|---|---|---|
| Pure text RAG, no figures, lowest RAM | `none` | text-only | `false` |
| Text RAG, store figure bytes for later | `image` | text-only | `false` |
| Text RAG, figures answered through descriptions | `description` | text-only | `false` |
| Vision QA on figure-rich docs (no cross-modal search) | `image` or `description` | text-only | `true` |
| Cross-modal search + vision QA | `image` or `description` | multimodal | `true` |
| Cross-modal search, text QA only | `description` | multimodal | `false` |

### Conversion timeout

```yaml
processing:
  conversion_timeout: 600   # seconds
```

Only `docling-local` reads this. A docling-serve conversion is bounded by
`providers.docling_serve.timeout` per HTTP call instead.

- **conversion_timeout**: How long one document may spend in conversion before
  it is abandoned with `ConversionTimeoutError`. It bounds the wait, not the
  work: the abandoned conversion keeps running in the background, holding a
  thread and its memory, and does not block process exit.

  PDFs and office formats share one docling converter, which an abandoned
  conversion keeps, so later conversions in that process raise
  `ConverterWedgedError`. Recovering needs a new process. The
  [ingester](../ingester.md#run-it-under-a-supervisor) exits for that reason,
  to be restarted. HTML and Markdown get a converter per call, so later
  conversions still run.

### Chunking strategies

`chunker_type` picks the docling chunker:

- `hybrid` (default): docling's `HybridChunker`. Starts from the document's structure, splits items longer than `chunk_size` tokens of `chunking_tokenizer`, and with `chunking_merge_peers` merges undersized neighbours under the same headings.
- `hierarchical`: docling's `HierarchicalChunker`. One chunk per document item (paragraph, list, table), with no token limit, so `chunk_size` does not apply.

### Chunk size

```yaml
processing:
  chunk_size: 256  # Maximum tokens per chunk
```

`chunk_size` applies to the `hybrid` chunker. How much surrounding content a search result carries is set in `search`, see [Search settings](qa.md#search-settings).

### Table serialization

Control how tables are represented in chunks:

```yaml
processing:
  chunking_use_markdown_tables: false  # Default: narrative format
```

- `false`: Tables as narrative text ("Value A, Column 2 = Value B")
- `true`: Tables as markdown (preserves table structure)

### Automatic title generation

Enable automatic title generation during document ingestion:

```yaml
processing:
  auto_title: true
  title_model:
    provider: ollama
    name: qwen3.8
    thinking: false
```

When `auto_title` is enabled, haiku.rag attempts to extract a title for each document during ingestion using a two-tier approach:

1. **Structural extraction** (free, no model calls): Scans the DoclingDocument for semantic labels (HTML `<title>` tags, `<h1>` headings, PDF title blocks, and section headers)
2. **LLM fallback**: When no structural title is found (e.g., plain text), generates a title using the configured `title_model`

Priority order: HTML `<title>` (furniture layer) → h1/PDF title (body layer) → first section header → LLM generation.

Explicit titles passed via `title=` parameter always take precedence and are never overridden. When updating documents, existing titles are preserved. Auto-generation only applies to untitled documents.

To generate titles for existing untitled documents, use [`rebuild --title-only`](../cli.md#rebuild).

### PDF embedded attachments

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
