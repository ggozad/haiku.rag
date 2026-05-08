# Document Processing & Monitoring

This guide covers how haiku.rag converts, chunks, and monitors documents.

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

    # VLM picture description (off by default; see "Picture Handling" below)
    picture_description:
      enabled: false
      model:
        provider: ollama
        name: ministral-3
```

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
```

- **images_scale**: Scale factor for extracted images. Higher values = better quality but larger size. Typical range: 1.0-3.0.
- **generate_page_images**: When `true` (default), rendered images of each PDF page are included in the document. Required for `visualize_chunk()` to show visual grounding. When `false`, page images are excluded to reduce document size.

#### Picture Handling

Picture bytes (figures, diagrams) are always extracted and stored in `document_items.picture_data` for every ingested document. The single configurable knob is whether a Vision Language Model (VLM) runs at ingest to generate textual descriptions:

```yaml
processing:
  conversion_options:
    picture_description:
      enabled: true              # default false
      model:
        provider: ollama         # any OpenAI-compatible /v1/chat/completions provider
        name: ministral-3
      timeout: 90
      max_tokens: 200
```

When `enabled: true`, each picture's description is woven into the chunk text and is searchable via FTS. The prompt is configurable under `prompts.picture_description` — see [Prompts](prompts.md).

**Switching the VLM on or off on an existing database** doesn't require reingesting (the bytes are already there):

- Off → on: `haiku-rag rebuild --descriptions` runs the VLM over stored bytes and re-chunks. Skips the docling parse entirely.
- On → off: `haiku-rag rebuild --rechunk` recomposes chunk text from the stripped docling blob without descriptions.

When using `converter: docling-serve`, the VLM is invoked from docling-serve rather than haiku.rag — see [Remote processing](../remote-processing.md#vlm-picture-description-with-docling-serve).

#### Pictures × embedder × QA model: how the pieces compose

Three independent settings drive ingest, retrieval, and QA:

| Setting | Question it answers | Values |
|---|---|---|
| `picture_description.enabled` | Should a VLM weave descriptions into chunk text at ingest? | `false` (default) / `true` |
| `embeddings.model.provider` | Can the embedder index image content? | text-only (`ollama`, `openai`, `cohere`, `sentence-transformers`) vs `vllm` (multimodal) |
| `qa.model.vision` | Can the QA model interpret images? | `false` (default) / `true` |

Picture bytes are always stored, regardless of these settings.

**What gets stored** by `enabled` × embedder:

| `enabled` | Embedder | Text chunks | Synthetic picture chunks |
|---|---|---|---|
| `false` | text-only | text only (caption/surrounding) | none |
| `false` | multimodal | text only | one per picture, vector = image embedding |
| `true` | text-only | text + descriptions | none |
| `true` | multimodal | text + descriptions | one per picture, vector = image embedding |

**What QA receives** at search time:

- `qa.model.vision: false` — text chunks only (descriptions, when present, answer figure questions in prose).
- `qa.model.vision: true` — text chunks + raw picture bytes via `BinaryContent`; the model reads figures directly.

`qa.model.vision` is independent of ingestion — flipping it never requires reingesting. Setting `vision: true` against a text-only model causes silent acceptance and confabulation on Ollama and a 400 on OpenAI; default `false` is the safe choice.

**Recommended combinations:**

| Use case | `picture_description.enabled` | Embedder | `qa.model.vision` |
|---|---|---|---|
| Pure text RAG, no figures | `false` | text-only | `false` |
| Text RAG, figures answered through descriptions | `true` | text-only | `false` |
| Vision QA on figure-rich docs (no cross-modal search) | `true` or `false` | text-only | `true` |
| Cross-modal search + vision QA | `true` or `false` | multimodal | `true` |
| Cross-modal search, text QA only | `true` | multimodal | `false` |

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

1. **Structural extraction** (free, no model calls): Scans the DoclingDocument for semantic labels — HTML `<title>` tags, `<h1>` headings, PDF title blocks, and section headers
2. **LLM fallback**: When no structural title is found (e.g., plain text), generates a title using the configured `title_model`

Priority order: HTML `<title>` (furniture layer) → h1/PDF title (body layer) → first section header → LLM generation.

Explicit titles passed via `title=` parameter always take precedence and are never overridden. When updating documents, existing titles are preserved — auto-generation only applies to untitled documents.

To generate titles for existing untitled documents, use [`rebuild --title-only`](../cli.md#rebuild-database).

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

Conversion options work identically for both local and remote processing.

**Note:** When using `chunker: docling-serve`, OCR options (`do_ocr`, `force_ocr`, `ocr_engine`, `ocr_lang`) from `conversion_options` are passed to the chunking API. This is useful when running docling-serve in a read-only container where OCR model downloads fail—set `do_ocr: false` to disable OCR entirely.

### Chunking Strategies

**Hybrid chunking** (default):
- Structure-aware chunking
- Respects document boundaries
- Best for most use cases

**Hierarchical chunking**:
- Creates hierarchical chunk structure
- Preserves document hierarchy
- Useful for complex documents

### Table Serialization

Control how tables are represented in chunks:

```yaml
processing:
  chunking_use_markdown_tables: false  # Default: narrative format
```

- `false`: Tables as narrative text ("Value A, Column 2 = Value B")
- `true`: Tables as markdown (preserves table structure)

### Chunk Size

```yaml
processing:
  chunk_size: 256  # Maximum tokens per chunk
```

Context expansion settings (for enriching search results with surrounding content) are configured in the `search` section. See [Search Settings](qa-research.md#search-settings).

## File Monitoring

Set directories to monitor for automatic indexing:

```yaml
monitor:
  directories:
    - /path/to/documents
    - /another_path/to/documents
```

### Filtering Monitored Files

Use gitignore-style patterns to control which files are monitored:

```yaml
monitor:
  directories:
    - /path/to/documents

  # Exclude specific files or directories
  ignore_patterns:
    - "*draft*"         # Ignore files with "draft" in the name
    - "temp/"           # Ignore temp directory
    - "**/archive/**"   # Ignore all archive directories
    - "*.backup"        # Ignore backup files

  # Only include specific files (whitelist mode)
  include_patterns:
    - "*.md"            # Only markdown files
    - "*.pdf"           # Only PDF files
    - "**/docs/**"      # Only files in docs directories
```

**How patterns work:**

1. **Extension filtering** - Only supported file types are considered
2. **Include patterns** - If specified, only matching files are included (whitelist)
3. **Ignore patterns** - Matching files are excluded (blacklist)
4. **Combining both** - Include patterns are applied first, then ignore patterns

**Common patterns:**

```yaml
# Only monitor markdown documentation, but ignore drafts
monitor:
  include_patterns:
    - "*.md"
  ignore_patterns:
    - "*draft*"
    - "*WIP*"

# Monitor all supported files except in specific directories
monitor:
  ignore_patterns:
    - "node_modules/"
    - ".git/"
    - "**/test/**"
    - "**/temp/**"
```

Patterns follow [gitignore syntax](https://git-scm.com/docs/gitignore#_pattern_format):

- `*` matches anything except `/`
- `**` matches zero or more directories
- `?` matches any single character
- `[abc]` matches any character in the set
