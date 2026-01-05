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

  # Conversion options (works with both local and remote converters)
  conversion_options:
    # OCR settings
    do_ocr: true                             # Enable OCR for bitmap content
    force_ocr: false                         # Replace existing text with OCR
    ocr_lang: []                             # OCR languages (e.g., ["en", "fr", "de"])

    # Table extraction
    do_table_structure: true                 # Extract table structure
    table_mode: accurate                     # fast or accurate
    table_cell_matching: true                # Match table cells back to PDF cells

    # Image settings
    images_scale: 2.0                        # Image scale factor
    generate_picture_images: false           # Include embedded images in output

    # VLM picture description (optional)
    picture_description:
      enabled: false                         # Enable VLM image descriptions
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
  ocr_lang: []          # List of OCR languages, e.g., ["en", "fr", "de"]
```

- **do_ocr**: When `true`, applies OCR to images and scanned pages. Disable for faster processing if documents contain only native text.
- **force_ocr**: When `true`, replaces existing text layers with OCR output. Useful for documents with poor text extraction.
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
  generate_picture_images: false  # Include embedded images in output
```

- **images_scale**: Scale factor for extracted images. Higher values = better quality but larger size. Typical range: 1.0-3.0.
- **generate_picture_images**: When `true`, embedded images (figures, diagrams) are included as base64-encoded data in the document. When `false` (default), images are excluded to reduce chunk size and avoid context bloat.

#### Picture Description (VLM)

Use a Vision Language Model (VLM) to automatically describe images in documents. Descriptions become searchable text, improving RAG retrieval for visual content.

```yaml
conversion_options:
  picture_description:
    enabled: true                  # Enable VLM picture description
    model:
      provider: ollama             # ollama, openai, or custom
      name: ministral-3            # VLM model name
    prompt: "Describe this image in detail. Be precise and concise."
    timeout: 90                    # Request timeout in seconds
    max_tokens: 200                # Maximum tokens in response
```

**Configuration options:**

- **enabled**: When `true`, each embedded image is sent to a VLM for description. Requires `generate_picture_images` to be `true` (automatically enabled).
- **model**: Standard model configuration
  - `provider`: `ollama` (default), `openai`, or use `base_url` for custom endpoints
  - `name`: Model name (e.g., `ministral-3`, `granite3.2-vision`, `gpt-4-vision`)
  - `base_url`: Optional custom API endpoint for vLLM, LM Studio, etc.
- **prompt**: Instruction for the VLM when describing images
- **timeout**: Request timeout in seconds
- **max_tokens**: Maximum tokens in the VLM response

**Using with Ollama:**

```yaml
conversion_options:
  picture_description:
    enabled: true
    model:
      provider: ollama
      name: ministral-3
```

Requires Ollama running with a vision-capable model:

```bash
ollama pull ministral-3
ollama serve
```

**Using with vLLM or custom endpoints:**

```yaml
conversion_options:
  picture_description:
    enabled: true
    model:
      provider: openai           # Use OpenAI-compatible API format
      name: granite-vision
      base_url: http://my-vllm-server:8000
```

**How it works:**

1. During PDF conversion, docling extracts embedded images
2. Each image is sent to the configured VLM for description
3. Descriptions are added as annotations on the image
4. When exported to markdown, descriptions appear as searchable text

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
    timeout: 300              # Request timeout in seconds
```

Conversion options work identically for both local and remote processing.

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
