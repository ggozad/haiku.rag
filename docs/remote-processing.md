# Remote Processing

`haiku.rag` can use [docling-serve](https://github.com/docling-project/docling-serve) for remote document processing and chunking, offloading resource-intensive operations to a dedicated service.

## Overview

docling-serve is a REST API service that provides:

- Document conversion (PDF, DOCX, PPTX, images, etc.)
- Intelligent chunking with structure preservation
- OCR capabilities for scanned documents
- Table and figure extraction

## When to Use docling-serve

**Use local processing (default) when:**

- Working with small to medium document volumes
- Running on development machines
- Want zero external dependencies
- Processing simple document formats

**Use docling-serve when:**

- Processing large volumes of documents
- Working with complex PDFs requiring OCR
- Running in production environments
- Want to separate compute-intensive tasks
- Need to scale document processing independently

## Setup

### Docker Compose (Recommended)

The easiest way to use haiku.rag with docling-serve is using the slim Docker image with docker-compose. See `examples/docker/docker-compose.yml` for a complete setup that includes both services.

### Running docling-serve Manually

See the [official docling-serve repository](https://github.com/docling-project/docling-serve) for installation options. The quickest way is using Docker:

```bash
docker run -p 5001:5001 -e DOCLING_SERVE_ENABLE_UI=1 quay.io/docling-project/docling-serve
```

### Configuration

Configure haiku.rag to use docling-serve. See the [Document Processing](configuration/processing.md) guide for all available options.

```yaml
# haiku.rag.yaml
processing:
  converter: docling-serve  # Use remote conversion
  chunker: docling-serve    # Use remote chunking

providers:
  docling_serve:
    base_url: http://localhost:5001
    api_key: ""     # Optional API key for authentication
    timeout: 300    # Request timeout in seconds
```

## Features

### Remote Document Conversion

When `converter: docling-serve` is configured, documents are sent to the docling-serve API for conversion:

```python
from haiku.rag.client import HaikuRAG

async with HaikuRAG() as client:
    # PDF is processed by docling-serve
    doc = await client.create_document_from_source("complex.pdf")
```

### Remote Chunking

When `chunker: docling-serve` is configured, chunking is performed remotely:

```yaml
processing:
  chunker: docling-serve
  chunker_type: hybrid              # or hierarchical
  chunk_size: 256
  chunking_tokenizer: "Qwen/Qwen3-Embedding-0.6B"
  chunking_merge_peers: true
  chunking_use_markdown_tables: false
```

## Advanced Configuration

### Custom Tokenizers

You can use any HuggingFace tokenizer model:

```yaml
processing:
  chunking_tokenizer: "bert-base-uncased"  # Or any HF model
```

### Chunking Strategies

**Hybrid Chunking** (default):

- Best for most documents
- Preserves semantic boundaries
- Structure-aware splitting

**Hierarchical Chunking**:

- Maintains document hierarchy
- Better for deeply nested documents
- Preserves parent-child relationships

```yaml
processing:
  chunker_type: hierarchical
```

### Table Handling

Control how tables are represented:

```yaml
processing:
  chunking_use_markdown_tables: true  # Preserve table structure
```

- `false` (default): Tables as narrative text
- `true`: Tables as markdown format

## Resources

- [docling-serve GitHub](https://github.com/docling-project/docling-serve)
- [docling-serve Documentation](https://github.com/docling-project/docling-serve#readme)
