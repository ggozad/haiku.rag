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
- Separating compute-intensive tasks
- Scaling document processing independently

## Setup

### Docker Compose (Recommended)

The slim Docker image with docker-compose is the recommended setup. See `examples/docker/docker-compose.yml` for a complete configuration that includes both services.

### Running docling-serve Manually

See the [official docling-serve repository](https://github.com/docling-project/docling-serve) for installation options. The quickest way is using Docker:

```bash
docker run -p 5001:5001 quay.io/docling-project/docling-serve
```

To enable the web UI for debugging:

```bash
docker run -p 5001:5001 -e DOCLING_SERVE_ENABLE_UI=true quay.io/docling-project/docling-serve
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
    api_key: ""  # Optional API key for authentication
```

For converter / chunker config options (chunking strategy, tokenizer,
OCR, table handling, picture description), see
[Document Processing](configuration/processing.md). The configuration is
identical between `docling-local` and `docling-serve` modes — this page
covers only what's specific to running docling-serve as a separate
service.

## VLM picture description with docling-serve

When `processing.pictures = "description"` and `converter: docling-serve`,
the VLM API calls are made by the docling-serve container, not by
haiku.rag. Two deployment caveats:

### Enable remote services

docling-serve blocks outbound calls by default. Enable them by setting
`DOCLING_SERVE_ENABLE_REMOTE_SERVICES=true` on the container:

```bash
docker run -p 5001:5001 \
  -e DOCLING_SERVE_ENABLE_REMOTE_SERVICES=true \
  quay.io/docling-project/docling-serve
```

### Reach host services from inside the container

If your VLM (e.g. Ollama) runs on the host while docling-serve runs in
Docker, set the VLM's `base_url` in
`processing.conversion_options.picture_description.model` to
`http://host.docker.internal:11434` rather than `localhost`. See
[Document Processing → Picture Handling](configuration/processing.md#picture-handling)
for the full config snippet.

## Operational notes

Long-running docling-serve containers see CPU memory grow monotonically
([docling-serve #366](https://github.com/docling-project/docling-serve/issues/366),
[#474](https://github.com/docling-project/docling-serve/issues/474)). The
underlying parser leaks are in core docling
([#2209](https://github.com/docling-project/docling/issues/2209),
[#1343](https://github.com/docling-project/docling/issues/1343)) and affect
docling-local too.

Recommended deployment shape:

- Set `mem_limit` on the docling-serve container (or `resources.limits.memory`
  in Kubernetes) at a value comfortably above your largest expected job.
- Combine with `restart: unless-stopped` so the runtime restarts when the
  kernel OOM-kills.
- Run multiple docling-serve replicas behind haiku.rag's round-robin
  `providers.docling_serve.base_url` list (see
  [Document Processing](configuration/processing.md)). A restart of one
  replica doesn't stop ingest.
- In haiku.rag, set `processing.split_pages` for large-PDF workloads so each
  slice is an independent docling-serve task and the per-task working set
  stays bounded.

## Resources

- [docling-serve GitHub](https://github.com/docling-project/docling-serve)
- [docling-serve Documentation](https://github.com/docling-project/docling-serve#readme)
