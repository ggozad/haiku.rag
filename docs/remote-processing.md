# Remote processing

haiku.rag can send conversion and chunking to [docling-serve](https://github.com/docling-project/docling-serve), a REST service running docling, instead of running docling in-process. It moves docling's memory and CPU out of the haiku.rag process, and lets `haiku.rag-slim` run without the `docling` extra. haiku.rag is tested against docling-serve 1.32.0.

## Running docling-serve

`examples/docker/docker-compose.yml` runs two docling-serve replicas beside the ingester and the MCP server. To run one by hand:

```bash
docker run -p 5001:5001 quay.io/docling-project/docling-serve:v1.32.0
```

`-e DOCLING_SERVE_ENABLE_UI=true` adds its web UI, for debugging.

## Configuration

```yaml
processing:
  converter: docling-serve
  chunker: docling-serve

providers:
  docling_serve:
    base_url: http://localhost:5001
    api_key: ""   # sent when docling-serve requires one
    timeout: 300  # seconds per HTTP call: submit, poll, result
```

`timeout` bounds each HTTP call, not the whole conversion. The [processing options](configuration/processing.md) apply as they do locally, except `fetch_remote_images`, `fetch_headers` and `infer_furniture`, which docling-serve ignores. With `chunker: docling-serve`, the OCR options (`do_ocr`, `force_ocr`, `ocr_engine`, `ocr_lang`) are sent to the chunking API too. `do_ocr: false` there avoids OCR model downloads in a read-only container.

## Several instances

`base_url` also takes a list. Jobs round-robin across the entries, and each job's submit, poll and result stay on one instance, since task ids are local to it:

```yaml
providers:
  docling_serve:
    base_url:
      - http://gpu-1:5001
      - http://cpu-1:5001
      - http://cpu-2:5001
    max_attempts: 3
    circuit_breaker:
      failure_threshold: 3
      cooldown_s: 30.0
```

The round-robin counter is per process, so separate ingester or client processes pick independently. When an instance fails or returns 5xx, the request moves to another, up to `max_attempts`, and that instance's circuit breaker opens so later jobs skip it until `cooldown_s` elapses. An external load balancer can front docling-serve only in its RQ mode, where task state is shared in Redis.

A default docling-serve instance processes one task at a time (`DOCLING_SERVE_ENG_LOC_NUM_WORKERS` raises it). Start `ingester.workers.worker_count` at 1–2× the number of `base_url` entries, which overlaps one job's fetch, embed and store with another's conversion.

## Picture descriptions

With `processing.pictures: description`, docling-serve calls the vision model itself, so:

- it must run with `DOCLING_SERVE_ENABLE_REMOTE_SERVICES=true`, since it blocks outbound calls by default
- the model's `base_url` in `processing.conversion_options.picture_description.model` must be reachable from the container: `http://host.docker.internal:11434` for Ollama on the host, not `localhost`

See [Picture handling](configuration/processing.md#picture-handling).

## Operations

Long-running docling-serve containers grow in memory ([docling-serve #366](https://github.com/docling-project/docling-serve/issues/366), [#474](https://github.com/docling-project/docling-serve/issues/474)), from leaks in docling itself. To keep ingesting through it:

- set a memory limit on each container (`mem_limit` in Compose, `resources.limits.memory` in Kubernetes) above your largest expected job, with `restart: unless-stopped`
- run several replicas in the `base_url` list, so one restarting does not stop ingest
- set `processing.split_pages` for large PDFs, so each slice is its own task, see [Large PDFs](configuration/processing.md#large-pdfs-and-docling-memory)
