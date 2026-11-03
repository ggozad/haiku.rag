# Local search API

The local search service is `haiku-ingester serve` with its HTTP API: it watches
the registered folders, ingests what changes, and answers searches over the
indexed files. This page is the contract clients (desktop search front ends,
scripts) build against. [haiku-rag-macos](https://github.com/tolgaerdonmez/haiku-rag-macos)
runs it as a launchd agent on macOS, with a Raycast extension as its client.

- **Base URL:** `http://127.0.0.1:8765` (`ingester.api.host` and `ingester.api.port`).
  The service binds the loopback interface only.
- **Format:** JSON in and out, UTF-8. Errors are FastAPI's `{"detail": "..."}`
  with a 4xx or 5xx status.
- **Auth:** none by default. When `ingester.api.auth_token` is set, every
  endpoint except `GET /health` needs `Authorization: Bearer <token>`.
- **Start-up:** a service manager can start the service on demand: a launcher
  holding the activated listening socket passes it on as `HAIKU_RAG_LISTEN_FD`,
  so the first request after login or after an idle exit starts it. That request, and the
  first search after the embedding model was unloaded, can take a few seconds.
  Use a 15 s timeout for searches, 5 s for everything else.
- **Versioning:** additive changes (new fields, new endpoints) do not bump
  anything. Clients must ignore fields they do not know.

## Coordinates

A hit locates its passage per page as boxes in the page's own coordinate
space:

- Units are the converter's page units: **PDF points** (1/72 inch) for PDFs,
  pixels for images. `width` and `height` give the page size in the same units.
- The origin is the page's **top-left corner**, `top < bottom`.
- To draw a box in PDF user space (origin bottom-left, as Skim and PDFKit use):
  `x = left`, `y = height - bottom`, `w = right - left`, `h = bottom - top`.
- Page numbers are 1-based and match the PDF's page index (page 1 is the first
  page of the file, not a printed page label).

Documents without page layout (Markdown, plain text, code, HTML, EPUB, Word)
have `page_numbers: []` and `pages: []`. PowerPoint pages are slides.

## Search

### `GET /search`

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `q` | string, required | | The query, natural language or keywords. |
| `limit` | int, 1-50 | 10 | Hits to return. |
| `type` | `hybrid`, `vector`, `fts` | `hybrid` | Search mode. `fts` needs no embedding model. |
| `database` | string, repeatable | all | Restrict to these databases (names from `lancedb.databases`). |
| `folder` | string, repeatable | all | Restrict to documents of these registered folders (folder ids). |
| `context` | bool | `false` | Also return each hit's passage expanded to its section, in `context`. |

Response:

```json
{
  "query": "travis picking exercises",
  "took_ms": 23.4,
  "hits": [
    {
      "rank": 1,
      "score": 0.0312,
      "chunk_id": "4f0c...",
      "document_id": "8a1e...",
      "database": "files",
      "folder_id": "guitar",
      "title": "Country Guitar",
      "path": "/Users/me/Documents/Guitar/Country Guitar.pdf",
      "uri": "file:///Users/me/Documents/Guitar/Country%20Guitar.pdf",
      "document_type": "pdf",
      "content_type": "application/pdf",
      "page_numbers": [40, 41],
      "line": null,
      "headings": ["Chapter 5", "Travis Picking"],
      "labels": ["text", "list_item"],
      "snippet": "...Travis picking is a fingerstyle technique named after...",
      "snippet_highlights": [[3, 9], [10, 17]],
      "text": "Travis picking is a fingerstyle technique named after Merle Travis...",
      "pages": [
        {
          "page": 40,
          "width": 612.0,
          "height": 792.0,
          "boxes": [
            {"left": 72.0, "top": 412.3, "right": 540.0, "bottom": 498.7}
          ]
        },
        {
          "page": 41,
          "width": 612.0,
          "height": 792.0,
          "boxes": [
            {"left": 72.0, "top": 60.1, "right": 540.0, "bottom": 96.4}
          ]
        }
      ],
      "context": null
    }
  ]
}
```

Hit fields:

| Field | Type | Meaning |
|---|---|---|
| `rank` | int | 1-based position, best first. |
| `score` | float | Ranking score. Only comparable within one response. |
| `chunk_id` | string | The matched chunk. Unique within its database. |
| `document_id` | string | The document; pass it to the `/documents/{id}` endpoints with `database`. |
| `database` | string | Database the hit came from. |
| `folder_id` | string or null | Registered folder that ingested the document; null for documents added another way. |
| `title` | string or null | Document title. |
| `path` | string or null | Absolute local file path; null when the document is not a local file. |
| `uri` | string or null | The stored URI (`file://...` for local files). |
| `document_type` | string | One of `pdf`, `word`, `powerpoint`, `excel`, `epub`, `html`, `markdown`, `text`, `code`, `image`, `other`. |
| `content_type` | string or null | MIME type recorded at ingest. |
| `page_numbers` | int[] | Pages the passage spans, ascending. |
| `line` | int or null | 1-based line in the file where the passage starts, for `markdown`, `text` and `code` documents when it can be found; null otherwise. |
| `headings` | string[] or null | Section heading path, outermost first. |
| `labels` | string[] | Docling element labels of the passage's items (`text`, `section_header`, `list_item`, `table`, `picture`, `caption`, `code`, `formula`, ...), in passage order. |
| `snippet` | string | At most about 300 characters of the passage around the best match, with `...` where cut. |
| `snippet_highlights` | `[int, int][]` | `[start, end)` character offsets into `snippet` of query words found in it. |
| `text` | string | The whole matched passage. |
| `pages` | object[] | Per page: `page`, `width`, `height`, and `boxes` (`left`, `top`, `right`, `bottom`), see Coordinates. One box per document element (paragraph, list item, table, picture) of the passage. |
| `context` | string or null | The passage expanded to its section, when `context=true`. |

### `GET /documents/{document_id}/search`

Search within one document. Takes `q`, `limit`, `type` and `context` as above,
plus an optional `database` to look in only one. Same response shape; every
hit has this `document_id`.

## Documents

### `GET /documents/{document_id}`

Query: `database`, optional: without it every database is asked.

```json
{
  "document_id": "8a1e...",
  "database": "files",
  "folder_id": "guitar",
  "title": "Country Guitar",
  "path": "/Users/me/Documents/Guitar/Country Guitar.pdf",
  "uri": "file:///Users/me/Documents/Guitar/Country%20Guitar.pdf",
  "document_type": "pdf",
  "content_type": "application/pdf",
  "created_at": "2026-11-03T10:12:00Z",
  "updated_at": "2026-11-03T10:12:00Z"
}
```

`404` when the document does not exist.

### `GET /documents/{document_id}/outline`

The document's heading tree. Query: `database` as above.

```json
{
  "document_id": "8a1e...",
  "outline": [
    {
      "id": "#/texts/12",
      "title": "Chapter 5: Country Styles",
      "level": 1,
      "page_numbers": [38],
      "children": [
        {"id": "#/texts/40", "title": "Travis Picking", "level": 2, "page_numbers": [40], "children": []}
      ]
    }
  ]
}
```

An empty `outline` means the document has no headings.

## Folders

Folders are the register of local directories the service indexes. A change
takes effect at once: the service starts watching an added folder and indexes
it, and stops watching a removed one.

Folder object:

```json
{
  "id": "guitar",
  "database": "files",
  "path": "/Users/me/Documents/Guitar",
  "include": ["**/*.pdf"],
  "exclude": ["**/drafts/**"],
  "status": {
    "watching": true,
    "documents": 507,
    "queued": 3,
    "failed": 2,
    "last_scan_at": "2026-11-03T10:12:00Z"
  }
}
```

`include` and `exclude` are gitignore-style patterns matched against each file's
absolute path, so `*.pdf` and `drafts/` match at any depth. An empty `include`
takes every supported file. `status` is null when the service is not running
that folder yet (just after it was added, or while it is being re-added).

### `GET /folders`

`{"folders": [<folder>, ...]}`

### `POST /folders`

Body: `{"path": "/abs/dir", "database": "files", "include": [], "exclude": [], "id": "guitar"}`.
Only `path` is required. `database` defaults to the only configured database;
`id` defaults to one derived from the directory name. Returns `201` with the
folder. `400` when the path is not an existing directory or the database is
unknown, `409` when the id is taken or the path is already registered.

### `PATCH /folders/{id}`

Body: any of `include`, `exclude`. Returns the folder.

### `DELETE /folders/{id}`

Query: `keep_documents` (bool, default `false`). Stops watching the folder; unless
`keep_documents=true`, its documents are removed from the index. Returns
`{"removed": "guitar"}`, `404` for an unknown id.

## Status and failures

### `GET /status`

```json
{
  "indexing": true,
  "queued": 12,
  "in_flight": 2,
  "failed": 2,
  "paused": false,
  "pause_reason": null,
  "embedder_loaded": true,
  "databases": [{"name": "files", "documents": 1204}]
}
```

`paused` is true while the service holds ingestion back, with `pause_reason`
`"on battery"`, `"system busy"` or `"low memory"`. Search keeps working.

### `GET /failures`

Documents that failed to ingest, newest first. Query: `folder`, `limit` (default
50), `offset`.

```json
{
  "failures": [
    {
      "job_id": "01J...",
      "folder_id": "guitar",
      "database": "files",
      "path": "/Users/me/Documents/Guitar/fingerpicking-disney.pdf",
      "uri": "file:///Users/me/Documents/Guitar/fingerpicking-disney.pdf",
      "reason": "conversion timed out after 300s (ocrmac), retried with rapidocr: timed out after 300s",
      "attempts": 1,
      "failed_at": "2026-11-03T10:12:00Z"
    }
  ]
}
```

### `POST /failures/{job_id}/retry`

Queues the document again. Returns `{"retried": ["01J..."]}`.

### `POST /failures/retry`

Queues every failure again, optionally only `folder`'s. Returns
`{"retried": [...]}`.

### `GET /health`

Unauthenticated liveness, the ingester's health response (`status` is `ok` or
`degraded`).

## Running it

The service is `haiku-ingester serve` with the API enabled, supervised
conversion on, and a folder register:

```yaml
lancedb:
  databases:
    files: ~/Library/Application Support/haiku.rag/files.lancedb
  read_consistency_interval_seconds: 2   # searches see new documents quickly
processing:
  conversion_options:
    ocr_engine: ocrmac
  supervisor:
    enabled: true
ingester:
  folders_register: folders.yaml
  api:
    host: 127.0.0.1
    port: 8765
  workers:
    worker_count: 3
    poll_idle_interval_s: 5
  service:
    search_idle_s: 300        # close the search client after 5 idle minutes
    throttle:
      on_battery: slow        # run | slow | pause
      max_load: 0.7           # pause conversions above this load per core
      min_free_memory_percent: 10
    vacuum_after_indexing:
      min_documents: 50       # vacuum once the queue drains after this many; null (default) never
      quiet_s: 60             # how long the queue must stay empty first
      retention_seconds: 0    # keep no old versions
```

While that vacuum runs, searches wait for it.

Pages and boxes come from what each chunk recorded when it was indexed. A
database indexed before chunks recorded them still answers with boxes, read
from the stored document at search time (slower); `haiku-rag rebuild
--rechunk` records them.

## MCP

The MCP server's `search_documents` output carries each hit's file `Path` and
`Pages` lines, and the server has `list_folders`, `add_folder` and
`remove_folder` tools over the same register.
