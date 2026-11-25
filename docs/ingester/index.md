# Ingester

`haiku-ingester` is a long-running service that watches sources for changes and keeps a haiku.rag database in step with them. It runs as its own process, keeps a job queue (SQLite by default, or Postgres) with retries and a dead-letter queue, and serves an HTTP control plane with a dashboard.

For one-off ingestion, `haiku-rag add-src` is enough, see [CLI](../cli.md#add).

One ingester writes every database its configuration names, with one queue and one control plane. haiku.rag allows one writing process per database (see [Operational constraints](../configuration/storage.md#operational-constraints)), so no other process writes them. MCP servers and other read-only consumers can run beside it.

[Running the ingester](running.md) covers serving, retries, maintenance and the HTTP API. [Queue and attribution](queue.md) covers the job queue, Postgres, and how documents are tied to their sources.

## Install

```bash
pip install 'haiku.rag-slim[ingester]'
# or, with the full package:
pip install 'haiku.rag[ingester]'
```

The extra pulls `fastapi`, `uvicorn`, `sqlalchemy`, `aiosqlite`, `asyncpg` and the `s3` extra. It does not pull docling. The default converter, `docling-local`, also needs the `docling` extra: `haiku.rag-slim[ingester,docling]`. The full package includes it. Without docling, `haiku-ingester` exits at startup and names the extra.

## Configure sources

Sources are listed under `ingester.databases` in `haiku.rag.yaml`, keyed by the database they are ingested into: a name in `lancedb.databases`, or `haiku.rag` when none is configured. The minimum is one source:

```yaml
ingester:
  databases:
    haiku.rag:
      sources:
        - type: fs
          id: local-docs
          root: /Users/you/docs
          delete_orphans: true
```

With `lancedb.databases` configured, each key names one of its entries. One process ingests every database it names:

```yaml
lancedb:
  databases:
    papers: /srv/haiku/papers.lancedb
    wiki: s3://my-bucket/wiki.lancedb
    papers-release: {location: /srv/haiku/papers.lancedb, tag: v1}

ingester:
  databases:
    papers:
      sources:
        - type: fs
          id: papers
          root: /srv/papers
    wiki:
      sources:
        - type: s3
          id: wiki-export
          uri: s3://my-bucket/wiki-export/
        - type: http
          id: wiki-pages
          urls: [https://example.com/handbook.html]
```

```bash
haiku-ingester serve                         # papers and wiki
haiku-ingester run-batch                     # one sweep over both, then exit
haiku-ingester run-batch --db-name wiki      # wiki's sources only
```

The ingester writes every database a key names, each through its own client, and leaves the other entries to readers, such as `papers-release` here. A key naming no configured database, or one configured at a tag, fails at load, and so does an entry without sources. A configuration carrying `ingester.sources` fails at load. `run-batch` refuses to start when `ingester.databases` names no database. `serve` starts, ingests nothing until a [reload](running.md#reloading-databases) adds a database, and deletes every queued and claimed job of its queue namespace, as it does for any source it does not configure.

One process serves every database. A [reload](running.md#reloading-databases) adds or removes one while the others keep ingesting. A database that fails to start is logged at ERROR and skipped, and the others start: for example when it needs a migration, when its recorded embedder differs from the configuration, when its store is unreachable, when a hook's lifespan or ingester lifespan raises, when a source names an unregistered metadata provider or one whose factory fails, or when reconciliation fails. The next reload retries it. `/health` lists each skipped database with its error and reports `degraded`. Startup fails when every configured database fails, and deletes no job then. Invalid configuration, an unreachable queue, two sources sharing an id and two databases at one location still fail startup. If a conversion stalls, the process exits (see [Run it under a supervisor](running.md#run-it-under-a-supervisor)), and every database waits for the restart. If a database's writes fail while the ingester runs, only that database's sources pause, since each source has its own worker circuit breaker. If databases must not stop together, run one ingester per database, each with its own queue or namespace.

Source ids are unique across every database: startup fails when two sources share an id, given or derived from their target. Two keys whose databases sit at one location fail startup too, a path and its `file://` URI included, and two names a case-insensitive filesystem treats as one directory, since a database has one writer. Each open database holds its own embedder client.

Every source takes `id`, `poll_interval_s`, `delete_orphans`, `max_file_size`, `retry`, `circuit_breaker` and `metadata_provider`. With `delete_orphans`, a document whose file disappears from the source is deleted. Without it, the document stays. A source deletes only the documents it owns: a document last ingested by another source, or one ingested without a source, is left in place. A sweep deletes nothing when an `fs` source has a directory it cannot list, the root included, or when a source lists nothing and its deletes would remove stored documents owned by the source. The sweep fails instead, until those documents are deleted by hand. When none would be removed, the deletes run as usual. A `--dry-run` reads no document store, so it fails whenever a source with known URIs lists nothing. `ignore_patterns` and `include_patterns` follow [gitignore syntax](https://git-scm.com/docs/gitignore#_pattern_format). Supported file extensions are matched case-insensitively (`.pdf`, `.PDF` and `.PdF` are equivalent). Include and ignore patterns are case-sensitive.

An `fs`, `s3` or `webdav` source lists only the files that pass its patterns and carry one of these extensions:

- Converted by docling, with either converter: `.adoc`, `.asc`, `.asciidoc`, `.bmp`, `.csv`, `.docx`, `.eml`, `.epub`, `.html`, `.xhtml`, `.jpeg`, `.jpg`, `.latex`, `.md`, `.msg`, `.pdf`, `.png`, `.pptx`, `.qmd`, `.rmd`, `.tex`, `.tiff`, `.xlsx`, `.xml`, `.webp`
- Read as text: `.astro`, `.bash`, `.c`, `.clj`, `.cljs`, `.cpp`, `.cs`, `.css`, `.dart`, `.elm`, `.ex`, `.exs`, `.fs`, `.fsx`, `.go`, `.gql`, `.graphql`, `.groovy`, `.h`, `.hcl`, `.hpp`, `.hs`, `.java`, `.jl`, `.js`, `.json`, `.kt`, `.less`, `.lua`, `.mdx`, `.mjs`, `.ml`, `.mli`, `.nim`, `.nix`, `.php`, `.pl`, `.plantuml`, `.pm`, `.proto`, `.pu`, `.puml`, `.ps1`, `.py`, `.r`, `.rb`, `.rs`, `.rst`, `.sass`, `.scala`, `.scss`, `.sh`, `.sql`, `.svelte`, `.swift`, `.tf`, `.toml`, `.ts`, `.tsx`, `.txt`, `.vue`, `.yaml`, `.yml`, `.zig`

Files that pass the patterns with any other extension are skipped and counted by extension in `unsupported_files` of `GET /sources`, as of its last successful sweep. Watch events do not change the counts. An `http` source fetches the URLs it is given and filters none of them.

### Removing or moving a source

To retire a source, remove it from the configuration, and delete its documents by hand if they should go. Its queued and claimed jobs are deleted when `serve`, `run-batch` or a manifest replay next starts, never by `--dry-run`, or when a reload removes its database. Its finished jobs stay until `retention_days` prunes them. This relies on one ingester per queue namespace (see [Postgres](queue.md#postgres)).

A source moved under another key keeps its id and its queue record. At the next start, reconciliation finds its documents missing from the new database and ingests them again there. The old database keeps its copies until they are deleted by hand.

### Filesystem

```yaml
ingester:
  databases:
    haiku.rag:
      sources:
        - type: fs
          id: local-docs                          # optional, derived from root
          root: /Users/you/docs
          poll_interval_s: 300
          delete_orphans: true
          ignore_patterns: ["**/.git/**", "**/node_modules/**"]
          include_patterns: ["*.md", "*.pdf"]    # optional allow-list
```

`watchfiles` delivers changes as they happen, and a periodic sweep catches anything missed while the service was down.

With `converter: docling-local`, a LaTeX document is converted where it sits, and `\input`, `\include` and `\includegraphics` read the files they name from its directory and below, whatever their extension and whatever `ignore_patterns` says. The files they pull in become searchable text. Do not watch a directory where untrusted users can write `.tex` files next to files that must stay private.

### S3 and object storage

```yaml
ingester:
  databases:
    haiku.rag:
      sources:
        - type: s3
          id: corp-docs
          uri: s3://my-bucket/incoming/
          poll_interval_s: 300
          delete_orphans: true
          ignore_patterns: ["draft*"]
          include_patterns: ["*.pdf", "*.md"]
          storage_options:
            endpoint: http://seaweed:8333         # omit for the AWS default chain
            aws_access_key_id: ${AWS_KEY}
            aws_secret_access_key: ${AWS_SECRET}
            region: us-east-1
            allow_http: "true"
```

Each sweep lists the prefix and fetches only keys whose ETag differs from the document's stored `metadata["source_revision"]`. The stored revision comes from the GET response that supplies the object's bytes. When the fetched bytes match the stored MD5, as after a multipart re-upload, only the revision is updated and nothing is re-chunked.

`storage_options` takes the same keys as `lancedb.storage_options`, so credentials written for the database work here too.

### HTTP

```yaml
ingester:
  databases:
    haiku.rag:
      sources:
        - type: http
          id: arxiv
          urls:
            - https://arxiv.org/pdf/2301.12345.pdf
          headers:
            Authorization: Bearer ${SOME_TOKEN}
          poll_interval_s: 86400
```

Change detection uses `HEAD`, or the headers of a `GET` closed before its body when `HEAD` answers 405 or 501. A URL removed from `urls`, or answering `410 Gone`, counts as gone. A URL answering `404` counts as gone when it also answered `404` to the previous sweep's probe and the fetch between them failed permanently. Under `delete_orphans`, every later sweep queues a DELETE for that URL, and the DELETE removes its dead jobs, so a URL that answers 404 from the start stays in the dead-letter queue for one poll interval. A transport error, 408, 429 or 5xx leaves the decision to the fetch. Any other error status is recorded as the URL's revision, `http-<status>`, so the fetch fails once and the URL is not fetched again until its status changes. A URL whose server sends neither `ETag` nor `Last-Modified` is not fetched again once ingested. The source fetches only the URLs in `urls`, so its `headers` reach no other URL.

Under `delete_orphans`, a sweep in which every URL counts as gone follows the empty-listing rule: when one of those URLs has a stored document owned by the source, the sweep deletes nothing and fails. `/sources` reports the error in `last_error`, and repeated failures open the source's poller circuit breaker, until a URL answers again or `urls` lists one that does. Removing the gone URLs from `urls` does not clear it: the removed URLs count as gone too. The failure clears once those documents are deleted by hand, or after a reload that drops the source from the configuration.

### WebDAV

```yaml
ingester:
  databases:
    haiku.rag:
      sources:
        - type: webdav
          id: nextcloud
          base_url: https://nextcloud.example.com/remote.php/dav/files/alice/Documents/
          username: alice
          password: ${NEXTCLOUD_APP_PASSWORD}
          ignore_patterns: ["**/Trash/**"]
          poll_interval_s: 600
```

Each sweep issues one `PROPFIND` with `Depth: infinity` against `base_url`. A file's `getetag`, or `getlastmodified` when the server sends no ETag, decides whether it changed. A file missing from the listing counts as gone. Fetches are plain `GET`s.

Redirects are followed for both `PROPFIND` and `GET`, so servers that redirect for a trailing slash or `http` to `https` work unchanged. A redirect that moves the collection to another path or host fails discovery, so the source can be pointed at the new location. Credentials are never sent to a different host.

`headers` can replace HTTP Basic with a bearer token:

```yaml
        - type: webdav
          id: kdrive
          base_url: https://kdrive.infomaniak.com/app/drive/123/
          headers:
            Authorization: Bearer ${KDRIVE_TOKEN}
```

### File size limits

`max_file_size` (bytes) rejects oversized files. They go straight to the dead-letter queue.

```yaml
        - type: fs
          root: /data/docs
          max_file_size: 104857600        # 100 MB
```

FS sources check the file size before reading. S3 sources check the GET response's size before reading its body. HTTP and WebDAV sources check the `Content-Length` of a `HEAD` response first, and stop a download once it passes the limit, with or without `Content-Length`.

### Metadata providers

A source can attach custom metadata to every document it ingests through a `metadata_provider`, a callable registered by an external package under the `haiku.rag.metadata_providers` entry-point group:

```yaml
        - type: webdav
          id: handbook
          base_url: https://dav.example.com/remote.php/dav/files/svc
          metadata_provider: example-provider
        - type: webdav
          id: policies
          base_url: https://dav.example.com/remote.php/dav/files/policies
          metadata_provider:
            name: example-provider
            options:
              department: legal
```

`metadata_provider` takes a name, or `name` and `options`. The entry point is a callable returning the provider, so a class is its own factory. It receives `options` as keyword arguments, and none for a bare name. The ingester calls the provider with `(source_id, uri, result)`, where `result` is the source's `FetchResult`, and merges the returned dict into the document's metadata:

```python
# example_pkg/__init__.py
from urllib.parse import urlparse

from haiku.rag.sources import FetchResult


class Provider:
    def __init__(self, department: str | None = None):
        self.department = department

    async def __call__(
        self, source_id: str, uri: str, result: FetchResult
    ) -> dict:
        path = urlparse(uri).path
        metadata = {
            "collection": source_id,
            "folder": path.rsplit("/", 1)[0] or "/",
            "bytes": str(len(result.body)),
        }
        if self.department:
            metadata["department"] = self.department
        return metadata
```

```toml
# in the provider package's pyproject.toml
[project.entry-points."haiku.rag.metadata_providers"]
example-provider = "example_pkg:Provider"
```

A metadata provider is metadata for one source: the source's configuration names it, and a reload changes it. Behaviour that applies to every ingested document, such as transforming, rejecting or observing it, belongs in a [hook](../hooks.md), and [Metadata lifetime](../hooks.md#metadata-lifetime) says how long provider and hook metadata last.

The provider is built once per source when its database starts, so it can hold a client or a cache. When a source's provider has no installed entry point, fails to import, or raises from its factory, the source's database does not start, and the error names the source and the provider (see [Configure sources](#configure-sources)). Options are part of the source, so a [reload](running.md#reloading-databases) that changes them removes and adds the database, which builds the provider again with the new options. The provider runs when a document is fetched for a new or changed revision. An unchanged document is skipped without a fetch and keeps its stored provider metadata, so changed options reach a document only when it changes or is re-ingested. The keys `md5`, `source_revision`, `content_type`, `source_id` and `parent_uri` are removed from provider output. A provider exception is handled like any ingestion error: network and timeout errors retry, others go to the dead-letter queue.

The provider is also called for each [PDF attachment](../configuration/processing.md#pdf-embedded-attachments) the ingester stores, new or with changed bytes, at any nesting depth. It receives the source id of the attachment's parent document, the attachment's URI (`<parent uri>#attachment=<percent-encoded name>`, with one fragment per nesting level), and a `FetchResult` holding the attachment's bytes and guessed content type, with `result.extra_metadata["parent_uri"]` set and `disk_path` and `revision` unset. A provider that should treat attachments differently can test `"parent_uri" in result.extra_metadata`. It is not called for an unsupported attachment, or for one whose bytes are unchanged, which keeps its stored provider metadata. A provider exception for an attachment is handled like one for its parent document. Attachments carry no `source_id` (see [Which source a document came from](queue.md#which-source-a-document-came-from)).

### Custom sources

To ingest from something the four built-in types (`fs`, `http`, `s3`, `webdav`) do not cover, such as a git host or a ticketing system, a package registers a source factory under the `haiku.rag.sources` entry-point group and the configuration names it with `type: plugin`:

```yaml
        - type: plugin
          id: api-docs
          plugin: git
          options:
            owner: acme
            repo: api
            branch: main
            token: ${SCM_TOKEN}
```

`plugin` is the entry-point name. `options` is passed to the factory as given, and the factory validates it. The base source fields are handled by the ingester and are not part of `options`.

The factory takes the source id, the options, and the extension and size limits, and returns a `Source`:

```python
def __call__(
    self,
    *,
    source_id: str,
    options: dict,
    supported_extensions: list[str] | None,
    max_file_size: int | None,
) -> Source: ...
```

```python
class Source(Protocol):
    source_id: str

    def supports(self, uri: str) -> bool: ...

    # Current revision for `uri`, or None when there is no cheap lookup.
    async def head(self, uri: str) -> str | None: ...

    # Release resources (connection pools, etc.). Called once at shutdown.
    async def aclose(self) -> None: ...

    async def fetch(self, uri: str) -> FetchResult: ...

    # Yield UPSERT / UNCHANGED / DELETE events. `since` is the uri -> revision
    # snapshot from the previous sweep, so the source can emit only changes.
    def discover(
        self,
        since: RevisionSnapshot | None = None,
        *,
        known_uris: set[str] | None = None,
    ) -> AsyncIterator[SourceEvent]: ...
```

`FetchResult`, `SourceEvent`, `SourceEventKind` and `RevisionSnapshot` are in `haiku.rag.sources`.

```toml
# in the source package's pyproject.toml
[project.entry-points."haiku.rag.sources"]
git = "example_pkg:build_git_source"
```

Only plugins a source references are imported. A `plugin` name with no installed entry point fails startup, as does a factory that returns something other than a `Source`.

Custom sources are reached through the ingester only, not through `haiku-rag add-src`. Change detection is per source and URI, so a source that needs one cursor for the whole source (a git commit SHA, say) keeps it in each URI's revision or under a sentinel URI.
