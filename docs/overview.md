# Architecture

haiku.rag ingests documents, retrieves from them with hybrid search, and answers
with citations. This page follows the data through the system. For a working
setup, start with the [Quickstart](tutorial.md).

## Ingestion

```text
source adapter -> converter -> chunker -> embedder -> LanceDB
```

A **source adapter** owns the I/O and the identity of a document: it fetches
bytes, reports the backend's revision (mtime for a file, ETag for S3 or HTTP),
and computes the content hash. The same adapters serve one-shot ingestion
(`haiku-rag add-src`, `HaikuRAG.create_document_from_source`) and the continuous
[`haiku-ingester`](ingester.md) service, so both agree on what a document is and
when it has changed.

The **converter** turns those bytes into a `DoclingDocument`, the structured form
that carries headings, tables, pictures and page provenance. It runs in-process
with the `docling` extra, or against a [docling-serve](remote-processing.md)
fleet.

The **chunker** splits that structure into chunks, each keeping the headings it
sits under, the page numbers it came from, and references to the document items
it covers. With a multimodal embedder, pictures become chunks of their own.

The **embedder** vectorizes them in batches. The document, its mutable metadata,
its chunks and its structural items are written under one process-local
transaction: it takes a version snapshot, and on failure restores each table to
it. A rollback that cannot complete raises rather than reporting success, and the
snapshot is only meaningful while this process is the only writer.

## Storage

LanceDB is embedded, so there is no server. The same code runs against a local
directory, S3, GCS, Azure or LanceDB Cloud by changing `lancedb.uri`.

Tables are versioned. Vacuum collapses old versions on a retention window, and
[tags](cli.md) name a state across all tables so a database can be restored to
it later.

One process writes at a time. Reads are unrestricted, and a reader sees another
process's writes after `lancedb.read_consistency_interval_seconds`.

## Retrieval

```text
query -> vector + full-text search -> fusion -> rerank -> context expansion
```

Search runs a vector query and a full-text query and fuses the rankings. With a
reranker configured, it retrieves ten times the requested limit and reranks down
to it, so quality improves without changing the caller's limit.

Results then expand: a chunk is returned with the section it belongs to, bounded
by `search.max_context_chars`. Sections that fit come back whole, larger ones
grow outward from the match, and small ones grow across boundaries. Every result
carries its page numbers and headings, which is what makes a citation checkable.

## Answering

Two [capabilities](capabilities/index.md) sit on top, both native Pydantic AI
capabilities you can attach to your own agent:

- The **RAG capability** searches and cites. Its citations carry page numbers and
  headings, and `haiku-rag visualize` draws the cited chunk on the page image.
- The **analysis capability** adds a sandboxed Python interpreter with the
  documents mounted as a filesystem, for questions that need computation across
  documents rather than retrieval.

Two optional capabilities compose with them: evidence compaction replaces older
turns' evidence with what was actually cited, and citation policy requires every
answer to declare what grounds it.

The same database is reachable from [Python](python.md), the [CLI](cli.md), and
the [MCP server](mcp.md).

## Running it

A laptop needs nothing but the package and Ollama. Production adds the
[`haiku-ingester`](ingester.md) service, which polls its sources, queues work in
SQLite or Postgres, and retries with a circuit breaker per source.

Before deploying, read the operational constraints in
[Storage](configuration/storage.md): one writer per database, `haiku-rag migrate`
after an upgrade that changes the schema, and a fixed embedding dimension per
database.
