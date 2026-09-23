# Multiple databases

`lancedb.databases` names databases that are searched together. A location is a local path or a URI:

```yaml
lancedb:
  databases:
    papers: s3://my-bucket/papers.lancedb
    wiki: s3://my-bucket/wiki.lancedb
    notes: /data/notes.lancedb
```

Placing a single database works the same way. See [Placing the database](storage.md#placing-the-database).

## Names and provenance

The configured name is the only identity that leaves the configuration. `SearchResult.source`, `Citation.source` and `Document.source` carry it, for a set and for one database alike. The model sees it as a `Collection:` line on each search result, only when the search spans more than one database.

An unavailable database raises `SourceUnavailableError`, which names the database and not its location. A migration, configuration or read-only failure keeps its own type, with the database named in the message. Commands that report on a database, such as `info`, still show where it is.

## Embedding compatibility

Embeddings are checked against two things:

- **On open**, each database against the configuration. A dimension mismatch raises `ConfigMismatchError`. A provider or model-name mismatch at the same dimension warns in read-only mode and raises in writable mode.
- **Across a selection**, the databases against each other. Vector and hybrid search embed the query once, so every database answering it must record the same provider, model and dimension, or the search raises `ConfigMismatchError`, read-only included. Only the databases searched together have to agree. Full-text search embeds nothing and is unaffected.

## CLI

- `search`, `ask`, `chat` and `mcp` cover the configured set, or the one database `--db-name` selects.
- `settings`, `init-config` and `download-models` open no database.
- Every other command works on one database. With several configured, it needs `--db-name NAME` or `--db PATH`. A configured set of one is selected automatically.

```bash
haiku-rag search "query"          # every configured database
haiku-rag --db-name papers list   # one of them
haiku-rag --db-name papers init   # each database is created, migrated and vacuumed on its own
haiku-rag --db-name wiki init
```

`--db-name` is global and precedes the command. It selects an entry from `lancedb.databases`, remote ones included, and `haiku.rag` when nothing is configured. `--db` follows the command and opens a local path, named by its stem, whatever is configured.

The CLI labels results and citations with their database only when the operation spans several.

## Python

A client covers the configured set. `search`, `ask` and `list_documents` take `sources` to select a subset:

```python
results = await client.search("machine learning")                      # all of them
results = await client.search("machine learning", sources=["papers"])  # one of them

answer, citations = await client.ask("What changed?", sources=["papers", "wiki"])
for cite in citations:
    print(f"[{cite.source}] {cite.document_title or cite.document_uri}")
```

A scoped question cites only the selected databases, and code run by the capability mounts only their documents. `sources=None` covers every database the client covers. `sources=[]` covers none: `search` returns no results, and `ask` runs with no evidence.

A name the client does not cover raises `UnknownDatabaseError`, a `KeyError`, wherever it is given: when the client opens, per query, and when placing a citation.

On the constructor, `sources` selects the databases the client covers. Beside a database path it raises `AmbiguousDatabaseError`. `sources=[]` alone raises `ValueError` when the client opens, since a client over no database has nothing to do.

Creating, writing, rebuilding and vacuuming need one database, and raise `AmbiguousDatabaseError` on a client covering several. Conversion, chunking and title generation touch no database and work on any client:

```python
async with HaikuRAG(config=config, create=True, sources=["papers"]) as papers:
    ...

async with HaikuRAG(config=config) as client:
    papers = await client.reader_for("papers")
```

The client describes its coverage:

```python
client.covers_multiple      # whether the client covers more than one database
client.source_names         # database names, in order, known before the client opens
client.source               # the one database's name, or None for a set

owner = await client.reader_for("papers")
papers, wiki = await client.clients_for(["papers", "wiki"])
covering = await client.clients_covering(["papers"])  # [] for [], every client for None
```

`reader_for`, `clients_for` and `clients_covering` open databases lazily and return borrowed clients. They stay valid while the covering client is open and inherit its read-only mode. The covering client closes them.

`DatabaseScope.resolve` reports what a configuration covers without opening anything:

```python
from haiku.rag.client import DatabaseScope

for ref in DatabaseScope.resolve(config).databases:
    print(ref.name, ref.location)   # "haiku.rag", Path(".../haiku.rag.lancedb") when nothing is configured
```

`get_document_by_id`, `get_chunk_by_id`, `get_picture_bytes` and `visualize_chunk` take an optional `source` and ask that database alone. Without one, the document and chunk lookups ask every covered database and answer from the first that holds the id. `get_picture_bytes` and `visualize_chunk` require `source` on a client covering several, since a chunk carries no database identity.

## Capability

The RAG capability covers the databases the configuration places. `sources` on `create_capability`, or in an agent spec, narrows that coverage:

```yaml
capabilities:
  - RAGCapability:
      sources: [manuals, specs]
```

An unknown name raises `UnknownDatabaseError`, an empty list `ValueError`. `sources` beside `db_path` or `rag=` raises `AmbiguousDatabaseError`, since each of those already says which databases the capability covers. The `sources` field of the capability state then selects among the covered databases for one question. See [Capabilities](../capabilities/index.md#database-selection).

## MCP server

The server covers the configured set. `sources` on the search tools, `list_documents` and `execute_code` restricts a call, and `source` on the document read tools names the database holding the document. `haiku-rag --db-name NAME mcp` serves one. See [MCP](../mcp.md#collections).

## Ingester

`haiku-ingester` writes one database. With several in `lancedb.databases` it refuses to start with `AmbiguousDatabaseError`. Run one ingester per database, each with a configuration naming only its own, or select one with `--db PATH`.

## Ranking

Results from several databases are fused into one list:

- **With a reranker**, the reranker scores the combined candidates directly. It is the strongest option, roughly 6 to 8 recall points above cosine fusion on the MTRAG retrieval benchmarks, and its cost grows with the number of databases.
- **Without one**, candidates are ordered by cosine similarity to the query vector. The databases share an embedder, so similarity in that space is comparable across them, where retrieval scores are not. Ties fall to rank within each database, then to configured order.
- **Full-text-only searches** have no query vector and order by retrieval score.
- **Image queries** are vector-only and skip the reranker, which takes a text query.

Results are not spread across databases. A database with nothing relevant contributes nothing, and a strong one can fill every slot.

## Duplicate ids

Ids are unique within a database, not across databases, so copies of a database share them.

A cited chunk id is rejected with `AmbiguousCitationError` when search returned it from more than one database, or it was cited earlier from another database. When only one retrieved result has the id, that result is cited. For an id no search returned, every selected database is checked and several holders are rejected. A shared id that nothing cites is ignored.

The sandbox rejects shared document ids, since its mount path is `/documents/{id}/`.

The chat document filter selects by document and database: the search is narrowed to the databases the selection names, and the id filter applies within them.
