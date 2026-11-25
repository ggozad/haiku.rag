# Queue and attribution

## The queue

The SQLite queue is `ingester.db` in the platform data directory (`~/Library/Application Support/haiku.rag/` on macOS), or `ingester.queue.path`. `serve` creates it. To create or migrate it ahead of time:

```bash
haiku-ingester queue init             # create the database and schema
haiku-ingester queue migrate          # apply pending schema changes
haiku-ingester queue init -q /var/lib/haiku-rag/ingester.db   # at another path
```

Succeeded, dead and rejected jobs are kept for history. The reaper deletes those completed more than `retention_days` ago:

```yaml
ingester:
  queue:
    path: /var/lib/haiku-rag/ingester.db
    retention_days: 30                # null keeps them all
```

### Postgres

`ingester.queue.dburi` points the queue at Postgres, with a SQLAlchemy async URL, and `namespace` names the ingester that owns its rows:

```yaml
ingester:
  queue:
    dburi: postgresql+asyncpg://ingester_rw:${QUEUE_DB_PASSWORD}@db:5432/haiku_queue
    namespace: papers
```

The `asyncpg` driver comes with the `ingester` extra. `dburi` overrides `path` and the `--queue` flag.

Several ingesters share one Postgres queue, each under its own namespace and each writing its own databases. An ingester claims, counts, prunes and serves through its API the rows of its namespace and no others, so source ids need to be unique within a namespace, not across the queue. A namespace is stable across restarts and upgrades: an ingester started under a new one does not see the rows recorded under the old. Two constraints hold independently. A namespace has one process: a second one deletes the first's queued and claimed jobs at startup, as jobs of sources it does not configure. A database has one writer (see [storage](../configuration/storage.md)): two namespaces never ingest into the same database. The database role sees every row: namespaces separate ingesters in the application, not in Postgres.

`serve`, `run-batch` and `--dry-run` refuse a server `dburi` without a namespace, and never create or migrate a server queue: they read its schema version and fail naming the command to run. A SQLite queue is owned by one process, migrates itself at startup and records its rows under `namespace`, empty unless configured.

#### Deployment

An account that owns the queue's tables creates and migrates them, through `haiku-ingester queue init` and `queue migrate`. The ingesters run as an account that can only read and write rows. `--dburi`, or `HAIKU_RAG_QUEUE_DBURI`, gives the two commands the owner's URL, so it never appears in the configuration the ingesters read:

```bash
haiku-ingester queue init --dburi postgresql+asyncpg://queue_owner:${OWNER_PASSWORD}@db:5432/haiku_queue
```

The owner grants the ingesters' account its rights once the tables exist, and again after a migration that adds a table:

```sql
GRANT USAGE ON SCHEMA public TO ingester_rw;
GRANT SELECT, INSERT, UPDATE, DELETE ON jobs, sync_state TO ingester_rw;
GRANT SELECT ON schema_version TO ingester_rw;
```

To keep the queue in a schema of its own, create the schema and point both accounts at it before `queue init`, with the `GRANT USAGE` above on that schema:

```sql
CREATE SCHEMA haiku_queue;
ALTER ROLE queue_owner SET search_path = haiku_queue;
ALTER ROLE ingester_rw SET search_path = haiku_queue;
```

A release that changes the queue schema rolls out in one order: stop every ingester, run `haiku-ingester queue migrate --dburi ...` as the owner, start the upgraded ingesters. Releases never share a queue: a new one refuses a queue that is behind, and an old one writes a migrated queue with its old statements. An upgrade from schema 3 also needs the steps below.

Each ingester process holds at most five connections to the queue: two kept open and three more under load, closed when returned. Size `max_connections` for five per process, plus the owner's migration and any dashboard. A sweep's `sync_state` writes go in one transaction, which holds its connection for the sweep's batch. PgBouncer in transaction mode is untested.

#### Upgrading a queue from schema 3

Queue schema 4 adds the namespace. The migration leaves every existing row under the empty namespace, and the ingester reads only its own, so hand the rows over before it starts:

1. Stop the ingester. One still running claims and prunes rows of every namespace.
2. `haiku-ingester queue migrate --dburi ...` as the owner.
3. Assign the namespace to the migrated rows, in one transaction:

    ```sql
    UPDATE jobs SET namespace = 'papers' WHERE namespace = '';
    UPDATE sync_state SET namespace = 'papers' WHERE namespace = '';
    ```

4. Start the ingester with `namespace: papers`.

A queue that several ingesters already wrote to needs a mapping from `source_id` to namespace in step 3. SQLite queues need nothing.

## Reading the queue from Python

A deployment that only runs `run-batch` has no HTTP API. The queue's read side is in `haiku.rag.ingester.queue`:

```python
from haiku.rag.ingester.queue import JobRepo, JobStatus, SyncStateRepo, open_queue

engine = await open_queue(config.ingester.queue)
try:
    jobs = JobRepo(engine, namespace=config.ingester.queue.namespace)
    rejected = await jobs.list_jobs(status=JobStatus.REJECTED, limit=100)
finally:
    await engine.dispose()
```

- `open_queue(queue_config)` returns the engine. It migrates a SQLite file and refuses one newer than the release. A server queue is only checked: one behind or ahead of the release is refused, naming the command to run. `haiku.rag.ingester.queue.migrations.verify_schema(engine)` makes the same check and writes nothing, for an account that can only read.
- `JobRepo(engine, namespace)` reads with `get_job(job_id)`, `list_jobs(status=, source_id=, uri=, limit=, offset=)` (newest enqueued first), `counts_by_status()`, and `counts_by_status_since(since)`, which counts the jobs completed at or after `since`, as the `run-batch` summary does.
- `SyncStateRepo(engine, namespace)` reads with `get_row(source_id, uri)`, `list_known_uris(source_id)` and `list_ingested_uris(source_id)`.

Pass the namespace the ingester runs under, `ingester.queue.namespace`, which is `""` unless configured. The methods that change rows belong to the ingester.

## Which source a document came from

A document the ingester fetches carries `metadata["source_id"]`, the id of the source that ingested it. Documents added with `haiku-rag add-src` have none. Neither do PDF attachments, which belong to their parent document and are deleted with it.

Only ingestion sets `source_id`, and `HaikuRAG.set_document_source`, which startup reconciliation uses. Passing it as metadata does nothing, and re-adding an ingested document by hand keeps it.

A source id is an identity. An `fs` source without an `id` derives it from its root (`fs:{resolved_root}`), and an `s3` source from its bucket and prefix (`s3:{bucket}/{prefix}`). Moving the directory or prefix, or renaming an `id`, detaches every document ingested under the old one. Set `id` on a source whose location may move:

```yaml
ingester:
  databases:
    haiku.rag:
      sources:
        - type: fs
          id: handbook
          root: /srv/handbook
```

When two sources of one database cover the same URI (nested roots or prefixes, one URL in two `http` sources), the latest ingestion takes ownership and a warning names both ids. Overlapping sources within a database are a configuration error. Sources of different databases may cover the same URI: each database holds its own copy.

## Reconciliation at startup

The ingester's state is in two places: the documents in the database, and the queue's record of what each source ingested, which deletion is computed from. Either can be restored or lost without the other, for example a queue file on container-local storage or a database restored from backup.

Every `serve` and `run-batch` reconciles them before its first sweep. For each source, against the database it is ingested into:

- A document the source owns with no queue record gets one back, so the next sweep can delete it if its file is gone.
- A recorded revision for a URI the database no longer holds is cleared, so the next sweep re-ingests it.
- A document without `source_id` that the source's queue records as ingested is attributed to the source, without a fetch.

A permanently failed job also records a revision, which keeps an unchanged failing file from being queued forever. Reconciliation leaves those alone.

A document two sources both ingested stays unattributed, with a warning naming them. Documents of a `source_id` no longer configured are counted in a warning and left alone. Reconciliation reads the document metadata table once and no content.

`run-batch --dry-run` opens no database, so a manifest written while the two disagreed misses deletions that reconciliation would find, and `--manifest` replay does not sweep. A normal sweep afterwards catches up.

## Documents with no source attribution

At startup, reconciliation reports documents that remain unattributed:

```
14 document(s) remain without source attribution. Review whether they are
intentionally unmanaged or have ambiguous or lost ownership.
```

PDF attachments of a stored document are not counted. An unattributed document was added by hand, was ingested by a version that did not record `source_id` and whose queue record is gone, is claimed by two sources and named in its own warning, or is a PDF attachment whose parent document is gone. A document claimed by two sources needs the sources separated, not the document deleted. While a lost document's file is still at its source, the next sweep attributes it. Once the file is gone, nothing identifies it.

To clear them:

**1. Run one batch and check it succeeds.**

```bash
haiku-ingester --config /etc/haiku/haiku.rag.yaml run-batch
echo $?    # must be 0
```

Reconciliation attributes what the queue records, and the sweep attributes what it finds. An unchanged document costs no fetch, conversion or embedding. A non-zero exit means a source failed to sweep or a job died, so some live documents may still be unattributed. Fix that and run again before step 2.

**2. List what is left.**

```bash
haiku-rag --config /etc/haiku/haiku.rag.yaml list \
  -f "metadata NOT LIKE '%\"source_id\"%' AND metadata NOT LIKE '%\"parent_uri\"%'"
```

Narrow it on a database that also holds hand-added documents:

```bash
  -f "metadata NOT LIKE '%\"source_id\"%' AND metadata NOT LIKE '%\"parent_uri\"%' AND uri LIKE 'file:///srv/handbook/%'"
```

The list holds candidates to review, not documents to delete. `metadata` is matched as JSON text, so a document whose own metadata contains the string `source_id` or `parent_uri` is left out.

PDF attachments are listed with `-f "metadata LIKE '%\"parent_uri\"%'"`. An attachment is orphaned only when no document in the database has the URI its `parent_uri` names. Check that with a filter on `uri`, not with the attachment filter, which leaves top-level parents out.

**3. Delete what you confirm.**

```bash
haiku-rag --config /etc/haiku/haiku.rag.yaml delete <id>
```

Read the URIs first. A hand-added document whose file has since moved looks the same as a document lost by the ingester.
