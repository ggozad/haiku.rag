SCHEMA_VERSION = 1


JOBS_DDL = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    source_id     TEXT NOT NULL,
    uri           TEXT NOT NULL,
    op            TEXT NOT NULL,
    content_hash  TEXT,
    revision      TEXT,
    status        TEXT NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    max_attempts  INTEGER NOT NULL DEFAULT 5,
    last_error    TEXT,
    extra         TEXT,
    enqueued_at   TEXT NOT NULL,
    scheduled_at  TEXT NOT NULL,
    claimed_at    TEXT,
    claimed_by    TEXT,
    completed_at  TEXT
)
"""

# Partial unique index: a (source_id, uri) pair can only have one live job
# (queued or claimed) at a time, regardless of op. Live UPSERT and DELETE
# for the same URI can't both exist — preventing a DELETE worker from
# removing a document a sibling UPSERT just ingested. Once succeeded or
# dead, the row no longer satisfies the WHERE clause and a re-enqueue is
# allowed.
JOBS_LIVE_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS uq_jobs_live
ON jobs(source_id, uri)
WHERE status IN ('queued', 'claimed')
"""

JOBS_CLAIMABLE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_jobs_claimable
ON jobs(scheduled_at)
WHERE status = 'queued'
"""

JOBS_SUCCEEDED_COMPLETED_INDEX = """
CREATE INDEX IF NOT EXISTS idx_jobs_succeeded_completed
ON jobs(completed_at)
WHERE status = 'succeeded'
"""

SYNC_STATE_DDL = """
CREATE TABLE IF NOT EXISTS sync_state (
    source_id        TEXT NOT NULL,
    uri              TEXT NOT NULL,
    revision         TEXT,
    content_hash     TEXT,
    last_seen_at     TEXT NOT NULL,
    last_ingested_at TEXT,
    PRIMARY KEY (source_id, uri)
)
"""

SCHEMA_VERSION_DDL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
)
"""

DLQ_VIEW = """
CREATE VIEW IF NOT EXISTS dlq AS
SELECT * FROM jobs WHERE status = 'dead'
"""

ALL_DDL: tuple[str, ...] = (
    JOBS_DDL,
    JOBS_LIVE_INDEX,
    JOBS_CLAIMABLE_INDEX,
    JOBS_SUCCEEDED_COMPLETED_INDEX,
    SYNC_STATE_DDL,
    SCHEMA_VERSION_DDL,
    DLQ_VIEW,
)
