import sqlalchemy as sa

SCHEMA_VERSION = 1

metadata = sa.MetaData()

schema_version = sa.Table(
    "schema_version",
    metadata,
    sa.Column("version", sa.Integer, primary_key=True),
)

sweeps = sa.Table(
    "sweeps",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("database", sa.Text, nullable=False),
    sa.Column("started_at", sa.Text, nullable=False),
    sa.Column("finished_at", sa.Text, nullable=False),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("table_versions", sa.Text),
    sa.Column("embedder", sa.Text),
    sa.Column("rebaseline", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Column("error", sa.Text),
    sa.Column("documents", sa.Integer),
    sa.Column("changed", sa.Integer),
    sa.Column("deleted", sa.Integer),
    sa.Index("ix_sweeps_database", "database", "id"),
)

fingerprints = sa.Table(
    "fingerprints",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("database", sa.Text, nullable=False),
    sa.Column("document_id", sa.Text, nullable=False),
    sa.Column("uri", sa.Text),
    sa.Column("title", sa.Text),
    sa.Column("change_key", sa.Text, nullable=False),
    sa.Column("md5", sa.Text),
    sa.Column("content_type", sa.Text),
    sa.Column("source_revision", sa.Text),
    sa.Column("metadata_keys", sa.Text, nullable=False),
    sa.Column("chunks", sa.Integer, nullable=False),
    sa.Column("embedded_chunks", sa.Integer, nullable=False),
    sa.Column("chars", sa.Integer, nullable=False),
    sa.Column("centroid", sa.LargeBinary),
    sa.Column("embedder", sa.Text),
    sa.Column("replacement_chars", sa.Integer, nullable=False),
    sa.Column("chunk_stats", sa.Text, nullable=False),
    sa.Column("created_at", sa.Text),
    sa.Column("updated_at", sa.Text),
    sa.Column(
        "became_current_sweep", sa.Integer, sa.ForeignKey("sweeps.id"), nullable=False
    ),
    sa.Column("ended_sweep", sa.Integer, sa.ForeignKey("sweeps.id")),
    sa.Column("deleted", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Index("ix_fingerprints_current", "database", "document_id", "ended_sweep"),
)

chunk_texts = sa.Table(
    "chunk_texts",
    metadata,
    sa.Column("database", sa.Text, nullable=False),
    sa.Column(
        "fingerprint_id",
        sa.Integer,
        sa.ForeignKey("fingerprints.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("text_hash", sa.Text, nullable=False),
    sa.Column("chars", sa.Integer, nullable=False),
    sa.Index("ix_chunk_texts_hash", "database", "text_hash"),
    sa.Index("ix_chunk_texts_fingerprint", "fingerprint_id"),
)

flags = sa.Table(
    "flags",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("identity", sa.Text, nullable=False, unique=True),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("database", sa.Text),
    sa.Column("subject", sa.Text),
    sa.Column("fingerprint_id", sa.Integer, sa.ForeignKey("fingerprints.id")),
    sa.Column("previous_fingerprint_id", sa.Integer, sa.ForeignKey("fingerprints.id")),
    sa.Column("members", sa.Text),
    sa.Column("reasons", sa.Text, nullable=False),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("raised_at", sa.Text, nullable=False),
    sa.Column("status_changed_at", sa.Text, nullable=False),
    sa.Column("note", sa.Text),
    sa.Index("ix_flags_database", "database", "status"),
)

watched = sa.Table(
    "watched",
    metadata,
    sa.Column("database", sa.Text, primary_key=True),
    sa.Column("uri", sa.Text, primary_key=True),
    sa.Column("note", sa.Text),
    sa.Column("added_at", sa.Text, nullable=False),
)

layout = sa.Table(
    "layout",
    metadata,
    sa.Column("database", sa.Text, primary_key=True),
    sa.Column("document_id", sa.Text, primary_key=True),
    sa.Column("sweep_id", sa.Integer, sa.ForeignKey("sweeps.id"), nullable=False),
    sa.Column("isolation", sa.Float),
    sa.Column("x", sa.Float),
    sa.Column("y", sa.Float),
)
