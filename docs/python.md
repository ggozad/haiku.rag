# Python API

Use `haiku.rag` directly in your Python applications.

## Basic Usage

```python
from pathlib import Path
from haiku.rag.client import HaikuRAG

# Use as async context manager (recommended)
async with HaikuRAG("Path(path/to/database.lancedb")) as client:
    # Your code here
    pass
```

## Document Management

### Creating Documents

From text:
```python
doc = await client.create_document(
    content="Your document content here",
    uri="doc://example",
    title="My Example Document",  # optional human‑readable title
    metadata={"source": "manual", "topic": "example"}
)
```

With custom externally generated chunks:
```python
from haiku.rag.store.models.chunk import Chunk

# Create custom chunks with optional embeddings
chunks = [
    Chunk(
        content="This is the first chunk",
        metadata={"section": "intro"}
    ),
    Chunk(
        content="This is the second chunk",
        metadata={"section": "body"},
        embedding=[0.1] * 1024  # Optional pre-computed embedding
    ),
]

doc = await client.create_document(
    content="Full document content",
    uri="doc://custom",
    metadata={"source": "manual"},
    chunks=chunks  # Use provided chunks instead of auto-generating
)
```

From file:
```python
doc = await client.create_document_from_source(
    "path/to/document.pdf", title="Project Brief"
)
```

From URL:
```python
doc = await client.create_document_from_source(
    "https://example.com/article.html", title="Example Article"
)
```

### Retrieving Documents

By ID:
```python
doc = await client.get_document_by_id(1)
```

By URI:
```python
doc = await client.get_document_by_uri("file:///path/to/document.pdf")
```

List all documents:
```python
docs = await client.list_documents(limit=10, offset=0)
```

Filter documents by properties:
```python
# Filter by URI pattern
docs = await client.list_documents(filter="uri LIKE '%arxiv%'")

# Filter by exact title
docs = await client.list_documents(filter="title = 'My Document'")

# Combine multiple conditions
docs = await client.list_documents(
    limit=10,
    filter="uri LIKE '%.pdf' AND title LIKE '%paper%'"
)
```

### Updating Documents

Update entire document:
```python
doc.content = "Updated content"
await client.update_document(doc)
```

Update specific fields:
```python
# Update only content (triggers re-chunking)
await client.update_document_fields(
    document_id=doc.id,
    content="New content"
)

# Update only metadata (no re-chunking)
await client.update_document_fields(
    document_id=doc.id,
    metadata={"version": "2.0", "updated_by": "admin"}
)

# Update only title (no re-chunking)
await client.update_document_fields(
    document_id=doc.id,
    title="New Title"
)

# Update multiple fields at once
await client.update_document_fields(
    document_id=doc.id,
    content="New content",
    title="Updated Title",
    metadata={"status": "final"}
)

# Use custom chunks instead of auto-generation
custom_chunks = [
    Chunk(content="Custom chunk 1"),
    Chunk(content="Custom chunk 2"),
]
await client.update_document_fields(
    document_id=doc.id,
    chunks=custom_chunks
)
```

**Performance Note:** Updates to only `metadata` or `title` skip re-chunking for efficiency. Updates to `content` or `chunks` will regenerate or replace the document's chunks.

### Deleting Documents

```python
await client.delete_document(doc.id)
```

### Rebuilding the Database

```python
from haiku.rag.client import RebuildMode

# Full rebuild (default) - re-converts from source files, re-chunks, re-embeds
async for doc_id in client.rebuild_database():
    print(f"Processed document {doc_id}")

# Re-chunk from stored content (no source file access)
async for doc_id in client.rebuild_database(mode=RebuildMode.RECHUNK):
    print(f"Processed document {doc_id}")

# Only regenerate embeddings (fastest, keeps existing chunks)
async for doc_id in client.rebuild_database(mode=RebuildMode.EMBED_ONLY):
    print(f"Processed document {doc_id}")
```

**Rebuild modes:**

- `RebuildMode.FULL` - Re-convert from source files, re-chunk, re-embed (default)
- `RebuildMode.RECHUNK` - Re-chunk from existing document content, re-embed
- `RebuildMode.EMBED_ONLY` - Keep existing chunks, only regenerate embeddings

## Maintenance

Run maintenance to optimize storage and prune old table versions:

```python
await client.vacuum()
```

This compacts tables and removes historical versions to keep disk usage in check. It’s safe to run anytime, for example after bulk imports or periodically in long‑running apps.

### Atomic Writes and Rollback

Document create and update operations take a snapshot of table versions before any write and automatically roll back to that snapshot if something fails (for example, during chunking or embedding). This restores both the `documents` and `chunks` tables to their pre‑operation state using LanceDB’s table versioning.

- Applies to: `create_document(...)`, `create_document_from_source(...)`, `update_document(...)`, and internal rebuild/update flows.
- Scope: Both document rows and all associated chunks are rolled back together.
- Vacuum: Running `vacuum()` later prunes old versions for disk efficiency; rollbacks occur immediately during the failing operation and are not impacted.

## Searching Documents

The search method performs native hybrid search (vector + full-text) using LanceDB with optional reranking for improved relevance:

Basic hybrid search (default):
```python
results = await client.search("machine learning algorithms", limit=5)
for chunk, score in results:
    print(f"Score: {score:.3f}")
    print(f"Content: {chunk.content}")
    print(f"Document ID: {chunk.document_id}")
```

Search with different search types:
```python
# Vector search only
results = await client.search(
    query="machine learning",
    limit=5,
    search_type="vector"
)

# Full-text search only
results = await client.search(
    query="machine learning",
    limit=5,
    search_type="fts"
)

# Hybrid search (default - combines vector + fts with native LanceDB RRF)
results = await client.search(
    query="machine learning",
    limit=5,
    search_type="hybrid"
)

# Process results
for chunk, relevance_score in results:
    print(f"Relevance: {relevance_score:.3f}")
    print(f"Content: {chunk.content}")
    print(f"From document: {chunk.document_id}")
    print(f"Document URI: {chunk.document_uri}")
    print(f"Document Title: {chunk.document_title}")  # when available
    print(f"Document metadata: {chunk.document_meta}")
```

### Filtering Search Results

Filter search results to only include chunks from documents matching specific criteria:

```python
# Filter by document URI pattern
results = await client.search(
    query="machine learning",
    limit=5,
    filter="uri LIKE '%arxiv%'"
)

# Filter by exact document title
results = await client.search(
    query="neural networks",
    limit=5,
    filter="title = 'Deep Learning Guide'"
)

# Combine multiple filter conditions
results = await client.search(
    query="AI research",
    limit=5,
    filter="uri LIKE '%.pdf' AND title LIKE '%paper%'"
)

# Filter with any search type
results = await client.search(
    query="transformers",
    limit=5,
    search_type="vector",
    filter="uri LIKE '%huggingface%'"
)
```

**Note:** Filters apply to document properties only. Available columns for filtering:
- `id` - Document ID
- `uri` - Document URI/URL
- `title` - Document title (if set)
- `created_at`, `updated_at` - Timestamps
- `metadata` - Document metadata (as string, use LIKE for pattern matching)

### Expanding Search Context

Expand search results with adjacent chunks for more complete context:

```python
# Get initial search results
search_results = await client.search("machine learning", limit=3)

# Expand with adjacent chunks using config setting
expanded_results = await client.expand_context(search_results)

# Or specify a custom radius
expanded_results = await client.expand_context(search_results, radius=2)

# The expanded results contain chunks with combined content from adjacent chunks
for chunk, score in expanded_results:
    print(f"Expanded content: {chunk.content}")  # Now includes before/after chunks
```

**Smart Merging**: When expanded chunks overlap or are adjacent within the same document, they are automatically merged into single chunks with continuous content. This eliminates duplication and provides coherent text blocks. The merged chunk uses the highest relevance score from the original chunks.

This is automatically used by the QA system when `processing.context_chunk_radius > 0` (configured in `haiku.rag.yaml`) to provide better answers with more complete context.

## Question Answering

Ask questions about your documents:

```python
answer = await client.ask("Who is the author of haiku.rag?")
print(answer)
```

Ask questions with citations showing source documents:

```python
answer = await client.ask("Who is the author of haiku.rag?", cite=True)
print(answer)
```

Customize the QA agent's behavior with a custom system prompt:

```python
custom_prompt = """You are a technical support expert for WIX.
Answer questions based on the knowledge base documents provided.
Be concise and helpful."""

answer = await client.ask(
    "How do I create a blog?",
    system_prompt=custom_prompt
)
```

The QA agent will search your documents for relevant information and use the configured LLM to generate a comprehensive answer. With `cite=True`, responses include citations showing which documents were used as sources. Citations prefer the document title when present, otherwise they use the URI.

The QA provider and model are configured in `haiku.rag.yaml` or can be passed directly to the client (see [Configuration](configuration/index.md)).

See also: [Agents](agents.md) for details on the QA agent and the multi‑agent research workflow.
