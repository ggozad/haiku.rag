# Python API

Use `haiku.rag` directly in your Python applications.

## Basic Usage

```python
from pathlib import Path
from haiku.rag.client import HaikuRAG

# Create a new database
async with HaikuRAG("path/to/database.lancedb", create=True) as client:
    # Your code here
    pass

# Open an existing database (will fail if database doesn't exist)
async with HaikuRAG("path/to/database.lancedb") as client:
    # Your code here
    pass
```

!!! note
    Databases must be explicitly created with `create=True` or via `haiku-rag init` before use. Operations on non-existent databases will raise `FileNotFoundError`.

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

From HTML content (preserves document structure):
```python
html_content = "<h1>Title</h1><p>Paragraph</p><ul><li>Item 1</li></ul>"
doc = await client.create_document(
    content=html_content,
    uri="doc://html-example",
    format="html"  # parse as HTML instead of markdown
)
```

The `format` parameter controls how text content is parsed:

- `"md"` (default) - Parse as Markdown
- `"html"` - Parse as HTML, preserving semantic structure (headings, lists, tables)

!!! note
    The document's `content` field stores the markdown export of the parsed document for consistent display. The original input is preserved in the `docling_document_json` field.

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

### Importing Pre-Processed Documents

If you process documents externally or need custom processing, use `import_document()`:

```python
from haiku.rag.store.models.chunk import Chunk

# Convert your source to a DoclingDocument
docling_doc = await client.convert("path/to/document.pdf")

# Create chunks (embeddings optional - will be generated if missing)
chunks = [
    Chunk(
        content="This is the first chunk",
        metadata={"section": "intro"},
        order=0,
    ),
    Chunk(
        content="This is the second chunk",
        metadata={"section": "body"},
        embedding=[0.1] * 1024,  # Optional: pre-computed embedding
        order=1,
    ),
]

# Import document with custom chunks
doc = await client.import_document(
    docling_document=docling_doc,
    chunks=chunks,
    uri="doc://custom",
    title="Custom Document",
    metadata={"source": "external-pipeline"},
)
```

The `docling_document` provides rich metadata for visual grounding, page numbers, and section headings. Content is automatically extracted from the DoclingDocument.

See [Custom Processing Pipelines](custom-pipelines.md) for building pipelines with `convert()`, `chunk()`, and `embed_chunks()`.

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

```python
# Update content (triggers re-chunking)
await client.update_document(document_id=doc.id, content="New content")

# Update metadata only (no re-chunking)
await client.update_document(
    document_id=doc.id,
    metadata={"version": "2.0", "updated_by": "admin"}
)

# Update title only (no re-chunking)
await client.update_document(document_id=doc.id, title="New Title")

# Update multiple fields at once
await client.update_document(
    document_id=doc.id,
    content="New content",
    title="Updated Title",
    metadata={"status": "final"}
)

# Use custom chunks (embeddings optional - will be generated if missing)
custom_chunks = [
    Chunk(content="Custom chunk 1"),
    Chunk(content="Custom chunk 2", embedding=[...]),  # Pre-computed embedding
]
await client.update_document(document_id=doc.id, chunks=custom_chunks)
```

**Notes:**

- Updates to only `metadata` or `title` skip re-chunking
- Updates to `content` trigger re-chunking and re-embedding
- Custom `chunks` with embeddings are stored as-is; missing embeddings are generated automatically

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
for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.content}")
    print(f"Document ID: {result.document_id}")
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
for result in results:
    print(f"Relevance: {result.score:.3f}")
    print(f"Content: {result.content}")
    print(f"From document: {result.document_id}")
    print(f"Document URI: {result.document_uri}")
    print(f"Document Title: {result.document_title}")  # when available
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

# Expand search results with adjacent content from the source document
expanded_results = await client.expand_context(search_results)

# The expanded results contain chunks with combined content
for result in expanded_results:
    print(f"Expanded content: {result.content}")
```

Context expansion uses your configuration settings:

- **search.context_radius**: For text content (paragraphs), includes N DocItems before and after
- **search.max_context_items**: Limits how many document items can be included
- **search.max_context_chars**: Hard limit on total characters

**Type-aware expansion**: Structural content (tables, code blocks, lists) automatically expands to include the complete structure, regardless of how it was split during chunking.

**Smart Merging**: When expanded chunks overlap or are adjacent within the same document, they are automatically merged into single chunks with continuous content. This eliminates duplication and provides coherent text blocks. The merged chunk uses the highest relevance score from the original chunks.

## Question Answering

Ask questions about your documents:

```python
answer, citations = await client.ask("Who is the author of haiku.rag?")
print(answer)
for cite in citations:
    print(f"  [{cite.chunk_id}] {cite.document_title or cite.document_uri}")
```

Customize the QA agent's behavior with a custom system prompt:

```python
custom_prompt = """You are a technical support expert for WIX.
Answer questions based on the knowledge base documents provided.
Be concise and helpful."""

answer, citations = await client.ask(
    "How do I create a blog?",
    system_prompt=custom_prompt
)
```

The QA agent searches your documents for relevant information and uses the configured LLM to generate an answer. The method returns a tuple of `(answer_text, list[Citation])`. Citations include page numbers, section headings, and document references.

The QA provider and model are configured in `haiku.rag.yaml` or can be passed directly to the client (see [Configuration](configuration/index.md)).

See also: [Agents](agents.md) for details on the QA agent and the multi‑agent research workflow.
