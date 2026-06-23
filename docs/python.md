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

# Open in read-only mode (blocks writes)
async with HaikuRAG("path/to/database.lancedb", read_only=True) as client:
    results = await client.search("query")  # Read operations work
    # await client.create_document(...)  # Would raise ReadOnlyError
```

!!! note
    Databases must be explicitly created with `create=True` or via `haiku-rag init` before use. Operations on non-existent databases will raise `FileNotFoundError`.

!!! note
    Read-only mode is useful for safely accessing databases without risk of modification. It blocks all write operations and downgrades an embedding provider/name mismatch to a warning instead of raising `ConfigMismatchError`.

!!! warning "Database Migrations"
    When upgrading haiku.rag to a version with schema changes, opening an existing database will raise `MigrationRequiredError`. Run `haiku-rag migrate` to apply pending migrations before using the database. See [CLI Database Management](cli.md#migrate-database) for details.

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
- `"plain"` - Plain text, no parsing (creates a simple text document)

!!! note
    The document's `content` field stores the markdown export of the parsed document for consistent display. The original DoclingDocument structure is preserved in the `docling_document` field (zstd-compressed, without page images). Page images are stored separately in `docling_pages`.

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

PDFs that carry attachments via the `/EmbeddedFiles` table are split into one Document per attachment, linked to the wrapper through `metadata.parent_uri`. See [PDF Embedded Attachments](configuration/processing.md#pdf-embedded-attachments).

### Retrieving Documents

By ID:
```python
doc = await client.get_document_by_id("document-id-string")
```

By URI:
```python
doc = await client.get_document_by_uri("file:///path/to/document.pdf")
```

List all documents:
```python
docs = await client.list_documents(limit=10, offset=0)

# Include full content and docling document (not loaded by default)
docs = await client.list_documents(include_content=True)
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

Count documents:
```python
# Count all documents
total = await client.count_documents()

# Count with filter
pdf_count = await client.count_documents(filter="uri LIKE '%.pdf'")
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
- Custom `chunks` with embeddings are stored as-is. Missing embeddings are generated automatically

### Deleting Documents

```python
await client.delete_document(doc.id)
```

Deleting a document also removes any child Documents linked to it via `metadata.parent_uri` (PDF attachment children, primarily). The cascade is transitive.

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

### Image queries

`client.search()` accepts an image instead of a text query when the configured embedder is multimodal (`embeddings.model.multimodal: true` on a vLLM, VoyageAI, or Cohere model). The image is embedded once and the chunks table is searched vector-only. Full-text search and reranking don't apply without a text query.

```python
from PIL import Image

# Bytes
results = await client.search(
    open("figure.png", "rb").read(),
    limit=5,
)

# PIL.Image works equivalently
results = await client.search(
    Image.open("figure.png"),
    limit=5,
)
```

Image queries surface picture chunks (synthetic per-figure chunks emitted at ingest under a multimodal embedder) and any text chunks whose vectors land near the image vector in the shared embedding space. Calling `client.search(bytes)` against a text-only embedder raises a `ValueError`.

### Expanding Search Context

Expand search results with surrounding content from the document:

```python
# Get initial search results
search_results = await client.search("machine learning", limit=3)

# Expand with section-bounded context
expanded_results = await client.expand_context(search_results)

for result in expanded_results:
    print(f"Expanded content: {result.content}")
```

Context expansion is automatic and section-aware. For structured documents (with section headers), expansion includes the entire section containing the match. For sections that exceed the budget or are too small (e.g., a title+authors area), expansion grows outward item-by-item from the match center, skipping noise labels (footnotes, page headers). This naturally crosses into adjacent sections until the budget is filled. For unstructured documents, expansion grows outward item-by-item. Results without `doc_item_refs` (e.g., custom chunks passed to `import_document`) pass through unexpanded.

Configuration:

- **search.max_context_chars**: Maximum characters in expanded context. Default: 10000.

**Smart Merging**: When expanded results overlap within the same document, they are automatically merged into a single result with continuous content and the highest relevance score.

## Question Answering

Ask questions about your documents:

```python
answer, citations = await client.ask("Who is the author of haiku.rag?")
print(answer)
for cite in citations:
    print(f"  [{cite.chunk_id}] {cite.document_title or cite.document_uri}")
```

Filter to specific documents:

```python
answer, citations = await client.ask(
    "What are the main findings?",
    filter="uri LIKE '%paper%'"
)
```

`client.ask` runs the [rag skill](skills/index.md) and returns `(answer_text, list[Citation])`. Citations include page numbers, section headings, and document references.

The QA provider and model are configured in `haiku.rag.yaml` or can be passed directly to the client (see [Configuration](configuration/index.md)).

See also: [Skills](skills/index.md) for details on the skills the client wraps.

## Analysis

Answer complex analytical questions via code execution:

```python
# Aggregation across documents
result = await client.analyze("Which quarter had the highest revenue?")
print(result.answer)
for citation in result.citations:
    print(citation.uri, citation.title)

# Computation within a document set
result = await client.analyze(
    "What is the average deal size mentioned in these contracts?",
    filter="uri LIKE '%contracts%'"
)
```

`client.analyze` runs the [analysis skill](skills/index.md), which writes and executes Python code in a sandboxed environment to solve problems that traditional RAG struggles with: aggregation, computation, and multi-document analysis.

See [Analysis skill](skills/analysis.md) for details on capabilities and configuration.

## Building custom agents

`client.ask` and `client.analyze` are the convenience wrappers. To build your own Pydantic AI agent against the same database, attach the rag and rag-analysis skills directly with `SkillToolset`. See [Skills](skills/index.md) for the full story and worked examples.

For the low-level toolset factories under `haiku.rag.tools` (one rung below the skill abstraction), see [Toolsets](tools.md).

## Importing Pre-Processed Documents

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

### Batch Import

Each `create_document*` / `import_document` call writes new versions of the `documents`, `document_meta`, `chunks`, and `document_items` tables. Ingesting many documents in a loop therefore creates a table version per document. Use `import_documents()` to write the whole batch in a single version per table:

```python
from haiku.rag.client import DocumentImport

imports = []
for path in paths:  # paths: list[Path]
    docling_doc = await client.convert(path)
    chunks = await client.chunk(docling_doc)
    imports.append(
        DocumentImport(
            docling_document=docling_doc,
            chunks=chunks,
            uri=path.absolute().as_uri(),
            metadata={"source": "external-pipeline"},
        )
    )

docs = await client.import_documents(imports)
```

Chunks without embeddings are embedded automatically. The import is all-or-nothing: if any document fails, all tables are restored to their pre-batch state.

See [Custom Processing Pipelines](custom-pipelines.md) for building pipelines with `convert()`, `chunk()`, and `embed_chunks()`.

## Maintenance

Run maintenance to optimize storage and prune old table versions:

```python
await client.vacuum()
```

This compacts tables and removes historical versions to keep disk usage in check. It’s safe to run anytime, for example after bulk imports or periodically in long‑running apps.

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

# Add VLM picture descriptions to an existing database. Runs the VLM
# over already-stored picture bytes, patches descriptions into the
# docling blob, then re-chunks + re-embeds. Requires
# processing.pictures='description' in the config.
async for doc_id in client.rebuild_database(mode=RebuildMode.DESCRIPTIONS):
    print(f"Described pictures in {doc_id}")
```

**Rebuild modes:**

- `RebuildMode.FULL` - Re-convert from source files, re-chunk, re-embed (default)
- `RebuildMode.RECHUNK` - Re-chunk from existing document content, re-embed
- `RebuildMode.EMBED_ONLY` - Keep existing chunks, only regenerate embeddings
- `RebuildMode.TITLE_ONLY` - Generate titles for untitled documents (no re-chunking or re-embedding)
- `RebuildMode.DESCRIPTIONS` - Run the VLM over picture bytes already stored on `document_items.picture_data`, patch descriptions into the docling blob, re-chunk + re-embed. Skips the docling parse entirely. Idempotent: pictures already carrying `meta.description.text` are not re-described, so the operation is safe to re-run.

### Generating Titles

Generate a title for an existing document on demand:

```python
title = await client.generate_title(doc)
if title:
    await client.update_document(document_id=doc.id, title=title)
```

Uses the same two-tier approach as automatic ingestion: structural extraction from DoclingDocument metadata first, with LLM fallback via `processing.title_model`. Unlike ingestion, this method does not catch exceptions. If the LLM call fails, the error propagates.

To batch-generate titles for all untitled documents, use `RebuildMode.TITLE_ONLY`:

```python
async for doc_id in client.rebuild_database(mode=RebuildMode.TITLE_ONLY):
    print(f"Generated title for {doc_id}")
```

See [Automatic Title Generation](configuration/processing.md#automatic-title-generation) for configuration details.

### Atomic Writes and Rollback

Document create, update, and delete operations take a snapshot of table versions before any write and automatically roll back to that snapshot if something fails (for example, during chunking or embedding). This restores the `documents`, `document_meta`, `chunks`, and `document_items` tables to their pre‑operation state using LanceDB’s table versioning. These writes are serialized under a single lock, so the rollback is safe under concurrent ingester workers.

- Applies to: `create_document(...)`, `create_document_from_source(...)`, `update_document(...)`, `delete_document(...)` (including the `parent_uri` cascade), and internal rebuild/update flows.
- Scope: Document rows, their mutable attributes, and all associated chunks and items are rolled back together.
- Vacuum: Running `vacuum()` later prunes old versions for disk efficiency. Rollbacks occur immediately during the failing operation and are not impacted.
