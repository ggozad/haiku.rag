# Custom Processing Pipelines

haiku.rag provides processing primitives that let you build custom document pipelines. Use these when you need control over conversion, chunking, or embedding—for example, to preprocess content, use external services, or implement custom chunking logic.

## Processing Primitives

The client exposes four primitives that can be composed into custom workflows:

| Primitive | Input | Output | Purpose |
|-----------|-------|--------|---------|
| `convert()` | file, URL, or text | `DoclingDocument` | Convert source to structured document |
| `chunk()` | `DoclingDocument` | `list[Chunk]` | Split document into chunks |
| `embed_chunks()` | `list[Chunk]` | `list[Chunk]` | Generate embeddings for chunks (includes contextualization) |
| `contextualize()` | `list[Chunk]` | `list[str]` | Get embedding-ready text (for custom embedders only) |

## Basic Pipeline

The standard pipeline mirrors what `create_document()` does internally:

```python
from haiku.rag.client import HaikuRAG
from haiku.rag.embeddings import embed_chunks

async with HaikuRAG("database.lancedb", create=True) as client:
    # 1. Convert source to DoclingDocument
    docling_doc = await client.convert("path/to/document.pdf")

    # 2. Chunk the document
    chunks = await client.chunk(docling_doc)

    # 3. Generate embeddings
    embedded_chunks = await embed_chunks(chunks)

    # 4. Store the document with chunks
    doc = await client.import_document(
        docling_document=docling_doc,
        chunks=embedded_chunks,
        uri="file:///path/to/document.pdf",
        title="My Document",
    )
```

## Convert

`convert()` accepts files, URLs, or plain text and returns a `DoclingDocument`:

```python
# From local file
docling_doc = await client.convert("report.pdf")
docling_doc = await client.convert(Path("/absolute/path/to/file.docx"))

# From URL (downloads and converts)
docling_doc = await client.convert("https://example.com/paper.pdf")

# From plain text (parsed as markdown by default)
docling_doc = await client.convert("# Title\n\nYour text content here")

# From HTML text (use format parameter to preserve structure)
html_content = "<h1>Title</h1><p>Paragraph</p><ul><li>Item</li></ul>"
docling_doc = await client.convert(html_content, format="html")

# From file:// URI
docling_doc = await client.convert("file:///path/to/document.md")
```

The `format` parameter controls how text content is parsed:

- `"md"` (default) - Parse as Markdown
- `"html"` - Parse as HTML, preserving semantic structure (headings, lists, tables)

!!! note
    The `format` parameter only applies to text content. Files and URLs determine their format from the file extension or content-type header.

Supported formats depend on your converter configuration (docling-local or docling-serve). Common formats include PDF, DOCX, HTML, Markdown, and images.

## Chunk

`chunk()` splits a `DoclingDocument` into `Chunk` objects with metadata:

```python
chunks = await client.chunk(docling_doc)

for chunk in chunks:
    print(f"Order: {chunk.order}")
    print(f"Content: {chunk.content[:100]}...")

    # Access structured metadata
    meta = chunk.get_chunk_metadata()
    print(f"Headings: {meta.headings}")
    print(f"Page numbers: {meta.page_numbers}")
    print(f"Labels: {meta.labels}")
```

Chunks are returned with:

- `content` - The chunk text
- `order` - Position in document (0-indexed)
- `metadata` - Dict with `doc_item_refs`, `headings`, `labels`, `page_numbers`
- `embedding` - `None` (not yet embedded)
- `document_id` - `None` (not yet stored)

## Embed

`embed_chunks()` generates embeddings for chunks. It automatically contextualizes chunks (prepends section headings) before embedding for better semantic search, without modifying the stored content:

```python
from haiku.rag.embeddings import embed_chunks

# Generate embeddings (returns new Chunk objects)
embedded_chunks = await embed_chunks(chunks)

# Original chunks unchanged
assert chunks[0].embedding is None

# New chunks have embeddings
assert embedded_chunks[0].embedding is not None
```

`embed_chunks()` returns **new** `Chunk` objects with embeddings set. The original chunks are not modified.

## Contextualize (for custom embedders)

`contextualize()` is a lower-level utility that prepares chunk content for embedding by prepending section headings. You only need this when implementing custom embedding logic—`embed_chunks()` already calls it internally.

```python
from haiku.rag.embeddings import contextualize

# Get embedding-ready text (only needed for custom embedders)
texts = contextualize(chunks)
# texts[0] might be: "Chapter 1\nIntroduction\nThe actual chunk content..."
```

See the [Custom Embeddings](#custom-embeddings) example below for when to use `contextualize()`.

## Custom Processing Examples

### Preprocessing Content

Transform content before chunking:

```python
def clean_markdown(text: str) -> str:
    """Remove HTML comments and normalize whitespace."""
    import re
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

async with HaikuRAG("database.lancedb", create=True) as client:
    # Convert to get raw content
    docling_doc = await client.convert("document.md")

    # Extract and preprocess markdown
    markdown = docling_doc.export_to_markdown()
    cleaned = clean_markdown(markdown)

    # Re-convert the cleaned content
    processed_doc = await client.convert(cleaned)

    # Continue with standard pipeline
    chunks = await client.chunk(processed_doc)
    embedded_chunks = await embed_chunks(chunks)

    await client.import_document(
        chunks=embedded_chunks,
        content=cleaned,
    )
```

### Filtering Chunks

Remove unwanted chunks before embedding:

```python
async with HaikuRAG("database.lancedb", create=True) as client:
    docling_doc = await client.convert("document.pdf")
    chunks = await client.chunk(docling_doc)

    # Filter out short chunks or boilerplate
    filtered = [
        c for c in chunks
        if len(c.content) > 50
        and "copyright" not in c.content.lower()
    ]

    # Re-number the order field after filtering
    for i, chunk in enumerate(filtered):
        chunk.order = i

    embedded_chunks = await embed_chunks(filtered)

    await client.import_document(
        docling_document=docling_doc,
        chunks=embedded_chunks,
    )
```

### Custom Embeddings

Use your own embedding service:

```python
async def my_embedder(texts: list[str]) -> list[list[float]]:
    """Your custom embedding function."""
    # Call your embedding API here
    ...

async with HaikuRAG("database.lancedb", create=True) as client:
    docling_doc = await client.convert("document.pdf")
    chunks = await client.chunk(docling_doc)

    # Use contextualize for consistent embedding input
    texts = contextualize(chunks)

    # Generate embeddings with your service
    embeddings = await my_embedder(texts)

    # Create chunks with embeddings
    from haiku.rag.store.models.chunk import Chunk

    embedded_chunks = [
        Chunk(
            content=chunk.content,
            metadata=chunk.metadata,
            order=chunk.order,
            embedding=embedding,
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    await client.import_document(
        docling_document=docling_doc,
        chunks=embedded_chunks,
    )
```

## When to Use Custom Pipelines

Use the primitives when you need to:

- Preprocess or clean content before chunking
- Filter or modify chunks before embedding
- Use external embedding services
- Implement custom chunking strategies
- Debug or inspect intermediate processing steps

For standard use cases, prefer the convenience methods:

- `create_document()` - Create from text content
- `create_document_from_source()` - Create from file or URL
- `import_document()` - Store pre-processed documents with custom chunks
