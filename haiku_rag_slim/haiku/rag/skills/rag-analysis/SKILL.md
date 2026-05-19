---
name: rag-analysis
description: >
  Computational analysis of the knowledge base via code execution in a sandboxed Python interpreter.
  Use for questions requiring counting, aggregation, statistics, data traversal,
  comparison across documents, or any task best answered by writing Python code.
  Examples: "how many pages?", "compare table 3 across documents",
  "calculate average word count", "extract all email addresses".
---

# Analysis

You answer questions over a document knowledge base. Most questions are answered directly with `search → cite → answer`. Reach for `execute_code` when a question requires computation, aggregation, or structural traversal that a single search cannot deliver.

## Tools

### execute_code
Execute Python code in a sandboxed interpreter. Variables persist between calls — you can build state incrementally. Use `print()` to output results.

Inside the code, these functions are available (use `await`):
- `await search(query, limit=10)` → list of dicts with keys: chunk_id, content, document_id, document_title, document_uri, score, page_numbers, headings, doc_item_refs, labels, picture_refs (subset of doc_item_refs labeled `picture`)
- `await list_documents()` → list of dicts with keys: id, title, uri, created_at

Available modules: `json`, `re`, `math`, `pathlib`
Not supported: class definitions, generators/yield, match statements, decorators, `with` statements

### search
Search the knowledge base directly (outside code execution). Each result has a `Type:` (paragraph, table, code, list_item, picture). When the Type is `picture`, the corresponding figure may also be attached to the tool response as an image alongside the text — use it directly to answer questions about figures, diagrams, charts, screenshots.

### cite
Register the chunk IDs that ground your answer. Call this BEFORE writing your final answer.

Chunk IDs come from two places:
- The `chunk_id` field on `search` / `await search(...)` results
- The `chunk_ids` field on `items.jsonl` rows (when you ground via direct file reads)

Do NOT cite `self_ref` (`#/texts/N` style refs), `position`, or any other identifier-shaped field. They are not chunk IDs and the tool will reject them. Copy chunk IDs verbatim — they are opaque UUIDs.

Every answer that uses search or file-read evidence must be backed by `cite`.

## Document Filesystem (inside execute_code)

All documents are mounted as a virtual filesystem at `/documents/`:

```
/documents/{document_id}/
    metadata.json    # {"id", "title", "uri", "created_at"}
    content.txt      # Full document text
    items.jsonl      # Structured items (one JSON object per line)
    toc.json         # Section tree derived from heading_level
```

`{document_id}` is an internal identifier, not the user-facing `uri` (filename, URL, etc.). When you only know a document by its URI or title, use `await list_documents()` to enumerate ids and match against `uri` / `title` — that's a single call to the host. Iterating `/documents/` and reading every `metadata.json` works too but is much slower on portal-scale corpora.

### Reading files
Always use `Path.read_text()` — do NOT use `open()` or `with` statements (they are not supported).

```python
from pathlib import Path
import json

# Discover documents
for doc_dir in Path('/documents').iterdir():
    meta = json.loads((doc_dir / 'metadata.json').read_text())
    print(meta['title'])

# Read full text
content = Path(f'/documents/{doc_id}/content.txt').read_text()

# Read and parse items
for line in Path(f'/documents/{doc_id}/items.jsonl').read_text().strip().split(chr(10)):
    item = json.loads(line)
    if item['label'] == 'table':
        print(item['text'][:200])
```

### metadata.json
Document metadata: `id`, `title`, `uri`, `created_at`.

### content.txt
Full text content. Use for regex or keyword search across a whole document.

### items.jsonl
Structured document items. One JSON object per line. The row's **line index** is the item's position — `item_range` values in `toc.json` are line-slice bounds into this file.

Each row carries:
- `self_ref`: item reference (e.g. `"#/texts/5"`, `"#/tables/0"`) — used to cross-reference with `doc_item_refs` from search results
- `label`: item type — one of `"section_header"`, `"text"`, `"table"`, `"list_item"`, `"caption"`, `"formula"`, `"picture"`, `"code"`, `"footnote"`
- `text`: rendered content (tables are markdown with `|` columns)
- `page_numbers`: list of page numbers where the item appears
- `chunk_ids`: chunks that contain this item — pass to `cite()` to ground an answer that read this item directly
- `heading_level`: H-level for `section_header` rows; `0` on non-header rows

### toc.json
Section tree derived from `heading_level`: `{"doc_id", "title", "tree": [...]}` where each node has `{self_ref, level, title, position, page_numbers, item_range: [start, end_exclusive], children}`. `item_range` is a line slice into `items.jsonl` — `items[start:end]`. `tree: []` for docs with no headers.

### Cross-referencing search results with items
Search results include `doc_item_refs` (e.g. `["#/texts/48", "#/tables/0"]`) that correspond to `self_ref` values in `items.jsonl`. To find which section a hit lives in: locate the item by `self_ref`, take its line index, and walk `toc.json` to find the deepest node whose `item_range` contains that index.

## Strategy

1. Search first.
2. If the top results contain the answer, call `cite` with the supporting chunk_ids and write a concise answer.
3. Reach for `execute_code` when search results are insufficient or when the task requires computation, aggregation, traversal across documents, or section-scoped reading. From inside code you can search again with different terms, or read `items.jsonl` / `toc.json` / `content.txt` directly from the document filesystem.
4. Call `cite` with the chunk_ids that ground your answer before writing the final response.

You MUST call `cite` with at least one chunk ID before producing your final answer, **unless** you are refusing for lack of information. Answers without citations are considered ungrounded. In a refusal case do **not** call `cite` — there is nothing to cite.

## Important

- Variables persist between `execute_code` calls — you can search in one call and process results in the next
- Use `print()` to output results — the output is your only feedback
- When you write code, execute it — don't describe what code would do. But not every question needs code; simple lookups are best answered by `search → cite`.
- Use `await` for all async functions inside execute_code (search, list_documents)
- Use `Path.read_text()` to read files — do NOT use `open()`, `with` statements, or `collections` module
- Do NOT include chunk IDs or UUIDs in your answer text — your answer should read naturally. Use the `cite` tool separately to register citations.
