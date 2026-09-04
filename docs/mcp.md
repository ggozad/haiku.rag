# Model Context Protocol (MCP)

The MCP server exposes `haiku.rag` as MCP tools for compatible MCP clients like Claude Desktop.

## Starting MCP Server

The MCP server supports Streamable HTTP and stdio transports:

```bash
# Default streamable HTTP transport on 127.0.0.1:8001
haiku-rag mcp

# Custom port
haiku-rag mcp --port 9000

# Bind to all interfaces (e.g. inside a container)
haiku-rag mcp --host 0.0.0.0 --port 8001

# stdio transport (for Claude Desktop)
haiku-rag mcp --stdio

# Without ask_question and analyze, which run a model on the server
haiku-rag mcp --stdio --no-agents
```

`--host` defaults to `127.0.0.1` (loopback only). Bind to `0.0.0.0` only
when you want the MCP server reachable from outside the local machine —
e.g. inside a Docker container with port mapping, or on a trusted LAN.

The server opens the database read-only. Ingestion goes through the CLI
(`haiku-rag add`, `add-src`, `delete`) or [`haiku-ingester`](ingester.md).

## Collections

With several databases in `lancedb.databases`, the server covers all of
them, as `haiku-rag search` does. Results, documents and citations name
theirs in `source`. `sources` on the search and question tools restricts a
call to a subset; `source` on `get_document` names the database holding the
document. A name the server does not cover is an error.
`haiku-rag --db-name NAME mcp` serves one. See
[Multiple Databases](configuration/storage.md#multiple-databases).

## Claude Code

The repository ships a plugin that registers the server and a skill telling
Claude when and how to use it:

```bash
claude plugin marketplace add ggozad/haiku.rag
claude plugin install haiku-rag
```

The plugin runs `haiku-rag mcp --stdio`, so `haiku-rag` must be on the PATH
and the configuration decides the database. The skill pre-approves every tool
and is also invocable as `/haiku-rag`. To register the server without the
plugin:

```bash
claude mcp add haiku-rag -- haiku-rag mcp --stdio
```

## Claude Desktop Integration

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "haiku-rag": {
      "command": "haiku-rag",
      "args": ["mcp", "--stdio"]
    }
  }
}
```

With a custom database path:

```json
{
  "mcpServers": {
    "haiku-rag": {
      "command": "haiku-rag",
      "args": ["mcp", "--stdio", "--db", "/path/to/database.lancedb"]
    }
  }
}
```

After restarting Claude Desktop, you can ask Claude to search your documents or answer questions using your knowledge base.

## Tools

Every tool is read-only and says so in its annotations. Each parameter carries
a description in the tool schema, so the listing below names them without
repeating it.

| Tool | Registered | Parameters |
|---|---|---|
| `search_documents` | always | `query`, `limit`, `include_images`, `filter`, `sources` |
| `search_documents_by_image` | multimodal embedder only | `image_base64`, `limit`, `include_images`, `filter`, `sources` |
| `get_document` | always | `document_id`, `source` |
| `get_document_outline` | always | `document_id`, `source` |
| `get_document_section` | always | `document_id`, `section_id`, `source` |
| `list_documents` | always | `limit`, `offset`, `filter` |
| `ask_question` | unless `--no-agents` | `question`, `images_base64`, `sources` |
| `analyze` | unless `--no-agents` | `question`, `filter`, `images_base64`, `sources` |

`search_documents` runs hybrid search, vector and full-text. Its text content
is the rendering the in-process agents read: results best first, each with its
rank, `Document ID`, `Collection` when the server covers several, the document
title, section headings, the matched chunk's metadata when it has any, and the
passage expanded to its section the way the agents get it
(`search.max_context_chars` caps it). Pictures in the results follow as
image blocks, one per distinct picture, each preceded by a line naming its
result; `include_images: false` leaves them out. The structured content is the
`SearchResult` list without picture bytes. Scores are not comparable across
queries or search types, so rank is the signal. `search_documents_by_image`
embeds the query image and searches by vector similarity alone.

`get_document` returns a document whole, in reading order. For a long one,
`get_document_outline` returns the heading tree with page numbers and
`get_document_section` the text of one section, subsections included; a
node's `id` in the outline is the `section_id`. A document without headings
has an empty outline. `list_documents` returns titles, URIs and metadata,
which is how a client learns what a filter can match.

`ask_question` runs the RAG agent on the server and returns an answer
followed by its citations. `analyze` writes and runs Python in a sandbox
over the documents, for counting, aggregation and computation across
documents. Both cost a model call.

### Filters

`filter` is a SQL WHERE clause over the document columns `id`, `uri`, `title`,
`metadata`, `created_at`, `updated_at`. `metadata` is a JSON string, so match
its keys with LIKE:

```sql
metadata LIKE '%"author": "Smith"%'
uri LIKE '%.pdf'
title = 'Q3 report'
```

### Errors

A failure is an MCP error, never an empty result. Expected failures carry a
message: a document or section id that matches nothing, a collection the
server does not cover, a filter the query engine rejects (with its message),
invalid base64,
and an `ask_question` or `analyze` failure naming only the exception type.
Anything else reaches the client as `Error calling tool 'name'` and its
traceback goes to the server log.

### Instructions

The server publishes `instructions` describing the knowledge base: what it
holds, when to reach for it, the collection names when it covers several, and
`prompts.domain_preamble` when set. Claude Code shows them to the model. Claude
Desktop does not, so every tool description stands on its own.

## Continuous ingestion

For continuous document ingestion (filesystem watch, S3 polling, HTTP
sources, a job queue with retries), run [`haiku-ingester`](ingester.md)
as a separate process against the same LanceDB.
