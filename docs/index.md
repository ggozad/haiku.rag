# haiku.rag

haiku.rag is an agentic RAG that runs locally and scales to production. Index PDFs, web pages, or whole directories. Ask questions and get cited answers. Build agents, skills, and MCP integrations on top.

haiku.rag is open-source first. The defaults run open models through [Ollama](https://ollama.com/) so the full pipeline works without external API keys. Any provider Pydantic AI supports works in its place.

Built on [LanceDB](https://lancedb.com/), [Pydantic AI](https://ai.pydantic.dev/), and [Docling](https://docling-project.github.io/docling/). Embedded database, no servers required.

## See it work

```bash
uv pip install haiku.rag

ollama pull qwen3-embedding:4b
ollama pull gpt-oss

haiku-rag init
haiku-rag add-src ~/Documents/some-paper.pdf
haiku-rag chat
```

The chat TUI is the fastest way to test retrieval and answer quality. `haiku-rag ask` and `haiku-rag search` cover one-shot CLI usage. Beyond that, the same database backs Python integrations, agents, skills, and the MCP server.

## What it does

**Ingest.** PDFs, DOCX, HTML, images, and 40+ formats via Docling. Add files, URLs, or whole directories. Monitor folders and reindex on change.

**Search.** Hybrid retrieval (vector + full-text with reciprocal rank fusion), optional cross-encoder reranking, structure-aware context expansion. Image-as-query and cross-modal retrieval when configured with a multimodal embedder.

**Answer.** RAG skill with citations including page numbers, section headings, and visual grounding. Vision-capable models receive figure bytes alongside chunk text. Analysis skill with a sandboxed Python interpreter for aggregation and computation across documents.

**Integrate.** Use it from Python, the CLI, the [MCP server](mcp.md), or as composable [skills](skills/index.md) built on haiku.skills. Skills bundle tools, prompts, and state for use inside any Pydantic AI agent.

**Operate.** Embedded LanceDB by default. Also runs on S3, GCS, Azure, or LanceDB Cloud. Time-travel queries via LanceDB versioning. File-monitoring mode for production deployments.

## Where to go next

- [Quickstart](tutorial.md): install through first chat in five minutes.
- [Skills](skills/index.md): the rag and rag-analysis skills you compose into Pydantic AI agents.
- [Python API](python.md): use haiku.rag from code.
- [MCP server](mcp.md): expose haiku.rag to Claude Desktop or other AI assistants.
- [Tuning](tuning.md): improve retrieval quality.
- [Configuration](configuration/index.md): every setting.

## License

MIT. Source on [GitHub](https://github.com/ggozad/haiku.rag).
