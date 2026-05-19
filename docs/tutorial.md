# Quickstart

Goal: install haiku.rag, index a document, and chat with it. Five minutes if you already have Ollama.

## Install

```bash
uv pip install haiku.rag
```

You also need [Ollama](https://ollama.com/) for the default embedding and answering models:

```bash
ollama pull qwen3-embedding:4b
ollama pull gpt-oss
```

!!! note "Prefer OpenAI?"
    Drop this into a `haiku.rag.yaml` next to where you'll run the CLI:

    ```yaml
    embeddings:
      model:
        provider: openai
        name: text-embedding-3-small
        vector_dim: 1536

    qa:
      model:
        provider: openai
        name: gpt-4o-mini
    ```

    Then `export OPENAI_API_KEY="sk-..."` and continue with the rest of this page. Any provider Pydantic AI supports works the same way. See [Providers](configuration/providers.md).

## Initialize

```bash
haiku-rag init
```

This creates a LanceDB database in your platform's user directory. Pass `--db` to any subcommand to use a different path:

```bash
haiku-rag init --db /tmp/test.lancedb
```

## Add a document

Add a file, a URL, or a whole folder:

```bash
haiku-rag add-src https://arxiv.org/pdf/2408.09134
haiku-rag add-src ~/Documents/papers/
```

Or paste text inline:

```bash
haiku-rag add "Yiorgis wrote haiku.rag in 2025."
```

Each `add-src` call converts the file with Docling, splits it into chunks, embeds them, and writes everything to LanceDB. Run `haiku-rag list` to see what you've added, `haiku-rag info` for a database summary.

## Chat

```bash
haiku-rag chat
```

Ask a question. The agent searches your documents, expands context around the hits, and answers with citations pointing back to the source page and section. Citations are expandable, with visual grounding so you can see the chunk highlighted on the original page. Follow-ups continue within the same session. Start a new session when you switch topics.

You can also ask a single question directly from the CLI without launching the TUI:

```bash
haiku-rag ask "Who wrote haiku.rag?"
```

## Where to go next

- [Chat](chat.md): sessions, citations, and the full TUI.
- [CLI reference](cli.md): every command.
- [Python API](python.md): use haiku.rag in your own code.
- [Skills](skills/index.md): the rag and rag-analysis skills the client wraps.
- [Tuning](tuning.md): better retrieval.
- [Configuration](configuration/index.md): every setting.
