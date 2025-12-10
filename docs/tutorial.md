# Tutorial

This tutorial provides quickstart instructions for getting familiar with `haiku.rag`. This tutorial is intended for people who are familiar with command line and Python, but not different AI ecosystem tools.

The tutorial covers:

- RAG and embeddings basics
- Installing `haiku.rag` Python package
- Configuring `haiku.rag` with YAML
- Adding and retrieving items
- Inspecting the database

The tutorial uses OpenAI API service - no local installation needed and will work on computers with any amount of RAM and GPU. The OpenAI API is pay-as-you-go, so you need to top it up with at least ~$5 when creating the API key.

## Introduction

Retrieval-Augmented Generation (RAG) lets you give AI models access to your own documents and data. Instead of relying solely on the model's training data, RAG finds relevant information from your documents and includes it in the AI's responses.

`haiku.rag` handles the mechanics: it converts your documents into searchable embeddings, stores them locally, and retrieves relevant chunks when you ask questions. You provide the documents and questions, and it coordinates between the embedding service (like OpenAI) and the AI model to give you accurate, grounded answers.

## Setup

First, [get an OpenAI API key](https://platform.openai.com/api-keys).

Install `haiku.rag` Python package using [uv](https://docs.astral.sh/uv/getting-started/installation/) or your favourite Python package manager:

```bash
# Python 3.12+ needed
uv pip install haiku.rag
```

Configure haiku.rag to use OpenAI. Create a `haiku.rag.yaml` file:

```yaml
embeddings:
  model:
    provider: openai
    name: text-embedding-3-small  # or text-embedding-3-large
    vector_dim: 1536

qa:
  model:
    provider: openai
    name: gpt-4o-mini  # or gpt-4o, gpt-4, etc.
```

Set your OpenAI API key as an environment variable (API keys should not be stored in the YAML file):

```bash
export OPENAI_API_KEY="<your OpenAI API key>"
```

For the list of available OpenAI models and their vector dimensions, see the [OpenAI documentation](https://platform.openai.com/docs/guides/embeddings).

See [Configuration](configuration/index.md) for all available options.

## Initialize the database

Before adding documents, initialize the database:

```bash
haiku-rag init
```

This creates an empty database with the configured settings.

## Adding the first documents

Now you can add some pieces of text in the database:

```bash
haiku-rag add "Python is the best programming language in the world, because it is flexible, with robust ecosystem, open source licensing and thousands of contributors"
haiku-rag add "JavaScript is a popular programming language, but has a lot of warts"
haiku-rag add "PHP is a bad programming language, because of spotted security history, horrible syntax and declining popularity"
```

What will happen:

- The piece of text is sent to OpenAI `/embeddings` API service
- OpenAI translates the free form text to RAG embedding vectors needed for the retrieval
- The vector values will be stored in a local database

Now you can view your [LanceDB](https://lancedb.com/) database, and the embeddings it is configured for:

```bash
haiku-rag info
```

You should see output like:

```
haiku.rag database info
  path: /Users/moo/Library/Application Support/haiku.rag/haiku.rag.lancedb
  haiku.rag version (db): 0.20.0
  embeddings: openai/text-embedding-3-small (dim: 1536)
  documents: 3 (storage: 48.0 KB)
  chunks: 3 (storage: 52.0 KB)
  vector index: not created
──────────────────────────────────────────────────────────────────────────────────
Versions
  haiku.rag: 0.20.0
  lancedb: 0.25.2
  docling: 2.58.0
```

## Asking questions and retrieving information

Now we can use OpenAI LLMs to retrieve information from our embeddings database.

In this example, we connect to a remote OpenAI API.

Behind the scenes [pydantic-ai](https://ai.pydantic.dev/) query is created
using `OpenAIChatModel.request()`.

The easiest way to do this is `ask` CLI command:

```bash
haiku-rag ask "What is the best programming language in the world"
```

```
Question: What is the best programming language in the world

Answer:
According to the document, Python is considered the best programming language in the world due to its flexibility, robust ecosystem, open-source licensing, and thousands of contributors.
```

## Programmatic interaction in Python

You can interact with haiku.rag from Python. Since the API is async, we'll use IPython which supports async/await directly.

```bash
uv pip install ipython
ipython
```

Then run:

```python
from haiku.rag.client import HaikuRAG

# Uses database from default location (must be initialized first)
async with HaikuRAG() as client:
    answer, citations = await client.ask("What is the best programming language in the world?")
    print(answer)
```

You should see:

```
According to the document, Python is considered the best programming language in the world due to its flexibility, robust ecosystem, open-source licensing, and support from thousands of contributors.
```

## Complex documents

Haiku RAG can also handle types beyond plain text, including PDF, DOCX, HTML, and 40+ other file formats.

Here we add research papers about Python from [arxiv](https://arxiv.org/search/?query=python&searchtype=all&source=header) using URL retriever.

```bash
# Better Python Programming for all: With the focus on Maintainability
haiku-rag add-src --meta collection="Interesting Python papers" "https://arxiv.org/pdf/2408.09134"

# Interoperability From OpenTelemetry to Kieker: Demonstrated as Export from the Astronomy Shop
haiku-rag add-src --meta collection="Interesting Python papers" "https://arxiv.org/pdf/2510.11179"
```

Then we can query this:

```bash
haiku-rag ask "Who wrote a paper about OpenTelemetry interoperability, and what was his take"
```

We should get something along the lines:

```
Answer:
David Georg Reichelt from Lancaster University wrote a paper titled "Interoperability From OpenTelemetry to Kieker: Demonstrated as Export from the Astronomy Shop." In his work, he indicates that there is a structural difference between Kieker’s synchronous traces and OpenTelemetry’s asynchronous traces, leading to  limited compatibility between the two systems. This highlights the challenges of interoperability in observability frameworks.
```

We can also add offline files, like PDFs. Here we add a local file to ensure OpenAI does not cheat - a file we know that should not be very well known in Internet:

```bash
# This static file is supplied in haiku.rag repo
haiku-rag add-src "examples/samples/PyCon Finland 2025 Schedule.html"
```

And then:

```bash
haiku-rag ask "Who were presenting talks in Pycon Finland 2025? Can you give at least five different people."
```

```
The following people are presenting talks at PyCon Finland 2025:

 1 Jeremy Mayeres - Talk: The Limits of Imagination: An Open Source Journey
 2 Aroma Rodrigues - Talk: Python and Rust, a Perfect Pairing
 3 Andreas Jung - Talk: Guillotina Volto: A New Backend for Volto
 4 Daniel Vahla - Talk: Experiences with AI in Software Projects
 5 Andreas Jung (also presenting another talk) - Talk: Debugging Python
```

## Next Steps

- **[CLI Reference](cli.md)** - All available commands and options
- **[Python API](python.md)** - Use haiku.rag in your Python applications
- **[Agents](agents.md)** - Deep QA and multi-agent research workflows
- **[Configuration](configuration/index.md)** - Complete YAML configuration reference
- **[Server Mode](server.md)** - File monitoring, MCP server, and AG-UI streaming
