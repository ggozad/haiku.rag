# Architecture

High-level overview of haiku.rag components and data flow.

## System Overview

```mermaid
flowchart TB
    subgraph Sources["Document Sources"]
        Files[Files]
        URLs[URLs]
        Text[Text]
    end

    subgraph Processing["Processing Pipeline"]
        Converter[Converter]
        Chunker[Chunker]
        Embedder[Embedder]
    end

    subgraph Storage["Storage Layer"]
        LanceDB[(LanceDB)]
    end

    subgraph Agents["Agent Layer"]
        QA[QA Agent]
        Chat[Chat Agent]
        Research[Research Graph]
        RLM[RLM Agent]
    end

    subgraph Apps["Applications"]
        CLI[CLI]
        ChatTUI[Chat TUI]
        WebApp[Web App]
        Inspector[Inspector]
        MCP[MCP Server]
    end

    Sources --> Converter
    Converter --> Chunker
    Chunker --> Embedder
    Embedder --> LanceDB

    LanceDB --> Agents
    Agents --> Apps
```

## Core Components

### Storage Layer

LanceDB provides vector storage with full-text search capabilities:

- **DocumentRecord** - Document metadata and full content
- **ChunkRecord** - Text chunks with embeddings and structural metadata
- **SettingsRecord** - Database configuration and version info

Repositories handle CRUD operations:

- `DocumentRepository` - Create, read, update, delete documents
- `ChunkRepository` - Chunk management and hybrid search
- `SettingsRepository` - Configuration persistence

### Processing Pipeline

```mermaid
flowchart LR
    Source[Source] --> Converter
    Converter --> DoclingDoc[DoclingDocument]
    DoclingDoc --> Chunker
    Chunker --> Chunks[Chunks]
    Chunks --> Embedder
    Embedder --> Vectors[Vectors]
    Vectors --> DB[(LanceDB)]
```

**Converters** transform sources into DoclingDocuments:

- `docling-local` - Local Docling processing
- `docling-serve` - Remote processing via docling-serve

**Chunkers** split documents into semantic chunks:

- Preserves document structure (tables, lists, code blocks)
- Maintains provenance (page numbers, headings)
- Configurable chunk size

**Embedders** generate vector representations:

| Provider | Models |
|----------|--------|
| Ollama | nomic-embed-text, mxbai-embed-large |
| OpenAI | text-embedding-3-small, text-embedding-3-large |
| VoyageAI | voyage-3, voyage-code-3 |
| vLLM | Any compatible model |
| LM Studio | Any compatible model |

### Agent Layer

Four agent types for different use cases:

```mermaid
flowchart TB
    subgraph QA["QA Agent"]
        Q1[Question] --> S1[Search]
        S1 --> A1[Answer]
    end

    subgraph Chat["Chat Agent"]
        Q2[Question] --> Tools[Tool Selection]
        Tools --> S2[Search / Ask / Get]
        S2 --> A2[Answer]
        A2 --> History[Session History]
        History -.-> Q2
    end

    subgraph Research["Research Graph"]
        Q3[Question] --> Plan[Plan Next]
        Plan --> SearchOne[Search One]
        SearchOne --> Eval[Evaluate]
        Eval -->|Continue| Plan
        Eval -->|Done| Synthesize[Synthesize]
    end

    subgraph RLM["RLM Agent"]
        Q4[Question] --> Code[Write Code]
        Code --> Execute[Execute]
        Execute --> Examine[Examine Results]
        Examine -->|Iterate| Code
        Examine -->|Done| A4[Answer]
    end
```

**QA Agent** - Single-turn question answering:

- Searches for relevant chunks
- Expands context around results
- Generates answer with optional citations

**Chat Agent** - Multi-turn conversational RAG:

- Composed from reusable [toolsets](tools.md) (search, documents, QA, analysis)
- Maintains session history with prior answer recall
- Background summarization for context continuity
- Session-level document filtering

**Research Graph** - Iterative research workflow:

- Proposes one question at a time, evaluates the answer, then decides whether to continue
- Session context resolves ambiguous references
- Prior answers let the planner skip redundant searches
- Synthesizes structured report or conversational answer

**RLM Agent** - Complex analytical tasks via code execution:

- Writes Python code to explore the knowledge base
- Executes in sandboxed environment
- Handles aggregation, computation, multi-document analysis
- Iterates until answer is found

### Applications

| Application | Interface | Use Case |
|-------------|-----------|----------|
| CLI | Command line | Scripts, one-off queries, batch processing |
| Chat TUI | Terminal | Interactive conversations |
| Web App | Browser | Team collaboration, visual interface |
| Inspector | Terminal | Database exploration, debugging |
| MCP Server | Protocol | AI assistant integration |

## Data Flow

### Document Ingestion

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Converter
    participant Chunker
    participant Embedder
    participant DB as LanceDB

    User->>CLI: add-src document.pdf
    CLI->>Converter: Convert to DoclingDocument
    Converter-->>CLI: DoclingDocument
    CLI->>Chunker: Split into chunks
    Chunker-->>CLI: Chunks with metadata
    CLI->>Embedder: Generate embeddings
    Embedder-->>CLI: Vectors
    CLI->>DB: Store document + chunks
    DB-->>User: Document ID
```

### Search and QA

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Embedder
    participant DB as LanceDB
    participant LLM

    User->>Agent: Ask question
    Agent->>Embedder: Embed query
    Embedder-->>Agent: Query vector
    Agent->>DB: Hybrid search
    DB-->>Agent: Relevant chunks
    Agent->>Agent: Expand context
    Agent->>LLM: Generate answer
    LLM-->>Agent: Answer + citations
    Agent-->>User: Response
```

## Configuration

Configuration flows through the system:

```
CLI args → Environment variables → haiku.rag.yaml → Defaults
```

Key configuration areas:

- **Storage** - Database path, vacuum settings
- **Embeddings** - Provider, model, dimensions
- **Processing** - Chunk size, converter, chunker
- **Search** - Limits, context expansion
- **QA/Research** - Model, iterations, concurrency
- **Providers** - Ollama, vLLM, docling-serve URLs

See [Configuration](configuration/index.md) for details.
