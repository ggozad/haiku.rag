## Agents

Two agentic flows are provided by haiku.rag:

- Simple QA Agent — a focused question answering agent
- Research Multi‑Agent — a multi‑step, analyzable research workflow


### Simple QA Agent

The simple QA agent answers a single question using the knowledge base. It retrieves relevant chunks, optionally expands context around them, and asks the model to answer strictly based on that context.

Key points:

- Uses a single `search_documents` tool to fetch relevant chunks
- Can be run with or without inline citations in the prompt
- Returns a plain string answer

Python usage:

```python
from haiku.rag.client import HaikuRAG
from haiku.rag.qa.agent import QuestionAnswerAgent

client = HaikuRAG(path_to_db)

# Choose a provider and model (see Configuration for env defaults)
agent = QuestionAnswerAgent(
    client=client,
    provider="openai",  # or "ollama", "vllm", etc.
    model="gpt-4o-mini",
    use_citations=False,  # set True to bias prompt towards citing sources
)

answer = await agent.answer("What is climate change?")
print(answer)
```

### Research Multi‑Agent

The research workflow coordinates specialized agents to plan, search, analyze, and synthesize a comprehensive answer. It is designed for deeper questions that benefit from iterative investigation and structured reporting.

Components:

- Orchestrator: Plans, coordinates, and loops until confidence is sufficient
- Search Specialist: Performs targeted RAG searches and answers sub‑questions
- Analysis & Evaluation: Extracts insights, identifies gaps, proposes new questions
- Synthesis: Produces a final structured research report

Primary models:

- `ResearchPlan` — produced by the orchestrator when planning
  - `main_question: str`
  - `sub_questions: list[str]` (standalone, self‑contained queries)
- `SearchAnswer` — produced by the search specialist for each sub‑question
  - `query: str` — the executed sub‑question
  - `answer: str` — the agent’s answer grounded in retrieved context
  - `context: list[str]` — minimal verbatim snippets used for the answer
  - `sources: list[str]` — document URIs aligned with `context`
- `EvaluationResult` — insights, new standalone questions, sufficiency & confidence
- `ResearchReport` — the final synthesized report


Python usage:

```python
from haiku.rag.client import HaikuRAG
from haiku.rag.research import ResearchOrchestrator

client = HaikuRAG(path_to_db)
orchestrator = ResearchOrchestrator(provider="openai", model="gpt-4o-mini")

report = await orchestrator.conduct_research(
    question="What are the main drivers and recent trends of global temperature anomalies since 1990?",
    client=client,
    max_iterations=2,
    confidence_threshold=0.8,
    verbose=False,
)

print(report.title)
print(report.executive_summary)
```
