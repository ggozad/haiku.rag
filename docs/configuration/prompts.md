# Prompt Customization

Customize the prompts used by haiku.rag's AI agents to better match your domain and use case.

## Configuration

```yaml
prompts:
  # Prepended to all agent prompts
  domain_preamble: |
    You are answering questions about our internal documentation.
    Technical terms like "time travel" refer to database versioning features.

  # Full replacement for QA agent prompt (optional)
  qa: null

  # Full replacement for research synthesis prompt (optional)
  synthesis: null

  # VLM prompt for image description during conversion (optional)
  picture_description: null  # Uses default prompt
```

## Domain Preamble

The `domain_preamble` field is prepended to **all** agent prompts (QA, research planning, search, evaluation, and synthesis). Use this to:

- Add domain context that clarifies terminology
- Set the tone or personality of responses
- Specify what the knowledge base contains

**Example:**

```yaml
prompts:
  domain_preamble: |
    You are a technical support assistant for Acme Corp products.
    The knowledge base contains product documentation, FAQs, and troubleshooting guides.
    Always be helpful and professional.
```

## Custom QA Prompt

Replace the default QA agent prompt entirely by setting `prompts.qa`. The prompt should instruct the agent how to:

1. Use the `search_documents` tool to find relevant content
2. Interpret search results with scores and metadata
3. Cite sources using chunk IDs
4. Handle insufficient information

**Example:**

```yaml
prompts:
  qa: |
    You are a concise technical assistant. Answer questions using only the knowledge base.

    Process:
    1. Search for relevant documents using the search_documents tool
    2. Review results and their relevance scores
    3. Provide a brief, direct answer based on retrieved content

    Guidelines:
    - Use only information from search results
    - Include chunk IDs in cited_chunks for sources you use
    - If information is insufficient, say so clearly
    - Be concise - avoid unnecessary elaboration
```

## Custom Synthesis Prompt

Replace the research report synthesis prompt by setting `prompts.synthesis`. This controls how the multi-agent research workflow generates its final report.

The prompt should produce a `ResearchReport` with: `title`, `executive_summary`, `main_findings`, `conclusions`, `recommendations`, `limitations`, and `sources_summary`.

**Example:**

```yaml
prompts:
  synthesis: |
    Generate a research report based on the gathered evidence.

    Output format:
    - title: 5-12 word title
    - executive_summary: 3-5 sentence overview
    - main_findings: 4-8 bullet points of key findings
    - conclusions: 2-4 bullet points
    - recommendations: 2-5 actionable recommendations
    - limitations: 1-3 limitations or gaps
    - sources_summary: Brief description of sources used

    Guidelines:
    - Base all content strictly on collected evidence
    - Be specific and objective
    - Avoid meta-commentary like "This report covers..."
```

## Picture Description Prompt

Customize the prompt used when generating VLM descriptions for embedded images during document conversion. This prompt is sent to the configured Vision Language Model for each image.

**Default prompt:**

```
Describe this image for a blind user. State the image type (screenshot, chart, photo, etc.),
what it depicts, any visible text, and key visual details. Be concise and accurate.
```

**Custom example:**

```yaml
prompts:
  picture_description: |
    Describe this image for a document search system.
    Focus on: image type, main content, any text, key visual elements.
    Be concise and factual.
```

The prompt is used when `processing.conversion_options.picture_description.enabled` is `true`. See [Picture Description (VLM)](processing.md#picture-description-vlm) for full configuration.

## Programmatic Configuration

```python
from haiku.rag.config import AppConfig
from haiku.rag.config.models import PromptsConfig

config = AppConfig(
    prompts=PromptsConfig(
        domain_preamble="You are answering questions about our product documentation.",
        qa=None,  # Use default QA prompt
        synthesis=None,  # Use default synthesis prompt
        picture_description="Describe this image for search indexing.",
    )
)
```
