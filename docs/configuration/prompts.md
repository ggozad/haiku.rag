# Prompt Customization

Customize the prompts used by haiku.rag's skills and research workflow to better match your domain and use case.

## Configuration

```yaml
prompts:
  # Domain context prepended to skill instructions and research prompts
  domain_preamble: |
    This knowledge base contains technical documentation for the Helios solar panel
    system, including installation manuals, maintenance procedures, and safety guidelines.
    Questions about "the system" or unqualified specs refer to the Helios panel.

  # Full replacement for research synthesis prompt (optional)
  synthesis: null

  # VLM prompt for image description during conversion (optional)
  picture_description: null  # Uses default prompt
```

## Domain Preamble

The `domain_preamble` field provides **domain context** prepended to the rag and rag-analysis skill instructions and to the research planner/search/synthesis prompts. Use this to:

- Describe what the knowledge base contains
- Clarify domain-specific terminology
- Provide context that helps the model interpret ambiguous queries

**Important:** `domain_preamble` is for domain context, not behavioral instructions. Descriptions of subject matter, terminology, and content scope belong here. Behavioral guidance (tone, response style, formatting rules) lives in the skill's SKILL.md — fork the skill via `haiku-rag create-skill` to customize behavior.

**Example:**

```yaml
prompts:
  domain_preamble: |
    This knowledge base contains product documentation, API references,
    and troubleshooting guides for Acme Corp's cloud platform.
    "Deployment" refers to Acme's managed deployment service, not general CI/CD.
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

The prompt is used when `processing.pictures` is `"description"`. See [Picture Handling](processing.md#picture-handling) for full configuration.

## Programmatic Configuration

```python
from haiku.rag.config import AppConfig
from haiku.rag.config.models import PromptsConfig

config = AppConfig(
    prompts=PromptsConfig(
        domain_preamble="This knowledge base contains Acme Corp product documentation and API references.",
        synthesis=None,  # Use default synthesis prompt
        picture_description="Describe this image for search indexing.",
    )
)
```
