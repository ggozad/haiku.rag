# Prompt customization

Two prompts are configurable: domain context for the RAG capability and the MCP server, and the prompt for picture descriptions.

## Configuration

```yaml
prompts:
  # Domain context prepended to capability instructions
  domain_preamble: |
    This knowledge base contains technical documentation for the Helios solar panel
    system, including installation manuals, maintenance procedures, and safety guidelines.
    Questions about "the system" or unqualified specs refer to the Helios panel.

  # VLM prompt for image description during conversion.
  # Omit the key to use the built-in prompt.
  picture_description: |
    Describe this figure in two sentences, naming any axis labels and units.
```

## Domain preamble

The `domain_preamble` field provides **domain context** prepended to the RAG capability instructions and appended to the MCP server's instructions. Use this to:

- Describe what the knowledge base contains
- Clarify domain-specific terminology
- Provide context that helps the model interpret ambiguous queries

Behavioral guidance belongs in the agent's own Pydantic AI instructions, not here.

## Picture description prompt

Customize the prompt used when generating VLM descriptions for embedded images during document conversion. This prompt is sent to the configured Vision Language Model for each image.

**Default prompt:**

```
Describe this image for a blind user. State the image type (screenshot, chart, photo, etc.),
what it depicts, any visible text, and key visual details. Be concise and accurate.
```

The prompt is used when `processing.pictures` is `"description"`, and by `haiku-rag rebuild --descriptions`. See [Picture Handling](processing.md#picture-handling) for full configuration.

## Programmatic configuration

```python
from haiku.rag.config import AppConfig
from haiku.rag.config.models import PromptsConfig

config = AppConfig(
    prompts=PromptsConfig(
        domain_preamble="This knowledge base contains Acme Corp product documentation and API references.",
        picture_description="Describe this image for search indexing.",
    )
)
```
