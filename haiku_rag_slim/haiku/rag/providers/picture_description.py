"""Direct VLM client for picture description, used by ``rebuild --descriptions``.

The docling-serve converter normally drives picture description as a side-effect
of conversion. When we need to run the VLM against pictures already stored in
the DB (skipping the docling parse entirely), we drive the VLM through
pydantic-ai with ``BinaryContent`` parts so model construction goes through the
same ``get_model`` plumbing as every other agent in the codebase.
"""

import logging

from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent

from haiku.rag.config import AppConfig
from haiku.rag.utils import get_model

logger = logging.getLogger(__name__)


async def describe_pictures(
    image_bytes_by_ref: dict[str, bytes],
    *,
    config: AppConfig,
) -> dict[str, str]:
    """Describe pictures sequentially; returns ``{self_ref: text}``.

    Pictures whose VLM call fails or returns empty content are silently
    dropped from the returned map so the caller can decide whether the
    partial result is acceptable.
    """
    pic_desc = config.processing.conversion_options.picture_description
    model = get_model(pic_desc.model, config)
    prompt = config.prompts.picture_description
    agent: Agent[None, str] = Agent(
        model=model,
        output_type=str,
        instructions=prompt,
    )

    out: dict[str, str] = {}
    for ref, blob in image_bytes_by_ref.items():
        try:
            result = await agent.run([BinaryContent(data=blob, media_type="image/png")])
        except Exception as e:
            logger.warning("VLM call failed for %s: %s", ref, e)
            continue
        text = (result.output or "").strip()
        if text:
            out[ref] = text

    return out
