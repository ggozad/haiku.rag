import yaml
from fastapi import APIRouter, Depends

from haiku.rag.config import redact_secrets
from haiku.rag.ingester.api.schemas import ConfigResponse
from haiku.rag.ingester.api.server import APIState, get_state

router = APIRouter(tags=["config"])


@router.get("/config", response_model=ConfigResponse)
async def config(state: APIState = Depends(get_state)) -> ConfigResponse:
    """The full effective configuration (defaults filled in, not the on-disk
    file) as YAML, with secret-bearing values redacted."""
    data = redact_secrets(state.config.model_dump(mode="json", exclude_none=False))
    text = yaml.dump(data, sort_keys=False, default_flow_style=False)
    return ConfigResponse(yaml=text)
