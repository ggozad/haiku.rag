import json
import logging

from haiku.rag.store.engine import SettingsRecord, Store
from haiku.rag.store.upgrades import Upgrade

logger = logging.getLogger(__name__)


def _apply_embeddings_model_config(store: Store) -> None:  # pragma: no cover
    """Migrate embeddings config from flat to nested EmbeddingModelConfig structure."""
    results = list(
        store.settings_table.search()
        .where("id = 'settings'")
        .limit(1)
        .to_pydantic(SettingsRecord)
    )

    if not results or not results[0].settings:
        return

    settings = json.loads(results[0].settings)
    embeddings = settings.get("embeddings", {})

    # Check if already migrated (model is a dict with nested structure)
    if isinstance(embeddings.get("model"), dict):
        return

    # Migrate from flat structure to nested EmbeddingModelConfig
    old_provider = embeddings.get("provider", "ollama")
    old_model = embeddings.get("model", "qwen3-embedding:4b")
    old_vector_dim = embeddings.get("vector_dim", 2560)

    logger.info(
        "Migrating embeddings config to new nested structure: "
        "embeddings.{provider,model,vector_dim} -> embeddings.model.{provider,name,vector_dim}"
    )

    # Create new nested structure
    settings["embeddings"] = {
        "model": {
            "provider": old_provider,
            "name": old_model,
            "vector_dim": old_vector_dim,
        }
    }

    store.settings_table.update(
        where="id = 'settings'",
        values={"settings": json.dumps(settings)},
    )

    logger.info(
        "Embeddings config migrated: provider=%s, name=%s, vector_dim=%d",
        old_provider,
        old_model,
        old_vector_dim,
    )


upgrade_embeddings_model_config = Upgrade(
    version="0.19.6",
    apply=_apply_embeddings_model_config,
    description="Migrate embeddings config to nested EmbeddingModelConfig structure",
)
