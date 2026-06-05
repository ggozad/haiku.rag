import json
import logging

from haiku.rag.store.engine import SettingsRecord, Store, query_to_pydantic

logger = logging.getLogger(__name__)


class ConfigMismatchError(Exception):
    """Raised when stored config doesn't match current config."""

    pass


class SettingsRepository:
    """Repository for Settings operations."""

    def __init__(self, store: Store) -> None:
        self.store = store

    async def create(self, entity: dict) -> dict:
        """Create settings in the database."""
        settings_record = SettingsRecord(id="settings", settings=json.dumps(entity))
        await self.store.settings_table.add([settings_record])
        return entity

    async def get_by_id(self, entity_id: str) -> dict | None:
        """Get settings by ID."""
        results = await query_to_pydantic(
            self.store.settings_table.query().where(f"id = '{entity_id}'").limit(1),
            SettingsRecord,
        )

        if not results:
            return None

        return json.loads(results[0].settings) if results[0].settings else {}

    async def update(self, entity: dict) -> dict:
        """Update existing settings."""
        await self.store.settings_table.update(
            {"settings": json.dumps(entity)}, where="id = 'settings'"
        )
        return entity

    async def delete(self, entity_id: str) -> bool:
        """Delete settings by ID."""
        await self.store.settings_table.delete(f"id = '{entity_id}'")
        return True

    async def list_all(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[dict]:
        """List all settings."""
        results = await query_to_pydantic(
            self.store.settings_table.query(), SettingsRecord
        )
        return [
            json.loads(record.settings) if record.settings else {} for record in results
        ]

    async def get_current_settings(self) -> dict:
        """Get the current settings."""
        results = await query_to_pydantic(
            self.store.settings_table.query().where("id = 'settings'").limit(1),
            SettingsRecord,
        )

        if not results:
            return {}

        return json.loads(results[0].settings) if results[0].settings else {}

    async def save_current_settings(self) -> None:
        """Save the current configuration to the database."""
        self.store._assert_writable()
        current_config = self.store._config.model_dump(mode="json")

        # Check if settings exist
        existing = await query_to_pydantic(
            self.store.settings_table.query().where("id = 'settings'").limit(1),
            SettingsRecord,
        )

        if existing:
            # Preserve existing version if present to avoid interfering with upgrade flow
            existing_settings = json.loads(existing[0].settings)
            if "version" in existing_settings:
                current_config["version"] = existing_settings["version"]

            # Update existing settings
            if existing_settings != current_config:
                await self.store.settings_table.update(
                    {"settings": json.dumps(current_config)},
                    where="id = 'settings'",
                )
        else:
            # Create new settings
            settings_record = SettingsRecord(
                id="settings", settings=json.dumps(current_config)
            )
            await self.store.settings_table.add([settings_record])

    async def validate_config_compatibility(self) -> None:
        """Validate the current configuration against stored settings without writing.

        Opening a database never modifies it. ``vector_dim`` mismatches raise —
        corpus and query vectors must live in the same dimensional space.
        ``provider`` and ``name`` drift (with matching ``vector_dim``) is soft:
        legitimate when the same model is served by a different stack (Ollama vs
        vLLM-via-openai, etc.). Drift is surfaced via a warning; a writable open
        then raises so a write cannot mix embedding identities in the corpus,
        while a read-only open continues. Stored settings are reconciled
        explicitly via ``haiku-rag rebuild --set-embedder``, never on open.
        """
        stored_settings = await self.get_current_settings()

        # Nothing stored to validate against — never write on open.
        if not stored_settings:
            return

        current_config = self.store._config.model_dump(mode="json")

        # Both stored and current use nested structure: embeddings.model.{provider,name,vector_dim}
        stored_model_obj = stored_settings.get("embeddings", {}).get("model", {})
        current_model_obj = current_config.get("embeddings", {}).get("model", {})

        stored_provider = stored_model_obj.get("provider")
        current_provider = current_model_obj.get("provider")

        stored_model = stored_model_obj.get("name")
        current_model = current_model_obj.get("name")

        stored_vector_dim = stored_model_obj.get("vector_dim")
        current_vector_dim = current_model_obj.get("vector_dim")

        if (
            stored_vector_dim
            and current_vector_dim
            and stored_vector_dim != current_vector_dim
        ):
            raise ConfigMismatchError(
                "Database configuration is incompatible with current settings:\n"
                f"  - Stored (db) embedding vector dimension {stored_vector_dim} -> "
                f"Environment (current) embedding vector dimension {current_vector_dim}\n"
                "\nPlease rebuild the database using: haiku-rag rebuild"
            )

        soft_changes: list[str] = []
        if stored_provider and stored_provider != current_provider:
            soft_changes.append(
                f"provider: '{stored_provider}' -> '{current_provider}'"
            )
        if stored_model and stored_model != current_model:
            soft_changes.append(f"model: '{stored_model}' -> '{current_model}'")

        if soft_changes:
            logger.warning(
                "Embedding identity changed (vector_dim matches): %s. If this is "
                "intentional, run 'haiku-rag rebuild --set-embedder' to update the "
                "stored settings; otherwise revert your config to match the database.",
                "; ".join(soft_changes),
            )
            if not self.store.is_read_only:
                raise ConfigMismatchError(
                    "Database embedding identity differs from current config "
                    f"(vector_dim matches): {'; '.join(soft_changes)}. "
                    "Run 'haiku-rag rebuild --set-embedder' to adopt the current "
                    "embedder, or revert your config to match the database."
                )
