import json

from haiku.rag.store.engine import SettingsRecord, Store


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
        self.store.settings_table.add([settings_record])
        return entity

    async def get_by_id(self, entity_id: str) -> dict | None:
        """Get settings by ID."""
        results = list(
            self.store.settings_table.search()
            .where(f"id = '{entity_id}'")
            .limit(1)
            .to_pydantic(SettingsRecord)
        )

        if not results:
            return None

        return json.loads(results[0].settings) if results[0].settings else {}

    async def update(self, entity: dict) -> dict:
        """Update existing settings."""
        self.store.settings_table.update(
            where="id = 'settings'", values={"settings": json.dumps(entity)}
        )
        return entity

    async def delete(self, entity_id: str) -> bool:
        """Delete settings by ID."""
        self.store.settings_table.delete(f"id = '{entity_id}'")
        return True

    async def list_all(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[dict]:
        """List all settings."""
        results = list(self.store.settings_table.search().to_pydantic(SettingsRecord))
        return [
            json.loads(record.settings) if record.settings else {} for record in results
        ]

    def get_current_settings(self) -> dict:
        """Get the current settings."""
        results = list(
            self.store.settings_table.search()
            .where("id = 'settings'")
            .limit(1)
            .to_pydantic(SettingsRecord)
        )

        if not results:
            return {}

        return json.loads(results[0].settings) if results[0].settings else {}

    def save_current_settings(self) -> None:
        """Save the current configuration to the database."""
        current_config = self.store._config.model_dump(mode="json")

        # Check if settings exist
        existing = list(
            self.store.settings_table.search()
            .where("id = 'settings'")
            .limit(1)
            .to_pydantic(SettingsRecord)
        )

        if existing:
            # Preserve existing version if present to avoid interfering with upgrade flow
            try:
                existing_settings = (
                    json.loads(existing[0].settings) if existing[0].settings else {}
                )
            except Exception:
                existing_settings = {}
            if "version" in existing_settings:
                current_config["version"] = existing_settings["version"]

            # Update existing settings
            if existing_settings != current_config:
                self.store.settings_table.update(
                    where="id = 'settings'",
                    values={"settings": json.dumps(current_config)},
                )
        else:
            # Create new settings
            settings_record = SettingsRecord(
                id="settings", settings=json.dumps(current_config)
            )
            self.store.settings_table.add([settings_record])

    def validate_config_compatibility(self) -> None:
        """Validate that the current configuration is compatible with stored settings."""
        stored_settings = self.get_current_settings()

        # If no stored settings, this is a new database - save current config and return
        if not stored_settings:
            self.save_current_settings()
            return

        current_config = self.store._config.model_dump(mode="json")

        # Check if embedding provider or model has changed
        # Support both old flat structure and new nested structure for backward compatibility
        stored_embeddings = stored_settings.get("embeddings", {})
        current_embeddings = current_config.get("embeddings", {})

        # Try nested structure first, fall back to flat for old databases
        stored_provider = stored_embeddings.get("provider") or stored_settings.get(
            "EMBEDDINGS_PROVIDER"
        )
        current_provider = current_embeddings.get("provider")

        stored_model = stored_embeddings.get("model") or stored_settings.get(
            "EMBEDDINGS_MODEL"
        )
        current_model = current_embeddings.get("model")

        stored_vector_dim = stored_embeddings.get("vector_dim") or stored_settings.get(
            "EMBEDDINGS_VECTOR_DIM"
        )
        current_vector_dim = current_embeddings.get("vector_dim")

        # Check for incompatible changes
        incompatible_changes = []

        if stored_provider and stored_provider != current_provider:
            incompatible_changes.append(
                f"Stored (db) embedding provider: '{stored_provider}' -> Environment (current) embedding provider: '{current_provider}'"
            )

        if stored_model and stored_model != current_model:
            incompatible_changes.append(
                f"Stored (db) embedding model '{stored_model}' -> Environment (current) embedding model '{current_model}'"
            )

        if stored_vector_dim and stored_vector_dim != current_vector_dim:
            incompatible_changes.append(
                f"Stored (db) embedding vector dimension {stored_vector_dim} -> Environment (current) embedding vector dimension {current_vector_dim}"
            )

        if incompatible_changes:
            error_msg = (
                "Database configuration is incompatible with current settings:\n"
                + "\n".join(f"  - {change}" for change in incompatible_changes)
            )
            error_msg += "\n\nPlease rebuild the database using: haiku-rag rebuild"
            raise ConfigMismatchError(error_msg)
