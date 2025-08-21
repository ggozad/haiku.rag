import json

from haiku.rag.config import Config
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
        current_config = Config.model_dump(mode="json")

        # Check if settings exist
        existing = list(
            self.store.settings_table.search()
            .where("id = 'settings'")
            .limit(1)
            .to_pydantic(SettingsRecord)
        )

        if existing:
            # Update existing settings
            self.store.settings_table.update(
                where="id = 'settings'", values={"settings": json.dumps(current_config)}
            )
        else:
            # Create new settings
            settings_record = SettingsRecord(
                id="settings", settings=json.dumps(current_config)
            )
            self.store.settings_table.add([settings_record])

    def validate_config_compatibility(self) -> None:
        """Validate that the current configuration is compatible with stored settings."""
        try:
            stored_settings = self.get_current_settings()
            current_config = Config.model_dump(mode="json")

            # Check if embedding provider or model has changed
            stored_provider = stored_settings.get("embedding_provider")
            current_provider = current_config.get("embedding_provider")

            stored_model = stored_settings.get("embedding_model")
            current_model = current_config.get("embedding_model")

            if (stored_provider and stored_provider != current_provider) or (
                stored_model and stored_model != current_model
            ):
                # Provider or model changed - need to recreate embeddings
                from rich.console import Console

                console = Console()
                console.print(
                    "[yellow]Warning: Embedding provider/model changed. "
                    "You may need to recreate embeddings for optimal performance.[/yellow]"
                )

                # Optionally recreate embeddings table
                # self.store.recreate_embeddings_table()

        except Exception:
            # If we can't validate, just continue
            pass
