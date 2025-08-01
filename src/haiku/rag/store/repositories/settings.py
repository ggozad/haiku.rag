import json
from typing import Any

from haiku.rag.store.engine import Store


class ConfigMismatchError(Exception):
    """Raised when current config doesn't match stored settings."""

    pass


class SettingsRepository:
    def __init__(self, store: Store):
        self.store = store

    def get(self) -> dict[str, Any]:
        """Get all settings from the database."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.execute("SELECT settings FROM settings LIMIT 1")
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return {}

    def save(self) -> None:
        """Sync settings from the current AppConfig to database."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        from haiku.rag.config import Config

        settings_json = Config.model_dump_json()

        self.store._connection.execute(
            "INSERT INTO settings (id, settings) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET settings = excluded.settings",
            (settings_json,),
        )

        self.store._connection.commit()

    def validate_config_compatibility(self) -> None:
        """Check if current config is compatible with stored settings.

        Raises ConfigMismatchError if there are incompatible differences.
        If no settings exist, saves current config.
        """
        db_settings = self.get()
        if not db_settings:
            # No settings in DB, save current config
            self.save()
            return

        from haiku.rag.config import Config

        current_config = Config.model_dump(mode="json")

        # Critical settings that must match
        critical_settings = [
            "EMBEDDINGS_PROVIDER",
            "EMBEDDINGS_MODEL",
            "EMBEDDINGS_VECTOR_DIM",
            "CHUNK_SIZE",
        ]

        errors = []
        for setting in critical_settings:
            if db_settings.get(setting) != current_config.get(setting):
                errors.append(
                    f"{setting}: current={current_config.get(setting)}, stored={db_settings.get(setting)}"
                )

        if errors:
            error_msg = f"Config mismatch detected: {'; '.join(errors)}. Consider rebuilding the database with the current configuration."
            raise ConfigMismatchError(error_msg)
