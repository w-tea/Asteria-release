from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from app.services.atomic_json import write_json_atomic


class ModelStorageSettings(BaseModel):
    root: str = ""


class ModelStorageSettingsRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> ModelStorageSettings:
        try:
            if not self.path.exists():
                return ModelStorageSettings()
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return ModelStorageSettings.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError):
            return ModelStorageSettings()

    def save(self, settings: ModelStorageSettings) -> ModelStorageSettings:
        self._ensure_existing_file_is_readable()
        write_json_atomic(self.path, settings.model_dump())
        return settings

    def _ensure_existing_file_is_readable(self) -> None:
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                ModelStorageSettings.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise OSError(f"model storage settings file could not be parsed: {exc}") from exc

    def resolve_root(self, fallback: Path) -> Path:
        root = self.load().root.strip()
        return Path(root) if root else fallback


def resolve_model_root(*, data_dir: Path, fallback: Path) -> Path:
    return ModelStorageSettingsRepository(data_dir / "model-storage-settings.json").resolve_root(fallback)
