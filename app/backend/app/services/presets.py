from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError


class Preset(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    kind: str
    name: str
    data: dict


class PresetRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def list(self, kind: str | None = None) -> list[Preset]:
        items = self._read()
        return [item for item in items if kind is None or item.kind == kind]

    def save(self, preset: Preset) -> Preset:
        items = [item for item in self._read(strict=True) if item.id != preset.id]
        items.insert(0, preset)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"items": [item.model_dump() for item in items]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return preset

    def delete(self, preset_id: str) -> bool:
        items = self._read(strict=True)
        next_items = [item for item in items if item.id != preset_id]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"items": [item.model_dump() for item in next_items]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return len(next_items) != len(items)

    def _read(self, *, strict: bool = False) -> list[Preset]:
        try:
            if not self.path.exists():
                return []
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except OSError:
            if strict:
                raise
            return []
        except json.JSONDecodeError as exc:
            if strict:
                raise OSError(f"preset file is not valid JSON: {exc}") from exc
            return []
        if not isinstance(payload, dict):
            if strict:
                raise OSError("preset file root must be an object")
            return []
        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list):
            if strict:
                raise OSError("preset file items must be a list")
            return []
        items: list[Preset] = []
        for item in raw_items:
            if not isinstance(item, dict):
                if strict:
                    raise OSError("preset file contains an invalid item")
                continue
            try:
                items.append(Preset.model_validate(item))
            except ValidationError:
                if strict:
                    raise OSError("preset file contains an invalid item")
                continue
        return items
