from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from app.services.atomic_json import write_json_atomic


class Preset(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    kind: str
    name: str
    data: dict


class PresetKindMismatchError(ValueError):
    pass


class PresetRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def list(self, kind: str | None = None) -> list[Preset]:
        items = self._read()
        return [item for item in items if kind is None or item.kind == kind]

    def save(self, preset: Preset) -> Preset:
        existing = self._read(strict=True)
        if any(item.id == preset.id and item.kind != preset.kind for item in existing):
            raise PresetKindMismatchError("preset ID belongs to another kind")
        items = [item for item in existing if item.id != preset.id]
        items.insert(0, preset)
        write_json_atomic(self.path, {"items": [item.model_dump() for item in items]})
        return preset

    def delete(self, preset_id: str, *, kind: str | None = None) -> bool:
        items = self._read(strict=True)
        next_items = [item for item in items if item.id != preset_id or (kind is not None and item.kind != kind)]
        if kind is not None and len(next_items) == len(items):
            return False
        write_json_atomic(self.path, {"items": [item.model_dump() for item in next_items]})
        return len(next_items) != len(items)

    def migrate_model_references(self, candidates: list[object]) -> int:
        from app.services.model_reference_migration import migrate_generation_reference_metadata

        items = self._read(strict=True)
        changed = 0
        migrated_items: list[Preset] = []
        for item in items:
            if item.kind not in {"generation", "model_profile"}:
                migrated_items.append(item)
                continue
            migrated_data = migrate_generation_reference_metadata(item.data, candidates)
            if migrated_data != item.data:
                item = item.model_copy(update={"data": migrated_data})
                changed += 1
            migrated_items.append(item)
        if changed:
            write_json_atomic(self.path, {"items": [item.model_dump() for item in migrated_items]})
        return changed

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
