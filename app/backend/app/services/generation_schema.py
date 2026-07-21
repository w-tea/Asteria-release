from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class GenerationSchemaNotFound(Exception):
    pass


@dataclass(frozen=True)
class GenerationSchema:
    profile: str
    id: str
    name: str
    mode: str
    model_family: str
    fields: dict[str, dict[str, Any]]

    @property
    def defaults(self) -> dict[str, Any]:
        return {
            field_id: spec["default"]
            for field_id, spec in self.fields.items()
            if "default" in spec
        }


class GenerationSchemaRepository:
    def __init__(self, schema_dir: Path) -> None:
        self.schema_dir = schema_dir

    def get_txt2img(self, profile: str) -> GenerationSchema:
        schema_profile = "flux" if profile == "flux1" else profile
        path = self.schema_dir / f"{schema_profile}_txt2img.yaml"
        try:
            exists = path.exists()
        except OSError:
            raise GenerationSchemaNotFound(profile) from None
        if not exists:
            raise GenerationSchemaNotFound(profile)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            raise GenerationSchemaNotFound(profile) from None
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            raise GenerationSchemaNotFound(profile)
        schema_id = raw["id"]
        name = raw.get("name")
        if not isinstance(name, str):
            name = schema_id
        mode = raw.get("mode")
        if not isinstance(mode, str):
            mode = "txt2img"
        model_family = raw.get("model_family")
        if not isinstance(model_family, str):
            model_family = profile
        fields = raw.get("fields", {})
        if not isinstance(fields, dict):
            fields = {}
        fields = {
            field_id: spec
            for field_id, spec in fields.items()
            if isinstance(field_id, str) and isinstance(spec, dict)
        }
        return GenerationSchema(
            profile=profile,
            id=schema_id,
            name=name,
            mode=mode,
            model_family="flux1" if profile == "flux1" else model_family,
            fields=fields,
        )
