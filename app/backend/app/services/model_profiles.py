from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelSlot:
    id: str
    label: str
    type: str
    required: bool


@dataclass(frozen=True)
class ModelProfile:
    id: str
    label: str
    dependency_ids: list[str]
    model_slots: list[ModelSlot]
    setting_groups: list[str]


class ModelProfileRepository:
    def __init__(self, profiles_path: Path) -> None:
        self.profiles_path = profiles_path

    def list(self) -> list[ModelProfile]:
        try:
            profiles_exists = self.profiles_path.exists()
        except OSError:
            profiles_exists = False
        if not profiles_exists:
            return []

        try:
            raw = yaml.safe_load(self.profiles_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return []
        if not isinstance(raw, dict):
            return []
        profiles = raw.get("profiles", [])
        if not isinstance(profiles, list):
            return []
        return [
            self._parse_profile(item)
            for item in profiles
            if isinstance(item, dict) and isinstance(item.get("id"), str) and isinstance(item.get("label"), str)
        ]

    def _parse_profile(self, item: dict) -> ModelProfile:
        model_slots = item.get("model_slots", [])
        if not isinstance(model_slots, list):
            model_slots = []
        dependency_ids = item.get("dependency_ids", [])
        if not isinstance(dependency_ids, list):
            dependency_ids = []
        setting_groups = item.get("setting_groups", [])
        if not isinstance(setting_groups, list):
            setting_groups = []
        return ModelProfile(
            id=item["id"],
            label=item["label"],
            dependency_ids=[value for value in dependency_ids if isinstance(value, str)],
            model_slots=[
                self._parse_slot(slot)
                for slot in model_slots
                if self._valid_slot(slot)
            ],
            setting_groups=[value for value in setting_groups if isinstance(value, str)],
        )

    def _valid_slot(self, item: object) -> bool:
        return (
            isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and isinstance(item.get("label"), str)
            and isinstance(item.get("type"), str)
            and ("required" not in item or isinstance(item.get("required"), bool))
        )

    def _parse_slot(self, item: dict) -> ModelSlot:
        return ModelSlot(
            id=item["id"],
            label=item["label"],
            type=item["type"],
            required=item.get("required", True),
        )
