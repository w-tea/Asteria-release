from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


PROMPT_SYNTAX_GRAMMAR = "asteria-prompt-v1"
PromptSyntaxFeature = Literal[
    "token_weight",
    "scheduled_switch",
    "alternating_switch",
    "conditioning_and",
    "conditioning_break",
]
SUPPORTED_PROMPT_SYNTAX_FEATURES: tuple[PromptSyntaxFeature, ...] = (
    "token_weight",
    "scheduled_switch",
    "alternating_switch",
    "conditioning_and",
    "conditioning_break",
)


@dataclass(frozen=True)
class ModelSlot:
    id: str
    label: str
    type: str
    required: bool


@dataclass(frozen=True)
class ManagedModelBundle:
    id: str
    variant: str
    components: dict[str, str]


@dataclass(frozen=True)
class PromptSyntaxCapability:
    grammar: Literal["asteria-prompt-v1"] | None = None
    features: tuple[PromptSyntaxFeature, ...] = ()


@dataclass(frozen=True)
class ModelProfile:
    id: str
    label: str
    dependency_ids: list[str]
    model_slots: list[ModelSlot]
    setting_groups: list[str]
    managed_model_bundles: list[ManagedModelBundle] = field(default_factory=list)
    prompt_syntax: PromptSyntaxCapability = field(default_factory=PromptSyntaxCapability)


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
        managed_model_bundles = item.get("managed_model_bundles", [])
        if not isinstance(managed_model_bundles, list):
            managed_model_bundles = []
        return ModelProfile(
            id=item["id"],
            label=item["label"],
            dependency_ids=[value for value in dependency_ids if isinstance(value, str)],
            model_slots=[
                self._parse_slot(slot)
                for slot in model_slots
                if self._valid_slot(slot)
            ],
            managed_model_bundles=[
                self._parse_managed_bundle(bundle)
                for bundle in managed_model_bundles
                if self._valid_managed_bundle(bundle)
            ],
            setting_groups=[value for value in setting_groups if isinstance(value, str)],
            prompt_syntax=self._parse_prompt_syntax(item.get("prompt_syntax")),
        )

    @staticmethod
    def _parse_prompt_syntax(item: object) -> PromptSyntaxCapability:
        if not isinstance(item, dict) or item.get("grammar") != PROMPT_SYNTAX_GRAMMAR:
            return PromptSyntaxCapability()
        raw_features = item.get("features")
        if not isinstance(raw_features, list):
            return PromptSyntaxCapability()
        features = tuple(
            feature
            for feature in SUPPORTED_PROMPT_SYNTAX_FEATURES
            if feature in raw_features
        )
        return PromptSyntaxCapability(
            grammar=PROMPT_SYNTAX_GRAMMAR,
            features=features,
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

    @staticmethod
    def _valid_managed_bundle(item: object) -> bool:
        return (
            isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and isinstance(item.get("variant"), str)
            and isinstance(item.get("components"), dict)
            and bool(item["components"])
            and all(
                isinstance(slot_id, str) and isinstance(dependency_id, str)
                for slot_id, dependency_id in item["components"].items()
            )
        )

    @staticmethod
    def _parse_managed_bundle(item: dict) -> ManagedModelBundle:
        return ManagedModelBundle(
            id=item["id"],
            variant=item["variant"],
            components=dict(item["components"]),
        )
