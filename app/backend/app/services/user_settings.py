from __future__ import annotations

import json
import os
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.services.atomic_json import write_json_atomic
from app.services.schema_versions import CURRENT_SCHEMA_VERSIONS


class SettingsSchemaTooNewError(RuntimeError):
    pass


class OutputSettings(BaseModel):
    root: str = ""
    filename_template: str = "{profile}_{seed}_{id}"
    organize_by_date: bool = True
    image_format: Literal["png", "jpg", "webp"] = "png"


class StorageSettings(BaseModel):
    merge_preset_dir: str = ""


class AIProviderSettings(BaseModel):
    id: str
    name: str = ""
    provider_mode: Literal["openai_compatible", "openai_responses", "anthropic", "gemini"] = "openai_compatible"
    endpoint: str = ""
    model: str = ""
    api_key_env_var: str = ""
    timeout_seconds: int = Field(default=60, ge=1, le=300)
    enabled: bool = True

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        provider_id = value.strip()
        if not provider_id:
            raise ValueError("AI provider id cannot be empty.")
        if any(char not in "-_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789." for char in provider_id):
            raise ValueError("AI provider id can only contain letters, numbers, dashes, underscores, and dots.")
        return provider_id

    @field_validator("name", "endpoint", "model", "api_key_env_var", mode="before")
    @classmethod
    def _coerce_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("api_key_env_var")
    @classmethod
    def _validate_api_key_env_var(cls, value: str) -> str:
        if value and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("API key environment variable must be a valid environment variable name.")
        return value


class AISettings(BaseModel):
    default_caption_provider_id: str = ""
    providers: list[AIProviderSettings] = Field(default_factory=list)

    @field_validator("default_caption_provider_id", mode="before")
    @classmethod
    def _coerce_default_provider_id(cls, value: object) -> str:
        return str(value or "").strip()

    @model_validator(mode="after")
    def _validate_provider_ids(self) -> "AISettings":
        seen: set[str] = set()
        for provider in self.providers:
            if provider.id in seen:
                raise ValueError(f"Duplicate AI provider id: {provider.id}")
            seen.add(provider.id)
        if self.default_caption_provider_id and self.default_caption_provider_id not in seen:
            raise ValueError("Default caption provider id must match a configured AI provider.")
        return self


class LibrarySavedFilter(BaseModel):
    id: str
    label: str
    query: str = ""
    profile: str = ""
    tag: str = ""
    tagSource: Literal["creator", "prompt", "all"] = "creator"
    favoriteOnly: bool = False
    showInternalTags: bool = False

    @field_validator("id", "label", "query", "profile", "tag", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> str:
        return " ".join(str(value or "").strip().split())

    @model_validator(mode="after")
    def _require_identity(self) -> "LibrarySavedFilter":
        if not self.id or not self.label:
            raise ValueError("Library saved filter id and label are required.")
        return self


class LibrarySettings(BaseModel):
    saved_filters: list[LibrarySavedFilter] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def _require_unique_ids(self) -> "LibrarySettings":
        ids = [item.id.casefold() for item in self.saved_filters]
        if len(ids) != len(set(ids)):
            raise ValueError("Library saved filter ids must be unique.")
        return self


class AppearanceSettings(BaseModel):
    theme: Literal["light", "dark", "system"] = "system"

    @field_validator("theme", mode="before")
    @classmethod
    def _normalize_theme(cls, value: object) -> str:
        normalized = str(value or "").strip().lower()
        return normalized if normalized in {"light", "dark", "system"} else "system"


class LauncherSettings(BaseModel):
    runtime_monitor_on_start: bool = False
    disable_pinned_memory: bool = False


class UserSettings(BaseModel):
    schema_version: Literal[3] = CURRENT_SCHEMA_VERSIONS.settings
    output: OutputSettings = Field(default_factory=OutputSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    ai: AISettings = Field(default_factory=AISettings)
    library: LibrarySettings = Field(default_factory=LibrarySettings)
    appearance: AppearanceSettings = Field(default_factory=AppearanceSettings)
    launcher: LauncherSettings = Field(default_factory=LauncherSettings)


@dataclass(frozen=True)
class UserSettingsLoadResult:
    settings: UserSettings
    reason_id: Literal["json_invalid", "access_denied", "field_type_invalid"] | None = None


class UserSettingsRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> UserSettings:
        return self.load_with_diagnostics().settings

    def load_with_diagnostics(self) -> UserSettingsLoadResult:
        try:
            if not self.path.exists():
                return UserSettingsLoadResult(UserSettings())
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                isinstance(payload, dict)
                and isinstance(payload.get("schema_version"), int)
                and payload["schema_version"] > CURRENT_SCHEMA_VERSIONS.settings
            ):
                raise SettingsSchemaTooNewError("user settings schema is newer than this application")
            payload, repaired = self._repair_disable_pinned_memory(payload)
            return UserSettingsLoadResult(
                UserSettings.model_validate(payload),
                "field_type_invalid" if repaired else None,
            )
        except SettingsSchemaTooNewError:
            raise
        except json.JSONDecodeError:
            return UserSettingsLoadResult(UserSettings(), "json_invalid")
        except OSError:
            return UserSettingsLoadResult(UserSettings(), "access_denied")
        except ValidationError:
            return UserSettingsLoadResult(UserSettings(), "json_invalid")

    def save(self, settings: UserSettings) -> UserSettings:
        self._ensure_existing_file_is_readable()
        write_json_atomic(self.path, settings.model_dump())
        return settings

    def _ensure_existing_file_is_readable(self) -> None:
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                payload, _ = self._repair_disable_pinned_memory(payload)
                UserSettings.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise OSError(f"user settings file could not be parsed: {exc}") from exc

    @staticmethod
    def _repair_disable_pinned_memory(payload: object) -> tuple[object, bool]:
        if not isinstance(payload, dict):
            return payload, False
        launcher = payload.get("launcher")
        if not isinstance(launcher, dict) or "disable_pinned_memory" not in launcher:
            return payload, False
        if isinstance(launcher["disable_pinned_memory"], bool):
            return payload, False
        repaired_payload = dict(payload)
        repaired_launcher = dict(launcher)
        repaired_launcher["disable_pinned_memory"] = False
        repaired_payload["launcher"] = repaired_launcher
        return repaired_payload, True


@dataclass(frozen=True)
class TemplateValidation:
    valid: bool
    code: str
    message: str


FILENAME_TEMPLATE_FIELDS = {"profile", "seed", "id", "index", "sampler", "scheduler"}


def validate_filename_template(template: str) -> TemplateValidation:
    value = template.strip()
    if not value:
        return TemplateValidation(False, "filename_template_empty", "Filename template cannot be empty.")
    try:
        parsed = list(string.Formatter().parse(value))
    except ValueError as exc:
        return TemplateValidation(False, "filename_template_invalid", f"Filename template is invalid: {exc}")
    unknown_fields = {
        field_name.split(".", 1)[0].split("[", 1)[0]
        for _, field_name, _, _ in parsed
        if field_name and field_name.split(".", 1)[0].split("[", 1)[0] not in FILENAME_TEMPLATE_FIELDS
    }
    if unknown_fields:
        allowed = ", ".join(sorted(FILENAME_TEMPLATE_FIELDS))
        unknown = ", ".join(sorted(unknown_fields))
        return TemplateValidation(False, "filename_template_unsupported_field", f"Unsupported filename template field: {unknown}. Allowed fields: {allowed}.")
    try:
        value.format(profile="sdxl", seed=1, id=1, index=1, sampler="Euler a", scheduler="Automatic")
    except (IndexError, KeyError, ValueError) as exc:
        return TemplateValidation(False, "filename_template_invalid", f"Filename template is invalid: {exc}")
    return TemplateValidation(True, "filename_template_valid", "Filename template is valid.")


def safe_segment(value: str, fallback: str = "default") -> str:
    segment = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return segment.strip("._") or fallback


def resolve_output_root(data_dir: Path, settings: OutputSettings) -> Path:
    raw_root = settings.root.strip()
    if not raw_root:
        return data_dir / "outputs"
    path = Path(raw_root).expanduser()
    if not path.is_absolute():
        path = data_dir / path
    return path


def resolve_current_output_root(data_dir: Path) -> Path:
    settings = UserSettingsRepository(data_dir / "user-settings.json").load().output
    return resolve_output_root(data_dir, settings)


def resolve_merge_preset_dir(data_dir: Path, settings: StorageSettings, default_dir: Path) -> Path:
    raw_root = settings.merge_preset_dir.strip()
    if not raw_root:
        return default_dir
    path = Path(raw_root).expanduser()
    if not path.is_absolute():
        path = data_dir / path
    return path


def directory_writable_status(path: Path, *, ensure: bool = False) -> tuple[bool, bool, str]:
    try:
        if ensure:
            path.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        if not exists:
            return False, False, "Directory does not exist yet."
        if not path.is_dir():
            return True, False, "Path exists but is not a directory."
        probe = path / f".asteria-write-test-{os.getpid()}.tmp"
        probe.write_text("ok", encoding="utf-8")
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        return True, True, "Directory is writable."
    except OSError as exc:
        return _path_exists_safely(path), False, f"Directory is not writable: {exc}"


def _path_exists_safely(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False
