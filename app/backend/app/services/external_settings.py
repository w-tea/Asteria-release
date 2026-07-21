from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, ValidationError, field_validator


DEFAULT_EAGLE_BASE_URL = "http://localhost:41595"


def normalize_eagle_base_url(value: str, *, allow_empty: bool = True) -> str:
    candidate = value.strip()
    if not candidate:
        if allow_empty:
            return ""
        raise ValueError("Eagle base URL is required")
    if len(candidate) > 2048 or any(character.isspace() or ord(character) < 32 for character in candidate):
        raise ValueError("Eagle base URL contains invalid characters")
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Eagle base URL is invalid") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Eagle base URL must use HTTP or HTTPS and include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Eagle base URL must not include credentials")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("Eagle base URL must not include a path, query, or fragment")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("Eagle base URL port is invalid")
    if parsed.netloc.endswith(":"):
        raise ValueError("Eagle base URL port is invalid")
    return candidate.rstrip("/")


class TopazIntegrationSettings(BaseModel):
    executable_path: str = ""


class EagleIntegrationSettings(BaseModel):
    base_url: str = Field(default="", max_length=2048)
    default_folder_id: str = Field(default="", max_length=256)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        return normalize_eagle_base_url(value)

    @field_validator("default_folder_id")
    @classmethod
    def validate_default_folder_id(cls, value: str) -> str:
        candidate = value.strip()
        if any(ord(character) < 32 for character in candidate):
            raise ValueError("Eagle folder id contains invalid characters")
        return candidate


class ExternalIntegrationSettings(BaseModel):
    topaz: TopazIntegrationSettings = Field(default_factory=TopazIntegrationSettings)
    eagle: EagleIntegrationSettings = Field(default_factory=EagleIntegrationSettings)


class ExternalIntegrationSettingsRepository:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> ExternalIntegrationSettings:
        try:
            if not self.path.exists():
                return ExternalIntegrationSettings()
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return ExternalIntegrationSettings()
            try:
                topaz = TopazIntegrationSettings.model_validate(payload.get("topaz", {}))
            except ValidationError:
                topaz = TopazIntegrationSettings()
            try:
                eagle = EagleIntegrationSettings.model_validate(payload.get("eagle", {}))
            except ValidationError:
                eagle = EagleIntegrationSettings()
            return ExternalIntegrationSettings(topaz=topaz, eagle=eagle)
        except (OSError, json.JSONDecodeError):
            return ExternalIntegrationSettings()

    def save(self, settings: ExternalIntegrationSettings) -> ExternalIntegrationSettings:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_existing_file_is_readable()
        self.path.write_text(
            json.dumps(settings.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return settings

    def _ensure_existing_file_is_readable(self) -> None:
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                ExternalIntegrationSettings.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise OSError(f"external integration settings file could not be parsed: {exc}") from exc

    def resolve_topaz_executable(self, fallback: Path) -> Path:
        configured = self.load().topaz.executable_path.strip()
        return Path(configured) if configured else fallback

    def resolve_eagle_base_url(self, fallback: str) -> str:
        configured = self.load().eagle.base_url
        if configured:
            return configured
        try:
            return normalize_eagle_base_url(fallback, allow_empty=False)
        except ValueError:
            return DEFAULT_EAGLE_BASE_URL
