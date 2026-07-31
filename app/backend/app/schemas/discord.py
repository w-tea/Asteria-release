from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.generation import Txt2ImgRequest


class DiscordSecretStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    configured: bool
    storage: Literal["windows_dpapi_current_user"] = "windows_dpapi_current_user"


class DiscordPublishedProfileSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile: str
    width: int
    height: int
    steps: int
    batch_size: int
    batch_count: int
    hires_enabled: bool
    detailer_enabled: bool
    topaz_enabled: bool


class DiscordPublishedProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    revision: int = Field(ge=1)
    published_at: datetime
    profile_name: str = Field(min_length=1, max_length=80)
    preset_id: str | None = Field(default=None, max_length=128)
    request: Txt2ImgRequest
    model_references: dict[str, str]
    summary: DiscordPublishedProfileSummary

    @field_validator("profile_name", mode="before")
    @classmethod
    def _normalize_profile_name(cls, value: object) -> str:
        return " ".join(str(value or "").split())

    @field_validator("preset_id", mode="before")
    @classmethod
    def _normalize_preset_id(cls, value: object) -> str | None:
        normalized = str(value or "").strip()
        return normalized or None

    @model_validator(mode="after")
    def _require_timezone_and_model_references(self) -> "DiscordPublishedProfile":
        if self.published_at.tzinfo is None:
            raise ValueError("published_at must include timezone")
        for slot_id, stable_model_id in self.model_references.items():
            if not slot_id.strip() or not stable_model_id.strip():
                raise ValueError("model references must be non-empty")
        return self


class DiscordPublishRequest(BaseModel):
    expected_revision: int = Field(ge=0)
    profile_name: str = Field(min_length=1, max_length=80)
    preset_id: str | None = Field(default=None, max_length=128)
    request: Txt2ImgRequest
    model_references: dict[str, str] = Field(default_factory=dict)

    @field_validator("profile_name", mode="before")
    @classmethod
    def _normalize_profile_name(cls, value: object) -> str:
        return " ".join(str(value or "").split())

    @field_validator("preset_id", mode="before")
    @classmethod
    def _normalize_preset_id(cls, value: object) -> str | None:
        normalized = str(value or "").strip()
        return normalized or None

    @model_validator(mode="after")
    def _require_txt2img_safe_base(self) -> "DiscordPublishRequest":
        if any(unit.enabled for unit in self.request.controlnet_units):
            raise ValueError("Discord V1.1 published profiles cannot use ControlNet images")
        if any(unit.enabled for unit in self.request.anima_lllite_units):
            raise ValueError("Discord V1.1 published profiles cannot use regional-control images")
        return self


class DiscordProfileStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    pointer_revision: int = Field(ge=0)
    active_revision: int | None = Field(default=None, ge=1)
    published: bool
    profile: DiscordPublishedProfile | None = None

    @model_validator(mode="after")
    def _validate_pointer(self) -> "DiscordProfileStatus":
        if self.published != (self.active_revision is not None and self.profile is not None):
            raise ValueError("published profile status is inconsistent")
        if self.profile is not None and self.profile.revision != self.active_revision:
            raise ValueError("active profile revision is inconsistent")
        return self


class DiscordUnpublishRequest(BaseModel):
    expected_revision: int = Field(ge=0)


class DiscordPolicy(BaseModel):
    enabled: bool = False
    application_id: str = Field(default="", max_length=64)
    registration_guild_id: str = Field(default="", max_length=64)
    allowed_guild_ids: tuple[str, ...] = ()
    allowed_channel_ids: tuple[str, ...] = ()
    allowed_user_ids: tuple[str, ...] = ()
    allowed_role_ids: tuple[str, ...] = ()
    admin_user_ids: tuple[str, ...] = ()
    allowed_model_ids: tuple[str, ...] = ()
    allowed_preset_ids: tuple[str, ...] = ()
    allow_dms: bool = False
    allow_nsfw_channels: bool = False
    cooldown_seconds: int = Field(default=30, ge=0, le=3600)
    max_concurrent: int = Field(default=1, ge=1, le=4)
    queue_limit: int = Field(default=4, ge=1, le=50)
    fallback_channel_id: str = Field(default="", max_length=64)

    @field_validator(
        "application_id",
        "registration_guild_id",
        "fallback_channel_id",
        mode="before",
    )
    @classmethod
    def _normalize_single_id(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator(
        "allowed_guild_ids",
        "allowed_channel_ids",
        "allowed_user_ids",
        "allowed_role_ids",
        "admin_user_ids",
        "allowed_model_ids",
        "allowed_preset_ids",
        mode="before",
    )
    @classmethod
    def _normalize_ids(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            values = value.replace(",", "\n").splitlines()
        else:
            values = value
        normalized = tuple(
            dict.fromkeys(
                str(item or "").strip()
                for item in values
                if str(item or "").strip()
            )
        )
        return normalized


class DiscordConnectionStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    configured: bool
    connected: bool
    commands_registered: bool
    application_id: str = ""
    bot_user_id: str = ""
    connected_at: datetime | None = None
    validated_at: datetime | None = None
    last_error_code: str = ""
    multi_instance_warning: bool = False


class DiscordInteractionContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    application_id: str
    interaction_id: str
    requester_id: str
    guild_id: str | None = None
    channel_id: str | None = None
    role_ids: tuple[str, ...] = ()
    channel_nsfw: bool = False
    has_attachment: bool = False


class DiscordGenerateCommand(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str | None = Field(default=None, max_length=4000)
    preset_id: str | None = Field(default=None, max_length=128)
    model_id: str | None = Field(default=None, max_length=256)
    overrides: dict[str, object] = Field(default_factory=dict)
