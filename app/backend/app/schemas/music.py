from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, PlainSerializer, field_validator, model_validator


MUSIC_REQUEST_SCHEMA_VERSION = "asteria.music.request.v1"
MUSIC_REQUEST_V2_SCHEMA_VERSION = "asteria.music.request.v2"
MusicMode = Literal["vocal", "instrumental"]
MusicProvider = Literal["local_music3"]
MusicModelComponentRole = Literal["diffusion_model", "text_encoder", "audio_vae"]
MusicExecutionProfile = Literal["automatic"]
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_FINGERPRINT_ALGORITHM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,191}$")
_SECRET = re.compile(r"(?:api[_-]?key|authorization|credential|password|secret|token)", re.I)


class _FrozenDict(dict[str, Any]):
    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("provider options are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


class _FrozenList(tuple[Any, ...]):
    append = lambda self, value: self._immutable()
    extend = lambda self, value: self._immutable()
    insert = lambda self, index, value: self._immutable()
    pop = lambda self, index=-1: self._immutable()
    remove = lambda self, value: self._immutable()
    clear = lambda self: self._immutable()

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("provider options are immutable")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenDict({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze(item) for item in value)
    return value


def _thaw_json(value: Any) -> JsonValue:
    if isinstance(value, dict):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


FrozenProviderOptions = Annotated[
    dict[str, JsonValue],
    PlainSerializer(_thaw_json, return_type=dict[str, JsonValue]),
]


class MusicModelComponentSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: MusicModelComponentRole
    stable_asset_id: str
    fingerprint_algorithm: str
    fingerprint_version: int = Field(ge=1)
    fingerprint_value: str

    @field_validator("stable_asset_id", "fingerprint_value")
    @classmethod
    def validate_receipt_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not _IDENTIFIER.fullmatch(normalized):
            raise ValueError("music model receipt identifier is invalid")
        return normalized

    @field_validator("fingerprint_algorithm")
    @classmethod
    def validate_fingerprint_algorithm(cls, value: str) -> str:
        normalized = value.strip()
        if not _FINGERPRINT_ALGORITHM.fullmatch(normalized):
            raise ValueError("music model fingerprint algorithm is invalid")
        return normalized


class MusicModelSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    architecture: Literal["minimax_music3"]
    bundle_id: str
    components: tuple[MusicModelComponentSelection, ...]

    @field_validator("profile_id", "bundle_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not _IDENTIFIER.fullmatch(normalized):
            raise ValueError("music model selection identifier is invalid")
        return normalized

    @model_validator(mode="after")
    def validate_components(self) -> "MusicModelSelection":
        roles = [component.role for component in self.components]
        if len(roles) != len(set(roles)):
            raise ValueError("music model component roles must be unique")
        if set(roles) != {"diffusion_model", "text_encoder", "audio_vae"}:
            raise ValueError("music model selection requires every Music3 component role")
        asset_ids = [component.stable_asset_id for component in self.components]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("music model component asset identifiers must be unique")
        return self


class MusicSamplingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text_guidance: float = Field(ge=0, le=100, allow_inf_nan=False)
    top_k: int = Field(ge=1, le=16384)
    steps: int = Field(ge=1, le=1000)
    sampler: str
    scheduler: str
    denoise: float = Field(ge=0, le=1, allow_inf_nan=False)
    tiled_decode: bool
    execution_profile: MusicExecutionProfile

    @field_validator("sampler", "scheduler")
    @classmethod
    def validate_sampling_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not _IDENTIFIER.fullmatch(normalized):
            raise ValueError("music sampling identifier is invalid")
        return normalized


class MusicLoraSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stable_asset_id: str
    fingerprint_algorithm: str
    fingerprint_version: int = Field(ge=1)
    fingerprint_value: str
    strength: float = Field(allow_inf_nan=False)

    @field_validator("stable_asset_id", "fingerprint_value")
    @classmethod
    def validate_receipt_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not _IDENTIFIER.fullmatch(normalized):
            raise ValueError("music LoRA receipt identifier is invalid")
        return normalized

    @field_validator("fingerprint_algorithm")
    @classmethod
    def validate_fingerprint_algorithm(cls, value: str) -> str:
        normalized = value.strip()
        if not _FINGERPRINT_ALGORITHM.fullmatch(normalized):
            raise ValueError("music LoRA fingerprint algorithm is invalid")
        return normalized


def contains_secret_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_SECRET.search(str(key)) or contains_secret_key(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(contains_secret_key(item) for item in value)
    return False


def contains_non_finite_number(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(contains_non_finite_number(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_non_finite_number(item) for item in value)
    return False


class MusicRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "asteria.music.request.v1",
        "asteria.music.request.v2",
    ] = MUSIC_REQUEST_SCHEMA_VERSION
    mode: MusicMode
    provider: MusicProvider
    description: str
    lyrics: str | None = None
    seed: int = Field(default=-1, ge=-1, le=9223372036854775807)
    duration_intent: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    language_intent: str | None = None
    model_profile: str = "minimax-music3"
    model_selection: MusicModelSelection | None = None
    sampling: MusicSamplingSettings | None = None
    loras: tuple[MusicLoraSelection, ...] = ()
    provider_options: FrozenProviderOptions = Field(default_factory=dict, validate_default=True)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> "MusicRequest":
        if not update and not deep:
            return super().model_copy(update=update, deep=deep)
        payload = self.model_dump(mode="python")
        if update:
            payload.update(update)
        return type(self).model_validate(payload)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("music description is required")
        return value

    @field_validator("lyrics", "language_intent")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("model_profile")
    @classmethod
    def validate_profile(cls, value: str) -> str:
        value = value.strip()
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("music model profile is invalid")
        return value

    @field_validator("provider_options")
    @classmethod
    def validate_options(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if contains_secret_key(value):
            raise ValueError("provider options cannot contain secret values")
        if contains_non_finite_number(value):
            raise ValueError("provider options must contain finite numbers")
        return _freeze(value)

    @model_validator(mode="after")
    def validate_mode(self) -> "MusicRequest":
        if self.mode == "instrumental" and self.lyrics is not None:
            raise ValueError("instrumental mode forbids lyrics")
        if self.schema_version == MUSIC_REQUEST_SCHEMA_VERSION:
            if self.model_selection is not None or self.sampling is not None or self.loras:
                raise ValueError("music request v1 cannot carry v2 production settings")
            return self
        if self.model_selection is None:
            raise ValueError("music v2 request requires model selection")
        if self.sampling is None:
            raise ValueError("music v2 request requires sampling settings")
        if self.model_selection.profile_id != self.model_profile:
            raise ValueError("music model selection profile must match model profile")
        lora_ids = [item.stable_asset_id for item in self.loras]
        if len(lora_ids) != len(set(lora_ids)):
            raise ValueError("music LoRA asset identifiers must be unique")
        return self
