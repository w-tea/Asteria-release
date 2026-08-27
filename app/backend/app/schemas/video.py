from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, PlainSerializer, field_validator, model_validator


VIDEO_REQUEST_SCHEMA_VERSION = "asteria.video.request.v1"
VIDEO_REQUEST_SCHEMA_VERSION_V2 = "asteria.video.request.v2"
VideoMode = Literal["text", "image", "first_last", "reference", "sequence"]
VideoProvider = Literal["local_h3"]
VideoReferenceKind = Literal["image", "video", "audio"]
VideoReferenceRole = Literal["identity", "style", "motion", "camera", "voice", "other"]
ResolutionIntent = Literal["preview", "quality", "custom"]
VideoModelVariant = Literal["fl2va", "ref2va"]
VideoModelComponentRole = Literal["diffusion_model", "text_encoder", "video_vae", "audio_vae"]

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_MODEL_PROFILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FINGERPRINT_ALGORITHM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,191}$")
_ASPECT_RATIO = re.compile(r"^(?:adaptive|auto|[1-9][0-9]*:[1-9][0-9]*)$")
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|credential|password|secret|token)",
    re.IGNORECASE,
)


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
    def append(self, _value: Any) -> None:
        raise TypeError("provider options are immutable")

    def extend(self, _value: Any) -> None:
        raise TypeError("provider options are immutable")

    def insert(self, _index: int, _value: Any) -> None:
        raise TypeError("provider options are immutable")

    def pop(self, _index: int = -1) -> None:
        raise TypeError("provider options are immutable")

    def remove(self, _value: Any) -> None:
        raise TypeError("provider options are immutable")

    def clear(self) -> None:
        raise TypeError("provider options are immutable")


def _thaw_json(value: Any) -> JsonValue:
    if isinstance(value, dict):
        return {str(key): _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(nested) for nested in value]
    return value


FrozenProviderOptions = Annotated[
    dict[str, JsonValue],
    PlainSerializer(_thaw_json, return_type=dict[str, JsonValue]),
]
FrozenJsonObject = FrozenProviderOptions


class VideoSize(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    aspect_ratio: str
    resolution_intent: ResolutionIntent

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _ASPECT_RATIO.fullmatch(normalized):
            raise ValueError("video aspect ratio is invalid")
        return normalized


class VideoAudioIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    prompt: str | None = None

    @field_validator("prompt")
    @classmethod
    def normalize_prompt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_intent(self) -> "VideoAudioIntent":
        if not self.enabled and self.prompt is not None:
            raise ValueError("disabled audio intent cannot carry a prompt")
        return self


class VideoReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    kind: VideoReferenceKind
    role: VideoReferenceRole

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("video reference artifact identifier is invalid")
        return value


class VideoModelComponentSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: VideoModelComponentRole
    stable_asset_id: str
    fingerprint_algorithm: str
    fingerprint_version: int = Field(ge=1)
    fingerprint_value: str

    @field_validator("stable_asset_id", "fingerprint_value")
    @classmethod
    def validate_receipt_token(cls, value: str) -> str:
        normalized = value.strip()
        if not _IDENTIFIER.fullmatch(normalized):
            raise ValueError("video model receipt identifier is invalid")
        return normalized

    @field_validator("fingerprint_algorithm")
    @classmethod
    def validate_fingerprint_algorithm(cls, value: str) -> str:
        normalized = value.strip()
        if not _FINGERPRINT_ALGORITHM.fullmatch(normalized):
            raise ValueError("video model fingerprint algorithm is invalid")
        return normalized


class VideoModelSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    architecture: Literal["minimax_h3"]
    variant: VideoModelVariant
    components: tuple[VideoModelComponentSelection, ...]

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, value: str) -> str:
        normalized = value.strip()
        if not _MODEL_PROFILE.fullmatch(normalized):
            raise ValueError("video model selection profile is invalid")
        return normalized

    @model_validator(mode="after")
    def validate_components(self) -> "VideoModelSelection":
        roles = [component.role for component in self.components]
        if len(roles) != len(set(roles)):
            raise ValueError("video model component roles must be unique")
        required = {"diffusion_model", "text_encoder", "video_vae", "audio_vae"}
        if set(roles) != required:
            raise ValueError("video model selection requires every H3 component role")
        asset_ids = [component.stable_asset_id for component in self.components]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("video model component asset identifiers must be unique")
        return self


class VideoSamplingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    steps: int = Field(ge=1)
    sampler: str
    scheduler: str
    video_shift: float = Field(ge=0, allow_inf_nan=False)
    audio_shift: float = Field(ge=0, allow_inf_nan=False)
    batch_count: int = Field(default=1, ge=1)
    seed_policy: Literal["increment", "random"] = "increment"

    @field_validator("sampler", "scheduler")
    @classmethod
    def validate_sampling_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not _IDENTIFIER.fullmatch(normalized):
            raise ValueError("video sampling identifier is invalid")
        return normalized


class VideoLoraSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stable_asset_id: str
    fingerprint_algorithm: str
    fingerprint_version: int = Field(ge=1)
    fingerprint_value: str
    weight: float = Field(allow_inf_nan=False)
    compatible_variants: tuple[VideoModelVariant, ...]

    @field_validator("stable_asset_id", "fingerprint_value")
    @classmethod
    def validate_receipt_token(cls, value: str) -> str:
        normalized = value.strip()
        if not _IDENTIFIER.fullmatch(normalized):
            raise ValueError("video LoRA receipt identifier is invalid")
        return normalized

    @field_validator("fingerprint_algorithm")
    @classmethod
    def validate_fingerprint_algorithm(cls, value: str) -> str:
        normalized = value.strip()
        if not _FINGERPRINT_ALGORITHM.fullmatch(normalized):
            raise ValueError("video LoRA fingerprint algorithm is invalid")
        return normalized

    @field_validator("compatible_variants")
    @classmethod
    def validate_compatible_variants(
        cls,
        value: tuple[VideoModelVariant, ...],
    ) -> tuple[VideoModelVariant, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("video LoRA compatible variants must be non-empty and unique")
        return value


class VideoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "asteria.video.request.v1",
        "asteria.video.request.v2",
    ] = VIDEO_REQUEST_SCHEMA_VERSION
    mode: VideoMode
    provider: VideoProvider
    prompt: str
    negative_prompt: str | None = None
    seed: int = Field(default=-1, ge=-1, le=9223372036854775807)
    size: VideoSize
    duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    fps: float = Field(gt=0, allow_inf_nan=False)
    audio: VideoAudioIntent = Field(default_factory=VideoAudioIntent)
    first_frame_artifact_id: str | None = None
    last_frame_artifact_id: str | None = None
    references: tuple[VideoReference, ...] = ()
    model_profile: str
    model_selection: VideoModelSelection | None = None
    sampling: VideoSamplingSettings | None = None
    loras: tuple[VideoLoraSelection, ...] = ()
    provider_options: FrozenProviderOptions = Field(default_factory=dict, validate_default=True)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("video prompt is required")
        return normalized

    @field_validator("negative_prompt")
    @classmethod
    def normalize_negative_prompt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("first_frame_artifact_id", "last_frame_artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str | None) -> str | None:
        if value is not None and not _IDENTIFIER.fullmatch(value):
            raise ValueError("video frame artifact identifier is invalid")
        return value

    @field_validator("model_profile")
    @classmethod
    def validate_model_profile(cls, value: str) -> str:
        normalized = value.strip()
        if not _MODEL_PROFILE.fullmatch(normalized):
            raise ValueError("video model profile is invalid")
        return normalized

    @field_validator("provider_options")
    @classmethod
    def validate_provider_options(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if contains_secret_key(value):
            raise ValueError("provider options cannot contain secret values")
        return _freeze_json(value)

    @model_validator(mode="after")
    def validate_sources(self) -> "VideoRequest":
        reference_ids = [reference.artifact_id for reference in self.references]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("video reference identifiers must be unique")
        if self.mode == "text":
            if self.first_frame_artifact_id or self.last_frame_artifact_id or self.references:
                raise ValueError("text mode does not accept source artifacts")
        elif self.mode == "image":
            if self.first_frame_artifact_id is None:
                raise ValueError("image mode requires a first frame artifact")
            if self.last_frame_artifact_id is not None:
                raise ValueError("image mode does not accept a last frame artifact")
            if self.references:
                raise ValueError("image mode does not accept reference artifacts")
        elif self.mode == "first_last":
            if self.first_frame_artifact_id is None or self.last_frame_artifact_id is None:
                raise ValueError("first-last mode requires both first frame and last frame artifacts")
            if self.references:
                raise ValueError("first-last mode does not accept reference artifacts")
        elif self.mode == "reference":
            if not self.references:
                raise ValueError("reference mode requires at least one reference artifact")
            if self.first_frame_artifact_id or self.last_frame_artifact_id:
                raise ValueError("reference mode does not accept frame inputs")
        if self.schema_version == VIDEO_REQUEST_SCHEMA_VERSION:
            if self.model_selection is not None or self.sampling is not None or self.loras:
                raise ValueError("video request v1 cannot carry v2 production settings")
            return self
        if self.model_selection is None:
            raise ValueError("video v2 request requires model selection")
        if self.sampling is None:
            raise ValueError("video v2 request requires sampling settings")
        if self.model_selection.profile_id != self.model_profile:
            raise ValueError("video model selection profile must match model profile")
        if self.mode in {"text", "image", "first_last"} and self.model_selection.variant != "fl2va":
            raise ValueError("video task and model variant are incompatible")
        if self.mode == "reference" and self.model_selection.variant != "ref2va":
            raise ValueError("video task and model variant are incompatible")
        if self.mode == "sequence" and self.model_selection.variant != "ref2va":
            raise ValueError("video sequence and model variant are incompatible")
        if any(
            self.model_selection.variant not in lora.compatible_variants
            for lora in self.loras
        ):
            raise ValueError("video LoRA is not compatible with the selected model variant")
        lora_ids = [lora.stable_asset_id for lora in self.loras]
        if len(lora_ids) != len(set(lora_ids)):
            raise ValueError("video LoRA asset identifiers must be unique")
        return self


def is_secret_field_name(value: str) -> bool:
    return _SECRET_KEY.search(value) is not None


def contains_secret_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if is_secret_field_name(str(key)) or contains_secret_key(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(contains_secret_key(item) for item in value)
    return False


def _freeze_json(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return _FrozenDict({str(key): _freeze_json(nested) for key, nested in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze_json(nested) for nested in value)
    return value


def freeze_json_object(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Return a recursively immutable JSON object for persisted snapshots."""
    return _freeze_json(value)  # type: ignore[return-value]
