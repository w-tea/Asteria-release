from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.video import FrozenJsonObject, VideoRequest


VIDEO_UPSCALE_REQUEST_SCHEMA_VERSION = "asteria.video-upscale.request.v1"
VIDEO_UPSCALE_BUNDLE_SCHEMA_VERSION = "asteria.video-upscale.work-bundle.v1"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_FINGERPRINT_ALGORITHM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,191}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TIME_BASE = re.compile(r"^[1-9][0-9]*/[1-9][0-9]*$")


def _snap_32(value: object) -> object:
    if isinstance(value, bool) or not isinstance(value, int):
        return value
    snapped = int(round(value / 32.0)) * 32
    if snapped < 64 or snapped > 4096:
        raise ValueError("video upscale dimension is outside the supported range")
    return snapped


class VideoUpscaleTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    width: int | None = Field(default=None, ge=64, le=4096)
    height: int | None = Field(default=None, ge=64, le=4096)
    scale: float | None = Field(default=None, gt=1.0, le=4.0, allow_inf_nan=False)

    @field_validator("width", "height", mode="before")
    @classmethod
    def snap_dimensions(cls, value: object) -> object:
        return _snap_32(value)

    @model_validator(mode="after")
    def validate_target(self) -> "VideoUpscaleTarget":
        has_dimensions = self.width is not None or self.height is not None
        if has_dimensions and (self.width is None or self.height is None):
            raise ValueError("video upscale target requires width and height together")
        if has_dimensions == (self.scale is not None):
            raise ValueError("video upscale target requires dimensions or scale")
        return self


class VideoUpscaleModelReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stable_asset_id: str
    fingerprint_algorithm: str
    fingerprint_version: int = Field(ge=1)
    fingerprint_value: str

    @field_validator("stable_asset_id", "fingerprint_value")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not _IDENTIFIER.fullmatch(normalized):
            raise ValueError("video upscale model receipt is invalid")
        return normalized

    @field_validator("fingerprint_algorithm")
    @classmethod
    def validate_algorithm(cls, value: str) -> str:
        normalized = value.strip()
        if not _FINGERPRINT_ALGORITHM.fullmatch(normalized):
            raise ValueError("video upscale fingerprint algorithm is invalid")
        return normalized


class VideoUpscaleTemporalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_length: int = Field(default=136, ge=17, le=100000)
    overlap: int = Field(default=17, ge=0, le=100000)
    anchor_strength: float = Field(default=0.999, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_temporal_grid(self) -> "VideoUpscaleTemporalSettings":
        if self.chunk_length % 17 != 0 or self.overlap % 17 != 0:
            raise ValueError("video upscale temporal settings require the 17-frame grid")
        if self.overlap >= self.chunk_length:
            raise ValueError("video upscale temporal overlap must be smaller than the chunk")
        return self


class VideoUpscaleSpatialSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tile_width: int = Field(default=512, ge=32, le=100000)
    tile_height: int = Field(default=512, ge=32, le=100000)
    width_overlap: int = Field(default=128, ge=0, le=100000)
    height_overlap: int = Field(default=128, ge=0, le=100000)
    fade_width: int = Field(default=32, ge=0, le=100000)
    fade_height: int = Field(default=32, ge=0, le=100000)
    minimum_tile_size: int = Field(default=256, ge=0, le=100000)
    overlap_mode: Literal["earlier", "later"] = "earlier"
    overlap_blend: Literal["linear", "smoothstep", "overwrite", "midpoint"] = "linear"

    @model_validator(mode="after")
    def validate_spatial_grid(self) -> "VideoUpscaleSpatialSettings":
        values = (
            self.tile_width,
            self.tile_height,
            self.width_overlap,
            self.height_overlap,
            self.fade_width,
            self.fade_height,
            self.minimum_tile_size,
        )
        if any(value % 32 != 0 for value in values):
            raise ValueError("video upscale spatial settings require the 32-pixel grid")
        if self.width_overlap >= self.tile_width or self.height_overlap >= self.tile_height:
            raise ValueError("video upscale spatial overlap must be smaller than the tile")
        if self.fade_width > self.width_overlap or self.fade_height > self.height_overlap:
            raise ValueError("video upscale fade must fit inside the overlap")
        if self.minimum_tile_size > min(self.tile_width, self.tile_height):
            raise ValueError("video upscale minimum tile size must fit inside the tile")
        return self


class VideoUpscaleSamplingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    steps: int = Field(ge=1, le=200)
    sampler: str
    scheduler: str
    cfg: float = Field(default=1, ge=0, le=100, allow_inf_nan=False)
    seed: int = Field(default=-1, ge=-1, le=9223372036854775807)

    @field_validator("sampler", "scheduler")
    @classmethod
    def validate_sampling_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not _IDENTIFIER.fullmatch(normalized):
            raise ValueError("video upscale sampling identifier is invalid")
        return normalized


class VideoUpscaleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["asteria.video-upscale.request.v1"] = (
        VIDEO_UPSCALE_REQUEST_SCHEMA_VERSION
    )
    source_generation_id: int = Field(ge=1)
    source_artifact_id: str
    mode: Literal["h3_3d_model", "model_free"]
    target: VideoUpscaleTarget
    upscaler_model: VideoUpscaleModelReceipt | None = None
    interpolation: Literal["nearest-exact", "bilinear", "area", "bicubic"] | None = None
    temporal: VideoUpscaleTemporalSettings | None = Field(
        default_factory=VideoUpscaleTemporalSettings
    )
    spatial: VideoUpscaleSpatialSettings | None = Field(
        default_factory=VideoUpscaleSpatialSettings
    )
    sampling: VideoUpscaleSamplingSettings
    device: Literal["cuda", "cpu"] = "cuda"
    precision: Literal["fp16", "fp32", "bf16"] = "fp16"
    offload: Literal["auto", "balanced", "maximum"] = "auto"

    @field_validator("source_artifact_id")
    @classmethod
    def validate_source_artifact_id(cls, value: str) -> str:
        normalized = value.strip()
        if not _IDENTIFIER.fullmatch(normalized):
            raise ValueError("video upscale source artifact identifier is invalid")
        return normalized

    @model_validator(mode="after")
    def validate_upscale_mode(self) -> "VideoUpscaleRequest":
        if self.mode == "h3_3d_model":
            if self.upscaler_model is None or self.interpolation is not None:
                raise ValueError("model-based video upscale requires exactly one model receipt")
        elif self.upscaler_model is not None or self.interpolation is None:
            raise ValueError("model-free video upscale requires interpolation only")
        return self


class VideoUpscaleWorkBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["asteria.video-upscale.work-bundle.v1"] = (
        VIDEO_UPSCALE_BUNDLE_SCHEMA_VERSION
    )
    source_generation_id: int = Field(ge=1)
    source_media_artifact_id: str
    source_media_sha256: str
    actual_output_root: str
    working_operation_id: str
    latent_artifact_id: str
    latent_sha256: str
    conditioning_sha256: str
    packet_sha256: str
    packet_snapshot: FrozenJsonObject
    request_snapshot: VideoRequest
    workflow_id: str
    mapping_id: str
    mapping_revision: int = Field(ge=1)
    actual_width: int = Field(gt=0)
    actual_height: int = Field(gt=0)
    actual_frame_count: int = Field(gt=0)
    fps_numerator: int = Field(gt=0)
    fps_denominator: int = Field(gt=0)
    audio_sample_rate: int | None = Field(default=None, gt=0)
    video_time_base: str
    audio_time_base: str | None = None
    created_at: datetime

    @field_validator(
        "source_media_artifact_id",
        "working_operation_id",
        "latent_artifact_id",
        "workflow_id",
        "mapping_id",
    )
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not _IDENTIFIER.fullmatch(normalized):
            raise ValueError("video upscale bundle identifier is invalid")
        return normalized

    @field_validator("actual_output_root")
    @classmethod
    def validate_output_root(cls, value: str) -> str:
        normalized = value.strip()
        path = Path(normalized)
        if not normalized or not path.is_absolute() or ".." in path.parts:
            raise ValueError("video upscale bundle output root is invalid")
        return str(path)

    @field_validator(
        "source_media_sha256",
        "latent_sha256",
        "conditioning_sha256",
        "packet_sha256",
    )
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.lower()
        if not _SHA256.fullmatch(normalized):
            raise ValueError("video upscale bundle hash is invalid")
        return normalized

    @field_validator("video_time_base", "audio_time_base")
    @classmethod
    def validate_time_base(cls, value: str | None) -> str | None:
        if value is not None and not _TIME_BASE.fullmatch(value):
            raise ValueError("video upscale bundle time base is invalid")
        return value

    @model_validator(mode="after")
    def validate_bundle(self) -> "VideoUpscaleWorkBundle":
        if self.created_at.tzinfo is None:
            raise ValueError("video upscale bundle timestamp requires timezone")
        if (self.audio_sample_rate is None) != (self.audio_time_base is None):
            raise ValueError("video upscale bundle audio receipt is incomplete")
        return self
