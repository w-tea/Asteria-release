from __future__ import annotations

from pathlib import PurePosixPath
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PROMPT_LAYOUT_SCHEMA_VERSION = "asteria.prompt-layout.v1"
SPATIAL_MASK_NORMALIZATION_VERSION = "asteria.spatial-mask.v1"
TARGET_ID_PATTERN = (
    r"^target_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
SHA256_PATTERN = r"^[0-9a-f]{64}$"
COLOR_PATTERN = r"^#[0-9A-Fa-f]{6}$"
SPATIAL_PROFILE_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,127}$"


class SpatialReferenceCanvas(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    width: int = Field(ge=1, le=8192, strict=True)
    height: int = Field(ge=1, le=8192, strict=True)


class SpatialAssetReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str = Field(pattern=r"^control-image:sha256:[0-9a-f]{64}$")
    storage_kind: Literal["control_image"]
    relative_path: str
    sha256: str = Field(pattern=SHA256_PATTERN)
    bytes: int = Field(ge=1, strict=True)
    width: int = Field(ge=1, strict=True)
    height: int = Field(ge=1, strict=True)
    mime_type: Literal["image/png", "image/jpeg", "image/webp"]

    @field_validator("relative_path")
    @classmethod
    def validate_managed_relative_path(cls, value: str) -> str:
        if (
            not value
            or "\\" in value
            or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
        ):
            raise ValueError("spatial asset path must be a POSIX relative path")
        candidate = PurePosixPath(value)
        if (
            candidate.is_absolute()
            or candidate.as_posix() != value
            or any(part in {"", ".", ".."} for part in candidate.parts)
            or not candidate.parts
            or candidate.parts[0] != "control_images"
        ):
            raise ValueError("spatial asset path must stay inside control_images")
        for part in candidate.parts:
            if any(character in '<>:"|?*' for character in part) or part.endswith((" ", ".")):
                raise ValueError("spatial asset path contains a Windows-invalid name")
            stem = part.split(".", 1)[0].casefold()
            if stem in {"con", "prn", "aux", "nul"} or re.fullmatch(
                r"(?:com|lpt)[1-9]",
                stem,
            ):
                raise ValueError("spatial asset path contains a reserved Windows name")
        return value

    @model_validator(mode="after")
    def validate_identity_and_media_type(self) -> "SpatialAssetReceipt":
        if self.asset_id != f"control-image:sha256:{self.sha256}":
            raise ValueError("spatial asset ID must match its SHA-256")
        suffix = PurePosixPath(self.relative_path).suffix.lower()
        expected_mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(suffix)
        if expected_mime != self.mime_type:
            raise ValueError("spatial asset MIME type must match its file extension")
        return self


class SpatialPixelRect(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x: int = Field(ge=0, strict=True)
    y: int = Field(ge=0, strict=True)
    width: int = Field(ge=1, strict=True)
    height: int = Field(ge=1, strict=True)

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


class SpatialMaskSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    asset: SpatialAssetReceipt
    source_color: str = Field(pattern=COLOR_PATTERN)
    source_rect_px: SpatialPixelRect
    source_bounds_px: SpatialPixelRect
    binary_mask_sha256: str = Field(pattern=SHA256_PATTERN)
    normalization_version: Literal["asteria.spatial-mask.v1"]

    @field_validator("source_color")
    @classmethod
    def normalize_source_color(cls, value: str) -> str:
        return value.upper()

    @model_validator(mode="after")
    def validate_source_bounds(self) -> "SpatialMaskSource":
        source_rect = self.source_rect_px
        source_bounds = self.source_bounds_px
        if source_rect.right > self.asset.width or source_rect.bottom > self.asset.height:
            raise ValueError("spatial source rectangle must stay inside its asset")
        if (
            source_bounds.x < source_rect.x
            or source_bounds.y < source_rect.y
            or source_bounds.right > source_rect.right
            or source_bounds.bottom > source_rect.bottom
        ):
            raise ValueError("spatial source bounds must stay inside the source rectangle")
        return self


class SpatialLayoutTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_id: str = Field(pattern=TARGET_ID_PATTERN)
    shape: Literal["ellipse", "rectangle", "mask"]
    center_x: float = Field(allow_inf_nan=False)
    center_y: float = Field(allow_inf_nan=False)
    width: float = Field(gt=0, le=16, allow_inf_nan=False)
    height: float = Field(gt=0, le=16, allow_inf_nan=False)
    size_mode: Literal["auto", "small", "standard", "large", "custom"]
    strength_mode: Literal["auto", "manual"]
    strength: int | None = Field(default=None, ge=0, le=100, strict=True)
    blending_mode: Literal["auto", "manual"]
    blending: int | None = Field(default=None, ge=0, le=100, strict=True)
    mask_source: SpatialMaskSource | None = None

    @model_validator(mode="after")
    def validate_geometry_and_modes(self) -> "SpatialLayoutTarget":
        left = self.center_x - self.width / 2
        right = self.center_x + self.width / 2
        top = self.center_y - self.height / 2
        bottom = self.center_y + self.height / 2
        if right <= 0 or bottom <= 0 or left >= 1 or top >= 1:
            raise ValueError("spatial target must intersect the reference canvas")

        if (self.strength_mode == "auto") != (self.strength is None):
            raise ValueError("spatial strength must be null only in auto mode")
        if (self.blending_mode == "auto") != (self.blending is None):
            raise ValueError("spatial blending must be null only in auto mode")
        if self.shape == "mask" and self.mask_source is None:
            raise ValueError("mask targets require a reproducible mask source")
        if self.shape != "mask" and self.mask_source is not None:
            raise ValueError("geometric targets must not contain a mask source")
        return self


class SpatialModelProfileSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base: str | None = Field(default=None, pattern=SPATIAL_PROFILE_ID_PATTERN)
    hires: str | None = Field(default=None, pattern=SPATIAL_PROFILE_ID_PATTERN)
    refiner: str | None = Field(default=None, pattern=SPATIAL_PROFILE_ID_PATTERN)

    @model_validator(mode="after")
    def require_at_least_one_stage(self) -> "SpatialModelProfileSelection":
        if self.base is None and self.hires is None and self.refiner is None:
            raise ValueError("spatial model profile selection must contain a stage")
        return self


class PromptLayout(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["asteria.prompt-layout.v1"]
    mode: Literal["enabled", "disabled"]
    reference_canvas: SpatialReferenceCanvas
    depth_mode: Literal["auto", "manual"]
    depth_order: tuple[str, ...] = ()
    excluded_target_ids: tuple[str, ...] = ()
    targets: tuple[SpatialLayoutTarget, ...] = Field(min_length=1)
    model_profile_selection: SpatialModelProfileSelection | None = None

    @field_validator("depth_order", "excluded_target_ids")
    @classmethod
    def validate_target_id_list(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for target_id in value:
            if re.fullmatch(TARGET_ID_PATTERN, target_id) is None:
                raise ValueError("spatial target references must use UUIDv7 target IDs")
        if len(value) != len(set(value)):
            raise ValueError("spatial target references must be unique")
        return value

    @model_validator(mode="after")
    def validate_target_references_and_depth(self) -> "PromptLayout":
        target_ids = tuple(target.target_id for target in self.targets)
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("spatial layout target IDs must be unique")
        target_id_set = set(target_ids)
        if not set(self.excluded_target_ids).issubset(target_id_set):
            raise ValueError("excluded spatial targets must exist in the layout")
        if self.depth_mode == "auto":
            if self.depth_order:
                raise ValueError("automatic depth must not contain a manual order")
        elif len(self.depth_order) != len(target_ids) or set(self.depth_order) != target_id_set:
            raise ValueError("manual depth must contain every spatial target exactly once")
        if self.mode == "enabled" and len(self.excluded_target_ids) == len(target_ids):
            raise ValueError("enabled spatial layout must retain at least one target")
        return self
