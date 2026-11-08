from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.prompt_layout import TARGET_ID_PATTERN

from app.schemas.spatial_prompt_runtime import (
    Sha256,
    SpatialCharacterBindingReceipt,
    SpatialExcludedTargetReceipt,
    SpatialPromptFragmentReceipt,
    SpatialScheduledPromptBinding,
    SpatialStagePromptPlan,
    canonical_runtime_sha256,
)


GuidanceSha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GuidanceTargetId = Annotated[str, Field(pattern=TARGET_ID_PATTERN)]


class _GuidanceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SpatialStagePromptPlanV2(SpatialStagePromptPlan):
    """A single positive prompt with target ranges for one generation stage."""

    schema_version: Literal["asteria.spatial-stage-prompt.v2"]
    full_text: str
    full_text_sha256: Sha256
    character_bindings: tuple[SpatialCharacterBindingReceipt, ...]
    global_character_ranges: tuple[tuple[int, int], ...]
    fragment_receipts: tuple[SpatialPromptFragmentReceipt, ...]
    scheduled_bindings: tuple[SpatialScheduledPromptBinding, ...]
    excluded_targets: tuple[SpatialExcludedTargetReceipt, ...] = ()


class SpatialLayoutFieldTargetReceipt(_GuidanceModel):
    target_id: GuidanceTargetId
    shape: Literal["ellipse", "rectangle", "mask"]
    center_x: float = Field(allow_inf_nan=False)
    center_y: float = Field(allow_inf_nan=False)
    width: float = Field(gt=0.0, le=16.0, allow_inf_nan=False)
    height: float = Field(gt=0.0, le=16.0, allow_inf_nan=False)
    variance_x: float = Field(gt=0.0, allow_inf_nan=False)
    variance_y: float = Field(gt=0.0, allow_inf_nan=False)
    depth_index: int = Field(ge=0)
    depth_prior: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    strength: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    blending: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    raw_field_sha256: GuidanceSha256
    normalized_field_sha256: GuidanceSha256
    overlap_pixel_count: int = Field(ge=0)


class SpatialLayoutFieldBundle(_GuidanceModel):
    schema_version: Literal["asteria.spatial-layout-field.v1"]
    stage: Literal["base", "hires", "refiner"]
    width: int = Field(ge=1, le=8192)
    height: int = Field(ge=1, le=8192)
    targets: tuple[SpatialLayoutFieldTargetReceipt, ...] = Field(min_length=1)
    raw_bundle_sha256: GuidanceSha256
    normalized_bundle_sha256: GuidanceSha256
    bundle_sha256: GuidanceSha256

    @model_validator(mode="after")
    def validate_bundle(self) -> "SpatialLayoutFieldBundle":
        target_ids = [item.target_id for item in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("layout field target ids must be unique")
        ordered_depth_indexes = [
            item.depth_index
            for item in sorted(self.targets, key=lambda item: item.depth_index)
        ]
        if ordered_depth_indexes != list(range(len(self.targets))):
            raise ValueError("layout field depth indexes must be contiguous")
        payload = self.model_dump(mode="json", exclude={"bundle_sha256"})
        if self.bundle_sha256 != canonical_runtime_sha256(payload):
            raise ValueError("bundle_sha256 does not match the layout field payload")
        return self
