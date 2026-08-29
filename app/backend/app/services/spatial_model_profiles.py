from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


Sha256Text = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class SpatialStageCoefficients(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    early_guidance_end: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    isolation_start: float = Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    isolation_end: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    late_blending_start: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_ranges(self) -> "SpatialStageCoefficients":
        if self.isolation_start >= self.isolation_end:
            raise ValueError("spatial isolation range must increase")
        return self


class SpatialModelProfileDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    model_family: str = Field(min_length=1)
    architectures: tuple[str, ...] = Field(min_length=1)
    variants: tuple[str, ...] = Field(min_length=1)
    encoder_kinds: tuple[str, ...] = Field(min_length=1)
    verified_structure_ids: tuple[str, ...] = Field(min_length=1)
    layout_fingerprints: tuple[Sha256Text, ...] = Field(min_length=1)
    diffusion_block_counts: tuple[int, ...] = Field(min_length=1)
    automatic_status: Literal["supported", "selection_required", "unsupported"]
    manual_selectable: bool
    coefficients: SpatialStageCoefficients


class SpatialModelProfileCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["asteria.spatial-model-profiles.v1"]
    profiles: tuple[SpatialModelProfileDefinition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "SpatialModelProfileCatalog":
        ids = [item.profile_id for item in self.profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("spatial profile IDs must be unique")
        return self


def load_spatial_model_profiles(path: str | Path) -> SpatialModelProfileCatalog:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SpatialModelProfileCatalog.model_validate(payload)
