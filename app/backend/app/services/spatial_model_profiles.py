from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


Sha256Text = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class SpatialLossWeightsDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    position: float = Field(ge=0.0, le=32.0, allow_inf_nan=False)
    size: float = Field(ge=0.0, le=32.0, allow_inf_nan=False)
    separation: float = Field(ge=0.0, le=32.0, allow_inf_nan=False)
    presence: float = Field(ge=0.0, le=32.0, allow_inf_nan=False)
    global_retention: float = Field(ge=0.0, le=32.0, allow_inf_nan=False)


class SpatialGuidanceDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start_percent: float = Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    end_percent: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    iterations: int = Field(ge=1, le=8, strict=True)
    step_size: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    attention_prior_scale: float = Field(ge=0.0, le=5.0, allow_inf_nan=False)
    attention_layers: tuple[str, ...] = Field(min_length=1)
    self_relation_layers: tuple[str, ...] = Field(min_length=1)
    global_retention_floor: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    loss_weights: SpatialLossWeightsDefinition

    @model_validator(mode="after")
    def validate_ranges(self) -> "SpatialGuidanceDefinition":
        if self.start_percent >= self.end_percent:
            raise ValueError("spatial guidance range must increase")
        if len(set(self.attention_layers)) != len(self.attention_layers):
            raise ValueError("spatial attention layers must be unique")
        if len(set(self.self_relation_layers)) != len(self.self_relation_layers):
            raise ValueError("spatial self relation layers must be unique")
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
    guidance: SpatialGuidanceDefinition


class SpatialModelProfileCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["asteria.spatial-model-profiles.v3"]
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
