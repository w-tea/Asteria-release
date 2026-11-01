from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


Sha256Text = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class SpatialRoutingDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cross_start_percent: float = Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    cross_end_percent: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    self_start_percent: float = Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    self_end_percent: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    cross_penalty: float = Field(gt=0.0, le=32.0, allow_inf_nan=False)
    cross_penalty_cap: float = Field(gt=0.0, le=32.0, allow_inf_nan=False)
    self_penalty: float = Field(ge=0.0, le=32.0, allow_inf_nan=False)
    self_penalty_cap: float = Field(ge=0.0, le=32.0, allow_inf_nan=False)
    core_threshold: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    ownership_margin: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    cross_attention_layers: tuple[str, ...] = Field(min_length=1)
    self_attention_layers: tuple[str, ...]
    self_attention_token_counts: tuple[int, ...]
    self_attention_token_limit: int = Field(ge=1, le=65536)
    hires_mode: Literal["base_only"]

    @model_validator(mode="after")
    def validate_ranges(self) -> "SpatialRoutingDefinition":
        if self.cross_start_percent >= self.cross_end_percent:
            raise ValueError("spatial cross routing range must increase")
        if self.self_attention_layers and self.self_start_percent >= self.self_end_percent:
            raise ValueError("spatial self routing range must increase")
        if self.cross_penalty > self.cross_penalty_cap:
            raise ValueError("spatial cross penalty must not exceed its cap")
        if self.self_penalty > self.self_penalty_cap:
            raise ValueError("spatial self penalty must not exceed its cap")
        if len(set(self.cross_attention_layers)) != len(
            self.cross_attention_layers
        ):
            raise ValueError("spatial cross attention layers must be unique")
        if len(set(self.self_attention_layers)) != len(self.self_attention_layers):
            raise ValueError("spatial self attention layers must be unique")
        if len(set(self.self_attention_token_counts)) != len(
            self.self_attention_token_counts
        ):
            raise ValueError("spatial self attention token counts must be unique")
        if any(
            value < 1 or value > self.self_attention_token_limit
            for value in self.self_attention_token_counts
        ):
            raise ValueError("spatial self attention token count is outside its limit")
        if "all" in self.cross_attention_layers or "all" in self.self_attention_layers:
            raise ValueError("spatial attention cannot target every layer")
        return self


class SpatialModelProfileDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    model_family: str = Field(min_length=1)
    architectures: tuple[str, ...] = Field(min_length=1)
    variants: tuple[str, ...] = Field(min_length=1)
    encoder_kinds: tuple[str, ...] = Field(min_length=1)
    prediction_types: tuple[str, ...] = Field(min_length=1)
    verified_structure_ids: tuple[str, ...] = Field(min_length=1)
    layout_fingerprints: tuple[Sha256Text, ...] = Field(min_length=1)
    diffusion_block_counts: tuple[int, ...] = Field(min_length=1)
    automatic_status: Literal["supported", "selection_required", "unsupported"]
    manual_selectable: bool
    routing: SpatialRoutingDefinition


class SpatialModelProfileCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["asteria.spatial-model-profiles.v5"]
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
