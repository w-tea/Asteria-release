from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ArchitectureDetectionStatus(StrEnum):
    SUPPORTED = "supported"
    PROVISIONAL_UNSUPPORTED = "provisional_unsupported"
    UNSUPPORTED = "unsupported"


class AnimaArchitectureComponent(StrEnum):
    DIFFUSION_MODEL = "diffusion_model"
    NATIVE_TEXT_ENCODER = "native_text_encoder"
    QWEN35_TEXT_ENCODER = "qwen35_text_encoder"
    PROGRESSIVE_ADAPTER = "progressive_adapter"
    UNKNOWN = "unknown"


class AnimaArchitectureVariant(StrEnum):
    BASE_28 = "anima_base_28"
    V2_9B_40 = "anima_2_9b_40"
    V3_8B_52 = "anima_3_8b_52"


class AnimaArchitectureDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["anima"] = "anima"
    component: AnimaArchitectureComponent = AnimaArchitectureComponent.UNKNOWN
    variant_id: AnimaArchitectureVariant | None = None
    diffusion_block_count: int | None = None
    llm_adapter_block_count: int | None = None
    progressive_adapter_required: bool = False
    progressive_adapter_signature: str | None = None
    text_encoder_requirements: tuple[str, ...] = ()
    key_layout_signature: str | None = None
    merge_schema_id: str | None = None
    detection_method: Literal["safetensors_header"] = "safetensors_header"
    status: ArchitectureDetectionStatus
    unsupported_reason: str | None = None
