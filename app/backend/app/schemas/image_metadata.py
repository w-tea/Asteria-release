from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetadataTextAbsent(StrictModel):
    presence: Literal["absent"] = "absent"
    value: None = None


class MetadataTextPresent(StrictModel):
    presence: Literal["present"] = "present"
    value: str


MetadataTextValue = Annotated[
    MetadataTextAbsent | MetadataTextPresent,
    Field(discriminator="presence"),
]

ResourceStatus = Literal["available", "missing", "ambiguous", "unsupported"]


class ProfileModelSlotReference(StrictModel):
    kind: Literal["checkpoint", "vae"]
    application: Literal["model_slot"] = "model_slot"
    slot_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    status: ResourceStatus
    resolved_id: str | None = None
    resolved_path: str | None = None


class RefinerCheckpointReference(StrictModel):
    kind: Literal["checkpoint"] = "checkpoint"
    application: Literal["refiner_checkpoint"] = "refiner_checkpoint"
    source_name: str = Field(min_length=1)
    status: ResourceStatus
    resolved_id: str | None = None
    resolved_path: str | None = None


class HiresCheckpointReference(StrictModel):
    kind: Literal["checkpoint"] = "checkpoint"
    application: Literal["hires_checkpoint"] = "hires_checkpoint"
    source_name: str = Field(min_length=1)
    status: ResourceStatus
    resolved_id: str | None = None
    resolved_path: str | None = None


class LoRAReference(StrictModel):
    kind: Literal["lora"] = "lora"
    application: Literal["lora_stack"] = "lora_stack"
    source_name: str = Field(min_length=1)
    status: ResourceStatus
    resolved_id: str | None = None
    resolved_path: str | None = None
    order: int = Field(ge=0)
    enabled: bool = True
    weight: float = 1.0
    clip_weight: float = 1.0
    block_weight: str = ""
    block_weight_preset: str = ""
    hierarchy_mode: str = "chain"


class InformationalReference(StrictModel):
    kind: Literal["embedding", "preset", "workflow"]
    application: Literal["prompt_embedded", "advanced_style", "preview_only"]
    source_name: str = Field(min_length=1)
    status: ResourceStatus
    resolved_id: str | None = None
    resolved_path: str | None = None
    order: int | None = Field(default=None, ge=0)
    weight: float | None = None


ResourceReference = Annotated[
    ProfileModelSlotReference
    | RefinerCheckpointReference
    | HiresCheckpointReference
    | LoRAReference
    | InformationalReference,
    Field(discriminator="application"),
]


class CompatibleTxt2ImgSettings(StrictModel):
    profile: str | None = None
    runtime_adapter: str | None = None
    steps: int | None = Field(default=None, ge=1, le=150)
    cfg_scale: float | None = Field(default=None, ge=0, le=30)
    distilled_cfg_scale: float | None = Field(default=None, ge=0, le=30)
    width: int | None = Field(default=None, ge=1, le=8192)
    height: int | None = Field(default=None, ge=1, le=8192)
    batch_size: int | None = Field(default=None, ge=1, le=16)
    batch_count: int | None = Field(default=None, ge=1, le=100)
    sampler: str | None = None
    scheduler: str | None = None
    restore_faces: bool | None = None
    tiling: bool | None = None
    clip_skip: int | None = Field(default=None, ge=1, le=12)
    refiner_enabled: bool | None = None
    refiner_switch_at: float | None = Field(default=None, ge=0, le=1)
    hires_enabled: bool | None = None
    hires_denoising_strength: float | None = Field(default=None, ge=0, le=1)
    hires_scale: float | None = Field(default=None, ge=1, le=8)
    hires_steps: int | None = Field(default=None, ge=0, le=150)
    hires_resize_width: int | None = Field(default=None, ge=0, le=8192)
    hires_resize_height: int | None = Field(default=None, ge=0, le=8192)
    hires_upscaler: str | None = None
    hires_sampler: str | None = None
    hires_scheduler: str | None = None
    hires_prompt: str | None = None
    hires_negative_prompt: str | None = None
    hires_cfg_scale: float | None = Field(default=None, ge=0, le=30)
    hires_distilled_cfg_scale: float | None = Field(default=None, ge=0, le=30)
    hires_keep_first_pass: bool | None = None
    detailer_units: list[dict[str, Any]] | None = None
    topaz_enabled: bool | None = None
    topaz_lighting: bool | None = None
    topaz_color: bool | None = None
    topaz_sharpen: bool | None = None
    topaz_denoise: bool | None = None
    topaz_override: bool | None = None
    topaz_show_settings: bool | None = None
    topaz_png_compression: int | None = Field(default=None, ge=0, le=10)


class ImageMetadataSource(StrictModel):
    filename: str
    format: Literal["png", "jpeg", "webp"]
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class ImageMetadataPreview(StrictModel):
    preview_token: str = Field(min_length=16)
    expires_at: datetime
    source: ImageMetadataSource
    source_family: Literal["asteria", "a1111_forge", "comfyui", "unknown"]
    status: Literal["complete", "partial", "unsupported"]
    positive_prompt: MetadataTextValue
    negative_prompt: MetadataTextValue
    settings: CompatibleTxt2ImgSettings = Field(default_factory=CompatibleTxt2ImgSettings)
    seed: int | None = None
    resources: list[ResourceReference] = Field(default_factory=list)
    recognized_codes: list[str] = Field(default_factory=list)
    warning_codes: list[str] = Field(default_factory=list)
    unsupported_codes: list[str] = Field(default_factory=list)


class ImageMetadataLibraryPreviewRequest(StrictModel):
    generation_id: int = Field(ge=1)
    output_path: str = Field(min_length=1)


class ImageMetadataUseAsInputRequest(StrictModel):
    preview_token: str = Field(min_length=16)


class ImageMetadataUseAsInputResponse(StrictModel):
    path: str
    filename: str


class ImageMetadataPickerSelected(StrictModel):
    status: Literal["selected"] = "selected"
    preview: ImageMetadataPreview


class ImageMetadataPickerCancelled(StrictModel):
    status: Literal["cancelled"] = "cancelled"


class ImageMetadataPickerUnavailable(StrictModel):
    status: Literal["unavailable"] = "unavailable"
    code: Literal["native_picker_unavailable"] = "native_picker_unavailable"


ImageMetadataPickerResult = Annotated[
    ImageMetadataPickerSelected | ImageMetadataPickerCancelled | ImageMetadataPickerUnavailable,
    Field(discriminator="status"),
]
