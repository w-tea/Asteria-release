from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import LoraSelection


RemasterProfile = Literal["sdxl", "anima", "flux1", "flux2"]
RemasterStatus = Literal[
    "queued",
    "running",
    "completed",
    "completed_with_warnings",
    "failed",
    "cancelled",
]
LatentInterpolation = Literal["bilinear", "bicubic", "nearest-exact"]


class RemasterSettingsInput(BaseModel):
    enabled: bool = False
    target_scale: float = Field(default=2.0, ge=1.0, le=4.0)
    anchor_upscaler: str = "R-ESGAN4x"
    buffer_ratio: float = Field(default=0.75, ge=0.5, le=0.95)
    latent_interpolation: LatentInterpolation = "bilinear"
    denoise: float | None = Field(default=None, ge=0, le=1)
    steps: int = Field(default=20, ge=1, le=150)
    sampler: str = ""
    scheduler: str = ""
    keep_original: bool = True
    create_comparison: bool = True


class ResolvedRemasterSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool
    target_scale: float
    anchor_upscaler: str
    buffer_ratio: float
    latent_interpolation: LatentInterpolation
    denoise: float
    steps: int
    sampler: str
    scheduler: str
    keep_original: bool
    create_comparison: bool


_PROFILE_DEFAULTS: dict[str, tuple[float, str, str]] = {
    "sdxl": (0.55, "DPM++ 3M SDE", "Karras"),
    "anima": (0.08, "ER SDE", "Beta"),
    "flux1": (0.05, "Euler", "Simple"),
    "flux2": (0.35, "Euler", "Flux2"),
}


def resolve_remaster_settings(
    profile: str,
    settings: RemasterSettingsInput,
) -> ResolvedRemasterSettings:
    defaults = _PROFILE_DEFAULTS.get(profile)
    if defaults is None:
        raise ValueError(f"unsupported remaster profile: {profile}")
    denoise, sampler, scheduler = defaults
    values = settings.model_dump()
    values["denoise"] = denoise if settings.denoise is None else settings.denoise
    values["sampler"] = settings.sampler or sampler
    values["scheduler"] = settings.scheduler or scheduler
    return ResolvedRemasterSettings.model_validate(values)


class PrimaryOutput(BaseModel):
    path: str
    seed: int
    output_index: int
    runtime_job_id: str = ""


class RemasterRequest(BaseModel):
    source_image_path: str
    source_generation_id: int | None = None
    source_output_index: int | None = None
    profile: RemasterProfile
    runtime: str = "comfy_workflow"
    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    seed: int = -1
    cfg_scale: float = Field(default=6.5, ge=1, le=30)
    distilled_cfg_scale: float = Field(default=3.5, ge=0, le=30)
    model_slot_overrides: dict[str, str] = Field(default_factory=dict)
    loras: list[LoraSelection] = Field(default_factory=list)
    remaster: RemasterSettingsInput


class RemasterResponse(BaseModel):
    id: int
    status: RemasterStatus
    source_generation_id: int | None = None
    source_image_path: str
    source_output_index: int | None = None
    output_path: str = ""
    contact_sheet_path: str = ""
    report_path: str = ""
    settings: dict = Field(default_factory=dict)
    model_slots: dict[str, str] = Field(default_factory=dict)
    loras: list[LoraSelection] = Field(default_factory=list)
    runtime_job_id: str = ""
    runtime_job_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
