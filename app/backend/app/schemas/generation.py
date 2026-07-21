from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import LoraSelection
from app.schemas.image_edit import ImageEditLineage
from app.schemas.remaster import PrimaryOutput, RemasterResponse, RemasterSettingsInput
from app.services.controlnet_mode import ControlNetMode


class AdvancedStyleSelection(BaseModel):
    id: str = ""
    group: str = ""
    name: str = ""
    prompt: str = ""
    negative: str = ""
    source_type: str = "style"
    selected_from_group: str = ""
    selection_seed: int | None = None
    order: int = 0


class ControlNetUnit(BaseModel):
    enabled: bool = False
    model: str = ""
    preprocessor: str = "none"
    image_path: str = ""
    weight: float = Field(default=1.0, ge=0, le=2)
    guidance_start: float = Field(default=0.0, ge=0, le=1)
    guidance_end: float = Field(default=1.0, ge=0, le=1)
    control_mode: ControlNetMode = "Balanced"
    resize_mode: str = "Crop and Resize"
    pixel_perfect: bool = False


class AnimaRegionalPrompt(BaseModel):
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    prompt: str = Field(min_length=1, max_length=4000)


class AnimaLLLiteUnit(BaseModel):
    enabled: bool = False
    model: str = ""
    image_path: str = ""
    mask_image_path: str = ""
    strength: float = Field(default=1.0, ge=-10, le=10)
    guidance_start: float = Field(default=0.0, ge=0, le=1)
    guidance_end: float = Field(default=0.45, ge=0, le=1)
    preserve_wrapper: bool = True
    regional_prompts: list[AnimaRegionalPrompt] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_regional_prompt_colors(self):
        colors = [region.color.lower() for region in self.regional_prompts]
        if len(colors) != len(set(colors)):
            raise ValueError("regional prompt colors must be unique")
        return self


class DetailerUnit(BaseModel):
    enabled: bool = False
    detector: str = "face_yolo"
    prompt: str = ""
    negative_prompt: str = ""
    confidence: float = Field(default=0.3, ge=0, le=1)
    mask_blur: int = Field(default=4, ge=0, le=64)
    denoising_strength: float = Field(default=0.4, ge=0, le=1)
    inpaint_only_masked: bool = True
    guide_size: int = Field(default=512, ge=64, le=2048)
    max_size: int = Field(default=1024, ge=64, le=4096)
    bbox_dilation: int = Field(default=10, ge=0, le=256)
    bbox_crop_factor: float = Field(default=3.0, ge=1, le=10)
    steps: int = Field(default=0, ge=0, le=150)
    cfg_scale: float = Field(default=0.0, ge=0, le=30)
    sampler: str = ""
    scheduler: str = ""


class Txt2ImgRequest(BaseModel):
    profile: str = "sdxl"
    runtime: str | None = None
    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    steps: int = Field(default=30, ge=1, le=150)
    cfg_scale: float = Field(default=6.5, ge=1, le=30)
    distilled_cfg_scale: float = Field(default=3.5, ge=0, le=30)
    seed: int = -1
    subseed: int = -1
    subseed_strength: float = Field(default=0.0, ge=0, le=1)
    seed_resize_from_w: int = Field(default=0, ge=0, le=4096)
    seed_resize_from_h: int = Field(default=0, ge=0, le=4096)
    width: int = Field(default=832, ge=256, le=4096)
    height: int = Field(default=1216, ge=256, le=4096)
    batch_size: int = Field(default=1, ge=1, le=16)
    batch_count: int = Field(default=1, ge=1, le=100)
    sampler: str = "Euler a"
    scheduler: str = "Automatic"
    restore_faces: bool = False
    tiling: bool = False
    vae: str = "Automatic"
    clip_skip: int = Field(default=2, ge=1, le=12)
    refiner_enabled: bool = True
    refiner_checkpoint: str = ""
    refiner_switch_at: float = Field(default=0.8, ge=0, le=1)
    hires_enabled: bool = True
    hires_denoising_strength: float = Field(default=0.38, ge=0, le=1)
    hires_scale: float = Field(default=2.0, ge=1, le=8)
    hires_steps: int = Field(default=18, ge=0, le=150)
    hires_resize_width: int = Field(default=0, ge=0, le=8192)
    hires_resize_height: int = Field(default=0, ge=0, le=8192)
    hires_upscaler: str = "R-ESGAN4x"
    hires_checkpoint: str = "Use same checkpoint"
    hires_sampler: str = "Use same sampler"
    hires_scheduler: str = "Use same scheduler"
    hires_prompt: str = ""
    hires_negative_prompt: str = ""
    hires_cfg_scale: float = Field(default=0.0, ge=0, le=30)
    hires_distilled_cfg_scale: float = Field(default=0.0, ge=0, le=30)
    hires_keep_first_pass: bool = False
    hires_noise_mode: str = ""
    topaz_enabled: bool = False
    topaz_lighting: bool = False
    topaz_color: bool = False
    topaz_sharpen: bool = False
    topaz_denoise: bool = False
    topaz_override: bool = False
    topaz_show_settings: bool = False
    topaz_png_compression: int = Field(default=2, ge=0, le=10)
    topaz_upscale_model: str = "Graphics"
    topaz_upscale_scale: int = Field(default=2, ge=1, le=8)
    advanced_styles: list[AdvancedStyleSelection] = Field(default_factory=list)
    loras: list[LoraSelection] = Field(default_factory=list)
    controlnet_units: list[ControlNetUnit] = Field(default_factory=list)
    anima_lllite_units: list[AnimaLLLiteUnit] = Field(default_factory=list)
    detailer_units: list[DetailerUnit] = Field(default_factory=list)
    model_slot_overrides: dict[str, str] = Field(default_factory=dict)
    remaster: RemasterSettingsInput = Field(default_factory=RemasterSettingsInput)

    @model_validator(mode="before")
    @classmethod
    def apply_profile_scheduler_default(cls, data):
        if not isinstance(data, dict):
            return data
        if data.get("profile", "sdxl") == "flux2" and "scheduler" not in data:
            return {**data, "scheduler": "Flux2"}
        return data


class Img2ImgRequest(Txt2ImgRequest):
    hires_enabled: bool = False
    init_image_path: str = Field(min_length=1)
    denoising_strength: float = Field(default=0.55, ge=0, le=1)
    image_edit_lineage: ImageEditLineage | None = None

    @model_validator(mode="after")
    def validate_image_edit_lineage_current_path(self):
        if self.image_edit_lineage and self.image_edit_lineage.current.path != self.init_image_path:
            raise ValueError("image edit lineage current path must match init_image_path")
        return self


class InpaintRequest(Img2ImgRequest):
    mask_image_path: str = Field(min_length=1)
    mask_blur: int = Field(default=4, ge=0, le=64)
    masked_content: str = "original"
    inpaint_area: str = "whole_picture"


class GenerationResponse(BaseModel):
    id: int
    status: str
    profile: str
    prompt: str
    expanded_prompt: str
    negative_prompt: str
    settings: dict
    model_slots: dict[str, str]
    loras: list[LoraSelection]
    embeddings: list[str]
    output_paths: list[str]
    primary_outputs: list[PrimaryOutput] = Field(default_factory=list)
    remasters: list[RemasterResponse] = Field(default_factory=list)


class XYZAxis(BaseModel):
    target: str
    values: list[str] = Field(min_length=1, max_length=16)
    label: str = ""


class XYZPlotRequest(BaseModel):
    mode: Literal["txt2img", "img2img", "inpaint"] = "txt2img"
    base: Txt2ImgRequest | Img2ImgRequest | InpaintRequest
    axes: list[XYZAxis] = Field(min_length=1, max_length=3)
    contact_sheet: bool = True

    @model_validator(mode="before")
    @classmethod
    def validate_base_for_mode(cls, data):
        if not isinstance(data, dict):
            return data
        mode = data.get("mode", "txt2img")
        request_type = {
            "txt2img": Txt2ImgRequest,
            "img2img": Img2ImgRequest,
            "inpaint": InpaintRequest,
        }.get(mode)
        if request_type is None:
            return data
        return {**data, "base": request_type.model_validate(data.get("base", {}))}


class XYZCellResult(BaseModel):
    index: int
    labels: dict[str, str]
    generation: GenerationResponse
    candidate: dict[str, Any] = Field(default_factory=dict)


class XYZPlotResponse(BaseModel):
    id: int
    status: str
    profile: str
    axes: list[XYZAxis]
    cells: list[XYZCellResult]
    contact_sheet_path: str = ""
    output_paths: list[str]


class GenerationHistoryItem(BaseModel):
    id: int
    status: str
    profile: str
    prompt: str
    expanded_prompt: str
    negative_prompt: str
    settings: dict
    model_slots: dict[str, str]
    loras: list[LoraSelection]
    embeddings: list[str]
    output_paths: list[str]
    rating: int = 0
    favorite: bool = False
    tags: list[str] = Field(default_factory=list)
    generated_tags: list[str] = Field(default_factory=list)
    notes: str = ""
    folder_id: int | None = None
    folder_name: str | None = None
    created_at: str


class GenerationHistoryResponse(BaseModel):
    items: list[GenerationHistoryItem]


class GenerationFolderItem(BaseModel):
    id: int | None = None
    name: str
    system_key: str | None = None
    created_at: str = ""
    item_count: int = 0


class GenerationFolderResponse(BaseModel):
    items: list[GenerationFolderItem]
