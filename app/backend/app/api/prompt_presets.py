from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.core.config import AppConfig
from app.schemas.prompt_input import PromptInput
from app.schemas.prompt_layout import PromptLayout
from app.services import api_errors
from app.services.spatial_prompt_contract import (
    SpatialPromptRecoveryStatus,
    validate_spatial_prompt_contract,
)
from app.services.prompt_presets import PromptPreset, PromptPresetRepository

router = APIRouter(prefix="/api/prompt-presets", tags=["prompt-presets"])


class PromptPresetItem(BaseModel):
    id: str
    name: str
    prompt: str
    negative_prompt: str = ""
    tags: list[str] = Field(default_factory=list)
    prompt_input: PromptInput | None = None
    prompt_layout: PromptLayout | None = None
    spatial_recovery_status: SpatialPromptRecoveryStatus = "ready"
    warning_codes: list[str] = Field(default_factory=list)


class PromptPresetListResponse(BaseModel):
    items: list[PromptPresetItem]


class PromptPresetCreateRequest(BaseModel):
    name: str
    prompt: str
    negative_prompt: str = ""
    tags: list[str] = Field(default_factory=list)
    prompt_input: PromptInput | None = None
    prompt_layout: PromptLayout | None = None

    @model_validator(mode="after")
    def validate_prompt_sections(self) -> "PromptPresetCreateRequest":
        validate_spatial_prompt_contract(
            self.prompt,
            self.prompt_input,
            self.prompt_layout,
            "txt2img",
        )
        return self


def _repository() -> PromptPresetRepository:
    return PromptPresetRepository(AppConfig().data_dir / "prompt-presets.json")


def _item(preset: PromptPreset) -> PromptPresetItem:
    return PromptPresetItem(
        id=preset.id,
        name=preset.name,
        prompt=preset.prompt,
        negative_prompt=preset.negative_prompt,
        tags=list(preset.tags),
        prompt_input=preset.prompt_input,
        prompt_layout=preset.prompt_layout,
        spatial_recovery_status=preset.spatial_recovery_status,
        warning_codes=list(preset.warning_codes),
    )


@router.get("", response_model=PromptPresetListResponse)
def list_presets() -> PromptPresetListResponse:
    return PromptPresetListResponse(items=[_item(preset) for preset in _repository().list()])


@router.post("", response_model=PromptPresetItem)
def create_preset(request: PromptPresetCreateRequest) -> PromptPresetItem:
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail={"code": api_errors.PROMPT_PRESET_PROMPT_REQUIRED})
    try:
        preset = _repository().add(
            request.name,
            request.prompt,
            request.negative_prompt,
            request.tags,
            request.prompt_input,
            request.prompt_layout,
        )
    except OSError as exc:
        raise HTTPException(status_code=409, detail={"code": api_errors.PROMPT_PRESET_SAVE_FAILED}) from exc
    return _item(preset)


@router.delete("/{preset_id}")
def delete_preset(preset_id: str) -> dict:
    try:
        deleted = _repository().delete(preset_id)
    except OSError as exc:
        raise HTTPException(status_code=409, detail={"code": api_errors.PROMPT_PRESET_SAVE_FAILED}) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail={"code": api_errors.PROMPT_PRESET_NOT_FOUND})
    return {"deleted": True, "id": preset_id}
