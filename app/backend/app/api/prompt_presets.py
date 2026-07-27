from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import AppConfig
from app.services import api_errors
from app.services.prompt_presets import PromptPresetRepository

router = APIRouter(prefix="/api/prompt-presets", tags=["prompt-presets"])


class PromptPresetItem(BaseModel):
    id: str
    name: str
    prompt: str
    negative_prompt: str = ""
    tags: list[str] = Field(default_factory=list)


class PromptPresetListResponse(BaseModel):
    items: list[PromptPresetItem]


class PromptPresetCreateRequest(BaseModel):
    name: str
    prompt: str
    negative_prompt: str = ""
    tags: list[str] = Field(default_factory=list)


def _repository() -> PromptPresetRepository:
    return PromptPresetRepository(AppConfig().data_dir / "prompt-presets.json")


@router.get("", response_model=PromptPresetListResponse)
def list_presets() -> PromptPresetListResponse:
    return PromptPresetListResponse(items=_repository().list())


@router.post("", response_model=PromptPresetItem)
def create_preset(request: PromptPresetCreateRequest) -> PromptPresetItem:
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail={"code": api_errors.PROMPT_PRESET_PROMPT_REQUIRED})
    try:
        item = _repository().add(request.name, request.prompt, request.negative_prompt, request.tags)
    except OSError as exc:
        raise HTTPException(status_code=409, detail={"code": api_errors.PROMPT_PRESET_SAVE_FAILED}) from exc
    return PromptPresetItem(**item)


@router.delete("/{preset_id}")
def delete_preset(preset_id: str) -> dict:
    try:
        deleted = _repository().delete(preset_id)
    except OSError as exc:
        raise HTTPException(status_code=409, detail={"code": api_errors.PROMPT_PRESET_SAVE_FAILED}) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail={"code": api_errors.PROMPT_PRESET_NOT_FOUND})
    return {"deleted": True, "id": preset_id}
