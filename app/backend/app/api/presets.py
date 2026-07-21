from fastapi import APIRouter, HTTPException

from app.core.config import AppConfig
from app.services import api_errors
from app.services.presets import Preset, PresetRepository

router = APIRouter(prefix="/api/presets", tags=["presets"])

ALLOWED_KINDS = {"generation", "model_profile", "controlnet", "detailer", "topaz"}


def _repository() -> PresetRepository:
    return PresetRepository(AppConfig().data_dir / "presets.json")


@router.get("/{kind}")
def list_presets(kind: str, profile: str = "") -> dict:
    if kind not in ALLOWED_KINDS:
        raise HTTPException(status_code=404, detail={"code": api_errors.PRESET_KIND_NOT_FOUND})
    items = _repository().list(kind)
    if profile:
        items = [item for item in items if str(item.data.get("profile", "")).strip() == profile]
    return {"items": items}


@router.post("/{kind}", response_model=Preset)
def save_preset(kind: str, preset: Preset) -> Preset:
    if kind not in ALLOWED_KINDS or preset.kind != kind:
        raise HTTPException(status_code=400, detail={"code": api_errors.INVALID_PRESET_KIND})
    try:
        return _repository().save(preset)
    except OSError as exc:
        raise HTTPException(status_code=409, detail={"code": api_errors.PRESET_SAVE_FAILED}) from exc


@router.delete("/{kind}/{preset_id}")
def delete_preset(kind: str, preset_id: str) -> dict:
    if kind not in ALLOWED_KINDS:
        raise HTTPException(status_code=404, detail={"code": api_errors.PRESET_KIND_NOT_FOUND})
    try:
        deleted = _repository().delete(preset_id)
    except OSError as exc:
        raise HTTPException(status_code=409, detail={"code": api_errors.PRESET_SAVE_FAILED}) from exc
    return {"deleted": deleted}
