from pathlib import Path
import threading
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

from app.core.config import AppConfig
from app.services.user_settings import (
    FILENAME_TEMPLATE_FIELDS,
    UserSettings,
    UserSettingsRepository,
    directory_writable_status,
    resolve_merge_preset_dir,
    resolve_output_root,
    validate_filename_template,
)

router = APIRouter(prefix="/api/settings", tags=["settings"])
_settings_update_lock = threading.Lock()


def _repository() -> UserSettingsRepository:
    return UserSettingsRepository(AppConfig().data_dir / "user-settings.json")


class DirectoryValidation(BaseModel):
    resolved: str
    exists: bool
    writable: bool
    code: str
    message: str


class FilenameTemplateValidation(BaseModel):
    valid: bool
    code: str
    message: str
    allowed_fields: list[str]


class SettingsResolvedPaths(BaseModel):
    output_root: str
    merge_preset_dir: str


class SettingsValidation(BaseModel):
    output_root: DirectoryValidation
    merge_preset_dir: DirectoryValidation
    filename_template: FilenameTemplateValidation


class UserSettingsResponse(UserSettings):
    resolved_paths: SettingsResolvedPaths
    validation: SettingsValidation


def _directory_validation(path: Path, *, ensure: bool) -> DirectoryValidation:
    exists, writable, message = directory_writable_status(path, ensure=ensure)
    if writable:
        code = "directory_writable"
    elif message == "Directory does not exist yet.":
        code = "directory_missing"
    elif message == "Path exists but is not a directory.":
        code = "not_directory"
    else:
        code = "directory_not_writable"
    return DirectoryValidation(resolved=str(path), exists=exists, writable=writable, code=code, message=message)


def _resolve_for_response(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _settings_validation_error_code(error: ValidationError) -> str:
    for item in error.errors():
        loc = tuple(str(part) for part in item.get("loc", ()))
        message = str(item.get("msg", ""))
        if "library" in loc or "Library saved filter" in message:
            return "settings_library_saved_filters_invalid"
        if loc[-1:] == ("api_key_env_var",):
            return "settings_ai_api_key_env_var_invalid"
        if loc[-1:] == ("id",):
            return "settings_ai_provider_id_invalid"
        if loc[-1:] == ("timeout_seconds",):
            return "settings_ai_timeout_invalid"
        if "Duplicate AI provider id" in message:
            return "settings_ai_provider_id_duplicate"
        if "Default caption provider id" in message:
            return "settings_ai_default_provider_missing"
    return "settings_validation_failed"


def _validate_settings_payload(payload: dict[str, Any]) -> UserSettings:
    try:
        return UserSettings.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"code": _settings_validation_error_code(exc)}) from exc


def _merge_settings_payload(current: UserSettings, payload: dict[str, Any]) -> dict[str, Any]:
    merged = current.model_dump()
    for section in ("output", "storage", "ai", "library", "appearance", "launcher"):
        if section not in payload:
            continue
        value = payload[section]
        if isinstance(value, dict):
            merged[section] = {**merged[section], **value}
        else:
            merged[section] = value
    return merged


def _settings_response(settings: UserSettings, *, ensure_directories: bool = False) -> UserSettingsResponse:
    config = AppConfig()
    output_root = _resolve_for_response(resolve_output_root(config.data_dir, settings.output))
    merge_preset_dir = _resolve_for_response(resolve_merge_preset_dir(config.data_dir, settings.storage, config.merge_preset_dir))
    template = validate_filename_template(settings.output.filename_template)
    return UserSettingsResponse(
        output=settings.output,
        storage=settings.storage,
        ai=settings.ai,
        library=settings.library,
        appearance=settings.appearance,
        launcher=settings.launcher,
        resolved_paths=SettingsResolvedPaths(
            output_root=str(output_root),
            merge_preset_dir=str(merge_preset_dir),
        ),
        validation=SettingsValidation(
            output_root=_directory_validation(output_root, ensure=ensure_directories),
            merge_preset_dir=_directory_validation(merge_preset_dir, ensure=ensure_directories),
            filename_template=FilenameTemplateValidation(
                valid=template.valid,
                code=template.code,
                message=template.message,
                allowed_fields=sorted(FILENAME_TEMPLATE_FIELDS),
            ),
        ),
    )


@router.get("", response_model=UserSettingsResponse)
def get_settings() -> UserSettingsResponse:
    return _settings_response(_repository().load())


@router.put("", response_model=UserSettingsResponse)
def update_settings(payload: dict[str, Any]) -> UserSettingsResponse:
    with _settings_update_lock:
        repository = _repository()
        settings = _validate_settings_payload(_merge_settings_payload(repository.load(), payload))
        template = validate_filename_template(settings.output.filename_template)
        if not template.valid:
            raise HTTPException(status_code=422, detail={"code": template.code})
        try:
            saved = repository.save(settings)
        except OSError as exc:
            raise HTTPException(status_code=409, detail={"code": "settings_save_failed"}) from exc
    return _settings_response(saved, ensure_directories=True)
