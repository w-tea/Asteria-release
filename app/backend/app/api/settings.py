import asyncio
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
import sqlite3
import threading
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ValidationError
from starlette.concurrency import run_in_threadpool

from app.adapters.image_check_codex import check_image_with_codex
from app.adapters.image_check_llm import check_image_with_llm
from app.core.config import AppConfig
from app.schemas.image_check import ImageCheckTestRequest, ImageCheckTestResponse
from app.services.history import HistoryRepository
from app.services.image_check_operation import capture_image_check_operation
from app.services.image_conditions import prepare_registered_image
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


@router.get("/image-check/codex")
def image_check_codex_status() -> dict:
    from app.adapters.image_check_codex import discover_codex, cached_image_models

    executable, version, reason = discover_codex()
    return {"available": executable is not None, "version": version, "reason_code": reason,
            "models": cached_image_models() if executable else []}


def _repository() -> UserSettingsRepository:
    return UserSettingsRepository(AppConfig().data_dir / "user-settings.json")


@router.post("/image-check/test", response_model=ImageCheckTestResponse)
async def test_image_check(body: ImageCheckTestRequest, request: Request) -> ImageCheckTestResponse:
    config = AppConfig()
    settings = _repository().load().model_copy(update={"image_check": body.image_check})
    operation = capture_image_check_operation(settings, data_dir=config.data_dir, enabled=True)
    history_path = config.data_dir / "asteria.db"

    def prepare():
        try:
            item = HistoryRepository.registered_output_source(history_path, body.generation_id)
            return prepare_registered_image(item, body.output_path, operation.output_root)
        except (ValueError, OSError, sqlite3.Error):
            raise HTTPException(404, detail={"code": "image_check_test_image_unavailable"}) from None

    image = await run_in_threadpool(prepare)
    source, root = Path(image.receipt.source_path), Path(image.receipt.output_root)
    cancel_event = asyncio.Event()
    if operation.settings.method == "codex":
        pending = check_image_with_codex(operation.settings, source, output_root=root, cancel_event=cancel_event)
    else:
        pending = check_image_with_llm(operation.settings, operation.providers, source,
            output_root=root, cancel_event=cancel_event)
    task = asyncio.create_task(pending)
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.2)
            if await request.is_disconnected():
                raise HTTPException(499, detail={"code": "image_check_cancelled"})
        result = await task
        try:
            current = await run_in_threadpool(prepare)
            checked_image = result.receipt.image
            digest = checked_image.source_sha256 if checked_image else image.receipt.source_sha256
            source_current = current.receipt.source_sha256 == digest
        except HTTPException:
            source_current = False
        return ImageCheckTestResponse(generation_id=body.generation_id, output_path=str(source),
            assessment=result.assessment, receipt=asdict(result.receipt), source_current=source_current)
    finally:
        cancel_event.set()
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task


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
        if "image_check" in loc:
            return "settings_image_check_invalid"
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
    for section in ("output", "storage", "ai", "image_check", "library", "appearance", "launcher"):
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
        image_check=settings.image_check,
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
