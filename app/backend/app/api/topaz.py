from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from PIL.PngImagePlugin import PngInfo

from app.core.config import AppConfig
from app.services.external_settings import (
    ExternalIntegrationSettings,
    ExternalIntegrationSettingsRepository,
    TopazIntegrationSettings,
)
from app.services.history import HistoryRepository
from app.services.artifact_lifecycle import artifact_write_operation
from app.services.metadata import MetadataWriter
from app.services.topaz import TopazError, TopazSettings, process_topaz_image
from app.services.user_settings import UserSettingsRepository, resolve_output_root
from app.services.api_errors import (
    EXTERNAL_SETTINGS_SAVE_FAILED,
    INVALID_OUTPUT_IMAGE,
    OUTPUT_IMAGE_NOT_FOUND,
    OUTPUT_IMAGE_UNREADABLE,
    TOPAZ_PROCESS_FAILED,
)

router = APIRouter(prefix="/api/topaz", tags=["topaz"])

_PATH_ACCESS_ERRORS = (TypeError, OSError, ValueError, RuntimeError)
_IMAGE_OPEN_ERRORS = (*_PATH_ACCESS_ERRORS, UnidentifiedImageError)
_SIDECAR_READ_ERRORS = (*_PATH_ACCESS_ERRORS, json.JSONDecodeError)
TOPAZ_HISTORY_REGISTRATION_FAILED = "topaz_history_registration_failed"


class TopazStatusResponse(BaseModel):
    configured_path: str
    installed: bool


class TopazSettingsResponse(BaseModel):
    executable_path: str
    installed: bool


class TopazSettingsRequest(BaseModel):
    executable_path: str


class TopazProcessRequest(BaseModel):
    filename: str
    settings: TopazSettings = Field(default_factory=lambda: TopazSettings(enabled=True))
    generation_id: int | None = None


class TopazProcessResponse(BaseModel):
    status: str
    output_path: str
    metadata_path: str | None = None
    source_artifact: dict[str, Any] | None = None
    output_artifact: dict[str, Any] | None = None
    source_size: dict[str, int] | None = None
    output_size: dict[str, int] | None = None
    logs: list[str] = []
    warnings: list[str] = []


def _topaz_log_messages(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return [str(value)]


def _topaz_warning_messages(logs: list[str]) -> list[str]:
    warning_markers = (
        "warning",
        "error",
        "failed",
        "failure",
        "exception",
        "cuda",
        "tensorrt",
        "driver shutting down",
        "smaller than requested",
    )
    warnings: list[str] = []
    for message in logs:
        lowered = message.lower()
        if any(marker in lowered for marker in warning_markers):
            warnings.append(message)
    return warnings


def _data_dir() -> Path:
    return AppConfig().data_dir


def _output_dir() -> Path:
    config = AppConfig()
    settings = UserSettingsRepository(config.data_dir / "user-settings.json").load().output
    return resolve_output_root(config.data_dir, settings)


def _settings_repository() -> ExternalIntegrationSettingsRepository:
    return ExternalIntegrationSettingsRepository(_data_dir() / "external-integrations.json")


def _topaz_executable() -> Path:
    return _settings_repository().resolve_topaz_executable(AppConfig().topaz_executable)


def _artifact_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except _PATH_ACCESS_ERRORS:
        return str(path)


def _path_exists_safely(path: Path) -> bool:
    try:
        return path.exists()
    except _PATH_ACCESS_ERRORS:
        return False


def _path_is_file_safely(path: Path) -> bool:
    try:
        return path.is_file()
    except _PATH_ACCESS_ERRORS:
        return False


def _first_safe_output_match(output_dir: Path, filename: str) -> Path | None:
    try:
        matches = list(output_dir.rglob(Path(filename).name))
    except _PATH_ACCESS_ERRORS:
        return None
    for match in matches:
        try:
            candidate = match.resolve()
        except _PATH_ACCESS_ERRORS:
            continue
        if _path_exists_safely(candidate) and _path_is_file_safely(candidate):
            return candidate
    return None


def _history_repository() -> HistoryRepository:
    return HistoryRepository(_data_dir() / "asteria.db")


def _append_external_event_safely(repository: HistoryRepository, generation_id: int, event: dict[str, Any]) -> None:
    try:
        repository.append_external_event(generation_id, event)
    except (OSError, sqlite3.Error, ValueError):
        return


def _paths_match(left: object, right: object) -> bool:
    try:
        return Path(str(left)).resolve() == Path(str(right)).resolve()
    except _PATH_ACCESS_ERRORS:
        return str(left) == str(right)


def _derived_source_lineage(item: dict[str, Any], source_path: str) -> tuple[int | None, int]:
    settings = item.get("settings") if isinstance(item.get("settings"), dict) else {}
    primary_outputs = settings.get("primary_outputs")
    if isinstance(primary_outputs, list):
        for output in primary_outputs:
            if not isinstance(output, dict) or not _paths_match(output.get("path"), source_path):
                continue
            try:
                output_index = int(output.get("output_index"))
            except (TypeError, ValueError):
                output_index = None
            try:
                seed = int(output.get("seed", settings.get("seed", -1)))
            except (TypeError, ValueError):
                seed = -1
            return output_index, seed
    derived_outputs = settings.get("derived_outputs")
    if isinstance(derived_outputs, list):
        for output in derived_outputs:
            if not isinstance(output, dict) or not _paths_match(output.get("path"), source_path):
                continue
            try:
                output_index = int(output.get("source_output_index"))
            except (TypeError, ValueError):
                output_index = None
            try:
                seed = int(output.get("seed", settings.get("seed", -1)))
            except (TypeError, ValueError):
                seed = -1
            return output_index, seed
    try:
        return None, int(settings.get("seed", -1))
    except (TypeError, ValueError):
        return None, -1


def _topaz_derived_output(
    item: dict[str, Any],
    *,
    source_path: str,
    output_path: str,
    metadata_path: str | None,
    output_artifact: dict[str, Any],
) -> dict[str, Any]:
    source_output_index, seed = _derived_source_lineage(item, source_path)
    image_size = output_artifact.get("image_size")
    size = image_size if isinstance(image_size, dict) else {}
    return {
        "kind": "topaz",
        "adapter": "topaz",
        "source_path": source_path,
        "path": output_path,
        "metadata_path": metadata_path,
        "source_output_index": source_output_index,
        "seed": seed,
        "size_bytes": int(output_artifact.get("size_bytes") or 0),
        "sha256": str(output_artifact.get("sha256") or ""),
        "width": int(size.get("width") or 0),
        "height": int(size.get("height") or 0),
    }


def _register_topaz_success(
    repository: HistoryRepository,
    generation_id: int,
    *,
    source_path: str,
    output_path: str,
    metadata_path: str | None,
    output_artifact: dict[str, Any],
    event: dict[str, Any],
) -> None:
    def register(item: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        settings = dict(item.get("settings") or {})
        derived = _topaz_derived_output(
            item,
            source_path=source_path,
            output_path=output_path,
            metadata_path=metadata_path,
            output_artifact=output_artifact,
        )
        current_derived = settings.get("derived_outputs")
        derived_outputs = (
            [entry for entry in current_derived if isinstance(entry, dict)]
            if isinstance(current_derived, list)
            else []
        )
        settings["derived_outputs"] = [
            *[entry for entry in derived_outputs if not _paths_match(entry.get("path"), output_path)],
            derived,
        ]

        current_events = settings.get("external_events")
        external_events = (
            [entry for entry in current_events if isinstance(entry, dict)]
            if isinstance(current_events, list)
            else []
        )
        event_payload = dict(event)
        event_payload.setdefault("created_at", datetime.now(UTC).isoformat())
        settings["external_events"] = [*external_events, event_payload]

        next_paths = list(item.get("output_paths") or [])
        for candidate in (output_path, metadata_path):
            if candidate and not any(_paths_match(existing, candidate) for existing in next_paths):
                next_paths.append(candidate)
        return settings, next_paths

    repository.mutate_generation_state(generation_id, register)


def _append_topaz_artifact_warning(
    output_path: str,
    metadata_path: str | None,
    warning: str,
) -> dict[str, Any] | None:
    if not metadata_path:
        return None
    try:
        sidecar_path = Path(metadata_path)
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        post_process = payload.get("post_process")
        if not isinstance(post_process, dict):
            return None
        current_warnings = post_process.get("warnings")
        warnings = [str(value) for value in current_warnings] if isinstance(current_warnings, list) else []
        if warning not in warnings:
            warnings.append(warning)
        post_process["warnings"] = warnings
    except _SIDECAR_READ_ERRORS:
        return None
    updated_artifact = None
    try:
        image_path = Path(output_path)
        if image_path.suffix.lower() == ".png":
            embedded_payload = dict(payload)
            embedded_post_process = dict(post_process)
            embedded_post_process.pop("output_artifact", None)
            embedded_payload["post_process"] = embedded_post_process
            with Image.open(image_path) as image:
                pnginfo = PngInfo()
                for key, value in image.info.items():
                    if isinstance(value, str):
                        pnginfo.add_text(key, value)
                pnginfo.add_text("Asteria", json.dumps(embedded_payload, ensure_ascii=False))
                image.save(image_path, pnginfo=pnginfo)
            updated_artifact = _image_artifact(image_path)
            post_process["output_artifact"] = updated_artifact
    except _IMAGE_OPEN_ERRORS:
        updated_artifact = None
    try:
        sidecar_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except _SIDECAR_READ_ERRORS:
        return updated_artifact
    return updated_artifact


def _image_artifact(path: Path) -> dict[str, Any]:
    try:
        content = path.read_bytes()
    except _PATH_ACCESS_ERRORS as exc:
        raise HTTPException(status_code=409, detail={"code": OUTPUT_IMAGE_UNREADABLE}) from exc
    image_size = None
    try:
        with Image.open(path) as image:
            image_size = {"width": int(image.width), "height": int(image.height)}
    except _IMAGE_OPEN_ERRORS:
        image_size = None
    return {
        "path": _artifact_path(path),
        "filename": path.name,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "image_size": image_size,
    }


def _output_path(filename: str, generation_id: int | None = None) -> Path:
    try:
        output_dir = _output_dir().resolve()
        if Path(filename).is_absolute():
            path = Path(filename).resolve()
        else:
            path = (output_dir / Path(filename)).resolve()
    except _PATH_ACCESS_ERRORS as exc:
        raise HTTPException(status_code=404, detail={"code": OUTPUT_IMAGE_NOT_FOUND}) from exc
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=400, detail={"code": INVALID_OUTPUT_IMAGE})
    if output_dir not in path.parents and path != output_dir:
        if generation_id is None:
            raise HTTPException(status_code=400, detail={"code": INVALID_OUTPUT_IMAGE})
        item = _history_repository().get(generation_id)
        referenced = False
        if item:
            for output_path in item["output_paths"]:
                try:
                    if Path(output_path).resolve() == path:
                        referenced = True
                        break
                except _PATH_ACCESS_ERRORS:
                    continue
        if not referenced:
            raise HTTPException(status_code=400, detail={"code": INVALID_OUTPUT_IMAGE})
    if not _path_exists_safely(path):
        match = _first_safe_output_match(output_dir, filename)
        if match is not None:
            path = match
    if not _path_exists_safely(path) or not _path_is_file_safely(path):
        raise HTTPException(status_code=404, detail={"code": OUTPUT_IMAGE_NOT_FOUND})
    return path


def _write_topaz_sidecar(
    generation_id: int,
    input_path: Path,
    output_path: Path,
    settings: TopazSettings,
    source_artifact: dict[str, Any],
    source_size: dict[str, int] | None,
    output_size: dict[str, int] | None,
    logs: list[str],
    warnings: list[str],
) -> tuple[str | None, dict[str, Any]]:
    repository = _history_repository()
    item = repository.get(generation_id)
    if item is None:
        return None, _image_artifact(output_path)
    sidecars = MetadataWriter().write_generation_sidecars(
        {
            "generation_id": generation_id,
            "status": item["status"],
            "profile": item["profile"],
            "prompt": item["prompt"],
            "expanded_prompt": item["expanded_prompt"],
            "negative_prompt": item["negative_prompt"],
            "settings": {
                **item["settings"],
                "topaz_manual_process": "completed",
                "topaz_manual_source_path": str(input_path),
                "topaz_manual_settings": settings.model_dump(),
            },
            "model_slots": item["model_slots"],
            "loras": item["loras"],
            "embeddings": item["embeddings"],
            "metadata_tags": [*item["generated_tags"], "topaz"],
            "post_process": {
                "adapter": "topaz",
                "source_path": str(input_path),
                "output_path": str(output_path),
                "source_artifact": source_artifact,
                "source_size": source_size,
                "output_size": output_size,
                "settings": settings.model_dump(),
                "logs": logs,
                "warnings": warnings,
            },
        },
        [str(output_path)],
    )
    output_artifact = _image_artifact(output_path)
    if sidecars:
        sidecar_path = Path(sidecars[0])
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            post_process = payload.get("post_process")
            if isinstance(post_process, dict):
                post_process["output_artifact"] = output_artifact
                sidecar_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except _SIDECAR_READ_ERRORS:
            pass
    return (sidecars[0] if sidecars else None, output_artifact)


@router.get("/status", response_model=TopazStatusResponse)
def status() -> TopazStatusResponse:
    executable = _topaz_executable()
    return TopazStatusResponse(configured_path=str(executable), installed=_path_is_file_safely(executable))


@router.get("/settings", response_model=TopazSettingsResponse)
def get_settings() -> TopazSettingsResponse:
    executable = _topaz_executable()
    return TopazSettingsResponse(executable_path=str(executable), installed=_path_is_file_safely(executable))


@router.put("/settings", response_model=TopazSettingsResponse)
def update_settings(request: TopazSettingsRequest) -> TopazSettingsResponse:
    path = request.executable_path.strip()
    repository = _settings_repository()
    settings = repository.load()
    settings.topaz = TopazIntegrationSettings(executable_path=path)
    try:
        repository.save(settings)
    except _PATH_ACCESS_ERRORS as exc:
        raise HTTPException(status_code=409, detail={"code": EXTERNAL_SETTINGS_SAVE_FAILED}) from exc
    executable = repository.resolve_topaz_executable(AppConfig().topaz_executable)
    return TopazSettingsResponse(executable_path=str(executable), installed=_path_is_file_safely(executable))


@router.post("/process", response_model=TopazProcessResponse)
@artifact_write_operation
def process(request: TopazProcessRequest) -> TopazProcessResponse:
    executable = _topaz_executable()
    path = _output_path(request.filename, request.generation_id)
    source_artifact = _image_artifact(path)
    repository = _history_repository()
    try:
        result = process_topaz_image(
            executable=executable,
            input_path=path,
            output_dir=path.parent,
            settings=request.settings,
        )
    except TopazError as exc:
        if request.generation_id:
            _append_external_event_safely(
                repository,
                request.generation_id,
                {
                    "adapter": "topaz",
                    "status": "failed",
                    "filename": path.name,
                    "source_artifact": source_artifact,
                    "error": str(exc),
                    "settings": request.settings.model_dump(),
                },
        )
        raise HTTPException(status_code=400, detail={"code": TOPAZ_PROCESS_FAILED}) from exc

    output_path = Path(str(result["output_path"]))
    source_size = result.get("source_size") if isinstance(result.get("source_size"), dict) else None
    output_size = result.get("output_size") if isinstance(result.get("output_size"), dict) else None
    result_logs = _topaz_log_messages(result.get("logs"))
    result_warnings = _topaz_warning_messages(result_logs)
    if request.generation_id:
        metadata_path, output_artifact = _write_topaz_sidecar(
            request.generation_id,
            path,
            output_path,
            request.settings,
            source_artifact,
            source_size,
            output_size,
            result_logs,
            result_warnings,
        )
    else:
        metadata_path = None
        output_artifact = _image_artifact(output_path)
    if request.generation_id:
        try:
            _register_topaz_success(
                repository,
                request.generation_id,
                source_path=str(path),
                output_path=str(output_path),
                metadata_path=metadata_path,
                output_artifact=output_artifact,
                event={
                    "adapter": "topaz",
                    "status": "success",
                    "filename": path.name,
                    "source_artifact": source_artifact,
                    "output_artifact": output_artifact,
                    "output_path": str(output_path),
                    "metadata_path": metadata_path,
                    "source_size": source_size,
                    "output_size": output_size,
                    "settings": request.settings.model_dump(),
                    "logs": result_logs,
                    "warnings": result_warnings,
                },
            )
        except (OSError, sqlite3.Error, TypeError, ValueError):
            if TOPAZ_HISTORY_REGISTRATION_FAILED not in result_warnings:
                result_warnings.append(TOPAZ_HISTORY_REGISTRATION_FAILED)
            updated_artifact = _append_topaz_artifact_warning(
                str(output_path),
                metadata_path,
                TOPAZ_HISTORY_REGISTRATION_FAILED,
            )
            if updated_artifact is not None:
                output_artifact = updated_artifact
    return TopazProcessResponse(
        status="success",
        output_path=str(output_path),
        metadata_path=metadata_path,
        source_artifact=source_artifact,
        output_artifact=output_artifact,
        source_size=source_size,
        output_size=output_size,
        logs=result_logs,
        warnings=result_warnings,
    )
