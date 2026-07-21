from pathlib import Path
from typing import Any
import hashlib
import sqlite3

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from app.core.config import AppConfig
from app.services.history import HistoryRepository
from app.services.eagle import EagleAdapter, EagleError, EagleImportOptions
from app.services.external_settings import (
    EagleIntegrationSettings,
    ExternalIntegrationSettingsRepository,
    normalize_eagle_base_url,
)
from app.services.api_errors import (
    EAGLE_SETTINGS_INVALID,
    EAGLE_IMPORT_FAILED,
    EAGLE_FOLDER_LIST_FAILED,
    EXTERNAL_SETTINGS_SAVE_FAILED,
    INVALID_OUTPUT_IMAGE,
    OUTPUT_IMAGE_NOT_FOUND,
    OUTPUT_IMAGE_UNREADABLE,
)
from app.services.user_settings import UserSettingsRepository, resolve_output_root

router = APIRouter(prefix="/api/eagle", tags=["eagle"])

_PATH_ACCESS_ERRORS = (TypeError, OSError, ValueError, RuntimeError)
_IMAGE_OPEN_ERRORS = (*_PATH_ACCESS_ERRORS, UnidentifiedImageError)


class EagleStatusResponse(BaseModel):
    base_url: str
    connected: bool
    info: dict[str, Any] | None = None
    error: str = ""


class EagleSettingsResponse(BaseModel):
    base_url: str
    default_folder_id: str = ""


class EagleSettingsRequest(BaseModel):
    base_url: str
    default_folder_id: str = Field(default="", max_length=256)


class EagleFolderItem(BaseModel):
    id: str
    name: str
    path: str
    depth: int


class EagleFolderListResponse(BaseModel):
    items: list[EagleFolderItem]


class EagleImportRequest(BaseModel):
    filename: str
    generation_id: int | None = None
    name: str | None = None
    annotation: str = ""
    tags: list[str] = Field(default_factory=lambda: ["asteria"])
    rating: int = Field(default=0, ge=0, le=5)
    folder_id: str | None = None
    skip_duplicates: bool = True


class EagleImportResponse(BaseModel):
    status: str
    filename: str
    eagle_response: dict[str, Any]
    source_artifact: dict[str, Any]
    duplicate: dict[str, Any] | None = None
    folder_id: str = ""
    transport: str
    folder_applied: bool


def _artifact_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except _PATH_ACCESS_ERRORS:
        return str(path)


def _settings_repository() -> ExternalIntegrationSettingsRepository:
    config = AppConfig()
    return ExternalIntegrationSettingsRepository(config.data_dir / "external-integrations.json")


def _eagle_base_url() -> str:
    return _settings_repository().resolve_eagle_base_url(AppConfig().eagle_base_url)


def _validated_eagle_base_url(value: str) -> str:
    try:
        return normalize_eagle_base_url(value, allow_empty=False)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": EAGLE_SETTINGS_INVALID}) from exc


def _public_eagle_info(info: dict[str, Any]) -> dict[str, Any]:
    public: dict[str, Any] = {}
    status_value = info.get("status")
    if isinstance(status_value, str):
        public["status"] = status_value
    data = info.get("data")
    if isinstance(data, dict):
        safe_data = {
            key: data[key]
            for key in ("version", "prereleaseVersion", "buildVersion", "platform")
            if key in data and (isinstance(data[key], str | int | float | bool) or data[key] is None)
        }
        public["data"] = safe_data
    return public


def _flatten_eagle_folders(payload: Any) -> list[EagleFolderItem]:
    if isinstance(payload, dict):
        roots = payload.get("items") or payload.get("list") or payload.get("folders") or []
    else:
        roots = payload
    if not isinstance(roots, list):
        return []
    items: list[EagleFolderItem] = []
    seen: set[str] = set()

    def visit(nodes: list[Any], parents: list[str], depth: int) -> None:
        if depth > 32 or len(items) >= 10000:
            return
        for node in nodes:
            if not isinstance(node, dict):
                continue
            folder_id = str(node.get("id") or "").strip()
            name = str(node.get("name") or "").strip()
            if not folder_id or not name or folder_id in seen:
                continue
            seen.add(folder_id)
            path_parts = [*parents, name]
            items.append(EagleFolderItem(id=folder_id, name=name, path=" / ".join(path_parts), depth=depth))
            children = node.get("children")
            if isinstance(children, list):
                visit(children, path_parts, depth + 1)

    visit(roots, [], 0)
    return items


def _status_for_base_url(base_url: str) -> EagleStatusResponse:
    adapter = EagleAdapter(base_url)
    try:
        info = adapter.status()
    except EagleError as exc:
        return EagleStatusResponse(base_url=base_url, connected=False, error=str(exc))
    return EagleStatusResponse(base_url=base_url, connected=True, info=_public_eagle_info(info))


def _source_artifact(path: Path) -> dict[str, Any]:
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


def _append_history_event_safely(data_dir: Path, generation_id: int, event: dict[str, Any]) -> None:
    try:
        HistoryRepository(data_dir / "asteria.db").append_external_event(generation_id, event)
    except (OSError, sqlite3.Error, ValueError):
        return


def _output_path(filename: str, generation_id: int | None = None) -> Path:
    config = AppConfig()
    settings = UserSettingsRepository(config.data_dir / "user-settings.json").load().output
    try:
        output_dir = resolve_output_root(config.data_dir, settings).resolve()
        if not Path(filename).is_absolute():
            path = (output_dir / Path(filename)).resolve()
        else:
            path = Path(filename).resolve()
    except _PATH_ACCESS_ERRORS as exc:
        raise HTTPException(status_code=404, detail={"code": OUTPUT_IMAGE_NOT_FOUND}) from exc
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=400, detail={"code": INVALID_OUTPUT_IMAGE})
    if output_dir not in path.parents and path != output_dir:
        if generation_id is None:
            raise HTTPException(status_code=400, detail={"code": INVALID_OUTPUT_IMAGE})
        item = HistoryRepository(config.data_dir / "asteria.db").get(generation_id)
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


@router.get("/status", response_model=EagleStatusResponse)
def status() -> EagleStatusResponse:
    return _status_for_base_url(_eagle_base_url())


@router.get("/settings", response_model=EagleSettingsResponse)
def get_settings() -> EagleSettingsResponse:
    settings = _settings_repository().load().eagle
    return EagleSettingsResponse(base_url=_eagle_base_url(), default_folder_id=settings.default_folder_id)


@router.put("/settings", response_model=EagleSettingsResponse)
def update_settings(request: EagleSettingsRequest) -> EagleSettingsResponse:
    base_url = _validated_eagle_base_url(request.base_url)
    repository = _settings_repository()
    settings = repository.load()
    settings.eagle = EagleIntegrationSettings(
        base_url=base_url,
        default_folder_id=request.default_folder_id,
    )
    try:
        repository.save(settings)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail={"code": EXTERNAL_SETTINGS_SAVE_FAILED}) from exc
    return EagleSettingsResponse(base_url=base_url, default_folder_id=settings.eagle.default_folder_id)


@router.post("/test", response_model=EagleStatusResponse)
def test_connection(request: EagleSettingsRequest) -> EagleStatusResponse:
    return _status_for_base_url(_validated_eagle_base_url(request.base_url))


@router.get("/folders", response_model=EagleFolderListResponse)
def list_folders() -> EagleFolderListResponse:
    try:
        result = EagleAdapter(_eagle_base_url()).list_folders()
    except EagleError as exc:
        raise HTTPException(status_code=502, detail={"code": EAGLE_FOLDER_LIST_FAILED}) from exc
    return EagleFolderListResponse(items=_flatten_eagle_folders(result.get("data")))


@router.post("/import", response_model=EagleImportResponse)
def import_output(request: EagleImportRequest) -> EagleImportResponse:
    config = AppConfig()
    eagle_settings = _settings_repository().load().eagle
    folder_id = request.folder_id if request.folder_id is not None else eagle_settings.default_folder_id
    path = _output_path(request.filename, request.generation_id)
    artifact = _source_artifact(path)
    options = EagleImportOptions(
        name=request.name or path.stem,
        annotation=request.annotation,
        tags=request.tags,
        rating=request.rating,
        folder_id=folder_id,
    )
    adapter = EagleAdapter(_eagle_base_url())
    transport = adapter.import_transport
    duplicate_check_error = ""
    try:
        duplicate = None
        if request.skip_duplicates:
            try:
                duplicate = adapter.find_duplicate_by_name(name=options.name or path.stem, ext=path.suffix)
            except EagleError as exc:
                duplicate_check_error = str(exc)
        if duplicate:
            duplicate_id = str(duplicate.get("id") or "")
            update_result = None
            if duplicate_id:
                update_result = adapter.update_item(
                    duplicate_id,
                    rating=request.rating,
                    tags=request.tags,
                    annotation=request.annotation,
                )
            result = {"status": "duplicate", "data": duplicate}
            if update_result is not None:
                result["update"] = update_result
            if request.generation_id:
                _append_history_event_safely(
                    config.data_dir,
                    request.generation_id,
                    {
                        "adapter": "eagle",
                        "status": "duplicate_skipped",
                        "filename": path.name,
                        "source_artifact": artifact,
                        "duplicate": duplicate,
                        "tags": request.tags,
                        "rating": request.rating,
                        "folder_id": folder_id,
                        "transport": transport,
                        "folder_applied": False,
                        "update": update_result,
                    },
                )
            return EagleImportResponse(
                status="duplicate_skipped",
                filename=path.name,
                eagle_response=result,
                source_artifact=artifact,
                duplicate=duplicate,
                folder_id=folder_id,
                transport=transport,
                folder_applied=False,
            )
        result = adapter.add_image(path, options)
        if duplicate_check_error:
            result = {**result, "duplicate_check_error": duplicate_check_error}
    except EagleError as exc:
        failure_event: dict[str, Any] = {
            "adapter": "eagle",
            "status": "failed",
            "filename": path.name,
            "source_artifact": artifact,
            "folder_id": folder_id,
            "transport": transport,
            "folder_applied": exc.folder_applied,
            "error": str(exc),
        }
        if duplicate_check_error:
            failure_event["duplicate_check_error"] = duplicate_check_error
        if request.generation_id:
            _append_history_event_safely(
                config.data_dir,
                request.generation_id,
                failure_event,
            )
        raise HTTPException(status_code=400, detail={"code": EAGLE_IMPORT_FAILED}) from exc
    if request.generation_id:
        success_event: dict[str, Any] = {
            "adapter": "eagle",
            "status": "success",
            "filename": path.name,
            "source_artifact": artifact,
            "tags": request.tags,
            "rating": request.rating,
            "folder_id": folder_id,
            "transport": transport,
            "folder_applied": True,
            "response": result,
        }
        if duplicate_check_error:
            success_event["duplicate_check_error"] = duplicate_check_error
        _append_history_event_safely(config.data_dir, request.generation_id, success_event)
    return EagleImportResponse(
        status="success",
        filename=path.name,
        eagle_response=result,
        source_artifact=artifact,
        folder_id=folder_id,
        transport=transport,
        folder_applied=True,
    )
