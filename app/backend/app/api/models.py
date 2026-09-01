from pathlib import Path, PurePosixPath
from dataclasses import replace
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core.config import AppConfig
from app.services.api_errors import (
    MANAGED_ARTIFACT_PATH_TOO_LONG,
    MANAGED_ARTIFACT_ROLLBACK_INCOMPLETE,
    MODEL_DEPENDENCY_INSTALL_FAILED,
    MODEL_DEPENDENCY_LICENSE_CONFIRMATION_REQUIRED,
    MODEL_DEPENDENCY_NOT_FOUND,
    MODEL_PROFILE_NOT_FOUND,
    MODEL_SELECTION_PROFILE_MISMATCH,
    MODEL_SELECTION_SAVE_FAILED,
    MODEL_SELECTION_SLOT_NOT_FOUND,
    MODEL_STORAGE_SETTINGS_SAVE_FAILED,
)
from app.services.managed_artifacts import (
    ManagedArtifactPathTooLong,
    ManagedArtifactRollbackIncomplete,
)
from app.services.model_dependencies import (
    LicenseConfirmationRequired,
    ModelBootstrapResult,
    ModelBootstrapper,
    ModelBootstrapProgress,
    ModelDependencyInstaller,
    ModelDependencyError,
    ModelDependencyNotFound,
    ModelDependencyRepository,
    ModelDependencyStatus,
)
from app.services.model_fingerprints import ModelFingerprintService
from app.services.model_library import ModelLibraryItem, ModelLinkHealth
from app.services.model_library_cache import (
    clear_model_library_cache,
    get_model_library,
    refresh_model_library,
)
from app.services.model_profiles import ModelProfile, ModelProfileRepository, ModelSlot
from app.services.model_selections import ModelSelection, ModelSelectionRepository
from app.services.model_storage import (
    ModelStorageSettings,
    ModelStorageSettingsRepository,
    external_model_root_enabled,
    resolve_model_root,
)

router = APIRouter(prefix="/api/models", tags=["models"])
_bootstrap_lock = threading.Lock()
_bootstrap_running = False
_bootstrap_result: ModelBootstrapResult | None = None
_bootstrap_progress = ModelBootstrapProgress(
    current_dependency_id=None,
    completed_count=0,
    total_count=0,
)


class ModelDependencyStatusResponse(BaseModel):
    id: str
    label: str
    profile: str
    type: str
    provider: str
    status: str
    path: Path
    slot_id: str | None
    source_url: str
    size_bytes: int | None
    local_size_bytes: int | None
    checksum_sha256: str | None
    local_checksum_sha256: str | None
    license_id: str | None
    requires_license_confirmation: bool
    auto_install: bool


class ModelDependencyListResponse(BaseModel):
    dependencies: list[ModelDependencyStatusResponse]


class InstallModelDependencyRequest(BaseModel):
    license_confirmed: bool = False


class BootstrapModelDependenciesRequest(BaseModel):
    profile: str | None = None


class BootstrapModelDependenciesResponse(BaseModel):
    running: bool = False
    current_dependency_id: str | None = None
    completed_count: int = 0
    total_count: int = 0
    installed: list[ModelDependencyStatusResponse]
    skipped: list[ModelDependencyStatusResponse]
    failed: dict[str, str]


class ModelProfileResponse(BaseModel):
    id: str
    label: str
    dependency_ids: list[str]
    model_slots: list["ModelSlotResponse"]
    setting_groups: list[str]


class ModelProfileListResponse(BaseModel):
    profiles: list[ModelProfileResponse]


class ModelSlotResponse(BaseModel):
    id: str
    label: str
    type: str
    required: bool


class ModelSelectionRequest(BaseModel):
    source_profile: str
    slots: dict[str, str]


class ModelSelectionResponse(BaseModel):
    profile: str
    slots: dict[str, str]


class ModelLibraryItemResponse(BaseModel):
    type: str
    name: str
    path: Path
    relative_path: Path
    size_bytes: int
    metadata: dict[str, str]
    source: str
    root_id: str = "asteria-local"
    stable_model_id: str | None = None
    fingerprint_algorithm: str | None = None
    fingerprint_version: int | None = None
    fingerprint_value: str | None = None


class ModelLinkHealthResponse(BaseModel):
    linked_model_count: int
    broken_link_count: int
    unreadable_link_count: int
    loop_link_count: int
    duplicate_target_count: int
    skipped_file_count: int


class ModelLibraryResponse(BaseModel):
    items: list[ModelLibraryItemResponse]
    link_health: ModelLinkHealthResponse


class ModelLibraryRefreshResponse(ModelLibraryResponse):
    counts: dict[str, int]


class ModelStorageSettingsResponse(BaseModel):
    root: Path
    exists: bool


class ModelStorageSettingsRequest(BaseModel):
    root: str


def get_model_root() -> Path:
    config = AppConfig()
    return resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)


def get_repository() -> ModelDependencyRepository:
    config = AppConfig()
    return ModelDependencyRepository(
        manifest_path=config.model_dependencies_path,
        model_root=get_model_root(),
        fingerprint_service=ModelFingerprintService(
            config.data_dir / "config" / "model-fingerprints.json"
        ),
    )


def get_profile_repository() -> ModelProfileRepository:
    config = AppConfig()
    return ModelProfileRepository(config.model_profiles_path)


def get_selection_repository() -> ModelSelectionRepository:
    config = AppConfig()
    return ModelSelectionRepository(config.model_selections_path)


def require_model_profile(profile: str) -> ModelProfile:
    active_profile = next((item for item in get_profile_repository().list() if item.id == profile), None)
    if active_profile is None:
        raise HTTPException(status_code=400, detail={"code": MODEL_PROFILE_NOT_FOUND, "profile": profile})
    return active_profile


def autofill_selection(profile: str) -> ModelSelection:
    repository = get_repository()
    active_profile = require_model_profile(profile)
    allowed_slot_ids = {slot.id for slot in active_profile.model_slots}
    return get_selection_repository().autofill(profile, repository.list_status(), allowed_slot_ids)


def to_response(status: ModelDependencyStatus) -> ModelDependencyStatusResponse:
    if status.provider == "manual" and status.slot_id and status.status != "installed":
        selection = get_selection_repository().get(status.profile)
        selected_path = Path(selection.slots.get(status.slot_id, ""))
        try:
            selected_is_file = selected_path.is_file()
        except OSError:
            status = replace(
                status,
                status="unreadable",
                path=selected_path,
                local_size_bytes=None,
            )
            return ModelDependencyStatusResponse(**status.__dict__)
        if selected_is_file:
            try:
                local_size = selected_path.stat().st_size
            except OSError:
                status = replace(
                    status,
                    status="unreadable",
                    path=selected_path,
                    local_size_bytes=None,
                )
                return ModelDependencyStatusResponse(**status.__dict__)
            resolved_status = "installed"
            if status.size_bytes is not None and local_size != status.size_bytes:
                resolved_status = "size_mismatch"
            status = replace(
                status,
                status=resolved_status,
                path=selected_path,
                local_size_bytes=local_size,
            )
    return ModelDependencyStatusResponse(**status.__dict__)


def bootstrap_to_response(result: ModelBootstrapResult) -> BootstrapModelDependenciesResponse:
    with _bootstrap_lock:
        progress = _bootstrap_progress
    return BootstrapModelDependenciesResponse(
        running=False,
        current_dependency_id=progress.current_dependency_id,
        completed_count=progress.completed_count,
        total_count=progress.total_count,
        installed=[to_response(status) for status in result.installed],
        skipped=[to_response(status) for status in result.skipped],
        failed=result.failed,
    )


def empty_bootstrap_response(running: bool) -> BootstrapModelDependenciesResponse:
    with _bootstrap_lock:
        progress = _bootstrap_progress
    return BootstrapModelDependenciesResponse(
        running=running,
        current_dependency_id=progress.current_dependency_id,
        completed_count=progress.completed_count,
        total_count=progress.total_count,
        installed=[],
        skipped=[],
        failed={},
    )


def profile_to_response(profile: ModelProfile) -> ModelProfileResponse:
    return ModelProfileResponse(
        id=profile.id,
        label=profile.label,
        dependency_ids=profile.dependency_ids,
        model_slots=[slot_to_response(slot) for slot in profile.model_slots],
        setting_groups=profile.setting_groups,
    )


def slot_to_response(slot: ModelSlot) -> ModelSlotResponse:
    return ModelSlotResponse(**slot.__dict__)


def library_item_to_response(item: ModelLibraryItem) -> ModelLibraryItemResponse:
    return ModelLibraryItemResponse(**item.__dict__)


def link_health_to_response(health: ModelLinkHealth) -> ModelLinkHealthResponse:
    return ModelLinkHealthResponse(**health.__dict__)


def library_counts(items: list[ModelLibraryItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.type] = counts.get(item.type, 0) + 1
    return counts


def path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def path_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def path_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def normalized_library_asset_path(path: str) -> str | None:
    normalized = PurePosixPath(path.replace("\\", "/"))
    if normalized.is_absolute() or not normalized.parts:
        return None
    if any(part in {"", ".", ".."} for part in normalized.parts):
        return None
    return normalized.as_posix()


@router.get("/storage", response_model=ModelStorageSettingsResponse)
def get_storage_settings() -> ModelStorageSettingsResponse:
    root = get_model_root()
    return ModelStorageSettingsResponse(root=root, exists=path_is_dir(root))


@router.put("/storage", response_model=ModelStorageSettingsResponse)
def update_storage_settings(request: ModelStorageSettingsRequest) -> ModelStorageSettingsResponse:
    config = AppConfig()
    if not external_model_root_enabled():
        return ModelStorageSettingsResponse(root=config.model_dir, exists=path_is_dir(config.model_dir))
    root = Path(request.root.strip()) if request.root.strip() else config.model_dir
    try:
        ModelStorageSettingsRepository(config.data_dir / "model-storage-settings.json").save(
            ModelStorageSettings(root=str(root))
        )
    except OSError as exc:
        raise HTTPException(status_code=409, detail={"code": MODEL_STORAGE_SETTINGS_SAVE_FAILED}) from exc
    clear_model_library_cache()
    return ModelStorageSettingsResponse(root=root, exists=path_is_dir(root))


@router.get("/dependencies", response_model=ModelDependencyListResponse)
def list_dependencies(profile: str | None = None) -> ModelDependencyListResponse:
    statuses = get_repository().list_status()
    if profile is not None:
        statuses = [status for status in statuses if status.profile == profile or status.profile == "shared"]
    return ModelDependencyListResponse(
        dependencies=[to_response(status) for status in statuses]
    )


@router.get("/profiles", response_model=ModelProfileListResponse)
def list_profiles() -> ModelProfileListResponse:
    return ModelProfileListResponse(
        profiles=[profile_to_response(profile) for profile in get_profile_repository().list()]
    )


@router.get("/library", response_model=ModelLibraryResponse)
def list_library(type: str | None = None) -> ModelLibraryResponse:
    library = get_model_library(get_model_root())
    items = library.by_type(type) if type else library.items
    return ModelLibraryResponse(
        items=[library_item_to_response(item) for item in items],
        link_health=link_health_to_response(library.link_health),
    )


@router.post("/library/refresh", response_model=ModelLibraryRefreshResponse)
def refresh_library(type: str | None = None) -> ModelLibraryRefreshResponse:
    library = refresh_model_library(get_model_root())
    items = library.by_type(type) if type else library.items
    return ModelLibraryRefreshResponse(
        items=[library_item_to_response(item) for item in items],
        link_health=link_health_to_response(library.link_health),
        counts=library_counts(library.items),
    )


@router.get("/library-asset")
def library_asset(path: str) -> FileResponse:
    relative_path = normalized_library_asset_path(path)
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    if relative_path is None or PurePosixPath(relative_path).suffix.lower() not in allowed_suffixes:
        raise HTTPException(status_code=404, detail="Model asset not found")

    model_root = get_model_root()
    library = get_model_library(model_root)
    indexed_previews = {
        normalized.casefold()
        for item in library.items
        if (normalized := normalized_library_asset_path(item.metadata.get("preview_image_path", ""))) is not None
    }
    if relative_path.casefold() not in indexed_previews:
        raise HTTPException(status_code=404, detail="Model asset not found")

    try:
        root = model_root.resolve()
        asset_path = (root / Path(*PurePosixPath(relative_path).parts)).resolve()
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Model asset not found") from exc
    if not path_exists(asset_path) or not path_is_file(asset_path):
        raise HTTPException(status_code=404, detail="Model asset not found")
    return FileResponse(asset_path)


@router.get("/selections/{profile}", response_model=ModelSelectionResponse)
def get_selection(profile: str) -> ModelSelectionResponse:
    selection = autofill_selection(profile)
    return ModelSelectionResponse(**selection.__dict__)


@router.put("/selections/{profile}", response_model=ModelSelectionResponse)
def save_selection(profile: str, request: ModelSelectionRequest) -> ModelSelectionResponse:
    active_profile = require_model_profile(profile)
    if request.source_profile != profile:
        raise HTTPException(
            status_code=400,
            detail={
                "code": MODEL_SELECTION_PROFILE_MISMATCH,
                "profile": profile,
                "source_profile": request.source_profile,
            },
        )
    allowed_slot_ids = {slot.id for slot in active_profile.model_slots}
    unknown_slot_ids = sorted(set(request.slots) - allowed_slot_ids)
    if unknown_slot_ids:
        raise HTTPException(
            status_code=400,
            detail={"code": MODEL_SELECTION_SLOT_NOT_FOUND, "slots": unknown_slot_ids},
        )
    try:
        selection = get_selection_repository().save(ModelSelection(profile=profile, slots=request.slots))
    except OSError as exc:
        raise HTTPException(status_code=409, detail={"code": MODEL_SELECTION_SAVE_FAILED}) from exc
    return ModelSelectionResponse(**selection.__dict__)


@router.post("/dependencies/{dependency_id}/install", response_model=ModelDependencyStatusResponse)
def install_dependency(
    dependency_id: str,
    request: InstallModelDependencyRequest,
) -> ModelDependencyStatusResponse:
    installer = ModelDependencyInstaller(repository=get_repository())
    try:
        return to_response(installer.install(dependency_id, request.license_confirmed))
    except LicenseConfirmationRequired as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": MODEL_DEPENDENCY_LICENSE_CONFIRMATION_REQUIRED},
        ) from exc
    except ModelDependencyNotFound as exc:
        raise HTTPException(status_code=404, detail={"code": MODEL_DEPENDENCY_NOT_FOUND}) from exc
    except ManagedArtifactPathTooLong as exc:
        raise HTTPException(status_code=409, detail={"code": MANAGED_ARTIFACT_PATH_TOO_LONG}) from exc
    except ManagedArtifactRollbackIncomplete as exc:
        raise HTTPException(status_code=409, detail={"code": MANAGED_ARTIFACT_ROLLBACK_INCOMPLETE}) from exc
    except ModelDependencyError as exc:
        raise HTTPException(status_code=409, detail={"code": MODEL_DEPENDENCY_INSTALL_FAILED}) from exc


@router.post("/dependencies/bootstrap", response_model=BootstrapModelDependenciesResponse)
def bootstrap_dependencies(request: BootstrapModelDependenciesRequest) -> BootstrapModelDependenciesResponse:
    started = start_model_bootstrap(profile=request.profile)
    return empty_bootstrap_response(running=started or is_model_bootstrap_running())


def start_model_bootstrap(profile: str | None = None) -> bool:
    global _bootstrap_running, _bootstrap_progress

    with _bootstrap_lock:
        if _bootstrap_running:
            return False
        _bootstrap_running = True
        _bootstrap_progress = ModelBootstrapProgress(
            current_dependency_id=None,
            completed_count=0,
            total_count=0,
        )

    thread = threading.Thread(target=_run_bootstrap, args=(profile,), daemon=True)
    thread.start()
    return True


def is_model_bootstrap_running() -> bool:
    with _bootstrap_lock:
        return _bootstrap_running


@router.get("/dependencies/bootstrap", response_model=BootstrapModelDependenciesResponse)
def bootstrap_status() -> BootstrapModelDependenciesResponse:
    with _bootstrap_lock:
        running = _bootstrap_running
        result = _bootstrap_result
    if running:
        return empty_bootstrap_response(running=True)
    if result is None:
        return empty_bootstrap_response(running=False)
    return bootstrap_to_response(result)


def _run_bootstrap(profile: str | None) -> None:
    global _bootstrap_running, _bootstrap_result
    try:
        result = ModelBootstrapper(
            repository=get_repository(),
            progress_callback=_set_bootstrap_progress,
        ).bootstrap(profile=profile)
        for profile_id in {status.profile for status in [*result.installed, *result.skipped]}:
            autofill_selection(profile_id)
    except Exception as exc:  # noqa: BLE001 - expose startup bootstrap failure through status.
        result = ModelBootstrapResult(installed=[], skipped=[], failed={"bootstrap": str(exc)})
    with _bootstrap_lock:
        _bootstrap_result = result
        _bootstrap_running = False


def _set_bootstrap_progress(progress: ModelBootstrapProgress) -> None:
    global _bootstrap_progress
    with _bootstrap_lock:
        _bootstrap_progress = progress
