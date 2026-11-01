from __future__ import annotations

import json
from pathlib import Path
import threading
from collections.abc import Callable, Mapping
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import yaml

from app.core.config import AppConfig
from app.services.model_dependencies import ModelDependencyRepository
from app.services.model_dependencies import ModelDependencyInstaller
from app.services.model_profiles import ModelProfileRepository
from app.services.model_selections import ModelSelectionRepository
from app.services.model_storage import resolve_model_root
from app.services.user_settings import resolve_current_output_root
from app.services.runtime_core import ComfyRuntimeCoreManager
from app.services.runtime_dependencies import RuntimeDependencyRepository
from app.services.runtime_dependencies import RuntimeDependencyInstaller
from app.services.release_manifest import ReleaseManifestRepository
from app.services.anima_lllite_runtime import anima_lllite_runtime_contract
from app.services.runtime import ComfyClient, RuntimeErrorResponse
from app.services.api_errors import (
    STARTUP_SETUP_CONSENT_REQUIRED,
    STARTUP_SETUP_BUSY,
    STARTUP_SETUP_EXECUTION_FAILED,
    STARTUP_SETUP_PLAN_STALE,
    STARTUP_SETUP_STEP_INVALID,
)
from app.services.startup_readiness import (
    RuntimeFeatureRequirement,
    StartupManifestHealth,
    StartupPathRequirement,
    StartupReadinessItem,
    StartupReadinessPolicy,
    StartupReadinessReport,
    StartupReadinessRequest,
    StartupReadinessService,
)
from app.services.startup_setup import (
    StartupStarterPack,
    StartupSetupConsentError,
    StartupSetupExecutor,
    StartupSetupPlan,
    StartupSetupPlanner,
    StartupSetupStep,
    StartupSetupStepError,
    StartupStarterPackSelectionError,
    build_starter_pack_options,
    validate_starter_pack_selection,
)
from app.services.windows_hardware import (
    WindowsHardwareProbe,
    WindowsHardwareStatus,
    probe_torch_cuda,
)


router = APIRouter(prefix="/api/startup", tags=["startup"])
_startup_setup_execution_lock = threading.Lock()


RUNTIME_FEATURE_POLICY = (
    RuntimeFeatureRequirement("merge_xyz", ("asteria-merge-runtime",), "all"),
    RuntimeFeatureRequirement("detailer", ("comfyui-impact-pack", "comfyui-impact-subpack"), "all"),
    RuntimeFeatureRequirement("controlnet_preprocessor", ("comfyui-controlnet-aux",), "all"),
    RuntimeFeatureRequirement("openpose", ("comfyui-controlnet-aux",), "all"),
    RuntimeFeatureRequirement("controlnet_advanced", ("comfyui-advanced-controlnet",), "all"),
    RuntimeFeatureRequirement("lora_block_weight", ("comfyui-inspire-pack",), "all"),
    RuntimeFeatureRequirement("seed_variation", ("comfyui-inspire-pack",), "all"),
    RuntimeFeatureRequirement("seed_resize", ("comfyui-inspire-pack",), "all"),
    RuntimeFeatureRequirement("anima_lllite", (), "core"),
)


class StartupReadinessRequestResponse(BaseModel):
    target_mode: str
    profile_id: str
    enabled_feature_ids: list[str]


class StartupReadinessItemResponse(BaseModel):
    code: str
    summary_key: str
    summary_params: dict[str, str]
    phase: str
    severity: str
    blocks: list[str]
    action: str | None
    dependency_id: str | None
    profile_id: str | None
    feature_ids: list[str]
    auto_install: bool | None
    requires_license_confirmation: bool
    download_size_bytes: int | None


class StartupReadinessSummaryResponse(BaseModel):
    app_ready: bool
    runtime_ready: bool
    generation_ready: bool
    blocking_count: int
    warning_count: int
    action_count: int
    required_license_dependency_ids: list[str]
    optional_license_dependency_ids: list[str]
    required_download_bytes: int
    optional_download_bytes: int


class StartupReadinessResponse(BaseModel):
    request: StartupReadinessRequestResponse
    items: list[StartupReadinessItemResponse]
    summary: StartupReadinessSummaryResponse


class StartupSetupStepResponse(BaseModel):
    id: str
    phase: str
    code: str
    operation: str | None
    dependency_id: str | None
    status: str
    required: bool
    executable: bool
    requires_license_confirmation: bool
    requires_download_confirmation: bool
    download_size_bytes: int | None


class StartupStarterPackResponse(BaseModel):
    id: str
    status: str
    selected_by_default: bool
    requires_license_confirmation: bool
    download_size_bytes: int
    source_urls: list[str]
    license_url: str | None


class StartupSetupPlanResponse(BaseModel):
    plan_id: str
    target_mode: str
    profile_id: str
    enabled_feature_ids: list[str]
    steps: list[StartupSetupStepResponse]
    required_license_dependency_ids: list[str]
    optional_license_dependency_ids: list[str]
    required_download_bytes: int
    optional_download_bytes: int
    has_unknown_required_download: bool
    has_unknown_optional_download: bool
    starter_packs: list[StartupStarterPackResponse]
    starter_pack_download_bytes: int


class StartupSetupExecuteRequest(BaseModel):
    target_mode: Literal["development", "packaged"] = "development"
    profile_id: str = "sdxl"
    enabled_feature_ids: list[str] = []
    plan_id: str
    step_id: str
    download_confirmed: bool = False
    accepted_license_dependency_ids: list[str] = []


class StartupSetupExecutionResponse(BaseModel):
    step_id: str
    status: Literal["completed"]
    operation_id: str | None = None


class StartupStarterPackExecuteRequest(BaseModel):
    target_mode: Literal["development", "packaged"] = "development"
    profile_id: str = "sdxl"
    enabled_feature_ids: list[str] = []
    plan_id: str
    selected_starter_pack_ids: list[str] = []
    download_confirmed: bool = False
    accepted_license_pack_ids: list[str] = []


class StartupStarterPackExecutionResponse(BaseModel):
    installed_starter_pack_ids: list[str]
    status: Literal["completed"]


@router.get("/readiness", response_model=StartupReadinessResponse)
def startup_readiness(
    target_mode: Literal["development", "packaged"] = "development",
    profile_id: str = "sdxl",
    feature: list[str] = Query(default=[]),
) -> StartupReadinessResponse:
    report = _inspect_startup_readiness(target_mode, profile_id, feature)
    return _readiness_response(report)


def _inspect_startup_readiness(
    target_mode: Literal["development", "packaged"],
    profile_id: str,
    feature: list[str],
) -> StartupReadinessReport:
    config = AppConfig()
    model_root = resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)
    model_dependencies = ModelDependencyRepository(config.model_dependencies_path, model_root)
    model_profiles = ModelProfileRepository(config.model_profiles_path)
    model_selections = ModelSelectionRepository(config.model_selections_path)
    runtime_dependencies = RuntimeDependencyRepository(
        config.runtime_dependencies_path,
        config.comfy_dir,
        python_path=config.comfy_python_path,
    )
    profiles_by_id = {profile.id: profile for profile in model_profiles.list()}
    service = StartupReadinessService(
        product_paths=(
            StartupPathRequirement("data_root", "startup.path.data", config.data_dir),
            StartupPathRequirement("model_root", "startup.path.models", model_root),
            StartupPathRequirement(
                "output_root",
                "startup.path.outputs",
                resolve_current_output_root(config.data_dir),
            ),
        ),
        runtime_python_path=config.comfy_python_path,
        core_status_provider=lambda: ComfyRuntimeCoreManager(
            comfy_dir=config.comfy_dir,
            python_path=config.comfy_python_path,
            repo_url=config.comfy_repo_url,
            ref=config.comfy_repo_ref,
        ).status(),
        runtime_dependency_status_provider=runtime_dependencies.list_status,
        core_feature_status_provider=lambda feature_id: _core_runtime_feature_available(
            config,
            feature_id,
        ),
        model_dependency_status_provider=model_dependencies.list_status,
        profile_provider=profiles_by_id.get,
        selection_provider=model_selections.get,
        manifest_health_provider=lambda: _manifest_health(config),
        policy=StartupReadinessPolicy(runtime_features=RUNTIME_FEATURE_POLICY),
        hardware_status_provider=lambda: _windows_hardware_status(config),
        cuda_status_provider=probe_torch_cuda,
    )
    return service.inspect(
        StartupReadinessRequest(
            target_mode=target_mode,
            profile_id=profile_id,
            enabled_feature_ids=tuple(feature),
        )
    )


def _core_runtime_feature_available(config: AppConfig, feature_id: str) -> bool:
    if feature_id != "anima_lllite":
        return False
    try:
        object_info = ComfyClient(config.comfy_base_url).object_info()
    except (OSError, RuntimeErrorResponse):
        return False
    return "LoadImage" in object_info and anima_lllite_runtime_contract(object_info) is not None


def _windows_hardware_status(config: AppConfig) -> WindowsHardwareStatus:
    try:
        minimum_driver = ReleaseManifestRepository(config.release_manifest_path).load().minimum_nvidia_driver
    except (OSError, ValueError):
        return WindowsHardwareStatus(
            supported=False,
            code="startup_hardware_manifest_invalid",
            minimum_driver="unknown",
        )
    return WindowsHardwareProbe(minimum_driver=minimum_driver).inspect()


def _readiness_response(report: StartupReadinessReport) -> StartupReadinessResponse:
    return StartupReadinessResponse(
        request=StartupReadinessRequestResponse(
            target_mode=report.request.target_mode,
            profile_id=report.request.profile_id,
            enabled_feature_ids=list(report.request.enabled_feature_ids),
        ),
        items=[_item_response(item) for item in report.items],
        summary=StartupReadinessSummaryResponse(
            app_ready=report.summary.app_ready,
            runtime_ready=report.summary.runtime_ready,
            generation_ready=report.summary.generation_ready,
            blocking_count=report.summary.blocking_count,
            warning_count=report.summary.warning_count,
            action_count=report.summary.action_count,
            required_license_dependency_ids=list(report.summary.required_license_dependency_ids),
            optional_license_dependency_ids=list(report.summary.optional_license_dependency_ids),
            required_download_bytes=report.summary.required_download_bytes,
            optional_download_bytes=report.summary.optional_download_bytes,
        ),
    )


@router.get("/setup/plan", response_model=StartupSetupPlanResponse)
def startup_setup_plan(
    target_mode: Literal["development", "packaged"] = "development",
    profile_id: str = "sdxl",
    feature: list[str] = Query(default=[]),
) -> StartupSetupPlanResponse:
    config = AppConfig()
    report = _inspect_startup_readiness(target_mode, profile_id, feature)
    return _setup_plan_response(StartupSetupPlanner(starter_packs=_starter_packs(config)).plan(report))


def _setup_plan_response(plan: StartupSetupPlan) -> StartupSetupPlanResponse:
    return StartupSetupPlanResponse(
        plan_id=plan.plan_id,
        target_mode=plan.target_mode,
        profile_id=plan.profile_id,
        enabled_feature_ids=list(plan.enabled_feature_ids),
        steps=[StartupSetupStepResponse(**step.__dict__) for step in plan.steps],
        required_license_dependency_ids=list(plan.required_license_dependency_ids),
        optional_license_dependency_ids=list(plan.optional_license_dependency_ids),
        required_download_bytes=plan.required_download_bytes,
        optional_download_bytes=plan.optional_download_bytes,
        has_unknown_required_download=plan.has_unknown_required_download,
        has_unknown_optional_download=plan.has_unknown_optional_download,
        starter_packs=[_starter_pack_response(pack) for pack in plan.starter_packs],
        starter_pack_download_bytes=plan.starter_pack_download_bytes,
    )


@router.post(
    "/setup/execute",
    response_model=StartupSetupExecutionResponse,
    response_model_exclude_none=True,
)
def execute_startup_setup(request: StartupSetupExecuteRequest) -> StartupSetupExecutionResponse:
    if not _startup_setup_execution_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail={"code": STARTUP_SETUP_BUSY})
    try:
        report = _inspect_startup_readiness(
            request.target_mode,
            request.profile_id,
            request.enabled_feature_ids,
        )
        plan = StartupSetupPlanner(starter_packs=_starter_packs(AppConfig())).plan(report)
        if plan.plan_id != request.plan_id:
            raise HTTPException(status_code=409, detail={"code": STARTUP_SETUP_PLAN_STALE})
        if request.target_mode == "packaged" and any(
            item.phase == "hardware" and bool(item.blocks) for item in report.items
        ):
            raise HTTPException(status_code=409, detail={"code": STARTUP_SETUP_STEP_INVALID})
        result = StartupSetupExecutor(handlers=_startup_setup_handlers()).execute(
            plan,
            request.step_id,
            download_confirmed=request.download_confirmed,
            accepted_license_dependency_ids=tuple(request.accepted_license_dependency_ids),
        )
    except StartupSetupConsentError as exc:
        raise HTTPException(status_code=409, detail={"code": STARTUP_SETUP_CONSENT_REQUIRED}) from exc
    except StartupSetupStepError as exc:
        raise HTTPException(status_code=409, detail={"code": STARTUP_SETUP_STEP_INVALID}) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=409, detail={"code": STARTUP_SETUP_EXECUTION_FAILED}) from exc
    finally:
        _startup_setup_execution_lock.release()
    return StartupSetupExecutionResponse(**result.__dict__)


@router.post("/setup/starter-packs/execute", response_model=StartupStarterPackExecutionResponse)
def execute_startup_starter_packs(
    request: StartupStarterPackExecuteRequest,
) -> StartupStarterPackExecutionResponse:
    if not _startup_setup_execution_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail={"code": STARTUP_SETUP_BUSY})
    try:
        config = AppConfig()
        report = _inspect_startup_readiness(
            request.target_mode,
            request.profile_id,
            request.enabled_feature_ids,
        )
        packs = _starter_packs(config)
        plan = StartupSetupPlanner(starter_packs=packs).plan(report)
        if plan.plan_id != request.plan_id:
            raise HTTPException(status_code=409, detail={"code": STARTUP_SETUP_PLAN_STALE})
        if request.target_mode == "packaged" and any(
            item.phase == "hardware" and bool(item.blocks) for item in report.items
        ):
            raise HTTPException(status_code=409, detail={"code": STARTUP_SETUP_STEP_INVALID})
        pending = validate_starter_pack_selection(
            packs,
            tuple(request.selected_starter_pack_ids),
            download_confirmed=request.download_confirmed,
            accepted_license_pack_ids=tuple(request.accepted_license_pack_ids),
        )
        _install_starter_packs(
            config,
            pending,
            accepted_license_pack_ids=set(request.accepted_license_pack_ids),
        )
    except StartupSetupConsentError as exc:
        raise HTTPException(status_code=409, detail={"code": STARTUP_SETUP_CONSENT_REQUIRED}) from exc
    except StartupStarterPackSelectionError as exc:
        raise HTTPException(status_code=409, detail={"code": STARTUP_SETUP_STEP_INVALID}) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=409, detail={"code": STARTUP_SETUP_EXECUTION_FAILED}) from exc
    finally:
        _startup_setup_execution_lock.release()
    return StartupStarterPackExecutionResponse(
        installed_starter_pack_ids=[pack.id for pack in pending],
        status="completed",
    )


def _startup_setup_handlers() -> Mapping[str, Callable[[StartupSetupStep], None]]:
    config = AppConfig()
    core = ComfyRuntimeCoreManager(
        comfy_dir=config.comfy_dir,
        python_path=config.comfy_python_path,
        repo_url=config.comfy_repo_url,
        ref=config.comfy_repo_ref,
    )
    runtime_repository = RuntimeDependencyRepository(
        config.runtime_dependencies_path,
        config.comfy_dir,
        python_path=config.comfy_python_path,
    )
    runtime_installer = RuntimeDependencyInstaller(runtime_repository, python_path=config.comfy_python_path)
    model_root = resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)
    model_installer = ModelDependencyInstaller(
        ModelDependencyRepository(config.model_dependencies_path, model_root)
    )

    def runtime_install(step: StartupSetupStep) -> None:
        if step.dependency_id is None:
            raise StartupSetupStepError("runtime dependency id is missing")
        runtime_installer.install(step.dependency_id)

    def runtime_repair(step: StartupSetupStep) -> None:
        if step.dependency_id is None:
            raise StartupSetupStepError("runtime dependency id is missing")
        runtime_installer.repair(step.dependency_id)

    def model_install(step: StartupSetupStep) -> None:
        if step.dependency_id is None:
            raise StartupSetupStepError("model dependency id is missing")
        model_installer.install(step.dependency_id, license_confirmed=True)

    return {
        "install_runtime_core": lambda step: core.install(),
        "repair_runtime_core": lambda step: core.repair(),
        "install_runtime_dependency": runtime_install,
        "repair_runtime_dependency": runtime_repair,
        "install_model_dependency": model_install,
        "confirm_model_license_then_install": model_install,
        "repair_model_dependency": model_install,
    }


def _starter_packs(config: AppConfig) -> tuple[StartupStarterPack, ...]:
    model_root = resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)
    statuses = ModelDependencyRepository(config.model_dependencies_path, model_root).list_status()
    return build_starter_pack_options(statuses)


def _starter_pack_response(pack: StartupStarterPack) -> StartupStarterPackResponse:
    return StartupStarterPackResponse(
        id=pack.id,
        status=pack.status,
        selected_by_default=pack.selected_by_default,
        requires_license_confirmation=pack.requires_license_confirmation,
        download_size_bytes=pack.download_size_bytes,
        source_urls=list(pack.source_urls),
        license_url=pack.license_url,
    )


def _install_starter_packs(
    config: AppConfig,
    packs: tuple[StartupStarterPack, ...],
    *,
    accepted_license_pack_ids: set[str],
) -> None:
    if not packs:
        return
    model_root = resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)
    repository = ModelDependencyRepository(config.model_dependencies_path, model_root)
    installer = ModelDependencyInstaller(repository)
    selected_ids = {pack.id for pack in packs}
    status_by_id = {status.id: status.status for status in repository.list_status()}
    failures: list[Exception] = []
    for dependency in repository.list():
        if dependency.release_class != "optional-starter" or dependency.starter_pack not in selected_ids:
            continue
        if status_by_id.get(dependency.id) == "installed":
            continue
        try:
            installer.install(
                dependency.id,
                license_confirmed=(
                    not dependency.requires_license_confirmation
                    or dependency.starter_pack in accepted_license_pack_ids
                ),
            )
        except Exception as exc:
            failures.append(exc)
    ModelSelectionRepository(config.model_selections_path).autofill_complete_starter_packs(
        tuple(pack.id for pack in packs),
        repository.list_status(),
    )
    if failures:
        raise failures[0]


def _item_response(item: StartupReadinessItem) -> StartupReadinessItemResponse:
    return StartupReadinessItemResponse(
        code=item.code,
        summary_key=item.summary_key,
        summary_params=dict(item.summary_params),
        phase=item.phase,
        severity=item.severity,
        blocks=list(item.blocks),
        action=item.action,
        dependency_id=item.dependency_id,
        profile_id=item.profile_id,
        feature_ids=list(item.feature_ids),
        auto_install=item.auto_install,
        requires_license_confirmation=item.requires_license_confirmation,
        download_size_bytes=item.download_size_bytes,
    )


def _manifest_health(config: AppConfig) -> list[StartupManifestHealth]:
    return [
        _yaml_manifest_health(
            config.runtime_dependencies_path,
            manifest_id="runtime_dependencies",
            phase="runtime_packages",
            root_key="dependencies",
        ),
        _yaml_manifest_health(
            config.model_dependencies_path,
            manifest_id="model_dependencies",
            phase="model_assets",
            root_key="dependencies",
        ),
        _yaml_manifest_health(
            config.model_profiles_path,
            manifest_id="model_profiles",
            phase="model_assets",
            root_key="profiles",
        ),
        _selection_health(config.model_selections_path),
    ]


def _yaml_manifest_health(
    path: Path,
    *,
    manifest_id: str,
    phase: Literal["runtime_packages", "model_assets"],
    root_key: str,
) -> StartupManifestHealth:
    try:
        if not path.is_file():
            return StartupManifestHealth(manifest_id, phase, False, "missing")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError:
        return StartupManifestHealth(manifest_id, phase, False, "unreadable")
    except yaml.YAMLError:
        return StartupManifestHealth(manifest_id, phase, False, "invalid_yaml")
    if not isinstance(payload, dict) or not isinstance(payload.get(root_key), list):
        return StartupManifestHealth(manifest_id, phase, False, "invalid_schema")
    return StartupManifestHealth(manifest_id, phase, True)


def _selection_health(path: Path) -> StartupManifestHealth:
    try:
        if not path.exists():
            return StartupManifestHealth("model_selections", "model_assets", True)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return StartupManifestHealth("model_selections", "model_assets", False, "unreadable")
    except json.JSONDecodeError:
        return StartupManifestHealth("model_selections", "model_assets", False, "invalid_json")
    if not isinstance(payload, dict):
        return StartupManifestHealth("model_selections", "model_assets", False, "invalid_schema")
    for slots in payload.values():
        if not isinstance(slots, dict) or any(not isinstance(value, str) for value in slots.values()):
            return StartupManifestHealth("model_selections", "model_assets", False, "invalid_schema")
    return StartupManifestHealth("model_selections", "model_assets", True)
