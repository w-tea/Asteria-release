from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import AppConfig
from app.api.generation import AVAILABLE_RUNTIMES
from app.services.api_errors import (
    RUNTIME_CORE_INSTALL_FAILED,
    RUNTIME_CORE_REPAIR_FAILED,
    RUNTIME_CORE_UPDATE_FAILED,
    RUNTIME_DEPENDENCY_INSTALL_FAILED,
    RUNTIME_DEPENDENCY_NOT_FOUND,
    RUNTIME_INTERRUPT_FAILED,
    RUNTIME_START_FAILED,
)
from app.services.controlnet_preprocessor import COMFY_CONTROLNET_PREPROCESSOR_NODES
from app.services.controlnet_mode import (
    CONTROLNET_MODE_LABELS,
    controlnet_mode_contract_status,
    resolve_controlnet_mode_contract,
)
from app.services.runtime import ComfyClient, RuntimeErrorResponse
from app.services.runtime import INSPIRE_KSAMPLER_CLASS, LORA_BLOCK_WEIGHT_CLASS
from app.services.seed_resize import resolve_seed_resize_contract, seed_resize_contract_status
from app.services.model_dependencies import ModelDependencyRepository
from app.services.sampling_catalog import SAMPLERS, SCHEDULERS
from app.services.upscaler_catalog import UPSCALERS
from app.services.runtime_dependencies import (
    RuntimeDependencyError,
    RuntimeDependencyInstaller,
    RuntimeDependencyNotFound,
    RuntimeDependencyRepository,
    RuntimeDependencyStatus,
)
from app.services.runtime_core import ComfyRuntimeCoreError, ComfyRuntimeCoreManager, ComfyRuntimeCoreStatus
from app.services.runtime_process import RuntimeProcessError, RuntimeProcessManager, RuntimeStatus
from app.services.workflow_mapping import WorkflowMappingRepository

router = APIRouter(prefix="/api/runtime", tags=["runtime"])


class RuntimeConfigResponse(BaseModel):
    generation_runtime: str
    runtime_base_url: str
    engine_command_configured: bool
    available_runtimes: dict[str, str]


class RuntimeFeatureCapability(BaseModel):
    available: bool
    runtime: str
    missing_nodes: list[str] = []
    message: str


class RuntimeCapabilitiesResponse(BaseModel):
    base_url: str
    features: dict[str, Any]


class RuntimeSamplingOptionResponse(BaseModel):
    label: str
    runtime_name: str
    available: bool


class RuntimeSamplingResponse(BaseModel):
    base_url: str
    samplers: list[RuntimeSamplingOptionResponse]
    schedulers: list[RuntimeSamplingOptionResponse]


class RuntimeUpscalerOptionResponse(BaseModel):
    label: str
    kind: str
    runtime_name: str
    model_name: str | None
    dependency_id: str | None
    available: bool
    installed: bool
    runtime_loaded: bool
    dependency_status: str | None = None


class RuntimeUpscalerResponse(BaseModel):
    base_url: str
    upscalers: list[RuntimeUpscalerOptionResponse]


class RuntimeDependencyStatusResponse(BaseModel):
    id: str
    label: str
    type: str
    provider: str
    status: str
    path: str
    source_url: str
    ref: str | None
    features: list[str]
    requirements_file: str | None
    install_script: str | None
    pip_packages: list[str]
    auto_install: bool


class RuntimeDependencyListResponse(BaseModel):
    dependencies: list[RuntimeDependencyStatusResponse]


class RuntimeMaintenanceResponse(BaseModel):
    status: str
    dependencies: list[RuntimeDependencyStatusResponse]
    errors: dict[str, str] = {}


class ComfyRuntimeCoreStatusResponse(BaseModel):
    status: str
    path: str
    repo_url: str
    ref: str
    message: str


class RuntimeLogsResponse(BaseModel):
    stdout: str
    stderr: str


CONTROLNET_REQUIRED_NODES = {
    "LoadImage",
    "ControlNetLoader",
    "ControlNetApplyAdvanced",
}
ANIMA_LLLITE_REQUIRED_NODES = {
    "LoadImage",
    "AnimaLLLiteApply",
}
ANIMA_REGIONAL_REQUIRED_NODES = {
    "AttentionCouplePPM",
    "ImageColorToMask",
    "SolidMask",
}
DETAILER_REQUIRED_NODES = {"FaceDetailer", "UltralyticsDetectorProvider"}
LORA_BLOCK_WEIGHT_REQUIRED_NODES = {LORA_BLOCK_WEIGHT_CLASS}
HIRES_FIX_REQUIRED_NODES = {
    "KSampler",
    "LatentUpscale",
    "LatentUpscaleBy",
    "ImageScale",
    "ImageScaleBy",
    "UpscaleModelLoader",
    "ImageUpscaleWithModel",
    "VAEEncode",
    "VAEDecode",
}
CLIP_SKIP_REQUIRED_NODES = {"CLIPSetLastLayer"}
CONTROLNET_PREPROCESSOR_NODES = COMFY_CONTROLNET_PREPROCESSOR_NODES


def _manager() -> RuntimeProcessManager:
    return RuntimeProcessManager(AppConfig())


def _dependency_repository() -> RuntimeDependencyRepository:
    app_config = AppConfig()
    return RuntimeDependencyRepository(
        manifest_path=app_config.runtime_dependencies_path,
        runtime_root=app_config.comfy_dir,
    )


def _model_dependency_repository() -> ModelDependencyRepository:
    app_config = AppConfig()
    return ModelDependencyRepository(
        manifest_path=app_config.model_dependencies_path,
        model_root=app_config.model_dir,
    )


def _runtime_dependency_runner():
    return None


def _runtime_core_runner():
    return None


def _dependency_to_response(status: RuntimeDependencyStatus) -> RuntimeDependencyStatusResponse:
    payload = {**status.__dict__, "path": str(status.path)}
    return RuntimeDependencyStatusResponse(**payload)


def _core_manager() -> ComfyRuntimeCoreManager:
    app_config = AppConfig()
    return ComfyRuntimeCoreManager(
        comfy_dir=app_config.comfy_dir,
        python_path=app_config.comfy_python_path,
        repo_url=app_config.comfy_repo_url,
        ref=app_config.comfy_repo_ref,
        runner=_runtime_core_runner(),
    )


def _core_to_response(status: ComfyRuntimeCoreStatus) -> ComfyRuntimeCoreStatusResponse:
    return ComfyRuntimeCoreStatusResponse(
        status=status.status,
        path=str(status.path),
        repo_url=status.repo_url,
        ref=status.ref,
        message=status.message,
    )


def _node_classes(object_info: dict) -> set[str]:
    return {key for key, value in object_info.items() if isinstance(key, str) and isinstance(value, dict)}


def _node_input_definition(object_info: dict, node_name: str, input_name: str) -> Any:
    node = object_info.get(node_name)
    if not isinstance(node, dict):
        return None
    inputs = node.get("input") or {}
    for section_name in ("required", "optional"):
        section = inputs.get(section_name) or {}
        if isinstance(section, dict) and input_name in section:
            return section.get(input_name)
    return None


def _node_input_values(object_info: dict, node_name: str, input_name: str) -> set[str]:
    return set(_node_input_value_list(object_info, node_name, input_name))


def _node_input_value_list(object_info: dict, node_name: str, input_name: str) -> list[str]:
    raw = _node_input_definition(object_info, node_name, input_name)
    if not isinstance(raw, list) or not raw:
        return []
    values = raw[0]
    if values == "COMBO" and len(raw) > 1 and isinstance(raw[1], dict):
        values = raw[1].get("options")
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def _node_has_input(object_info: dict, node_name: str, input_name: str) -> bool:
    return _node_input_definition(object_info, node_name, input_name) is not None


def _all_required_capability(name: str, nodes: set[str], required: set[str]) -> RuntimeFeatureCapability:
    missing = sorted(required - nodes)
    return RuntimeFeatureCapability(
        available=not missing,
        runtime="comfy_workflow",
        missing_nodes=missing,
        message=f"{name} ready" if not missing else f"{name} missing ComfyUI nodes",
    )


def _controlnet_capability(object_info: dict, nodes: set[str]) -> dict[str, Any]:
    missing = sorted(CONTROLNET_REQUIRED_NODES - nodes)
    controlnet_models = (
        sorted(_node_input_values(object_info, "ControlNetLoader", "control_net_name"))
        if not missing
        else []
    )
    mode_ready = False
    if not missing:
        try:
            contract = resolve_controlnet_mode_contract(WorkflowMappingRepository(AppConfig().mapping_dir), "sdxl")
            mode_ready = controlnet_mode_contract_status(object_info, contract).ready
        except ValueError:
            mode_ready = False
    control_mode_runtime = (
        "unavailable"
        if missing
        else "advanced_controlnet_soft_weights"
        if mode_ready
        else "balanced_standard_controlnet_apply"
    )
    control_mode_message = (
        "Control mode unavailable until ControlNet nodes are installed"
        if missing
        else "Control mode priority workflows ready"
        if mode_ready
        else "Priority modes require Advanced-ControlNet"
    )
    return {
        "available": not missing,
        "runtime": "comfy_workflow",
        "missing_nodes": missing,
        "message": "ControlNet ready" if not missing else "ControlNet missing ComfyUI nodes",
        "controlnet_models": controlnet_models,
        "control_mode": {
            "available": mode_ready,
            "runtime": control_mode_runtime,
            "native_node_input": False,
            "modes": list(CONTROLNET_MODE_LABELS) if mode_ready else [CONTROLNET_MODE_LABELS[0]] if not missing else [],
            "message": control_mode_message,
        },
    }


def _anima_lllite_capability(object_info: dict, nodes: set[str]) -> dict[str, Any]:
    missing = sorted(ANIMA_LLLITE_REQUIRED_NODES - nodes)
    lllite_models = (
        sorted(_anima_lllite_model_values(object_info))
        if not missing
        else []
    )
    return {
        "available": not missing and bool(lllite_models),
        "runtime": "comfy_workflow",
        "missing_nodes": missing,
        "message": "Anima LLLite control ready"
        if not missing and lllite_models
        else "Anima LLLite control missing node or model",
        "lllite_models": lllite_models,
    }


def _anima_lllite_model_values(object_info: dict) -> set[str]:
    values = _node_input_values(object_info, "AnimaLLLiteApply", "lllite_name")
    return {
        value
        for value in values
        if "lllite" in value.lower() or "anima" in value.lower()
    }


def _detailer_capability(object_info: dict, nodes: set[str]) -> dict[str, Any]:
    missing = sorted(DETAILER_REQUIRED_NODES - nodes)
    detector_models = (
        sorted(_node_input_values(object_info, "UltralyticsDetectorProvider", "model_name"))
        if not missing
        else []
    )
    return {
        "available": not missing,
        "runtime": "comfy_workflow",
        "missing_nodes": missing,
        "message": "Detailer ready" if not missing else "Detailer custom nodes are not installed",
        "detector_models": detector_models,
    }


def _controlnet_preprocessor_capability(nodes: set[str]) -> dict[str, Any]:
    preprocessors = {
        name: {
            "available": node_class in nodes,
            "runtime": "comfy_node" if node_class in nodes else "local_fallback",
            "node_class": node_class,
        }
        for name, node_class in CONTROLNET_PREPROCESSOR_NODES.items()
    }
    all_available = all(item["available"] for item in preprocessors.values())
    return {
        "available": all_available,
        "runtime": "comfy_workflow",
        "missing_nodes": sorted(item["node_class"] for item in preprocessors.values() if not item["available"]),
        "message": "ControlNet preprocessors ready"
        if all_available
        else "ControlNet preprocessors use local fallback until custom nodes are installed",
        "preprocessors": preprocessors,
    }


def _unmapped_generation_settings_capability(object_info: dict | None = None, nodes: set[str] | None = None) -> dict[str, Any]:
    nodes = nodes or set()
    object_info = object_info or {}
    settings: list[dict[str, str]] = []
    if DETAILER_REQUIRED_NODES - nodes or "bbox/face_yolov8m.pt" not in _node_input_values(
        object_info,
        "UltralyticsDetectorProvider",
        "model_name",
    ):
        settings.append(
            {
                "field": "restore_faces",
                "reason": "restore_faces_requires_face_detailer_runtime",
            }
        )
    return {
        "available": not settings,
        "runtime": "comfy_workflow",
        "missing_nodes": [],
        "message": "All common generation settings are mapped" if not settings else "Some generation settings are blocked until mapped workflow support is added",
        "unsupported_settings": settings,
    }


def _seed_variation_capability(nodes: set[str], unavailable_message: str | None = None) -> dict[str, Any]:
    available = INSPIRE_KSAMPLER_CLASS in nodes
    return {
        "available": available,
        "runtime": "comfy_workflow",
        "missing_nodes": [] if available else [INSPIRE_KSAMPLER_CLASS],
        "message": "Seed variation ready for SDXL"
        if available
        else unavailable_message or "Seed variation requires the Inspire KSampler",
        "profiles": ["sdxl"],
    }


def _seed_resize_capability(
    config: AppConfig,
    object_info: dict[str, Any],
    unavailable_message: str | None = None,
) -> dict[str, Any]:
    try:
        contract = resolve_seed_resize_contract(WorkflowMappingRepository(config.mapping_dir))
        status = seed_resize_contract_status(object_info, contract)
        missing_nodes = sorted(
            node.class_type
            for node in contract.nodes
            if not isinstance(object_info.get(node.class_type), dict)
        )
        issues = list(status.issues)
        available = status.ready
    except ValueError:
        available = False
        missing_nodes = []
        issues = ["mapping"]
    return {
        "available": available,
        "runtime": "comfy_workflow",
        "missing_nodes": missing_nodes,
        "message": "Seed resize ready for SDXL txt2img"
        if available
        else unavailable_message or "Seed resize requires compatible Inspire noise nodes",
        "profiles": ["sdxl"],
        "modes": ["txt2img"],
        "issues": issues,
    }


def _clip_skip_capability(nodes: set[str], unavailable_message: str | None = None) -> dict[str, Any]:
    available = not (CLIP_SKIP_REQUIRED_NODES - nodes)
    return {
        "available": available,
        "runtime": "comfy_workflow",
        "missing_nodes": [] if available else sorted(CLIP_SKIP_REQUIRED_NODES),
        "message": "CLIP skip ready for SDXL"
        if available
        else unavailable_message or "CLIP skip requires a ComfyUI runtime node",
        "profiles": ["sdxl"],
    }


def _unreachable_capabilities(base_url: str, message: str) -> RuntimeCapabilitiesResponse:
    return RuntimeCapabilitiesResponse(
        base_url=base_url,
        features={
            "controlnet": RuntimeFeatureCapability(
                available=False,
                runtime="comfy_workflow",
                missing_nodes=sorted(CONTROLNET_REQUIRED_NODES),
                message=message,
            ),
            "detailer": RuntimeFeatureCapability(
                available=False,
                runtime="comfy_workflow",
                missing_nodes=sorted(DETAILER_REQUIRED_NODES),
                message=message,
            ),
            "controlnet_preprocessors": {
                "available": False,
                "runtime": "comfy_workflow",
                "missing_nodes": sorted(CONTROLNET_PREPROCESSOR_NODES.values()),
                "message": message,
                "preprocessors": {},
            },
            "anima_lllite": {
                "available": False,
                "runtime": "comfy_workflow",
                "missing_nodes": sorted(ANIMA_LLLITE_REQUIRED_NODES),
                "message": message,
                "lllite_models": [],
            },
            "anima_regional_prompting": RuntimeFeatureCapability(
                available=False,
                runtime="comfy_workflow",
                missing_nodes=sorted(ANIMA_REGIONAL_REQUIRED_NODES),
                message=message,
            ),
            "lora_block_weight": RuntimeFeatureCapability(
                available=False,
                runtime="comfy_workflow",
                missing_nodes=sorted(LORA_BLOCK_WEIGHT_REQUIRED_NODES),
                message=message,
            ),
            "hires_fix": RuntimeFeatureCapability(
                available=False,
                runtime="comfy_workflow",
                missing_nodes=sorted(HIRES_FIX_REQUIRED_NODES),
                message=message,
            ),
            "seed_variation": _seed_variation_capability(set(), message),
            "seed_resize": _seed_resize_capability(AppConfig(), {}, message),
            "clip_skip": _clip_skip_capability(set(), message),
            "unmapped_generation_settings": _unmapped_generation_settings_capability(),
        },
    )


@router.get("/config", response_model=RuntimeConfigResponse)
def config() -> RuntimeConfigResponse:
    app_config = AppConfig()
    return RuntimeConfigResponse(
        generation_runtime=app_config.generation_runtime,
        runtime_base_url=app_config.runtime_base_url,
        engine_command_configured=bool(app_config.engine_command.strip()),
        available_runtimes=AVAILABLE_RUNTIMES,
    )


@router.get("/status", response_model=RuntimeStatus)
def status() -> RuntimeStatus:
    return _manager().status()


@router.get("/capabilities", response_model=RuntimeCapabilitiesResponse)
def capabilities() -> RuntimeCapabilitiesResponse:
    app_config = AppConfig()
    client = ComfyClient(app_config.comfy_base_url)
    try:
        object_info = client.object_info()
        nodes = _node_classes(object_info)
    except RuntimeErrorResponse:
        return _unreachable_capabilities(app_config.comfy_base_url, "ComfyUI not reachable")
    return RuntimeCapabilitiesResponse(
        base_url=app_config.comfy_base_url,
        features={
            "controlnet": _controlnet_capability(object_info, nodes),
            "anima_lllite": _anima_lllite_capability(object_info, nodes),
            "anima_regional_prompting": _all_required_capability(
                "Anima regional prompting",
                nodes,
                ANIMA_REGIONAL_REQUIRED_NODES,
            ),
            "detailer": _detailer_capability(object_info, nodes),
            "controlnet_preprocessors": _controlnet_preprocessor_capability(nodes),
            "lora_block_weight": _all_required_capability(
                "LoRA block weight",
                nodes,
                LORA_BLOCK_WEIGHT_REQUIRED_NODES,
            ),
            "hires_fix": _all_required_capability("Hires.fix", nodes, HIRES_FIX_REQUIRED_NODES),
            "seed_variation": _seed_variation_capability(nodes),
            "seed_resize": _seed_resize_capability(app_config, object_info),
            "clip_skip": _clip_skip_capability(nodes),
            "unmapped_generation_settings": _unmapped_generation_settings_capability(object_info, nodes),
        },
    )


@router.get("/sampling", response_model=RuntimeSamplingResponse)
def sampling_options() -> RuntimeSamplingResponse:
    app_config = AppConfig()
    try:
        object_info = ComfyClient(app_config.comfy_base_url).object_info()
    except RuntimeErrorResponse:
        object_info = {}
    sampler_values = _node_input_values(object_info, "KSampler", "sampler_name")
    scheduler_values = _node_input_values(object_info, "KSampler", "scheduler")
    return RuntimeSamplingResponse(
        base_url=app_config.comfy_base_url,
        samplers=[
            RuntimeSamplingOptionResponse(
                label=option.label,
                runtime_name=option.runtime_name,
                available=not sampler_values or option.runtime_name in sampler_values,
            )
            for option in SAMPLERS
        ],
        schedulers=[
            RuntimeSamplingOptionResponse(
                label=option.label,
                runtime_name=option.runtime_name,
                available=not scheduler_values or option.runtime_name in scheduler_values,
            )
            for option in SCHEDULERS
        ],
    )


@router.get("/upscalers", response_model=RuntimeUpscalerResponse)
def upscaler_options() -> RuntimeUpscalerResponse:
    app_config = AppConfig()
    try:
        object_info = ComfyClient(app_config.comfy_base_url).object_info()
    except RuntimeErrorResponse:
        object_info = {}
    model_values = _node_input_values(object_info, "UpscaleModelLoader", "model_name")
    dependency_statuses = {status.id: status for status in _model_dependency_repository().list_status()}
    upscalers: list[RuntimeUpscalerOptionResponse] = []
    for option in UPSCALERS:
        dependency_status = dependency_statuses.get(option.dependency_id) if option.dependency_id else None
        installed = option.kind != "model" or (dependency_status is not None and dependency_status.status == "installed")
        runtime_loaded = option.kind != "model" or (option.model_name in model_values)
        upscalers.append(
            RuntimeUpscalerOptionResponse(
                label=option.label,
                kind=option.kind,
                runtime_name=option.runtime_name,
                model_name=option.model_name,
                dependency_id=option.dependency_id,
                available=runtime_loaded,
                installed=installed,
                runtime_loaded=runtime_loaded,
                dependency_status=dependency_status.status if dependency_status else None,
            )
        )
    return RuntimeUpscalerResponse(
        base_url=app_config.comfy_base_url,
        upscalers=upscalers,
    )


@router.get("/dependencies", response_model=RuntimeDependencyListResponse)
def list_dependencies(feature: str | None = None) -> RuntimeDependencyListResponse:
    statuses = _dependency_repository().list_status()
    if feature is not None:
        statuses = [status for status in statuses if feature in status.features]
    return RuntimeDependencyListResponse(dependencies=[_dependency_to_response(status) for status in statuses])


@router.get("/core", response_model=ComfyRuntimeCoreStatusResponse)
def runtime_core_status() -> ComfyRuntimeCoreStatusResponse:
    return _core_to_response(_core_manager().status())


@router.post("/core/install", response_model=ComfyRuntimeCoreStatusResponse)
def install_runtime_core() -> ComfyRuntimeCoreStatusResponse:
    try:
        return _core_to_response(_core_manager().install())
    except ComfyRuntimeCoreError as exc:
        raise HTTPException(status_code=409, detail={"code": RUNTIME_CORE_INSTALL_FAILED}) from exc


@router.post("/core/repair", response_model=ComfyRuntimeCoreStatusResponse)
def repair_runtime_core() -> ComfyRuntimeCoreStatusResponse:
    try:
        return _core_to_response(_core_manager().repair())
    except ComfyRuntimeCoreError as exc:
        raise HTTPException(status_code=409, detail={"code": RUNTIME_CORE_REPAIR_FAILED}) from exc


@router.post("/core/update", response_model=ComfyRuntimeCoreStatusResponse)
def update_runtime_core() -> ComfyRuntimeCoreStatusResponse:
    try:
        return _core_to_response(_core_manager().update())
    except ComfyRuntimeCoreError as exc:
        raise HTTPException(status_code=409, detail={"code": RUNTIME_CORE_UPDATE_FAILED}) from exc


@router.post("/dependencies/{dependency_id}/install", response_model=RuntimeDependencyStatusResponse)
def install_dependency(dependency_id: str) -> RuntimeDependencyStatusResponse:
    app_config = AppConfig()
    repository = RuntimeDependencyRepository(
        manifest_path=app_config.runtime_dependencies_path,
        runtime_root=app_config.comfy_dir,
    )
    try:
        status = RuntimeDependencyInstaller(repository, runner=_runtime_dependency_runner(), python_path=app_config.comfy_python_path).install(dependency_id)
    except RuntimeDependencyNotFound as exc:
        raise HTTPException(status_code=404, detail={"code": RUNTIME_DEPENDENCY_NOT_FOUND}) from exc
    except RuntimeDependencyError as exc:
        raise HTTPException(status_code=409, detail={"code": RUNTIME_DEPENDENCY_INSTALL_FAILED}) from exc
    return _dependency_to_response(status)


def _install_dependencies(only_missing: bool, update: bool = False, repair: bool = False) -> RuntimeMaintenanceResponse:
    app_config = AppConfig()
    repository = RuntimeDependencyRepository(
        manifest_path=app_config.runtime_dependencies_path,
        runtime_root=app_config.comfy_dir,
    )
    installer = RuntimeDependencyInstaller(repository, runner=_runtime_dependency_runner(), python_path=app_config.comfy_python_path)
    statuses: list[RuntimeDependencyStatusResponse] = []
    errors: dict[str, str] = {}
    for status in repository.list_status():
        if only_missing and status.status == "installed":
            statuses.append(_dependency_to_response(status))
            continue
        try:
            if update:
                status = installer.update(status.id)
            elif repair:
                status = installer.repair(status.id)
            else:
                status = installer.install(status.id)
        except RuntimeDependencyError as exc:
            errors[status.id] = str(exc)
        statuses.append(_dependency_to_response(status))
    return RuntimeMaintenanceResponse(
        status="completed_with_errors" if errors else "completed",
        dependencies=statuses,
        errors=errors,
    )


@router.post("/install", response_model=RuntimeMaintenanceResponse)
def install_runtime_dependencies() -> RuntimeMaintenanceResponse:
    return _install_dependencies(only_missing=True)


@router.post("/repair", response_model=RuntimeMaintenanceResponse)
def repair_runtime_dependencies() -> RuntimeMaintenanceResponse:
    return _install_dependencies(only_missing=False, repair=True)


@router.post("/update", response_model=RuntimeMaintenanceResponse)
def update_runtime_dependencies() -> RuntimeMaintenanceResponse:
    return _install_dependencies(only_missing=False, update=True)


@router.post("/start", response_model=RuntimeStatus)
def start() -> RuntimeStatus:
    try:
        return _manager().start()
    except RuntimeProcessError as exc:
        raise HTTPException(status_code=409, detail={"code": RUNTIME_START_FAILED}) from exc


@router.post("/stop", response_model=RuntimeStatus)
def stop() -> RuntimeStatus:
    return _manager().stop()


@router.post("/restart", response_model=RuntimeStatus)
def restart() -> RuntimeStatus:
    manager = _manager()
    manager.stop()
    try:
        return manager.start()
    except RuntimeProcessError as exc:
        raise HTTPException(status_code=409, detail={"code": RUNTIME_START_FAILED}) from exc


@router.post("/interrupt")
def interrupt() -> dict:
    try:
        return ComfyClient(AppConfig().comfy_base_url).interrupt()
    except RuntimeErrorResponse as exc:
        raise HTTPException(status_code=502, detail={"code": RUNTIME_INTERRUPT_FAILED}) from exc


def _tail(path: str, max_bytes: int = 20000) -> str:
    log_path = AppConfig().comfy_dir / path
    try:
        if not log_path.exists():
            return ""
        data = log_path.read_bytes()[-max_bytes:]
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


@router.get("/logs", response_model=RuntimeLogsResponse)
def logs() -> RuntimeLogsResponse:
    return RuntimeLogsResponse(stdout=_tail("comfy.out.log"), stderr=_tail("comfy.err.log"))
