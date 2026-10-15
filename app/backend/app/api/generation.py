import json
from dataclasses import asdict
import logging
import os
from pathlib import Path
import asyncio
from datetime import UTC, datetime
import hashlib
import secrets
from itertools import product
import sqlite3
import threading
import time
import unicodedata
from typing import Any, Literal, Mapping
from urllib.parse import quote
from uuid import uuid4
import zipfile

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.core.config import AppConfig
from app.schemas.generation import (
    AdvancedStyleSelection,
    GenerationFolderResponse,
    GenerationHistoryResponse,
    GenerationTagCatalogResponse,
    GenerationResponse,
    Img2ImgRequest,
    InpaintRequest,
    PromptSyntaxGenerationRecord,
    Txt2ImgRequest,
    XYZAxis,
    XYZCellResult,
    XYZPlotRequest,
    XYZPlotResponse,
)
from app.schemas.common import LoraSelection
from app.schemas.prompt_choices import PromptChoicePins
from app.services.prompt_choices import PromptChoice
from app.schemas.media import MediaArtifact
from app.schemas.remaster import RemasterRequest, RemasterResponse
from app.schemas.operations import OperationChildSummary, OperationRecord
from app.services.history import HistoryBulkDeleteConflictError, HistoryRepository, HistoryTagLimitExceededError
from app.services.media_artifacts import MediaArtifactAccessError, resolve_registered_media_artifact
from app.services.video_sequence_history import VideoSequenceHistoryProjector
from app.services.artifact_lifecycle import (
    artifact_write_operation as _artifact_write_operation,
    begin_orphan_cleanup as _begin_orphan_cleanup,
    end_orphan_cleanup as _end_orphan_cleanup,
)
from app.services.image_validation import InvalidImageContentError, validate_image_file
from app.services.image_check_operation import ImageCheckOperation, capture_image_check_operation
from app.services.image_check import remap_checked_image_paths
from app.services.image_check_finalization import FinalImageSession, is_private_image_candidate
from app.services.library_thumbnails import LibraryThumbnailError, LibraryThumbnailService
from app.services.inpaint_region import InpaintRegionError, validate_only_masked_inputs
from app.services.external_settings import ExternalIntegrationSettingsRepository
from app.services.external_model_libraries import (
    ExternalModelLibraryError,
    ExternalModelLibraryRepository,
    ExternalModelRoot,
)
from app.services.generation_schema import GenerationSchema, GenerationSchemaNotFound, GenerationSchemaRepository
from app.services import api_errors
from app.services.comfy_model_paths import comfy_model_name
from app.services.controlnet_preprocessor import SUPPORTED_PREPROCESSORS
from app.services.anima_lllite_runtime import (
    anima_lllite_missing_nodes,
    anima_lllite_model_directory,
    anima_lllite_model_values,
    anima_lllite_runtime_contract,
    is_standard_controlnet_model_name,
)
from app.services.controlnet_mode import (
    controlnet_mode_contract_status,
    enabled_priority_controlnet_units,
    resolve_controlnet_mode_contract,
)
from app.services.detailer_catalog import detailer_detector_model
from app.services.eagle import EagleImportOptions, register_eagle_image
from app.services.generation_metadata import (
    finalize_merge_xyz_metadata,
    metadata_tags,
    record_anima_lllite_artifacts,
    record_controlnet_artifacts,
    record_detailer_metadata,
    record_hires_metadata,
    record_merge_xyz_metadata,
)
from app.services.generation_artifacts import (
    OutputFinalizationError,
    apply_filename_template as _apply_filename_template,
    apply_output_image_format as _apply_output_image_format,
    remap_primary_outputs,
    unique_output_path as _unique_output_path,
)
from app.services.generation_artifact_deletion import delete_generation_artifacts, skip_generation_artifacts
from app.services.lora_training import LoRATrainingDatasetRepository
from app.services.metadata import MetadataWriter, read_generation_parameters_from_image
from app.services.remaster import REMASTER_REQUIRED_NODES, RemasterExecutionError
from app.services.reference_mutation import (
    ReferenceMutationLease,
    reference_mutation_coordinator,
)
from app.services.model_library_cache import get_model_library
from app.services.model_library import ModelLibrary
from app.services.model_fingerprints import ModelFingerprintService
from app.services.generation_reproduction import capture_generation_assets, compare_generation_assets
from app.services.prompt_styles import (
    apply_advanced_style_text as _apply_advanced_style_text,
    compose_with_advanced_styles as _compose_with_advanced_styles,
)
from app.services.anima_lora_remap import (
    AnimaLoraRemapRegistryError,
    prepare_anima_lora_runtime_entries,
)
from app.services.anima_generation_bundle import (
    AnimaGenerationBundleError,
    prepare_anima_generation_bundle,
)
from app.services.model_profiles import ModelProfileRepository
from app.services.model_selections import ModelSelectionRepository
from app.services.model_storage import resolve_model_root
from app.services.operation_coordinator import OperationCoordinator
from app.services.operation_executor import (
    OperationCancelled,
    OperationExecutionContext,
    OperationExecutionError,
    OperationExecutionResult,
)
from app.services.operation_repository import (
    OperationNotFoundError,
    OperationRepository,
    OperationRetryError,
)
from app.services.operation_retention import OperationRetentionService
from app.services.preview_registry import PreviewRegistry
from app.services.parameter_mapper import ParameterMapper
from app.services.prompt_pipeline import expand_dynamic_prompt, extract_embeddings, extract_loras, list_wildcards
from app.services.prompt_syntax import (
    PROMPT_PARSER_GENERATION,
    PromptSyntaxError,
    PromptSyntaxPlan,
    parse_prompt_syntax,
)
from app.services.runtime import (
    ComfyClient,
    ComfyGenerationRuntime,
    INSPIRE_KSAMPLER_CLASS,
    LORA_BLOCK_WEIGHT_CLASS,
    RuntimeCancellationConfirmed,
    RuntimeErrorResponse,
)
from app.services.seed_resize import (
    resolve_seed_resize_contract,
    seed_resize_active,
    seed_resize_contract_status,
    seed_resize_requested,
    seed_resize_request_issues,
)
from app.services.spatial_runtime_contract import (
    SPATIAL_PROMPT_CONTRACT_VERSION,
    spatial_runtime_contract_status,
    uses_spatial_prompt_layout,
)
from app.services.spatial_prompt_contract import validate_spatial_prompt_contract
from app.services.spatial_generation_context import (
    PreparedSpatialGenerationContext,
    SpatialGenerationContextError,
    SpatialStagePreparation,
    prepare_spatial_generation_context,
)
from app.services.spatial_mask_compiler import SpatialMaskCompilerError
from app.services.spatial_profile_resolver import (
    SpatialModelEvidence,
    infer_sdxl_spatial_profile_id,
    sdxl_spatial_model_evidence,
)
from app.services.media_model_catalog import (
    SafetensorsHeaderError,
    read_safetensors_header,
)
from app.services.anima_model_classifier import AnimaModelClassifier
from app.services.model_library import ModelLibraryItem
from app.services.runtime_adapters import AsteriaGenerationRuntime, AsteriaNativeEngineAdapter, ExternalProcessRuntimeAdapter, RuntimeAdapterError
from app.services.sampling_catalog import sampler_runtime_name, scheduler_runtime_name
from app.services.topaz import TopazError, TopazSettings, process_topaz_image
from app.services.upscaler_catalog import upscaler_model_name, upscaler_option
from app.services.user_settings import UserSettingsRepository, resolve_output_root, safe_segment
from app.services.xyz_grid import XYZGridAxis, XYZGridCell, create_xyz_contact_sheet_streaming
from app.services.workflow_mapping import WorkflowMappingNotFound, WorkflowMappingRepository
from app.services.schema_versions import CURRENT_SCHEMA_VERSIONS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/generations", tags=["generations"])


class GenerationSchemaResponse(BaseModel):
    profile: str
    id: str
    name: str
    mode: str
    model_family: str
    fields: dict[str, dict[str, Any]]
    defaults: dict[str, Any]


class WildcardItemResponse(BaseModel):
    name: str
    path: str
    option_count: int


class WildcardListResponse(BaseModel):
    items: list[WildcardItemResponse]


class PromptExpandRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    seed: int = -1
    advanced_styles: list[AdvancedStyleSelection] = Field(default_factory=list)
    prompt_pins: PromptChoicePins | None = None


class PromptExpandResponse(BaseModel):
    prompt: str
    expanded_prompt: str
    selections: list[str]
    choices: list[PromptChoice] = Field(default_factory=list)
    prompt_hash: str = ''


class DeleteGenerationResponse(BaseModel):
    id: int
    deleted: bool
    deleted_files: list[str]
    missing_files: list[str]
    skipped_files: list[str]


class BulkDeleteGenerationRequest(BaseModel):
    ids: list[int] = Field(min_length=1, max_length=500)
    delete_files: bool = False


class BulkDeleteGenerationResponse(BaseModel):
    deleted_ids: list[int]
    delete_files: bool
    deleted_files: list[str]
    missing_files: list[str]
    skipped_files: list[str]
    protected_files: list[str]


class HistoryIntegrityResponse(BaseModel):
    missing_files: dict[int, list[str]]
    orphan_files: list[str]


class CleanupResponse(BaseModel):
    deleted_files: list[str]
    skipped_files: list[str]


class ExportedBundleFile(BaseModel):
    source_path: str
    archive_path: str
    size_bytes: int
    sha256: str


class ExportBundleResponse(BaseModel):
    path: str
    file_count: int = 0
    missing_files: list[str] = Field(default_factory=list)
    bundle_size_bytes: int = 0
    bundle_sha256: str = ""
    files: list[ExportedBundleFile] = Field(default_factory=list)


class UpdateGenerationCurationRequest(BaseModel):
    rating: int | None = Field(default=None, ge=0, le=5)
    favorite: bool | None = None
    tags: list[str] | None = None
    notes: str | None = None
    folder_id: int | None = None
    update_folder: bool = False


class CreateGenerationFolderRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class BulkGenerationCurationRequest(BaseModel):
    ids: list[int] = Field(min_length=1, max_length=500)
    rating: int | None = Field(default=None, ge=0, le=5)
    favorite: bool | None = None
    tags_add: list[str] = Field(default_factory=list, max_length=100)
    tags_remove: list[str] = Field(default_factory=list, max_length=100)
    folder_id: int | None = None
    update_folder: bool = False

    @field_validator("tags_add", "tags_remove")
    @classmethod
    def normalize_bulk_tags(cls, values: list[str]) -> list[str]:
        normalized = [" ".join(str(value).strip().split()) for value in values]
        if any(len(value) > 120 for value in normalized):
            raise ValueError("bulk tag entries must be 120 characters or fewer")
        return normalized


class LibraryScanResponse(BaseModel):
    imported: int
    skipped: int
    items: list[dict[str, Any]]


class GenerationJobResponse(BaseModel):
    id: str
    status: str
    mode: str
    progress: float = 0
    current: int = 0
    total: int = 1
    step: int = 0
    steps: int = 0
    phase: str = ""
    generation: GenerationResponse | None = None
    remaster: RemasterResponse | None = None
    preview_path: str = ""
    preview_paths: list[str] = []
    error: str = ""
    error_target_ids: list[str] = Field(default_factory=list)
    cancelled: bool = False


class GenerationJobCleanupResponse(BaseModel):
    removed_jobs: int
    remaining_active_jobs: int
    remaining_terminal_jobs: int
    removed_preview_references: int


def _data_dir() -> Path:
    return Path(os.environ.get("ASTERIA_DATA_DIR", Path(__file__).resolve().parents[3] / "data"))


AVAILABLE_RUNTIMES = {
    "asteria": "Packet preview",
    "asteria_engine": "Asteria engine preview",
    "external_process": "External process",
    "comfy_workflow": "ComfyUI real generation",
}
CONTROLNET_REQUIRED_NODES = {
    "LoadImage",
    "ControlNetLoader",
    "ControlNetApplyAdvanced",
}
CONTROLNET_UNION_REQUIRED_NODES = {"SetUnionControlNetType"}
ANIMA_LLLITE_REQUIRED_NODES = {
    "LoadImage",
    "AnimaLLLiteApply",
}
ANIMA_REGIONAL_REQUIRED_NODES = {
    "AttentionCouplePPM",
    "ImageColorToMask",
    "SolidMask",
}
SDXL_BASE_REQUIRED_NODES = {
    "CheckpointLoaderSimple",
    "CLIPTextEncodeSDXL",
    "KSampler",
    "VAEDecode",
    "SaveImage",
}
SDXL_REFINER_REQUIRED_NODES = {"KSamplerAdvanced", "CLIPTextEncodeSDXLRefiner"}
DETAILER_REQUIRED_NODES = {"FaceDetailer", "UltralyticsDetectorProvider"}
LORA_BLOCK_WEIGHT_REQUIRED_NODES = {LORA_BLOCK_WEIGHT_CLASS}
HIRES_COMMON_REQUIRED_NODES = {"KSampler", "VAEDecode"}
_jobs_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_publisher_leases: dict[str, ReferenceMutationLease] = {}
_operation_coordinators: dict[Path, OperationCoordinator] = {}


def _generation_operation_coordinator() -> OperationCoordinator:
    db_path = (_data_dir() / "asteria.db").absolute()
    coordinator = _operation_coordinators.get(db_path)
    if coordinator is None:
        coordinator = OperationCoordinator(OperationRepository(db_path), auto_start=False)
        _register_generation_executors(coordinator)
        _operation_coordinators[db_path] = coordinator
    return coordinator


def configure_generation_operation_coordinator(
    coordinator: OperationCoordinator,
) -> OperationCoordinator:
    _register_generation_executors(coordinator)
    _operation_coordinators[coordinator.repository.db_path.absolute()] = coordinator
    return coordinator


def _register_generation_executors(coordinator: OperationCoordinator) -> None:
    from app.services.prompt_comparison_jobs import PromptComparisonExecutor

    for mode in ("txt2img", "img2img", "inpaint"):
        coordinator.register_executor(f"generation.{mode}", _GenerationOperationExecutor(mode))
    coordinator.register_executor("generation.remaster", _RemasterOperationExecutor())
    coordinator.register_executor("generation.prompt_comparison", PromptComparisonExecutor(coordinator.repository.db_path))


class _GenerationOperationExecutor:
    version = "generation-v1.1"

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def resource_claims(self, _snapshot: dict) -> tuple[str, ...]:
        return ("comfy_gpu_exclusive",)

    def execute(self, context: OperationExecutionContext) -> OperationExecutionResult:
        return _execute_generation_operation(context, self.mode)


class _RemasterOperationExecutor:
    version = "remaster-v1.1"

    def resource_claims(self, _snapshot: dict) -> tuple[str, ...]:
        return ("comfy_gpu_exclusive",)

    def execute(self, context: OperationExecutionContext) -> OperationExecutionResult:
        return _execute_generation_operation(context, "remaster")


def _execute_generation_operation(
    context: OperationExecutionContext,
    mode: str,
) -> OperationExecutionResult:
        job_id = context.operation.operation_id
        payload = context.request_snapshot
        request_type = {
            "txt2img": Txt2ImgRequest,
            "img2img": Img2ImgRequest,
            "inpaint": InpaintRequest,
            "remaster": RemasterRequest,
        }[mode]
        request_payload = dict(payload["request"])
        submitted_fields = payload.get("request_fields_set")
        if isinstance(submitted_fields, list):
            request_payload = {
                key: value
                for key, value in request_payload.items()
                if key in set(submitted_fields)
            }
        discord_snapshot = payload.get("discord")
        if isinstance(discord_snapshot, dict):
            model_references = discord_snapshot.get("model_references")
            if isinstance(model_references, dict) and model_references:
                request_payload["model_slot_overrides"] = _resolve_discord_model_slots(
                    {
                        str(slot_id): str(stable_model_id)
                        for slot_id, stable_model_id in model_references.items()
                    }
                )
        request = request_type.model_validate(request_payload)
        with _jobs_lock:
            cache = _jobs.setdefault(job_id, _new_job_cache(mode, request))
            cache["request"] = request
            cache["generation_operation_snapshot"] = payload.get("generation_operation_snapshot")
        lease = _publisher_leases.pop(job_id, None)
        if lease is None:
            lease = reference_mutation_coordinator.acquire_publisher()
        try:
            context.raise_if_cancelled()
            if mode == "remaster":
                from app.api.remaster import execute_remaster

                result = execute_remaster(
                    request,
                    cancel_requested=lambda: context.cancel_requested
                    or bool(_jobs.get(job_id, {}).get("cancelled")),
                    runtime_job_callback=lambda prompt_id: _set_job(job_id, runtime_job_id=prompt_id),
                    runtime_stop_confirmed=lambda: bool(
                        _jobs.get(job_id, {}).get("runtime_stop_confirmed")
                    ),
                )
            elif mode == "img2img":
                result = _generate(request, "img2img", job_id=job_id)
            elif mode == "inpaint":
                result = _generate(request, "inpaint", job_id=job_id)
            else:
                result = _generate(request, "txt2img", job_id=job_id)
            if mode != "remaster" and getattr(result, "status", None) == "cancelled":
                _set_job(job_id, generation=result)
                context.acknowledge_cancel()
                raise OperationCancelled(artifact_refs=_generation_artifact_refs(result))
            context.raise_if_cancelled()
            result_field = "remaster" if mode == "remaster" else "generation"
            _set_job(job_id, **{result_field: result})
            artifact_refs = _generation_artifact_refs(result)
            child_summaries, warning_codes = _generation_child_summaries(
                context,
                request,
                result,
                artifact_refs,
            )
            return OperationExecutionResult(
                artifact_refs=artifact_refs,
                warning_codes=warning_codes,
                child_summaries=child_summaries,
            )
        except OperationCancelled:
            raise
        except RemasterExecutionError as exc:
            code = exc.code if isinstance(exc.code, str) else "remaster_failed"
            if code == "remaster_cancelled":
                raise OperationCancelled() from exc
            raise OperationExecutionError(code, retryable=True) from exc
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            code = detail.get("code")
            if not isinstance(code, str):
                code = "generation_request_failed"
            safe_parts = [code]
            unsupported = detail.get("unsupported")
            if isinstance(unsupported, list):
                for item in unsupported:
                    reason = item.get("reason") if isinstance(item, dict) else None
                    if isinstance(reason, str) and all(
                        character.isalnum() or character in "_.:-" for character in reason
                    ):
                        safe_parts.append(reason[:127])
            missing_nodes = detail.get("missing_nodes")
            if isinstance(missing_nodes, list):
                safe_parts.extend(
                    value
                    for value in missing_nodes
                    if isinstance(value, str)
                    and 0 < len(value) <= 128
                    and all(character.isalnum() or character in "_.:-" for character in value)
                )
            target_ids = detail.get("target_ids")
            safe_target_ids = [
                value for value in target_ids
                if isinstance(value, str)
                and 0 < len(value) <= 128
                and all(character.isalnum() or character in "_-" for character in value)
            ] if isinstance(target_ids, list) else []
            _set_job(
                job_id,
                error=":".join(dict.fromkeys(safe_parts)),
                error_target_ids=list(dict.fromkeys(safe_target_ids)),
            )
            raise OperationExecutionError(code, retryable=False) from exc
        finally:
            try:
                PreviewRegistry(_data_dir() / "asteria.db").mark_terminal(job_id)
            except (OSError, sqlite3.Error):
                pass
            if lease is not None:
                lease.release()


def _generation_child_summaries(
    context: OperationExecutionContext,
    request: Txt2ImgRequest | RemasterRequest,
    result: Any,
    artifact_refs: tuple[str, ...],
) -> tuple[tuple[OperationChildSummary, ...], tuple[str, ...]]:
    if not isinstance(request, Txt2ImgRequest) or not isinstance(result, GenerationResponse):
        return (), ()
    completed_at = datetime.now(UTC)
    started_at = context.operation.started_at or completed_at
    summaries: list[OperationChildSummary] = []
    warning_codes: list[str] = []
    if request.restore_faces or any(unit.enabled for unit in request.detailer_units):
        summaries.append(OperationChildSummary(
            child_summary_id="detailer",
            ordinal=len(summaries),
            kind="detailer",
            status="completed",
            started_at=started_at,
            completed_at=completed_at,
            artifact_refs=artifact_refs,
        ))
    if request.topaz_enabled or request.hires_upscaler == "Topaz Photo AI":
        topaz_failed = str(result.settings.get("topaz_postprocess", "")).startswith("failed:")
        status = "failed" if topaz_failed else "completed"
        error_code = "topaz_process_failed" if topaz_failed else None
        if error_code:
            warning_codes.append(error_code)
        summaries.append(OperationChildSummary(
            child_summary_id="topaz",
            ordinal=len(summaries),
            kind="topaz",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            artifact_refs=artifact_refs,
            warning_codes=(error_code,) if error_code else (),
            error_code=error_code,
        ))
    if request.remaster.enabled:
        remaster_failed = bool(result.settings.get("remaster_partial_failure"))
        error_code = "remaster_partial_failure" if remaster_failed else None
        if error_code:
            warning_codes.append(error_code)
        summaries.append(OperationChildSummary(
            child_summary_id="remaster",
            ordinal=len(summaries),
            kind="remaster",
            status="failed" if remaster_failed else "completed",
            started_at=started_at,
            completed_at=completed_at,
            artifact_refs=artifact_refs,
            warning_codes=(error_code,) if error_code else (),
            error_code=error_code,
        ))
    return tuple(summaries), tuple(warning_codes)


def _generation_artifact_refs(result: Any) -> tuple[str, ...]:
    if isinstance(result, GenerationResponse):
        return (f"history:{result.id}",)
    if isinstance(result, RemasterResponse):
        return (f"remaster:{result.id}",)
    return ()


def _new_job_cache(mode: str, request: Txt2ImgRequest | RemasterRequest) -> dict[str, Any]:
    return {
        "mode": mode,
        "request": request,
        "generation": None,
        "remaster": None,
        "preview_path": "",
        "preview_paths": [],
        "step": 0,
        "steps": 0,
        "runtime_job_id": "",
        "runtime_stop_confirmed": False,
        "cancel_warnings": [],
        "error_target_ids": [],
        "cancelled": False,
    }


def _runtime(config: AppConfig, runtime_id: str | None = None):
    selected_runtime = runtime_id or config.generation_runtime
    if selected_runtime not in AVAILABLE_RUNTIMES:
        raise HTTPException(
            status_code=400,
            detail={"code": api_errors.GENERATION_RUNTIME_NOT_FOUND, "runtime": selected_runtime},
        )
    if selected_runtime == "asteria":
        return AsteriaGenerationRuntime()
    if selected_runtime == "asteria_engine":
        return AsteriaGenerationRuntime(adapter=AsteriaNativeEngineAdapter())
    if selected_runtime == "external_process":
        return AsteriaGenerationRuntime(adapter=ExternalProcessRuntimeAdapter(config.engine_command))
    if selected_runtime == "comfy_workflow":
        return ComfyGenerationRuntime(
            ComfyClient(config.comfy_base_url),
            model_root=config.model_dir,
            external_roots=_registered_external_model_roots(config),
            comfy_output_dir=config.comfy_output_dir,
            topaz_executable=_topaz_executable(config),
        )
    raise HTTPException(
        status_code=500,
        detail={"code": api_errors.GENERATION_RUNTIME_UNSUPPORTED, "runtime": selected_runtime},
    )


def _selected_runtime_id(config: AppConfig, runtime_id: str | None = None) -> str:
    selected_runtime = runtime_id or config.generation_runtime
    if selected_runtime not in AVAILABLE_RUNTIMES:
        raise HTTPException(
            status_code=400,
            detail={"code": api_errors.GENERATION_RUNTIME_NOT_FOUND, "runtime": selected_runtime},
        )
    return selected_runtime


def _validate_controlnet_mode_object_info(
    config: AppConfig,
    object_info: dict[str, Any],
    settings: dict[str, Any],
) -> None:
    if not enabled_priority_controlnet_units(settings.get("controlnet_units")):
        return
    try:
        contract = resolve_controlnet_mode_contract(WorkflowMappingRepository(config.mapping_dir), "sdxl")
        status = controlnet_mode_contract_status(object_info, contract)
    except ValueError:
        status = None
    if status is not None and status.ready:
        return
    raise HTTPException(
        status_code=409,
        detail={
            "code": api_errors.MISSING_CONTROLNET_MODE_RUNTIME_NODES,
            "issues": list(status.issues) if status is not None else ["mapping"],
        },
    )


def _preflight_controlnet_mode_request(request: Txt2ImgRequest, mode: str = "txt2img") -> None:
    priority_units = enabled_priority_controlnet_units(request.controlnet_units)
    if not priority_units:
        return
    config = AppConfig()
    selected_runtime = _selected_runtime_id(config, request.runtime)
    if request.profile != "sdxl" or selected_runtime != "comfy_workflow":
        raise HTTPException(
            status_code=409,
            detail={
                "code": api_errors.MISSING_CONTROLNET_MODE_RUNTIME_NODES,
                "profile": request.profile,
                "runtime": selected_runtime,
                "issues": ["sdxl_comfy_workflow_required"],
            },
        )
    try:
        object_info = ComfyClient(config.comfy_base_url).object_info()
    except RuntimeErrorResponse as exc:
        raise HTTPException(status_code=502, detail={"code": api_errors.GENERATION_RUNTIME_UNREACHABLE}) from exc
    settings, _negative_prompt = _settings_from_request(request, config, mode)
    _validate_controlnet_mode_object_info(config, object_info, settings)


def _validate_spatial_prompt_runtime_contract(
    object_info: dict[str, Any],
    *,
    required: bool,
) -> None:
    if not required:
        return
    status = spatial_runtime_contract_status(object_info)
    if not status.available:
        raise HTTPException(
            status_code=409,
            detail={
                "code": api_errors.MISSING_SPATIAL_PROMPT_RUNTIME_CONTRACT,
                "missing_components": list(status.missing_components),
                "contract_version": SPATIAL_PROMPT_CONTRACT_VERSION,
            },
        )


_SPATIAL_PROMPT_SUPPORTED_SAMPLERS = {
    "euler_ancestral": "Euler a",
    "dpmpp_2m": "DPM++ 2M",
    "er_sde": "ER SDE",
}


def _validate_spatial_prompt_sampling_options(settings: dict) -> None:
    if not uses_spatial_prompt_layout(settings):
        return
    candidates = [("sampler", str(settings.get("sampler") or "Euler a"))]
    if settings.get("hires_enabled"):
        hires_sampler = str(settings.get("hires_sampler") or "Use same sampler")
        if hires_sampler != "Use same sampler":
            candidates.append(("hires_sampler", hires_sampler))
    unsupported = [
        {
            "field": field,
            "label": label,
            "reason": "spatial_prompt_sampler_not_supported",
            "supported": list(_SPATIAL_PROMPT_SUPPORTED_SAMPLERS.values()),
        }
        for field, label in candidates
        if sampler_runtime_name(label) not in _SPATIAL_PROMPT_SUPPORTED_SAMPLERS
    ]
    if unsupported:
        raise HTTPException(
            status_code=409,
            detail={
                "code": api_errors.UNSUPPORTED_COMFY_RUNTIME_SETTING,
                "unsupported": unsupported,
            },
        )


def _validate_runtime_feature_requirements(
    config: AppConfig,
    selected_runtime: str,
    profile: str,
    settings: dict,
    loras: list[dict] | None = None,
    model_slots: dict[str, str] | None = None,
) -> None:
    requires_spatial_prompt = uses_spatial_prompt_layout(settings)
    remaster_settings = settings.get("remaster", {})
    requires_remaster = isinstance(remaster_settings, dict) and bool(remaster_settings.get("enabled"))
    if requires_remaster and selected_runtime != "comfy_workflow":
        raise HTTPException(
            status_code=409,
            detail={"code": api_errors.REMASTER_RUNTIME_NOT_SUPPORTED, "runtime": selected_runtime},
        )
    if seed_resize_requested(settings) and selected_runtime != "comfy_workflow":
        raise HTTPException(
            status_code=409,
            detail={
                "code": api_errors.UNSUPPORTED_COMFY_RUNTIME_SETTING,
                "unsupported": [
                    {
                        "field": "seed_resize",
                        "reason": "seed_resize_requires_comfy_workflow",
                    }
                ],
            },
        )
    if requires_spatial_prompt and selected_runtime != "comfy_workflow":
        raise HTTPException(
            status_code=409,
            detail={
                "code": api_errors.UNSUPPORTED_COMFY_RUNTIME_SETTING,
                "unsupported": [
                    {
                        "field": "prompt_layout",
                        "reason": "spatial_prompt_requires_comfy_workflow",
                    }
                ],
            },
        )
    _validate_spatial_prompt_sampling_options(settings)
    if selected_runtime != "comfy_workflow":
        return
    requires_anima_lllite = bool(settings.get("anima_lllite_units"))
    requires_anima_regional = _uses_anima_regional_prompting(settings)
    requires_detailer = bool(settings.get("detailer_units")) or bool(settings.get("restore_faces"))
    if requires_anima_lllite and profile != "anima":
        raise HTTPException(
            status_code=409,
            detail={
                "code": api_errors.UNSUPPORTED_COMFY_RUNTIME_SETTING,
                "unsupported": [
                    {
                        "field": "anima_lllite_units",
                        "reason": "anima_lllite_is_mapped_only_for_anima_profile",
                    }
                ],
            },
        )
    missing_anima_lllite_images = _missing_anima_lllite_images(settings)
    if missing_anima_lllite_images:
        raise HTTPException(
            status_code=400,
            detail={"code": api_errors.MISSING_ANIMA_LLLITE_IMAGE, "missing": missing_anima_lllite_images},
        )
    if profile != "sdxl":
        unsupported_profile_settings = _unsupported_profile_runtime_settings(
            profile,
            settings,
            model_root=config.model_dir,
        )
        if unsupported_profile_settings:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": api_errors.UNSUPPORTED_COMFY_RUNTIME_SETTING,
                    "unsupported": unsupported_profile_settings,
                },
            )
        requires_runtime_model_check = bool(_runtime_model_slot_checks(profile))
        if (
            not loras
            and not requires_runtime_model_check
            and not requires_anima_lllite
            and not requires_remaster
            and not requires_detailer
            and not requires_spatial_prompt
        ):
            return
        try:
            object_info = ComfyClient(config.comfy_base_url).object_info()
        except RuntimeErrorResponse as exc:
            raise HTTPException(status_code=502, detail={"code": api_errors.GENERATION_RUNTIME_UNREACHABLE}) from exc
        missing_runtime_models = _missing_runtime_model_slots(
            profile,
            object_info,
            model_slots or {},
            model_root=config.model_dir,
        )
        if missing_runtime_models:
            raise HTTPException(
                status_code=409,
                detail={"code": api_errors.MISSING_RUNTIME_MODEL, "missing": missing_runtime_models},
            )
        missing_anima_bundle_models = _missing_anima_bundle_runtime_models(
            object_info,
            settings,
            model_slots or {},
            model_root=config.model_dir,
        )
        if missing_anima_bundle_models:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": api_errors.MISSING_RUNTIME_MODEL,
                    "missing": missing_anima_bundle_models,
                },
            )
        missing_hires_checkpoint = _missing_hires_checkpoint_model(
            object_info,
            settings,
            profile,
            model_root=config.model_dir,
        )
        if missing_hires_checkpoint:
            raise HTTPException(
                status_code=409,
                detail={"code": api_errors.MISSING_RUNTIME_MODEL, "missing": [missing_hires_checkpoint]},
            )
        nodes = {key for key, value in object_info.items() if isinstance(key, str) and isinstance(value, dict)}
        _validate_spatial_prompt_runtime_contract(
            object_info,
            required=requires_spatial_prompt,
        )
        unsupported_seed_variation = _unsupported_seed_variation_runtime_settings(profile, nodes, settings)
        if unsupported_seed_variation:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": api_errors.UNSUPPORTED_COMFY_RUNTIME_SETTING,
                    "unsupported": unsupported_seed_variation,
                },
            )
        missing_profile_nodes = sorted(
            _profile_base_required_nodes(profile, settings, requires_lora=bool(loras)) - nodes
        )
        if missing_profile_nodes:
            raise HTTPException(
                status_code=409,
                detail={"code": api_errors.MISSING_PROFILE_RUNTIME_NODES, "missing_nodes": missing_profile_nodes},
            )
        missing_profile_hires_nodes = sorted(_profile_hires_required_nodes(settings) - nodes)
        if missing_profile_hires_nodes:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": api_errors.MISSING_HIRES_FIX_RUNTIME_NODES,
                    "missing_nodes": missing_profile_hires_nodes,
                },
            )
        _validate_hires_upscaler_model(object_info, settings)
        _validate_remaster_runtime_requirements(profile, object_info, settings)
        _validate_profile_sampling_options(profile, object_info, settings)
        if requires_detailer:
            missing = sorted(DETAILER_REQUIRED_NODES - nodes)
            if missing:
                raise HTTPException(
                    status_code=409,
                    detail={"code": api_errors.MISSING_DETAILER_RUNTIME_NODES, "missing_nodes": missing},
                )
            missing_detector_models = _missing_detailer_detector_models(object_info, settings)
            if missing_detector_models:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": api_errors.MISSING_DETAILER_DETECTOR_MODEL,
                        "missing": missing_detector_models,
                    },
                )
        if requires_anima_lllite:
            missing = sorted(
                set(ANIMA_LLLITE_REQUIRED_NODES - nodes)
                | set(anima_lllite_missing_nodes(object_info, nodes))
            )
            if missing:
                raise HTTPException(
                    status_code=409,
                    detail={"code": api_errors.MISSING_ANIMA_LLLITE_RUNTIME_NODES, "missing_nodes": missing},
                )
            lllite_contract = anima_lllite_runtime_contract(object_info)
            if lllite_contract is None:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": api_errors.MISSING_ANIMA_LLLITE_RUNTIME_NODES,
                        "missing_nodes": ["AnimaLLLiteApply"],
                    },
                )
            settings["anima_lllite_runtime_contract"] = lllite_contract
            if requires_anima_regional:
                missing = sorted(ANIMA_REGIONAL_REQUIRED_NODES - nodes)
                if missing:
                    raise HTTPException(
                        status_code=409,
                        detail={"code": api_errors.MISSING_ANIMA_REGIONAL_RUNTIME_NODES, "missing_nodes": missing},
                    )
            missing_lllite_models = _missing_anima_lllite_models(
                object_info,
                settings,
                model_root=config.model_dir,
                external_roots=_registered_external_model_roots(config),
            )
            if missing_lllite_models:
                raise HTTPException(
                    status_code=409,
                    detail={"code": api_errors.MISSING_ANIMA_LLLITE_MODEL, "missing": missing_lllite_models},
                )
        missing_lora_models = _missing_lora_models(
            object_info,
            loras or [],
            model_root=config.model_dir,
            external_roots=_registered_external_model_roots(config),
        )
        if missing_lora_models:
            raise HTTPException(
                status_code=409,
                detail={"code": api_errors.MISSING_LORA_MODEL, "missing": missing_lora_models},
            )
        return
    unsupported_settings = _unsupported_comfy_runtime_settings(settings, profile)
    if unsupported_settings:
        raise HTTPException(
            status_code=409,
            detail={"code": api_errors.UNSUPPORTED_COMFY_RUNTIME_SETTING, "unsupported": unsupported_settings},
        )
    missing_controlnet_images = _missing_controlnet_images(settings)
    if missing_controlnet_images:
        raise HTTPException(
            status_code=400,
            detail={"code": api_errors.MISSING_CONTROLNET_IMAGE, "missing": missing_controlnet_images},
        )
    missing_image_inputs = _missing_generation_image_inputs(settings)
    if missing_image_inputs:
        raise HTTPException(
            status_code=400,
            detail={"code": api_errors.MISSING_GENERATION_IMAGE_INPUT, "missing": missing_image_inputs},
        )
    unsupported_controlnet_preprocessors = _unsupported_controlnet_preprocessors(settings)
    if unsupported_controlnet_preprocessors:
        raise HTTPException(
            status_code=409,
            detail={
                "code": api_errors.UNSUPPORTED_CONTROLNET_PREPROCESSOR,
                "unsupported": unsupported_controlnet_preprocessors,
                "supported": sorted(SUPPORTED_PREPROCESSORS),
            },
        )
    requires_controlnet = bool(settings.get("controlnet_units"))
    requires_lora_block_weight = settings.get("lora_block_weight_runtime") == "requested"
    requires_lora = bool(loras)
    hires_required_nodes = _hires_fix_required_nodes(settings)
    try:
        object_info = ComfyClient(config.comfy_base_url).object_info()
    except RuntimeErrorResponse as exc:
        raise HTTPException(status_code=502, detail={"code": api_errors.GENERATION_RUNTIME_UNREACHABLE}) from exc
    nodes = {key for key, value in object_info.items() if isinstance(key, str) and isinstance(value, dict)}
    _validate_spatial_prompt_runtime_contract(
        object_info,
        required=requires_spatial_prompt,
    )
    _validate_seed_resize_object_info(config, object_info, settings)
    unsupported_seed_variation = _unsupported_seed_variation_runtime_settings(profile, nodes, settings)
    if unsupported_seed_variation:
        raise HTTPException(
            status_code=409,
            detail={
                "code": api_errors.UNSUPPORTED_COMFY_RUNTIME_SETTING,
                "unsupported": unsupported_seed_variation,
            },
        )
    missing_runtime_models = _missing_runtime_model_slots(
        profile,
        object_info,
        model_slots or {},
        model_root=config.model_dir,
    )
    if missing_runtime_models:
        raise HTTPException(
            status_code=409,
            detail={"code": api_errors.MISSING_RUNTIME_MODEL, "missing": missing_runtime_models},
        )
    missing_hires_checkpoint = _missing_hires_checkpoint_model(
        object_info,
        settings,
        profile,
        model_root=config.model_dir,
    )
    if missing_hires_checkpoint:
        raise HTTPException(
            status_code=409,
            detail={"code": api_errors.MISSING_RUNTIME_MODEL, "missing": [missing_hires_checkpoint]},
        )
    missing_refiner_checkpoint = _missing_refiner_checkpoint_model(
        object_info,
        settings,
        model_root=config.model_dir,
    )
    if missing_refiner_checkpoint:
        raise HTTPException(
            status_code=409,
            detail={"code": api_errors.MISSING_RUNTIME_MODEL, "missing": [missing_refiner_checkpoint]},
        )
    _validate_comfy_sampling_options(object_info, settings)
    _validate_hires_upscaler_model(object_info, settings)
    _validate_remaster_runtime_requirements(profile, object_info, settings)
    if requires_controlnet:
        required_nodes = set(CONTROLNET_REQUIRED_NODES)
        if _uses_union_controlnet(settings):
            required_nodes.update(CONTROLNET_UNION_REQUIRED_NODES)
        missing = sorted(required_nodes - nodes)
        if missing:
            raise HTTPException(
                status_code=409,
                detail={"code": api_errors.MISSING_CONTROLNET_RUNTIME_NODES, "missing_nodes": missing},
            )
        _validate_controlnet_mode_object_info(config, object_info, settings)
        missing_controlnet_models = _missing_controlnet_models(
            object_info,
            settings,
            model_root=config.model_dir,
            external_roots=_registered_external_model_roots(config),
        )
        if missing_controlnet_models:
            raise HTTPException(
                status_code=409,
                detail={"code": api_errors.MISSING_CONTROLNET_MODEL, "missing": missing_controlnet_models},
            )
    if requires_detailer:
        missing = sorted(DETAILER_REQUIRED_NODES - nodes)
        if missing:
            raise HTTPException(
                status_code=409,
                detail={"code": api_errors.MISSING_DETAILER_RUNTIME_NODES, "missing_nodes": missing},
            )
        missing_detector_models = _missing_detailer_detector_models(object_info, settings)
        if missing_detector_models:
            raise HTTPException(
                status_code=409,
                detail={"code": api_errors.MISSING_DETAILER_DETECTOR_MODEL, "missing": missing_detector_models},
            )
    if requires_lora_block_weight:
        missing = sorted(LORA_BLOCK_WEIGHT_REQUIRED_NODES - nodes)
        if missing:
            raise HTTPException(
                status_code=409,
                detail={"code": api_errors.MISSING_LORA_BLOCK_WEIGHT_RUNTIME_NODES, "missing_nodes": missing},
            )
    if requires_lora:
        missing_lora_models = _missing_lora_models(
            object_info,
            loras or [],
            model_root=config.model_dir,
            external_roots=_registered_external_model_roots(config),
        )
        if missing_lora_models:
            raise HTTPException(
                status_code=409,
                detail={"code": api_errors.MISSING_LORA_MODEL, "missing": missing_lora_models},
            )
    if hires_required_nodes:
        missing = sorted(hires_required_nodes - nodes)
        if missing:
            raise HTTPException(
                status_code=409,
                detail={"code": api_errors.MISSING_HIRES_FIX_RUNTIME_NODES, "missing_nodes": missing},
            )
    missing_base_nodes = sorted(_sdxl_base_required_nodes(settings, model_slots or {}) - nodes)
    if missing_base_nodes:
        raise HTTPException(
            status_code=409,
            detail={"code": api_errors.MISSING_SDXL_BASE_RUNTIME_NODES, "missing_nodes": missing_base_nodes},
        )
    if _uses_refiner(settings):
        missing_refiner_nodes = sorted(SDXL_REFINER_REQUIRED_NODES - nodes)
        if missing_refiner_nodes:
            raise HTTPException(
                status_code=409,
                detail={"code": api_errors.MISSING_SDXL_BASE_RUNTIME_NODES, "missing_nodes": missing_refiner_nodes},
            )


def _unsupported_comfy_runtime_settings(
    settings: dict,
    profile: str = "sdxl",
) -> list[dict[str, str]]:
    return seed_resize_request_issues(
        profile,
        str(settings.get("mode") or "txt2img"),
        settings,
    )


def _validate_seed_resize_object_info(
    config: AppConfig,
    object_info: dict[str, Any],
    settings: dict[str, Any],
) -> None:
    if not seed_resize_active(settings):
        return
    try:
        contract = resolve_seed_resize_contract(WorkflowMappingRepository(config.mapping_dir))
        status = seed_resize_contract_status(object_info, contract)
    except ValueError:
        status = None
    if status is not None and status.ready:
        return
    raise HTTPException(
        status_code=409,
        detail={
            "code": api_errors.UNSUPPORTED_COMFY_RUNTIME_SETTING,
            "unsupported": [
                {
                    "field": "seed_resize",
                    "reason": "seed_resize_requires_compatible_inspire_nodes",
                    "issues": list(status.issues) if status is not None else ["mapping"],
                }
            ],
        },
    )


def _preflight_seed_resize_request(
    request: Txt2ImgRequest,
    mode: str = "txt2img",
) -> None:
    config = AppConfig()
    settings, _negative_prompt = _settings_from_request(request, config, mode)
    if not seed_resize_requested(settings):
        return
    selected_runtime = _selected_runtime_id(config, request.runtime)
    request_issues = seed_resize_request_issues(request.profile, mode, settings)
    if selected_runtime != "comfy_workflow":
        request_issues = [
            {
                "field": "seed_resize",
                "reason": "seed_resize_requires_comfy_workflow",
            }
        ]
    if request_issues:
        raise HTTPException(
            status_code=409,
            detail={
                "code": api_errors.UNSUPPORTED_COMFY_RUNTIME_SETTING,
                "unsupported": request_issues,
            },
        )
    try:
        object_info = ComfyClient(config.comfy_base_url).object_info()
    except RuntimeErrorResponse as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": api_errors.GENERATION_RUNTIME_UNREACHABLE},
        ) from exc
    _validate_seed_resize_object_info(config, object_info, settings)


def _seed_variation_requested(settings: dict) -> bool:
    subseed = int(settings.get("subseed") if settings.get("subseed") is not None else -1)
    return subseed != -1 or float(settings.get("subseed_strength") or 0) != 0


def _unsupported_seed_variation_runtime_settings(
    profile: str,
    nodes: set[str],
    settings: dict,
) -> list[dict[str, str]]:
    if not _seed_variation_requested(settings):
        return []
    if profile != "sdxl":
        reason = "seed_variation_is_mapped_only_for_sdxl_with_inspire_ksampler"
    elif INSPIRE_KSAMPLER_CLASS not in nodes:
        reason = "seed_variation_requires_inspire_ksampler"
    else:
        return []
    return [{"field": "seed_variation", "reason": reason}]


def _unsupported_profile_runtime_settings(
    profile: str,
    settings: dict,
    *,
    model_root: Path | None = None,
) -> list[dict[str, str]]:
    if profile == "sdxl":
        return []
    unsupported = _unsupported_comfy_runtime_settings(settings, profile)
    unsupported.extend(_unsupported_seed_variation_runtime_settings(profile, set(), settings))
    if int(settings.get("clip_skip") or 1) > 1:
        unsupported.append(
            {
                "field": "clip_skip",
                "reason": "clip_skip_is_mapped_only_for_sdxl_profile",
            }
        )
    if settings.get("restore_faces") and profile != "anima":
        unsupported.append(
            {
                "field": "restore_faces",
                "reason": "restore_faces_is_not_mapped_to_this_comfy_profile",
            }
        )
    if settings.get("controlnet_units"):
        unsupported.append(
            {
                "field": "controlnet_units",
                "reason": "controlnet_is_not_mapped_to_this_comfy_profile",
            }
        )
    if settings.get("detailer_units") and profile != "anima":
        unsupported.append(
            {
                "field": "detailer_units",
                "reason": "detailer_is_not_mapped_to_this_comfy_profile",
            }
        )
    if settings.get("lora_block_weight_runtime") == "requested":
        unsupported.append(
            {
                "field": "lora_block_weight",
                "reason": "lora_block_weight_is_not_mapped_to_this_comfy_profile",
            }
        )
    if settings.get("refiner_enabled"):
        unsupported.append({"field": "refiner", "reason": "refiner_is_not_mapped_to_this_comfy_profile"})
    if settings.get("hires_enabled"):
        hires_checkpoint = str(settings.get("hires_checkpoint") or "").strip()
        if hires_checkpoint and hires_checkpoint != "Use same checkpoint":
            normalized_checkpoint = comfy_model_name(
                hires_checkpoint,
                "diffusion_models",
                model_root=model_root,
            )
            if profile == "anima":
                if not normalized_checkpoint.lower().startswith("anima\\"):
                    unsupported.append(
                        {
                            "field": "hires_checkpoint",
                            "reason": "hires_diffusion_model_must_be_anima_family",
                        }
                    )
            elif profile not in {"flux1", "flux2"}:
                unsupported.append(
                    {
                        "field": "hires_checkpoint",
                        "reason": "hires_checkpoint_is_not_mapped_to_the_current_comfy_workflow",
                    }
                )
        if profile in {"anima", "flux1", "flux2"} and str(settings.get("hires_prompt") or "").strip():
            unsupported.append(
                {
                    "field": "hires_prompt",
                    "reason": f"hires_prompt_is_not_mapped_to_{profile}_second_pass",
                }
            )
        if profile in {"anima", "flux1", "flux2"} and str(settings.get("hires_negative_prompt") or "").strip():
            unsupported.append(
                {
                    "field": "hires_negative_prompt",
                    "reason": f"hires_negative_prompt_is_not_mapped_to_{profile}_second_pass",
                }
            )
    return unsupported


def _unsupported_controlnet_preprocessors(settings: dict) -> list[dict[str, Any]]:
    unsupported: list[dict[str, Any]] = []
    units = settings.get("controlnet_units")
    if not isinstance(units, list):
        return unsupported
    for index, unit in enumerate(units):
        if not isinstance(unit, dict) or not unit.get("enabled", True):
            continue
        preprocessor = str(unit.get("preprocessor") or "none").lower()
        if preprocessor not in SUPPORTED_PREPROCESSORS:
            unsupported.append({"index": index, "preprocessor": preprocessor})
    return unsupported


def _uses_union_controlnet(settings: dict) -> bool:
    units = settings.get("controlnet_units")
    if not isinstance(units, list):
        return False
    return any(
        isinstance(unit, dict)
        and unit.get("enabled", True)
        and "union" in str(unit.get("model") or "").lower()
        for unit in units
    )


_PATH_ACCESS_ERRORS = (TypeError, OSError, ValueError, RuntimeError)


def _path_exists_safely(path: Path | str) -> bool:
    try:
        return Path(path).exists()
    except _PATH_ACCESS_ERRORS:
        return False


def _path_is_file_safely(path: Path | str) -> bool:
    try:
        return Path(path).is_file()
    except _PATH_ACCESS_ERRORS:
        return False


def _resolved_path_string_safely(path: Path | str) -> str | None:
    try:
        return str(Path(path).resolve())
    except _PATH_ACCESS_ERRORS:
        return None


def _resolve_path_safely(path: Path | str) -> Path | None:
    try:
        return Path(path).resolve()
    except _PATH_ACCESS_ERRORS:
        return None


def _missing_controlnet_images(settings: dict) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    units = settings.get("controlnet_units")
    if not isinstance(units, list):
        return missing
    for index, unit in enumerate(units):
        if not isinstance(unit, dict) or not unit.get("enabled", True):
            continue
        image_path = str(unit.get("image_path") or "").strip()
        if not image_path:
            missing.append({"index": index, "reason": "image_path_required"})
        elif not _path_exists_safely(image_path) or not _path_is_file_safely(image_path):
            missing.append({"index": index, "reason": "image_path_not_found"})
    return missing


def _missing_anima_lllite_images(settings: dict) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    units = settings.get("anima_lllite_units")
    if not isinstance(units, list):
        return missing
    for index, unit in enumerate(units):
        if not isinstance(unit, dict) or not unit.get("enabled", True):
            continue
        for field in ("image_path", "mask_image_path"):
            image_path = str(unit.get(field) or "").strip()
            if not image_path and field == "mask_image_path":
                continue
            if not image_path:
                missing.append({"index": index, "field": field, "reason": "image_path_required"})
            elif not _path_exists_safely(image_path) or not _path_is_file_safely(image_path):
                missing.append({"index": index, "field": field, "reason": "image_path_not_found"})
    return missing


def _uses_anima_regional_prompting(settings: dict) -> bool:
    units = settings.get("anima_lllite_units")
    if not isinstance(units, list):
        return False
    return any(
        isinstance(unit, dict)
        and unit.get("enabled", True)
        and isinstance(unit.get("regional_prompts"), list)
        and any(
            isinstance(region, dict) and str(region.get("prompt") or "").strip()
            for region in unit["regional_prompts"]
        )
        for unit in units
    )


def _record_anima_lllite_runtime_settings(settings: dict) -> None:
    regional_runtime = False
    for unit in settings.get("anima_lllite_units", []):
        if not isinstance(unit, dict) or not unit.get("enabled", True):
            continue
        unit["runtime"] = "comfy_workflow"
        active_regions = [
            region
            for region in unit.get("regional_prompts", [])
            if isinstance(region, dict) and str(region.get("prompt") or "").strip()
        ]
        if not active_regions:
            continue
        unit["regional_prompt_runtime"] = "attention_couple_ppm"
        for region in active_regions:
            region["runtime"] = "attention_couple_ppm"
        regional_runtime = True
    if regional_runtime:
        settings["anima_regional_prompt_runtime"] = "attention_couple_ppm"


def _missing_generation_image_inputs(settings: dict) -> list[dict[str, str]]:
    mode = str(settings.get("mode") or "txt2img")
    checks: list[tuple[str, str]] = []
    if mode in {"img2img", "inpaint"}:
        checks.append(("init_image_path", str(settings.get("init_image_path") or "").strip()))
    if mode == "inpaint":
        checks.append(("mask_image_path", str(settings.get("mask_image_path") or "").strip()))

    missing: list[dict[str, str]] = []
    for field, value in checks:
        if not value:
            missing.append({"field": field, "reason": "image_path_required"})
        elif not _path_exists_safely(value) or not _path_is_file_safely(value):
            missing.append({"field": field, "reason": "image_path_not_found"})
    return missing


def _missing_controlnet_models(
    object_info: dict,
    settings: dict,
    *,
    model_root: Path | None = None,
    external_roots: tuple[ExternalModelRoot, ...] = (),
) -> list[dict[str, Any]]:
    units = settings.get("controlnet_units")
    if not isinstance(units, list) or "ControlNetLoader" not in object_info:
        return []
    model_values = {
        value
        for value in _node_input_values(object_info, "ControlNetLoader", "control_net_name")
        if is_standard_controlnet_model_name(value)
    }
    missing: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        if not isinstance(unit, dict) or not unit.get("enabled", True):
            continue
        model = str(unit.get("model") or "").strip()
        if not model:
            continue
        model_name = comfy_model_name(
            model,
            "controlnet",
            model_root=model_root,
            external_roots=external_roots,
        )
        if model_name not in model_values:
            missing.append({"index": index, "model": model, "model_name": model_name})
    return missing


def _missing_anima_lllite_models(
    object_info: dict,
    settings: dict,
    *,
    model_root: Path | None = None,
    external_roots: tuple[ExternalModelRoot, ...] = (),
) -> list[dict[str, Any]]:
    units = settings.get("anima_lllite_units")
    contract = anima_lllite_runtime_contract(object_info)
    if not isinstance(units, list) or contract is None:
        return []
    model_values = _anima_lllite_model_values(object_info)
    missing: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        if not isinstance(unit, dict) or not unit.get("enabled", True):
            continue
        model = str(unit.get("model") or "").strip()
        if not model:
            missing.append({"index": index, "model": model, "reason": "model_required"})
            continue
        model_name = comfy_model_name(
            model,
            anima_lllite_model_directory(contract),
            model_root=model_root,
            external_roots=external_roots,
        )
        if model_name not in model_values:
            missing.append({"index": index, "model": model, "model_name": model_name})
    return missing


def _anima_lllite_model_values(object_info: dict) -> set[str]:
    return anima_lllite_model_values(object_info)


def _missing_detailer_detector_models(object_info: dict, settings: dict) -> list[dict[str, Any]]:
    units = settings.get("detailer_units")
    if "UltralyticsDetectorProvider" not in object_info:
        return []
    model_values = _node_input_values(object_info, "UltralyticsDetectorProvider", "model_name")
    missing: list[dict[str, Any]] = []
    normalized_units: list[tuple[int | str, dict[str, Any]]] = []
    if settings.get("restore_faces"):
        normalized_units.append(("restore_faces", {"enabled": True, "detector": "face_yolo"}))
    if isinstance(units, list):
        normalized_units.extend((index, unit) for index, unit in enumerate(units))
    for index, unit in normalized_units:
        if not isinstance(unit, dict) or not unit.get("enabled", True):
            continue
        detector = str(unit.get("detector") or "face_yolo")
        model_name = detailer_detector_model(detector)
        if model_name not in model_values:
            missing.append({"index": index, "detector": detector, "model_name": model_name})
    return missing


def _missing_lora_models(
    object_info: dict,
    loras: list[dict],
    *,
    model_root: Path | None = None,
    external_roots: tuple[ExternalModelRoot, ...] = (),
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    lora_loader_present = "LoraLoader" in object_info
    anima_remap_loader_present = "AsteriaAnimaLoraRemap" in object_info
    block_loader_present = LORA_BLOCK_WEIGHT_CLASS in object_info
    lora_loader_values = _node_model_input_values(object_info, "LoraLoader", "lora_name")
    block_loader_values = _node_model_input_values(object_info, LORA_BLOCK_WEIGHT_CLASS, "lora_name")
    anima_remap_loader_values = _node_model_input_values(
        object_info,
        "AsteriaAnimaLoraRemap",
        "lora_name",
    )
    for index, lora in enumerate(loras):
        if not isinstance(lora, dict):
            continue
        name = str(lora.get("name") or "").strip()
        if not name:
            continue
        model_name = comfy_model_name(
            name,
            "loras",
            model_root=model_root,
            external_roots=external_roots,
        )
        block_vector = str(lora.get("block_weight") or lora.get("block_weight_preset") or "").strip()
        uses_block_loader = bool(block_vector and lora.get("hierarchy_mode") != "disabled")
        uses_anima_remap = isinstance(lora.get("anima_remap"), dict)
        loader_present = (
            anima_remap_loader_present
            if uses_anima_remap
            else block_loader_present
            if uses_block_loader
            else lora_loader_present
        )
        available = (
            anima_remap_loader_values
            if uses_anima_remap
            else block_loader_values
            if uses_block_loader
            else lora_loader_values
        )
        if loader_present and model_name not in available:
            missing.append({"index": index, "name": name, "model_name": model_name})
    return missing


def _missing_runtime_model_slots(
    profile: str,
    object_info: dict,
    model_slots: dict[str, str],
    *,
    model_root: Path | None = None,
) -> list[dict[str, str]]:
    checks = _runtime_model_slot_checks(profile)
    missing: list[dict[str, str]] = []
    for slot, model_dir_name, node_name, input_name in checks:
        value = str(model_slots.get(slot) or "").strip()
        if not value or node_name not in object_info:
            continue
        model_name = comfy_model_name(value, model_dir_name, model_root=model_root)
        available = _node_model_input_values(object_info, node_name, input_name)
        if model_name not in available:
            missing.append({"slot": slot, "model_name": model_name})
    return missing


def _missing_hires_checkpoint_model(
    object_info: dict,
    settings: dict,
    profile: str = "sdxl",
    *,
    model_root: Path | None = None,
) -> dict[str, str] | None:
    if not settings.get("hires_enabled"):
        return None
    checkpoint = str(settings.get("hires_checkpoint") or "").strip()
    if not checkpoint or checkpoint == "Use same checkpoint":
        return None
    if profile in {"anima", "flux1", "flux2"}:
        model_dir_name, node_name, input_name = "diffusion_models", "UNETLoader", "unet_name"
    elif profile == "sdxl":
        model_dir_name, node_name, input_name = "checkpoints", "CheckpointLoaderSimple", "ckpt_name"
    else:
        return None
    if node_name not in object_info:
        return None
    model_name = comfy_model_name(checkpoint, model_dir_name, model_root=model_root)
    available = _node_model_input_values(object_info, node_name, input_name)
    if model_name not in available:
        return {"slot": "hires_checkpoint", "model_name": model_name}
    return None


def _uses_refiner(settings: dict) -> bool:
    return bool(settings.get("refiner_enabled") and str(settings.get("refiner_checkpoint") or "").strip())


def _missing_refiner_checkpoint_model(
    object_info: dict,
    settings: dict,
    *,
    model_root: Path | None = None,
) -> dict[str, str] | None:
    checkpoint = str(settings.get("refiner_checkpoint") or "").strip()
    if not _uses_refiner(settings) or "CheckpointLoaderSimple" not in object_info:
        return None
    model_name = comfy_model_name(checkpoint, "checkpoints", model_root=model_root)
    available = _node_model_input_values(object_info, "CheckpointLoaderSimple", "ckpt_name")
    if model_name not in available:
        return {"slot": "refiner_checkpoint", "model_name": model_name}
    return None


def _record_hires_checkpoint_runtime(settings: dict, selected_runtime: str, profile: str) -> None:
    if (
        selected_runtime != "comfy_workflow"
        or profile not in {"sdxl", "anima", "flux1", "flux2"}
        or not settings.get("hires_enabled")
    ):
        return
    checkpoint = str(settings.get("hires_checkpoint") or "").strip()
    if not checkpoint or checkpoint == "Use same checkpoint":
        return
    if profile in {"anima", "flux1", "flux2"}:
        settings["hires_checkpoint_runtime"] = (
            "topaz_second_pass_diffusion_model_slot"
            if settings.get("hires_upscaler") == "Topaz Photo AI"
            else "second_pass_diffusion_model_loader"
        )
        settings["hires_checkpoint_kind"] = "diffusion_model"
    else:
        settings["hires_checkpoint_runtime"] = (
            "topaz_second_pass_model_slot"
            if settings.get("hires_upscaler") == "Topaz Photo AI"
            else "second_pass_checkpoint_loader"
        )


def _record_hires_runtime(settings: dict, selected_runtime: str, profile: str) -> None:
    if selected_runtime != "comfy_workflow" or not settings.get("hires_enabled"):
        return
    if settings.get("hires_upscaler") == "Topaz Photo AI":
        return
    settings["hires_runtime"] = {
        "sdxl": "sdxl_ksampler_second_pass",
        "flux1": "flux1_custom_sampler_second_pass",
        "flux2": "flux2_custom_sampler_second_pass",
        "anima": "anima_custom_sampler_second_pass",
    }.get(profile, "comfy_custom_sampler_second_pass")


def _record_refiner_runtime(settings: dict, selected_runtime: str, profile: str) -> None:
    if selected_runtime != "comfy_workflow" or profile != "sdxl" or not _uses_refiner(settings):
        return
    settings["refiner_runtime"] = "sdxl_advanced_sampler"


def _runtime_model_slot_checks(profile: str) -> tuple[tuple[str, str, str, str], ...]:
    if profile == "sdxl":
        return (
            ("checkpoint", "checkpoints", "CheckpointLoaderSimple", "ckpt_name"),
            ("vae", "vae", "VAELoader", "vae_name"),
        )
    if profile == "anima":
        return (
            ("diffusion_model", "diffusion_models", "UNETLoader", "unet_name"),
            ("text_encoder", "text_encoders", "CLIPLoader", "clip_name"),
            ("vae", "vae", "VAELoader", "vae_name"),
        )
    if profile == "flux2":
        return (
            ("diffusion_model", "diffusion_models", "UNETLoader", "unet_name"),
            ("text_encoder", "text_encoders", "CLIPLoader", "clip_name"),
            ("vae", "vae", "VAELoader", "vae_name"),
        )
    if profile == "flux1":
        return (
            ("diffusion_model", "diffusion_models", "UNETLoader", "unet_name"),
            ("clip_l", "text_encoders", "DualCLIPLoader", "clip_name1"),
            ("t5xxl", "text_encoders", "DualCLIPLoader", "clip_name2"),
            ("vae", "vae", "VAELoader", "vae_name"),
        )
    return ()


def _runtime_model_slots_for_request(
    profile: str,
    model_slots: dict[str, str],
    *,
    model_root: Path,
    external_roots: tuple[ExternalModelRoot, ...] = (),
) -> dict[str, str]:
    model_directories = {
        slot: model_dir_name
        for slot, model_dir_name, _node_name, _input_name in _runtime_model_slot_checks(profile)
    }
    if profile == "anima":
        model_directories.update(
            {
                "qwen35_text_encoder": "text_encoders",
                "progressive_adapter": "text_encoders",
            }
        )
    projected = dict(model_slots)
    for slot, model_dir_name in model_directories.items():
        value = str(projected.get(slot) or "").strip()
        if value:
            projected[slot] = comfy_model_name(
                value,
                model_dir_name,
                model_root=model_root,
                external_roots=external_roots,
            )
    return projected


def _registered_external_model_roots(config: AppConfig) -> tuple[ExternalModelRoot, ...]:
    try:
        return tuple(
            ExternalModelLibraryRepository(
                config.data_dir / "config" / "external-model-libraries.json"
            ).list()
        )
    except ExternalModelLibraryError:
        return ()


def _missing_anima_bundle_runtime_models(
    object_info: dict,
    settings: dict,
    model_slots: dict[str, str],
    *,
    model_root: Path | None = None,
) -> list[dict[str, str]]:
    bundle = settings.get("anima_generation_bundle")
    if not isinstance(bundle, dict) or bundle.get("variant_id") != "anima_3_8b_52":
        return []
    checks = (
        (
            "qwen35_text_encoder",
            "AnimaQwen35Loader",
            "qwen35_model",
        ),
        (
            "progressive_adapter",
            "AnimaQwen35UnifiedPrompt",
            "adapter_name",
        ),
    )
    missing: list[dict[str, str]] = []
    for slot, node, input_name in checks:
        value = str(model_slots.get(slot) or "").strip()
        if not value or node not in object_info:
            continue
        model_name = comfy_model_name(
            value,
            "text_encoders",
            model_root=model_root,
        )
        if model_name not in _node_model_input_values(object_info, node, input_name):
            missing.append({"slot": slot, "model_name": model_name})
    return missing


def _profile_base_required_nodes(profile: str, settings: dict, *, requires_lora: bool = False) -> set[str]:
    if profile == "anima":
        bundle = settings.get("anima_generation_bundle")
        is_anima_3_8b = (
            isinstance(bundle, dict)
            and bundle.get("variant_id") == "anima_3_8b_52"
        )
        required = {
            "UNETLoader",
            "CLIPLoader",
            "VAELoader",
            "RandomNoise",
            "CFGGuider",
            "KSamplerSelect",
            "BasicScheduler",
            "SamplerCustomAdvanced",
            "VAEDecode",
            "SaveImage",
        }
        if is_anima_3_8b:
            required.update(
                {
                    "AnimaQwen35Loader",
                    "AnimaQwen35UnifiedPrompt",
                    "CLIPTextEncode",
                    "EmptySD3LatentImage",
                }
            )
        else:
            required.add("TextEncodeQwenImageEdit")
        mode = str(settings.get("mode") or "txt2img")
        if mode == "inpaint":
            required.update({"LoadImage", "VAEEncode", "SetLatentNoiseMask"})
        elif mode == "img2img":
            required.update({"LoadImage", "VAEEncode"})
        elif not is_anima_3_8b:
            required.add("EmptyQwenImageLayeredLatentImage")
        if requires_lora:
            if int(settings.get("anima_remap_lora_count") or 0) > 0:
                required.add("AsteriaAnimaLoraRemap")
            if int(settings.get("anima_standard_lora_count") or 0) > 0:
                required.add("LoraLoader")
        return required
    if profile == "flux1":
        required = {
            "UNETLoader",
            "DualCLIPLoader",
            "VAELoader",
            "ModelSamplingFlux",
            "CLIPTextEncodeFlux",
            "RandomNoise",
            "BasicGuider",
            "KSamplerSelect",
            "BasicScheduler",
            "SamplerCustomAdvanced",
            "VAEDecode",
            "SaveImage",
        }
        mode = str(settings.get("mode") or "txt2img")
        if mode == "inpaint":
            required.update({"LoadImage", "SetLatentNoiseMask", "VAEEncode"})
        elif mode == "img2img":
            required.update({"LoadImage", "VAEEncode"})
        else:
            required.add("EmptySD3LatentImage")
        if requires_lora:
            required.add("LoraLoader")
        return required
    if profile == "flux2":
        required = {
            "UNETLoader",
            "CLIPLoader",
            "VAELoader",
            "CLIPTextEncode",
            "FluxGuidance",
            "RandomNoise",
            "BasicGuider",
            "KSamplerSelect",
            "Flux2Scheduler",
            "SamplerCustomAdvanced",
            "VAEDecode",
            "SaveImage",
        }
        mode = str(settings.get("mode") or "txt2img")
        if mode == "inpaint":
            required.update({"LoadImage", "SetLatentNoiseMask", "VAEEncode"})
        elif mode == "img2img":
            required.update({"LoadImage", "VAEEncode"})
        else:
            required.add("EmptyFlux2LatentImage")
        denoise = settings.get("denoise", settings.get("denoising_strength", 1.0))
        if float(denoise) < 1.0:
            required.add("SplitSigmas")
        if requires_lora:
            required.add("LoraLoader")
        return required
    return set()


def _profile_hires_required_nodes(settings: dict) -> set[str]:
    if not settings.get("hires_enabled"):
        return set()
    if settings.get("hires_upscaler") == "Topaz Photo AI":
        return {"LoadImage", "VAEEncode"}
    option = upscaler_option(settings.get("hires_upscaler"))
    has_explicit_resize = bool(int(settings.get("hires_resize_width") or 0) or int(settings.get("hires_resize_height") or 0))
    if option.kind == "latent":
        return {"LatentUpscale" if has_explicit_resize else "LatentUpscaleBy"}
    if option.kind == "image":
        return {"VAEEncode", "ImageScale" if has_explicit_resize else "ImageScaleBy"}
    if option.kind == "model":
        return {"VAEEncode", "UpscaleModelLoader", "ImageUpscaleWithModel", "ImageScale"}
    return set()


def _node_input_values(object_info: dict, node_name: str, input_name: str) -> set[str]:
    node = object_info.get(node_name)
    if not isinstance(node, dict):
        return set()
    required = ((node.get("input") or {}).get("required") or {})
    raw = required.get(input_name)
    if not isinstance(raw, list) or not raw:
        return set()
    values = raw[0]
    if values == "COMBO" and len(raw) > 1 and isinstance(raw[1], dict):
        values = raw[1].get("options")
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values}


def _node_model_input_values(object_info: dict, node_name: str, input_name: str) -> set[str]:
    return {value.replace("/", "\\") for value in _node_input_values(object_info, node_name, input_name)}


def _validate_comfy_sampling_options(object_info: dict, settings: dict) -> None:
    sampler_values = _node_input_values(object_info, "KSampler", "sampler_name")
    scheduler_values = _node_input_values(object_info, "KSampler", "scheduler")
    unsupported: list[dict[str, str]] = []
    _append_unsupported_sampling_option(
        unsupported,
        field="sampler",
        label=str(settings.get("sampler") or "Euler a"),
        runtime_name=sampler_runtime_name(str(settings.get("sampler") or "Euler a")),
        available=sampler_values,
    )
    _append_unsupported_sampling_option(
        unsupported,
        field="scheduler",
        label=str(settings.get("scheduler") or "Automatic"),
        runtime_name=scheduler_runtime_name(str(settings.get("scheduler") or "Automatic")),
        available=scheduler_values,
    )
    if settings.get("hires_enabled"):
        hires_sampler = str(settings.get("hires_sampler") or "Use same sampler")
        hires_scheduler = str(settings.get("hires_scheduler") or "Use same scheduler")
        if hires_sampler != "Use same sampler":
            _append_unsupported_sampling_option(
                unsupported,
                field="hires_sampler",
                label=hires_sampler,
                runtime_name=sampler_runtime_name(hires_sampler),
                available=sampler_values,
            )
        if hires_scheduler != "Use same scheduler":
            _append_unsupported_sampling_option(
                unsupported,
                field="hires_scheduler",
                label=hires_scheduler,
                runtime_name=scheduler_runtime_name(hires_scheduler),
                available=scheduler_values,
            )
    detailer_units = settings.get("detailer_units")
    if isinstance(detailer_units, list):
        for index, unit in enumerate(detailer_units):
            if not isinstance(unit, dict) or not unit.get("enabled", True):
                continue
            detailer_sampler = str(unit.get("sampler") or "")
            detailer_scheduler = str(unit.get("scheduler") or "")
            if detailer_sampler:
                _append_unsupported_sampling_option(
                    unsupported,
                    field="detailer_sampler",
                    label=detailer_sampler,
                    runtime_name=sampler_runtime_name(detailer_sampler),
                    available=sampler_values,
                    index=index,
                )
            if detailer_scheduler:
                _append_unsupported_sampling_option(
                    unsupported,
                    field="detailer_scheduler",
                    label=detailer_scheduler,
                    runtime_name=scheduler_runtime_name(detailer_scheduler),
                    available=scheduler_values,
                    index=index,
                )
    if unsupported:
        raise HTTPException(
            status_code=409,
            detail={"code": api_errors.UNSUPPORTED_COMFY_SAMPLING_OPTION, "unsupported": unsupported},
        )


def _validate_profile_sampling_options(profile: str, object_info: dict, settings: dict) -> None:
    if profile not in {"anima", "flux1", "flux2"}:
        return
    sampler_values = _node_input_values(object_info, "KSamplerSelect", "sampler_name")
    unsupported: list[dict[str, str]] = []
    _append_unsupported_sampling_option(
        unsupported,
        field="sampler",
        label=str(settings.get("sampler") or "Euler"),
        runtime_name=sampler_runtime_name(str(settings.get("sampler") or "Euler")),
        available=sampler_values,
    )
    if profile != "flux2":
        scheduler_values = _node_input_values(object_info, "BasicScheduler", "scheduler")
        _append_unsupported_sampling_option(
            unsupported,
            field="scheduler",
            label=str(settings.get("scheduler") or "Simple"),
            runtime_name=scheduler_runtime_name(str(settings.get("scheduler") or "Simple")),
            available=scheduler_values,
        )
    if settings.get("hires_enabled"):
        hires_sampler = str(settings.get("hires_sampler") or "Use same sampler")
        if hires_sampler != "Use same sampler":
            _append_unsupported_sampling_option(
                unsupported,
                field="hires_sampler",
                label=hires_sampler,
                runtime_name=sampler_runtime_name(hires_sampler),
                available=sampler_values,
            )
        if profile != "flux2":
            hires_scheduler = str(settings.get("hires_scheduler") or "Use same scheduler")
            if hires_scheduler != "Use same scheduler":
                _append_unsupported_sampling_option(
                    unsupported,
                    field="hires_scheduler",
                    label=hires_scheduler,
                    runtime_name=scheduler_runtime_name(hires_scheduler),
                    available=scheduler_values,
                )
    if profile == "anima":
        detailer_units = settings.get("detailer_units")
        if isinstance(detailer_units, list):
            for index, unit in enumerate(detailer_units):
                if not isinstance(unit, dict) or not unit.get("enabled", True):
                    continue
                detailer_sampler = str(unit.get("sampler") or "")
                detailer_scheduler = str(unit.get("scheduler") or "")
                if detailer_sampler:
                    _append_unsupported_sampling_option(
                        unsupported,
                        field="detailer_sampler",
                        label=detailer_sampler,
                        runtime_name=sampler_runtime_name(detailer_sampler),
                        available=sampler_values,
                        index=index,
                    )
                if detailer_scheduler:
                    _append_unsupported_sampling_option(
                        unsupported,
                        field="detailer_scheduler",
                        label=detailer_scheduler,
                        runtime_name=scheduler_runtime_name(detailer_scheduler),
                        available=scheduler_values,
                        index=index,
                    )
    if unsupported:
        raise HTTPException(
            status_code=409,
            detail={"code": api_errors.UNSUPPORTED_COMFY_SAMPLING_OPTION, "unsupported": unsupported},
        )


def _append_unsupported_sampling_option(
    unsupported: list[dict[str, str]],
    *,
    field: str,
    label: str,
    runtime_name: str,
    available: set[str],
    index: int | None = None,
) -> None:
    if available and runtime_name not in available:
        item: dict[str, Any] = {"field": field, "label": label, "runtime_name": runtime_name}
        if index is not None:
            item["index"] = index
        unsupported.append(item)


def _validate_hires_upscaler_model(object_info: dict, settings: dict) -> None:
    if not settings.get("hires_enabled"):
        return
    option = upscaler_option(settings.get("hires_upscaler"))
    if option.kind != "model":
        return
    model_name = upscaler_model_name(settings.get("hires_upscaler"), settings.get("hires_upscale_model"))
    if "UpscaleModelLoader" not in object_info:
        return
    model_values = _node_input_values(object_info, "UpscaleModelLoader", "model_name")
    if model_name and model_name not in model_values:
        raise HTTPException(
            status_code=409,
            detail={"code": api_errors.MISSING_HIRES_UPSCALER_MODEL, "model_name": model_name},
        )


def _validate_remaster_runtime_requirements(profile: str, object_info: dict, settings: dict) -> None:
    remaster = settings.get("remaster", {})
    if not isinstance(remaster, dict) or not remaster.get("enabled"):
        return
    nodes = {key for key, value in object_info.items() if isinstance(key, str) and isinstance(value, dict)}
    required_nodes = set(REMASTER_REQUIRED_NODES)
    denoise = remaster.get("denoise")
    if profile == "flux2" and float(0.35 if denoise is None else denoise) < 1:
        required_nodes.add("SplitSigmas")
    missing_nodes = sorted(required_nodes - nodes)
    if missing_nodes:
        raise HTTPException(
            status_code=409,
            detail={"code": api_errors.MISSING_REMASTER_RUNTIME_NODES, "missing_nodes": missing_nodes},
        )
    anchor_upscaler = str(remaster.get("anchor_upscaler") or "R-ESGAN4x")
    option = upscaler_option(anchor_upscaler)
    model_name = upscaler_model_name(anchor_upscaler)
    if option.kind != "model" or not model_name:
        raise HTTPException(
            status_code=409,
            detail={"code": api_errors.REMASTER_UPSCALER_UNAVAILABLE, "model": model_name or anchor_upscaler},
        )
    available = _node_input_values(object_info, "UpscaleModelLoader", "model_name")
    normalized = {value.replace("/", "\\") for value in available}
    if not available or model_name.replace("/", "\\") not in normalized:
        raise HTTPException(
            status_code=409,
            detail={"code": api_errors.REMASTER_UPSCALER_UNAVAILABLE, "model": model_name},
        )


def _hires_fix_required_nodes(settings: dict) -> set[str]:
    if not settings.get("hires_enabled"):
        return set()
    if settings.get("hires_upscaler") == "Topaz Photo AI":
        return {"LoadImage", "VAEEncode"}
    required = set(HIRES_COMMON_REQUIRED_NODES)
    option = upscaler_option(settings.get("hires_upscaler"))
    has_explicit_resize = bool(int(settings.get("hires_resize_width") or 0) or int(settings.get("hires_resize_height") or 0))
    if option.kind == "latent":
        required.add("LatentUpscale" if has_explicit_resize else "LatentUpscaleBy")
    elif option.kind == "image":
        required.update({"VAEEncode", "ImageScale" if has_explicit_resize else "ImageScaleBy"})
    elif option.kind == "model":
        required.update({"VAEEncode", "UpscaleModelLoader", "ImageUpscaleWithModel", "ImageScale"})
    return required


def _sdxl_base_required_nodes(settings: dict, model_slots: dict[str, str]) -> set[str]:
    required = set(SDXL_BASE_REQUIRED_NODES)
    if model_slots.get("vae"):
        required.add("VAELoader")
    if int(settings.get("clip_skip") or 1) > 1:
        required.add("CLIPSetLastLayer")
    mode = str(settings.get("mode") or "txt2img")
    if mode == "inpaint":
        required.update({"LoadImage", "VAEEncode", "SetLatentNoiseMask"})
    elif mode == "img2img":
        required.update({"LoadImage", "VAEEncode"})
    else:
        required.add("EmptyLatentImage")
    return required

def _schema_repository(config: AppConfig) -> GenerationSchemaRepository:
    return GenerationSchemaRepository(config.schema_dir)


def _model_root(config: AppConfig) -> Path:
    return resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)


def _resolve_anima_hires_diffusion_path(
    settings: dict[str, Any],
    *,
    model_root: Path,
    library: ModelLibrary,
) -> Path | None:
    if not settings.get("hires_enabled"):
        return None
    value = str(settings.get("hires_checkpoint") or "").strip()
    if not value or value == "Use same checkpoint":
        return None
    direct = Path(value)
    if direct.is_absolute() and _path_is_file_safely(str(direct)):
        return direct
    expected_name = comfy_model_name(
        value,
        "diffusion_models",
        model_root=model_root,
    )
    matches = [
        item.path
        for item in library.by_type("checkpoint")
        if comfy_model_name(
            str(item.path),
            "diffusion_models",
            model_root=model_root,
        ).casefold()
        == expected_name.casefold()
    ]
    if len(matches) != 1:
        raise AnimaLoraRemapRegistryError(
            "anima_hires_target_model_ambiguous"
            if matches
            else "anima_hires_target_model_unresolved"
        )
    return matches[0]


def _resolve_discord_model_slots(model_references: dict[str, str]) -> dict[str, str]:
    config = AppConfig()
    library = get_model_library(_model_root(config))
    by_stable_id: dict[str, Path] = {}
    duplicate_ids: set[str] = set()
    for item in library.items:
        stable_model_id = str(item.stable_model_id or "").strip()
        if not stable_model_id:
            continue
        if stable_model_id in by_stable_id:
            duplicate_ids.add(stable_model_id)
            continue
        by_stable_id[stable_model_id] = item.path

    resolved: dict[str, str] = {}
    for slot_id, stable_model_id in model_references.items():
        normalized_slot = str(slot_id).strip()
        normalized_model_id = str(stable_model_id).strip()
        if (
            not normalized_slot
            or not normalized_model_id
            or normalized_model_id in duplicate_ids
            or normalized_model_id not in by_stable_id
        ):
            raise OperationExecutionError("discord_model_unresolved", retryable=False)
        resolved[normalized_slot] = str(by_stable_id[normalized_model_id])
    return resolved


def _schema_to_response(schema: GenerationSchema) -> GenerationSchemaResponse:
    return GenerationSchemaResponse(
        profile=schema.profile,
        id=schema.id,
        name=schema.name,
        mode=schema.mode,
        model_family=schema.model_family,
        fields=schema.fields,
        defaults=schema.defaults,
    )


def _validate_required_model_slots(profile: str, model_slots: dict[str, str], config: AppConfig) -> None:
    profiles = ModelProfileRepository(config.model_profiles_path).list()
    active_profile = next((item for item in profiles if item.id == profile), None)
    if active_profile is None and profile == "flux":
        active_profile = next((item for item in profiles if item.id == "flux1"), None)
    if active_profile is None:
        raise HTTPException(status_code=400, detail={"code": api_errors.MODEL_PROFILE_NOT_FOUND, "profile": profile})

    missing = [
        slot.id
        for slot in active_profile.model_slots
        if slot.required and not model_slots.get(slot.id)
    ]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={"code": api_errors.MISSING_REQUIRED_MODEL_SLOTS, "slots": missing},
        )
    missing_files = []
    for slot in active_profile.model_slots:
        if not slot.required or not model_slots.get(slot.id):
            continue
        if not _path_is_file_safely(model_slots[slot.id]):
            missing_files.append(slot.id)
    if missing_files:
        raise HTTPException(
            status_code=400,
            detail={"code": api_errors.MISSING_REQUIRED_MODEL_FILES, "slots": missing_files},
        )


def _external_model_reference_snapshots(
    model_slots: dict[str, str],
    config: AppConfig,
) -> list[dict[str, Any]]:
    library = get_model_library(_model_root(config))
    by_path = {
        os.path.normcase(os.path.abspath(os.fspath(item.path))): item
        for item in library.items
        if item.stable_model_id
    }
    snapshots: list[dict[str, Any]] = []
    for slot_id, value in sorted(model_slots.items()):
        item = by_path.get(os.path.normcase(os.path.abspath(value)))
        if item is None:
            continue
        snapshots.append(
            {
                "slot_id": slot_id,
                "stable_model_id": item.stable_model_id,
                "root_id": item.root_id,
                "model_type": item.type,
                "normalized_relative_path": item.metadata.get(
                    "external_relative_path",
                    item.relative_path.as_posix(),
                ),
                "fingerprint_algorithm": item.fingerprint_algorithm,
                "fingerprint_version": item.fingerprint_version,
                "fingerprint_value": item.fingerprint_value,
                "execution_path": str(item.path),
            }
        )
    return snapshots


def _revalidate_external_model_reference_snapshots(
    snapshots: list[dict[str, Any]],
    config: AppConfig,
) -> None:
    fingerprints = ModelFingerprintService(config.data_dir / "config" / "model-fingerprints.json")
    for snapshot in snapshots:
        path = Path(str(snapshot["execution_path"]))
        try:
            current = fingerprints.fast_fingerprint(
                path,
                root_id=str(snapshot["root_id"]),
                model_type=str(snapshot["model_type"]),
                relative_path=str(snapshot["normalized_relative_path"]),
            )
        except OSError as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "external_model_offline", "slot": snapshot["slot_id"]},
            ) from exc
        if (
            snapshot.get("fingerprint_algorithm") != current.algorithm
            or snapshot.get("fingerprint_version") != current.version
            or snapshot.get("fingerprint_value") != current.value
        ):
            raise HTTPException(
                status_code=409,
                detail={"code": "external_model_fingerprint_changed", "slot": snapshot["slot_id"]},
            )


def _settings_from_request(request: Txt2ImgRequest, config: AppConfig, mode: str = "txt2img") -> tuple[dict, str]:
    schema_profile = "flux" if request.profile == "flux1" else request.profile
    schema_path = config.schema_dir / f"{schema_profile}_txt2img.yaml"
    try:
        schema_exists = schema_path.exists()
    except OSError:
        raise HTTPException(status_code=400, detail={"code": api_errors.GENERATION_SCHEMA_LOAD_FAILED}) from None
    if not schema_exists:
        raise HTTPException(
            status_code=400,
            detail={"code": api_errors.GENERATION_SCHEMA_NOT_FOUND, "profile": request.profile},
        )

    values = request.model_dump(
        exclude={
            "profile",
            "runtime",
            "prompt_input",
            "prompt_layout",
            "prompt_pins",
            "loras",
            "controlnet_units",
            "anima_lllite_units",
            "anima_qwen35_adapter_strength",
            "anima_qwen35_negative_adapter_strength",
            "detailer_units",
            "model_slot_overrides",
            "remaster",
            "init_image_path",
            "denoising_strength",
            "image_edit_lineage",
        },
        exclude_unset=True,
    )
    values['record_branch_step'] = request.record_branch_step or 0
    try:
        mapped = ParameterMapper(schema_path).map(values)
    except ValueError as error:
        if str(error) == "Generation schema could not be loaded":
            raise HTTPException(status_code=400, detail={"code": api_errors.GENERATION_SCHEMA_LOAD_FAILED}) from error
        raise HTTPException(status_code=400, detail=str(error)) from error
    mapped.pop("prompt", None)
    negative_prompt = mapped.pop("negative_prompt", request.negative_prompt)
    mapped["mode"] = mode
    if (
        request.profile == "sdxl"
        and mapped.get("hires_enabled")
        and mapped.get("hires_upscaler") != "Topaz Photo AI"
        and not mapped.get("hires_noise_mode")
    ):
        mapped["hires_noise_mode"] = "CPU"
    if isinstance(request, Img2ImgRequest):
        mapped["init_image_path"] = request.init_image_path
        mapped["denoising_strength"] = request.denoising_strength
        mapped["denoise"] = request.denoising_strength
        mapped["hires_enabled"] = request.hires_enabled
        if request.image_edit_lineage:
            mapped["image_edit_lineage"] = request.image_edit_lineage.model_dump(mode="json")
    if isinstance(request, InpaintRequest):
        mapped["mask_image_path"] = request.mask_image_path
        mapped["mask_blur"] = request.mask_blur
        mapped["masked_content"] = request.masked_content
        mapped["inpaint_area"] = request.inpaint_area
    controlnet_units = [unit.model_dump() for unit in request.controlnet_units if unit.enabled]
    anima_lllite_units = [unit.model_dump() for unit in request.anima_lllite_units if unit.enabled]
    detailer_units = [unit.model_dump() for unit in request.detailer_units if unit.enabled]
    if controlnet_units:
        mapped["controlnet_units"] = controlnet_units
        mapped["controlnet_runtime"] = "not_implemented"
    if anima_lllite_units:
        mapped["anima_lllite_units"] = anima_lllite_units
        mapped["anima_lllite_runtime"] = "not_implemented"
    if request.profile == "anima":
        mapped["anima_qwen35_adapter_strength"] = (
            request.anima_qwen35_adapter_strength
        )
        mapped["anima_qwen35_negative_adapter_strength"] = (
            request.anima_qwen35_negative_adapter_strength
        )
    if detailer_units:
        mapped["detailer_units"] = detailer_units
        mapped["detailer_runtime"] = "not_implemented"
    if request.model_slot_overrides:
        mapped["model_slot_overrides"] = dict(request.model_slot_overrides)
    mapped["remaster"] = request.remaster.model_dump()
    if request.prompt_pins is not None:
        mapped["prompt_pins"] = request.prompt_pins.model_dump(mode="json")
    if request.prompt_input is not None:
        mapped["prompt_input"] = request.prompt_input.model_dump(
            mode="json",
            exclude_none=True,
        )
    if request.prompt_layout is not None:
        mapped["prompt_layout"] = request.prompt_layout.model_dump(
            mode="json",
            exclude_none=True,
        )
    if request.remaster.enabled:
        mapped["requested_batch_size"] = request.batch_size
        mapped["requested_batch_count"] = request.batch_count
        mapped["batch_size"] = 1
        mapped["batch_count"] = request.batch_size * request.batch_count
    return mapped, negative_prompt


def _spatial_request_enabled(request: Txt2ImgRequest) -> bool:
    return request.prompt_layout is not None and request.prompt_layout.mode == "enabled"


def _freeze_spatial_request_seed(request: Txt2ImgRequest) -> Txt2ImgRequest:
    if not _spatial_request_enabled(request) or request.seed >= 0:
        return request
    return request.model_copy(update={"seed": secrets.randbelow(2**32)}, deep=True)


def _resolve_spatial_model_path(
    value: str,
    *,
    model_root: Path,
    model_directory: str,
    library: ModelLibrary,
) -> Path:
    normalized = str(value or "").strip()
    direct = Path(normalized)
    if direct.is_absolute() and _path_is_file_safely(direct):
        return direct
    rooted = model_root / model_directory / normalized
    if _path_is_file_safely(rooted):
        return rooted
    expected = comfy_model_name(
        normalized,
        model_directory,
        model_root=model_root,
    ).casefold()
    matches = [
        item.path
        for item in library.by_type("checkpoint")
        if comfy_model_name(
            str(item.path),
            model_directory,
            model_root=model_root,
        ).casefold()
        == expected
    ]
    if len(matches) != 1:
        raise SpatialGenerationContextError(
            "spatial_model_ambiguous" if matches else "spatial_model_unavailable"
        )
    return matches[0]


def _spatial_model_evidence(
    *,
    profile: str,
    path: Path,
    stage: str,
    anima_variant: str | None = None,
    manual_profile_id: str | None = None,
    catalog_metadata: Mapping[str, str] | None = None,
) -> SpatialModelEvidence:
    try:
        header = read_safetensors_header(path)
    except SafetensorsHeaderError as exc:
        raise SpatialGenerationContextError(exc.reason_code) from exc
    slot_id = f"model:{stage}:{path.name}"
    if profile == "sdxl":
        lineage_metadata = {**header.metadata, **dict(catalog_metadata or {})}
        selected_profile_id = infer_sdxl_spatial_profile_id(lineage_metadata)
        return sdxl_spatial_model_evidence(
            header,
            model_slot_id=slot_id,
            variant=(
                "illustrious"
                if selected_profile_id == "illustrious.boundary.v5"
                else "sdxl"
            ),
        )
    if profile != "anima":
        raise SpatialGenerationContextError("spatial_profile_unsupported")
    encoder_kind = (
        "qwen35_progressive"
        if anima_variant == "anima_3_8b_52"
        else "native_qwen"
    )
    item = ModelLibraryItem(
        type="checkpoint",
        name=path.name,
        path=path,
        relative_path=Path(path.name),
        size_bytes=path.stat().st_size,
        metadata={},
    )
    evidence = AnimaModelClassifier().spatial_model_evidence(
        item,
        header,
        model_slot_id=slot_id,
        encoder_kind=encoder_kind,
    )
    if evidence is None:
        raise SpatialGenerationContextError("spatial_model_structure_unverified")
    return evidence


def _spatial_catalog_metadata(
    library,
    path: Path,
) -> Mapping[str, str]:
    item = next((candidate for candidate in library.items if candidate.path == path), None)
    return item.metadata if item is not None else {}


def _positive_inpaint_mask(path: str | Path, size: tuple[int, int]) -> Image.Image:
    try:
        with Image.open(path) as source:
            source.load()
            if source.mode == "RGBA":
                alpha = source.getchannel("A")
                luminance = ImageOps.grayscale(source.convert("RGB"))
                mask = (
                    luminance
                    if alpha.getextrema() == (255, 255)
                    else Image.composite(luminance, alpha, alpha)
                )
            else:
                mask = ImageOps.grayscale(source.convert("RGB"))
            if mask.size != size:
                mask = mask.resize(size, Image.Resampling.NEAREST)
            return mask.copy()
    except OSError as exc:
        raise SpatialGenerationContextError("spatial_inpaint_mask_unavailable") from exc


def _spatial_base_geometry(
    request: Txt2ImgRequest,
) -> tuple[int, int, Image.Image | None, tuple[int, int, int, int] | None]:
    if not isinstance(request, Img2ImgRequest):
        return request.width, request.height, None, None
    if isinstance(request, InpaintRequest) and request.inpaint_area.strip().lower() == "only_masked":
        try:
            analysis = validate_only_masked_inputs(
                request.init_image_path,
                request.mask_image_path,
                mask_blur=request.mask_blur,
                requested_size=(request.width, request.height),
            )
        except InpaintRegionError as exc:
            raise SpatialGenerationContextError(exc.code) from exc
        with analysis.open_positive_mask() as image:
            mask = image.convert("L").copy()
        return (*analysis.canvas_size, mask, analysis.region.rect)
    try:
        with Image.open(request.init_image_path) as source:
            source_width, source_height = source.size
    except OSError as exc:
        raise SpatialGenerationContextError("spatial_source_image_unavailable") from exc
    width = source_width // 8 * 8
    height = source_height // 8 * 8
    inpaint_mask = None
    if isinstance(request, InpaintRequest):
        inpaint_mask = _positive_inpaint_mask(
            request.mask_image_path,
            (width, height),
        )
    return width, height, inpaint_mask, None


def _prepare_spatial_generation(
    *,
    request: Txt2ImgRequest,
    settings: dict[str, Any],
    config: AppConfig,
    model_slots: dict[str, str],
    known_embeddings: list[str],
    output_dir: Path,
) -> PreparedSpatialGenerationContext | None:
    if not _spatial_request_enabled(request):
        return None
    if request.profile not in {"sdxl", "anima"}:
        raise HTTPException(
            status_code=409,
            detail={"code": "spatial_profile_unsupported", "profile": request.profile},
        )
    snapshot = validate_spatial_prompt_contract(
        request.prompt,
        request.prompt_input,
        request.prompt_layout,
        request._spatial_generation_mode,
    )
    model_root = _model_root(config)
    library = get_model_library(model_root, config=config)
    base_slot = "checkpoint" if request.profile == "sdxl" else "diffusion_model"
    model_directory = "checkpoints" if request.profile == "sdxl" else "diffusion_models"
    base_path = _resolve_spatial_model_path(
        model_slots[base_slot],
        model_root=model_root,
        model_directory=model_directory,
        library=library,
    )
    anima_bundle = settings.get("anima_generation_bundle")
    anima_variant = (
        str(anima_bundle.get("variant_id"))
        if isinstance(anima_bundle, dict)
        else None
    )
    # Runtime tensors are authoritative. SDXL uses its verified 24-block
    # structure; Anima resolves the 28/40/52-block profile from its tensors.
    # Older histories may retain a manual selection, but it is no longer used.
    base_manual_profile = None
    base_width, base_height, inpaint_mask, crop_rect = _spatial_base_geometry(request)
    stage_inputs: list[SpatialStagePreparation] = [
        SpatialStagePreparation(
            stage="base",
            layout=request.prompt_layout,
            width=base_width,
            height=base_height,
            evidence=_spatial_model_evidence(
                profile=request.profile,
                path=base_path,
                stage="base",
                anima_variant=anima_variant,
                manual_profile_id=base_manual_profile,
                catalog_metadata=_spatial_catalog_metadata(library, base_path),
            ),
            inpaint_mask=inpaint_mask,
            crop_rect=crop_rect,
            manual_profile_id=base_manual_profile,
        )
    ]
    hires_size = _hires_target_size(settings)
    if hires_size is not None:
        hires_value = str(settings.get("hires_checkpoint") or "").strip()
        hires_path = base_path
        if hires_value and hires_value != "Use same checkpoint":
            hires_path = _resolve_spatial_model_path(
                hires_value,
                model_root=model_root,
                model_directory=model_directory,
                library=library,
            )
        hires_manual_profile = None
        hires_mask = (
            inpaint_mask.resize(hires_size, Image.Resampling.NEAREST)
            if inpaint_mask is not None and crop_rect is None
            else None
        )
        stage_inputs.append(
            SpatialStagePreparation(
                stage="hires",
                layout=request.prompt_layout,
                width=hires_size[0],
                height=hires_size[1],
                evidence=_spatial_model_evidence(
                    profile=request.profile,
                    path=hires_path,
                    stage="hires",
                    anima_variant=(
                        str(settings.get("anima_hires_generation_bundle", {}).get("variant_id"))
                        if isinstance(settings.get("anima_hires_generation_bundle"), dict)
                        else anima_variant
                    ),
                    manual_profile_id=hires_manual_profile,
                    catalog_metadata=_spatial_catalog_metadata(library, hires_path),
                ),
                inpaint_mask=hires_mask,
                manual_profile_id=hires_manual_profile,
            )
        )
    if request.profile == "sdxl" and _uses_refiner(settings):
        refiner_path = _resolve_spatial_model_path(
            str(settings["refiner_checkpoint"]),
            model_root=model_root,
            model_directory="checkpoints",
            library=library,
        )
        refiner_size = hires_size or (base_width, base_height)
        refiner_manual_profile = None
        refiner_mask = (
            inpaint_mask.resize(refiner_size, Image.Resampling.NEAREST)
            if inpaint_mask is not None and crop_rect is None
            else inpaint_mask
        )
        stage_inputs.append(
            SpatialStagePreparation(
                stage="refiner",
                layout=request.prompt_layout,
                width=refiner_size[0],
                height=refiner_size[1],
                evidence=_spatial_model_evidence(
                    profile="sdxl",
                    path=refiner_path,
                    stage="refiner",
                    manual_profile_id=refiner_manual_profile,
                    catalog_metadata=_spatial_catalog_metadata(library, refiner_path),
                ),
                inpaint_mask=refiner_mask,
                crop_rect=crop_rect if hires_size is None else None,
                manual_profile_id=refiner_manual_profile,
            )
        )
    try:
        return prepare_spatial_generation_context(
            snapshot,
            profile_id=request.profile,
            generation_seed=request.seed,
            stages=stage_inputs,
            profiles_path=config.mapping_dir / "spatial-model-profiles.yaml",
            working_directory=output_dir / "_spatial" / uuid4().hex,
            advanced_styles=request.advanced_styles,
            hires_prompt=str(settings.get("hires_prompt") or ""),
            wildcard_dir=config.wildcard_dir,
            known_embedding_names=known_embeddings,
            data_dir=config.data_dir,
        )
    except SpatialMaskCompilerError as exc:
        detail: dict[str, Any] = {"code": exc.code}
        if exc.target_ids:
            detail["target_ids"] = list(exc.target_ids)
        raise HTTPException(status_code=409, detail=detail) from exc
    except SpatialGenerationContextError as exc:
        detail = {"code": exc.code}
        if exc.target_ids:
            detail["target_ids"] = list(exc.target_ids)
        if exc.stage is not None:
            detail["stage"] = exc.stage
        if exc.profile_ids:
            detail["profile_options"] = list(exc.profile_ids)
        raise HTTPException(status_code=409, detail=detail) from exc


def _spatial_prompt_selections(plan) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    stages = [plan.base]
    if plan.hires is not None and plan.hires.reused_from_stage is None:
        stages.append(plan.hires)
    for stage in stages:
        for fragment in (stage.global_fragment, *stage.target_fragments):
            for selection in fragment.selections:
                records.append(
                    {
                        "fragment_id": fragment.fragment_id,
                        "stage": fragment.stage,
                        **selection.model_dump(mode="json"),
                    }
                )
    return records


def _topaz_settings_from_request(request: Txt2ImgRequest) -> TopazSettings:
    force_hires_override = request.hires_enabled and request.hires_upscaler == "Topaz Photo AI"
    return TopazSettings(
        enabled=request.topaz_enabled,
        lighting=request.topaz_lighting,
        color=request.topaz_color,
        sharpen=request.topaz_sharpen,
        denoise=request.topaz_denoise,
        override=request.topaz_override or force_hires_override,
        show_settings=request.topaz_show_settings,
        png_compression=request.topaz_png_compression,
    )


def _topaz_executable(config: AppConfig) -> Path:
    return ExternalIntegrationSettingsRepository(config.data_dir / "external-integrations.json").resolve_topaz_executable(
        config.topaz_executable
    )



def _hires_target_size(settings: dict) -> tuple[int, int] | None:
    if not settings.get("hires_enabled"):
        return None
    resize_width = int(settings.get("hires_resize_width") or 0)
    resize_height = int(settings.get("hires_resize_height") or 0)
    width = int(settings.get("width") or 0)
    height = int(settings.get("height") or 0)
    if resize_width > 0 and resize_height > 0:
        return _round_hires_dimension(resize_width), _round_hires_dimension(resize_height)
    if resize_width > 0 and width > 0 and height > 0:
        return _round_hires_dimension(resize_width), _round_hires_dimension(resize_width * (height / width))
    if resize_height > 0 and width > 0 and height > 0:
        return _round_hires_dimension(resize_height * (width / height)), _round_hires_dimension(resize_height)
    scale = float(settings.get("hires_scale") or 1)
    if width <= 0 or height <= 0 or scale <= 0:
        return None
    return _round_hires_dimension(width * scale), _round_hires_dimension(height * scale)


def _round_hires_dimension(value: int | float, step: int = 8) -> int:
    return int(float(value) / step + 0.5) * step


def _apply_topaz_hires_if_needed(output_paths: list[str], settings: dict, request: Txt2ImgRequest, config: AppConfig, output_dir: Path) -> list[str]:
    if isinstance(settings.get("image_check"), dict) and settings["image_check"].get("target_stage") == "final_output":
        # A final decision must never be followed by another image correction.
        return output_paths
    if not settings.get("hires_enabled") or settings.get("hires_upscaler") != "Topaz Photo AI":
        return output_paths
    if settings.get("topaz_hires_runtime") in {"completed", "completed_topaz_only"}:
        settings["topaz_postprocess"] = "handled_in_hires_upscale_stage"
        return output_paths
    if settings.get("topaz_hires_runtime") == "skipped_missing_executable":
        settings["topaz_postprocess"] = "skipped_missing_executable"
        return output_paths
    settings["topaz"] = _topaz_settings_from_request(request).model_dump()
    settings["topaz_source_output_paths"] = output_paths
    executable = _topaz_executable(config)
    if not _path_is_file_safely(executable):
        settings["topaz_postprocess"] = "skipped_missing_executable"
        return output_paths

    processed_paths: list[str] = []
    for output_path in output_paths:
        path = Path(output_path)
        if (
            path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}
            or not _path_exists_safely(path)
            or not _path_is_file_safely(path)
        ):
            continue
        try:
            result = process_topaz_image(
                executable=executable,
                input_path=path,
                output_dir=output_dir,
                settings=_topaz_settings_from_request(request),
                target_size=_hires_target_size(settings),
            )
        except TopazError as exc:
            settings["topaz_postprocess"] = f"failed: {exc}"
            return output_paths
        processed_paths.append(str(result["output_path"]))
    if processed_paths:
        settings["topaz_postprocess"] = "completed"
        return processed_paths
    else:
        settings["topaz_postprocess"] = "skipped_no_image_outputs"
    return output_paths


@router.get("/schemas/{profile}", response_model=GenerationSchemaResponse)
def generation_schema(profile: str) -> GenerationSchemaResponse:
    try:
        schema = _schema_repository(AppConfig()).get_txt2img(profile)
    except GenerationSchemaNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": api_errors.GENERATION_SCHEMA_NOT_FOUND, "profile": profile},
        ) from exc
    return _schema_to_response(schema)


@router.get("/wildcards", response_model=WildcardListResponse)
def wildcards() -> WildcardListResponse:
    return WildcardListResponse(
        items=[
            WildcardItemResponse(name=item.name, path=item.path, option_count=item.option_count)
            for item in list_wildcards(AppConfig().wildcard_dir)
        ]
    )


def _expand_with_pins(prompt: str, request: Txt2ImgRequest | PromptExpandRequest, config: AppConfig):
    try:
        return expand_dynamic_prompt(
            prompt,
            seed=request.seed if request.seed >= 0 else None,
            wildcard_dir=config.wildcard_dir,
            **({"pins": request.prompt_pins} if request.prompt_pins is not None else {}),
        )
    except ValueError as exc:
        if str(exc).startswith('prompt_choices_'):
            raise HTTPException(status_code=409, detail={"code": str(exc)}) from exc
        raise


@router.post("/prompt/expand", response_model=PromptExpandResponse)
def expand_prompt_preview(request: PromptExpandRequest) -> PromptExpandResponse:
    prompt, _negative_prompt = _compose_with_advanced_styles(
        request.prompt,
        request.negative_prompt,
        request.advanced_styles,
    )
    expansion = _expand_with_pins(prompt, request, AppConfig())
    return PromptExpandResponse(
        prompt=request.prompt,
        expanded_prompt=expansion.expanded,
        selections=expansion.selections,
        choices=expansion.choices,
        prompt_hash=expansion.prompt_hash,
    )


def _merge_loras(prompt_loras: list[dict], request_loras: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for lora in [*prompt_loras, *request_loras]:
        merged[str(lora["name"])] = lora
    return list(merged.values())


def _unsupported_profile_prompt_settings(profile: str, negative_prompt: str) -> list[dict[str, str]]:
    if profile not in {"flux1", "flux2"} or not negative_prompt.strip():
        return []
    return [
        {
            "field": "negative_prompt",
            "reason": f"negative_prompt_is_not_supported_by_{profile}",
        }
    ]


def _unsupported_flux2_scheduler_settings(request: Txt2ImgRequest) -> list[dict[str, str]]:
    if request.profile != "flux2":
        return []
    unsupported: list[dict[str, str]] = []
    if request.scheduler != "Flux2":
        unsupported.append({"field": "scheduler", "label": request.scheduler, "accepted": "Flux2"})
    if request.hires_enabled and request.hires_scheduler not in {"Use same scheduler", "Flux2"}:
        unsupported.append(
            {
                "field": "hires_scheduler",
                "label": request.hires_scheduler,
                "accepted": "Use same scheduler",
            }
        )
    if request.remaster.enabled and request.remaster.scheduler not in {"", "Flux2"}:
        unsupported.append(
            {
                "field": "remaster.scheduler",
                "label": request.remaster.scheduler,
                "accepted": "Flux2",
            }
        )
    return unsupported


def _preflight_profile_prompt_request(request: Txt2ImgRequest) -> None:
    _styled_prompt, styled_negative_prompt = _compose_with_advanced_styles(
        request.prompt,
        request.negative_prompt,
        request.advanced_styles,
    )
    unsupported = [
        *_unsupported_profile_prompt_settings(request.profile, styled_negative_prompt),
        *_unsupported_flux2_scheduler_settings(request),
    ]
    if unsupported:
        raise HTTPException(
            status_code=409,
            detail={
                "code": api_errors.UNSUPPORTED_COMFY_RUNTIME_SETTING,
                "unsupported": unsupported,
            },
        )
    if request.profile == "flux2":
        if request.hires_enabled and request.hires_scheduler == "Flux2":
            request.hires_scheduler = "Use same scheduler"
        if request.remaster.enabled:
            request.remaster.scheduler = "Flux2"


_PROMPT_SYNTAX_CONTRACTS = {
    "scheduled_switch": {"conditioning_time_range", "conditioning_combine"},
    "alternating_switch": {"prompt_step_schedule"},
    "conditioning_and": {"conditioning_multiply", "conditioning_combine"},
    "conditioning_break": {"conditioning_concat"},
}


def _prompt_payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prompt_syntax_http_error(exc: PromptSyntaxError) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "code": api_errors.PROMPT_SYNTAX_INVALID,
            "reason": exc.code,
            "message": exc.message,
            "recovery": exc.recovery,
        },
    )


def _prompt_syntax_assignments(plan: PromptSyntaxPlan) -> list[Literal["A", "B"]]:
    return [
        segment.assignment
        for segment in plan.segments
        if segment.assignment in {"A", "B"}
    ]


def _prompt_syntax_pass_plan(
    *,
    pass_id: Literal["base", "hires"],
    profile_id: str,
    steps: int,
    sampler: str,
    scheduler: str,
    denoise: float,
    positive_plan: PromptSyntaxPlan,
    negative_plan: PromptSyntaxPlan,
) -> dict[str, Any]:
    assignments = {
        key: values
        for key, plan in (("positive", positive_plan), ("negative", negative_plan))
        if (values := _prompt_syntax_assignments(plan))
    }
    return {
        "schema_version": "asteria.prompt-pass.v1",
        "node_contract_revision": "asteria.prompt.step-schedule.v1",
        "profile_id": profile_id,
        "pass_id": pass_id,
        "steps": steps,
        "sampler": sampler_runtime_name(sampler),
        "scheduler": scheduler_runtime_name(scheduler),
        "denoise": denoise,
        "positive_plan_hash": positive_plan.sha256,
        "negative_plan_hash": negative_plan.sha256,
        "alternating_assignments": assignments,
    }


def _prepare_prompt_syntax_record(
    *,
    request: Txt2ImgRequest,
    config: AppConfig,
    settings: dict[str, Any],
    expanded_prompt: str,
    expanded_negative_prompt: str,
    selected_runtime: str,
) -> dict[str, Any]:
    try:
        positive_plan = parse_prompt_syntax(expanded_prompt, profile_id=request.profile)
        negative_plan = parse_prompt_syntax(expanded_negative_prompt, profile_id=request.profile)
        semantic_plans: dict[str, PromptSyntaxPlan] = {
            "positive": positive_plan,
            "negative": negative_plan,
        }
        hires_positive_plan = positive_plan
        hires_negative_plan = negative_plan
        if settings.get("hires_enabled") and settings.get("hires_prompt"):
            hires_positive_plan = parse_prompt_syntax(
                str(settings["hires_prompt"]),
                profile_id=request.profile,
            )
            semantic_plans["hires_positive"] = hires_positive_plan
        if settings.get("hires_enabled") and settings.get("hires_negative_prompt"):
            hires_negative_plan = parse_prompt_syntax(
                str(settings["hires_negative_prompt"]),
                profile_id=request.profile,
            )
            semantic_plans["hires_negative"] = hires_negative_plan
    except PromptSyntaxError as exc:
        raise _prompt_syntax_http_error(exc) from exc

    enabled_features = {
        feature
        for plan in semantic_plans.values()
        for feature in plan.enabled_features
    }
    if enabled_features:
        profiles = ModelProfileRepository(config.model_profiles_path).list()
        active_profile = next((item for item in profiles if item.id == request.profile), None)
        if active_profile is None:
            raise HTTPException(
                status_code=400,
                detail={"code": api_errors.MODEL_PROFILE_NOT_FOUND, "profile": request.profile},
            )
        unsupported = sorted(enabled_features - set(active_profile.prompt_syntax.features))
        if (
            active_profile.prompt_syntax.grammar != PROMPT_PARSER_GENERATION
            or unsupported
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": api_errors.PROMPT_SYNTAX_PROFILE_UNSUPPORTED,
                    "profile": request.profile,
                    "features": unsupported or sorted(enabled_features),
                },
            )
        if selected_runtime != "comfy_workflow":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": api_errors.PROMPT_SYNTAX_RUNTIME_UNSUPPORTED,
                    "runtime": selected_runtime,
                },
            )
        try:
            mapping = WorkflowMappingRepository(config.mapping_dir).get_txt2img(request.profile)
        except WorkflowMappingNotFound as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": api_errors.PROMPT_SYNTAX_MAPPING_UNAVAILABLE},
            ) from exc
        required_contracts = {
            contract
            for plan in semantic_plans.values()
            for contract in _PROMPT_SYNTAX_CONTRACTS.get(plan.operation, set())
        }
        missing_contracts = sorted(required_contracts - set(mapping.node_contracts))
        if missing_contracts:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": api_errors.PROMPT_SYNTAX_MAPPING_UNAVAILABLE,
                    "contracts": missing_contracts,
                },
            )

    semantic_payloads = {
        key: plan.to_dict()
        for key, plan in semantic_plans.items()
    }
    pass_plans = {
        "base": _prompt_syntax_pass_plan(
            pass_id="base",
            profile_id=request.profile,
            steps=int(settings["steps"]),
            sampler=str(settings.get("sampler", "Euler a")),
            scheduler=str(settings.get("scheduler", "Automatic")),
            denoise=float(settings.get("denoise", 1.0)),
            positive_plan=positive_plan,
            negative_plan=negative_plan,
        )
    }
    if settings.get("hires_enabled"):
        pass_plans["hires"] = _prompt_syntax_pass_plan(
            pass_id="hires",
            profile_id=request.profile,
            steps=int(settings.get("hires_steps") or settings["steps"]),
            sampler=str(
                settings.get("sampler")
                if settings.get("hires_sampler") in {None, "", "Use same sampler"}
                else settings["hires_sampler"]
            ),
            scheduler=str(
                settings.get("scheduler")
                if settings.get("hires_scheduler") in {None, "", "Use same scheduler"}
                else settings["hires_scheduler"]
            ),
            denoise=float(settings.get("hires_denoising_strength", 0.7)),
            positive_plan=hires_positive_plan,
            negative_plan=hires_negative_plan,
        )
    payload = {
        "schema_version": "asteria.prompt-generation.v1",
        "parser_generation": PROMPT_PARSER_GENERATION,
        "profile_id": request.profile,
        "raw_prompt": request.prompt,
        "raw_negative_prompt": request.negative_prompt,
        "expanded_prompt": expanded_prompt,
        "expanded_negative_prompt": expanded_negative_prompt,
        "semantic_plans": semantic_payloads,
        "semantic_hashes": {
            key: plan.sha256
            for key, plan in semantic_plans.items()
        },
        "pass_plans": pass_plans,
        "pass_plan_hashes": {
            key: _prompt_payload_sha256(plan)
            for key, plan in pass_plans.items()
        },
    }
    return PromptSyntaxGenerationRecord.model_validate(payload).model_dump(mode="json")


def _known_embedding_names(config: AppConfig) -> list[str]:
    library = get_model_library(_model_root(config))
    return [item.path.stem for item in library.by_type("embedding")]


def _output_dir_for(data_dir: Path, profile: str, operation: ImageCheckOperation | None = None) -> Path:
    from datetime import datetime

    output_settings = (operation.output_settings if operation is not None
                       else UserSettingsRepository(data_dir / "user-settings.json").load().output)
    parts = [operation.output_root if operation is not None else resolve_output_root(data_dir, output_settings)]
    if output_settings.organize_by_date:
        parts.append(Path(datetime.now().strftime("%Y-%m-%d")))
    parts.append(Path(safe_segment(profile)))
    return Path(*parts)


def _generation_operation_snapshot(request: Txt2ImgRequest) -> dict:
    data_dir = _data_dir()
    current = UserSettingsRepository(data_dir / "user-settings.json").load()
    try:
        return capture_image_check_operation(current, data_dir=data_dir, enabled=request.image_check_enabled).snapshot()
    except ValidationError as exc:
        raise HTTPException(status_code=409, detail={"code": "image_check_settings_invalid"}) from exc


def _delete_history_artifacts(
    items: list[dict],
    data_dir: Path,
    repository: HistoryRepository,
):
    try:
        remaining_items: list[dict] | None = repository.list_all()
    except Exception:
        remaining_items = None
    try:
        lora_datasets: list[Any] | None = LoRATrainingDatasetRepository(
            data_dir / "lora-training" / "datasets.json"
        ).list()
    except Exception:
        lora_datasets = None
    try:
        return delete_generation_artifacts(
            items,
            data_dir,
            remaining_items=remaining_items,
            lora_datasets=lora_datasets,
        )
    except Exception:
        logger.exception("Generation artifact file phase failed after history commit")
        return skip_generation_artifacts(items)


def _sync_curation_sidecars(items: list[dict]) -> None:
    writer = MetadataWriter()
    for item in items:
        writer.update_curation_sidecars(item)


def _sidecar_rating(value: Any) -> int:
    try:
        return max(0, min(5, int(value)))
    except (TypeError, ValueError):
        return 0


def _sidecar_favorite(value: Any) -> bool:
    return value if isinstance(value, bool) else False


def _sidecar_notes(value: Any) -> str:
    return value if isinstance(value, str) else ""


@_artifact_write_operation
def _scan_sidecar_metadata(repository: HistoryRepository, output_roots: list[Path]) -> LibraryScanResponse:
    referenced = {
        resolved
        for path in repository.referenced_output_paths()
        if (resolved := _resolved_path_string_safely(path)) is not None
    }
    imported_items: list[dict[str, Any]] = []
    skipped = 0
    for output_root in output_roots:
        if not _path_exists_safely(output_root):
            continue
        try:
            sidecar_paths = list(output_root.rglob("*.asteria.json"))
        except _PATH_ACCESS_ERRORS:
            skipped += 1
            continue
        for sidecar_path in sidecar_paths:
            if is_private_image_candidate(sidecar_path):
                continue
            try:
                payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                skipped += 1
                continue
            if not isinstance(payload, dict):
                skipped += 1
                continue
            image_path = Path(str(payload.get("image_path") or sidecar_path.with_suffix(".png")))
            if not image_path.is_absolute():
                image_path = sidecar_path.parent / image_path
            try:
                resolved_image = str(image_path.resolve())
            except _PATH_ACCESS_ERRORS:
                skipped += 1
                continue
            if (
                resolved_image in referenced
                or is_private_image_candidate(resolved_image)
                or not _path_exists_safely(image_path)
                or not _path_is_file_safely(image_path)
            ):
                skipped += 1
                continue
            try:
                validate_image_file(image_path)
            except InvalidImageContentError:
                skipped += 1
                continue
            output_paths = [str(image_path), str(sidecar_path)]
            scan_tags = (
                payload.get("tags")
                if isinstance(payload.get("tags"), list)
                else []
            )
            scan_generated_tags = (
                payload.get("metadata_tags")
                if isinstance(payload.get("metadata_tags"), list)
                else metadata_tags(
                    profile=str(payload.get("profile") or "unknown"),
                    prompt=str(payload.get("expanded_prompt") or payload.get("prompt") or ""),
                    model_slots=payload.get("model_slots") if isinstance(payload.get("model_slots"), dict) else {},
                    embeddings=payload.get("embeddings") if isinstance(payload.get("embeddings"), list) else [],
                    loras=payload.get("loras") if isinstance(payload.get("loras"), list) else [],
                )
            )
            generation_id = repository.add(
                status=str(payload.get("status") or "completed"),
                profile=str(payload.get("profile") or "unknown"),
                prompt=str(payload.get("prompt") or ""),
                expanded_prompt=str(payload.get("expanded_prompt") or payload.get("prompt") or ""),
                negative_prompt=str(payload.get("negative_prompt") or ""),
                settings=payload.get("settings") if isinstance(payload.get("settings"), dict) else {},
                model_slots=payload.get("model_slots") if isinstance(payload.get("model_slots"), dict) else {},
                loras=payload.get("loras") if isinstance(payload.get("loras"), list) else [],
                embeddings=payload.get("embeddings") if isinstance(payload.get("embeddings"), list) else [],
                output_paths=output_paths,
                tags=scan_tags,
                generated_tags=scan_generated_tags,
            )
            item = repository.update_curation(
                generation_id,
                rating=_sidecar_rating(payload.get("rating")),
                favorite=_sidecar_favorite(payload.get("favorite")),
                notes=_sidecar_notes(payload.get("notes")),
            )
            if item:
                imported_items.append(item)
                referenced.update(
                    resolved
                    for path in output_paths
                    if (resolved := _resolved_path_string_safely(path)) is not None
                )
        try:
            png_paths = list(output_root.rglob("*.png"))
        except _PATH_ACCESS_ERRORS:
            skipped += 1
            continue
        for image_path in png_paths:
            if is_private_image_candidate(image_path):
                continue
            try:
                resolved_image = str(image_path.resolve())
            except _PATH_ACCESS_ERRORS:
                continue
            if (
                resolved_image in referenced
                or not _path_exists_safely(image_path)
                or not _path_is_file_safely(image_path)
            ):
                continue
            payload = read_generation_parameters_from_image(image_path)
            if payload is None:
                continue
            output_paths = [str(image_path)]
            scan_generated_tags = (
                payload.get("metadata_tags")
                if isinstance(payload.get("metadata_tags"), list)
                else metadata_tags(
                    profile=str(payload.get("profile") or "unknown"),
                    prompt=str(payload.get("expanded_prompt") or payload.get("prompt") or ""),
                    model_slots=payload.get("model_slots") if isinstance(payload.get("model_slots"), dict) else {},
                    embeddings=payload.get("embeddings") if isinstance(payload.get("embeddings"), list) else [],
                    loras=payload.get("loras") if isinstance(payload.get("loras"), list) else [],
                )
            )
            generation_id = repository.add(
                status=str(payload.get("status") or "completed"),
                profile=str(payload.get("profile") or "unknown"),
                prompt=str(payload.get("prompt") or ""),
                expanded_prompt=str(payload.get("expanded_prompt") or payload.get("prompt") or ""),
                negative_prompt=str(payload.get("negative_prompt") or ""),
                settings=payload.get("settings") if isinstance(payload.get("settings"), dict) else {},
                model_slots=payload.get("model_slots") if isinstance(payload.get("model_slots"), dict) else {},
                loras=payload.get("loras") if isinstance(payload.get("loras"), list) else [],
                embeddings=payload.get("embeddings") if isinstance(payload.get("embeddings"), list) else [],
                output_paths=output_paths,
                generated_tags=scan_generated_tags,
            )
            item = repository.get(generation_id)
            if item:
                imported_items.append(item)
                referenced.add(resolved_image)
    return LibraryScanResponse(imported=len(imported_items), skipped=skipped, items=imported_items)


def _history_output_file_set(items: list[dict]) -> set[Path]:
    paths: set[Path] = set()
    for item in items:
        for output_path in item["output_paths"]:
            try:
                path = Path(output_path).resolve()
            except _PATH_ACCESS_ERRORS:
                continue
            if path.is_file():
                paths.add(path)
    return paths


def _auto_send_primary_outputs_to_eagle(
    *,
    data_dir: Path,
    config: AppConfig,
    primary_outputs: list[dict],
    generated_tags: list[str],
    prompt: str,
    negative_prompt: str,
    settings: dict[str, Any],
) -> None:
    repository = ExternalIntegrationSettingsRepository(data_dir / "external-integrations.json")
    eagle_settings = repository.load().eagle
    if not eagle_settings.auto_send:
        return
    base_url = repository.resolve_eagle_base_url(config.eagle_base_url)
    events = list(settings.get("external_events") or [])
    failure_count = 0
    attempted = 0
    annotation_parts = [prompt.strip()]
    if negative_prompt.strip():
        annotation_parts.append(f"Negative: {negative_prompt.strip()}")
    annotation = "\n\n".join(part for part in annotation_parts if part)
    tags = list(dict.fromkeys(["asteria", *generated_tags]))
    for primary in primary_outputs:
        path_value = str(primary.get("path") or "").strip()
        if not path_value:
            continue
        path = Path(path_value)
        attempted += 1
        try:
            registration = register_eagle_image(
                path,
                base_url=base_url,
                options=EagleImportOptions(
                    name=path.stem,
                    annotation=annotation,
                    tags=tags,
                    folder_id=eagle_settings.default_folder_id,
                ),
                skip_duplicates=True,
            )
            event = {**registration["event"], "automatic": True}
        except (OSError, ValueError, RuntimeError):
            event = {
                "adapter": "eagle",
                "status": "failed",
                "filename": path.name,
                "automatic": True,
                "error": "auto_send_internal_error",
            }
        if event["status"] == "failed":
            failure_count += 1
        events.append(event)
    settings["external_events"] = events
    settings["eagle_auto_send_attempted"] = attempted
    if attempted == 0:
        settings["eagle_auto_send_warning"] = "no_primary_outputs"
    elif failure_count == attempted:
        settings["eagle_auto_send_warning"] = "failed"
    elif failure_count:
        settings["eagle_auto_send_warning"] = "partial_failure"
    else:
        settings["eagle_auto_send_status"] = "completed"


@_artifact_write_operation
def _generate(
    request: Txt2ImgRequest,
    mode: str,
    job_id: str | None = None,
    *,
    merge_runtime_context: dict[str, Any] | None = None,
    comfy_admission_lease_token: str | None = None,
    merge_xyz_metadata: dict[str, Any] | None = None,
    comparison_metadata: dict[str, Any] | None = None,
) -> GenerationResponse:
    from contextlib import ExitStack, nullcontext
    from app.services.making_video import MakingRecorder
    from app.services.generation_branch import BranchSession
    from app.services.word_masks import WordMapSession

    with ExitStack() as finalization_stack, \
            (MakingRecorder() if request.record_making_video else nullcontext(None)) as making, \
            (BranchSession() if request.record_branch_step is not None or request.branch_source_id else nullcontext(None)) as branch, \
            (WordMapSession() if request.record_word_masks else nullcontext(None)) as word_maps:
        return _generate_with_recording(
            request, mode, job_id, making=making, branch=branch, word_maps=word_maps, merge_runtime_context=merge_runtime_context,
            comfy_admission_lease_token=comfy_admission_lease_token,
            merge_xyz_metadata=merge_xyz_metadata, comparison_metadata=comparison_metadata,
            finalization_stack=finalization_stack,
        )


def _generate_with_recording(
    request: Txt2ImgRequest,
    mode: str,
    job_id: str | None = None,
    *,
    making=None,
    branch=None,
    word_maps=None,
    merge_runtime_context: dict[str, Any] | None = None,
    comfy_admission_lease_token: str | None = None,
    merge_xyz_metadata: dict[str, Any] | None = None,
    comparison_metadata: dict[str, Any] | None = None,
    finalization_stack=None,
) -> GenerationResponse:
    from contextlib import nullcontext
    generation_started = time.perf_counter()
    request = _freeze_spatial_request_seed(request)
    _preflight_profile_prompt_request(request)
    data_dir = _data_dir()
    config = AppConfig()
    stored_operation = _jobs.get(job_id, {}).get("generation_operation_snapshot") if job_id else None
    check_operation = ImageCheckOperation.from_snapshot(stored_operation or _generation_operation_snapshot(request))
    final_output_dir = _output_dir_for(data_dir, request.profile, check_operation)
    history = HistoryRepository(data_dir / 'asteria.db') if check_operation.settings.enabled else None
    image_session = (finalization_stack.enter_context(FinalImageSession(check_operation, final_output_dir, history=history))
                     if check_operation.settings.enabled else None)
    output_dir = image_session.directory if image_session else final_output_dir
    if making is not None:
        making.begin(check_operation.output_root, expected_images=request.batch_count * request.batch_size)
    model_slots = _model_slots_for_request(request, config)
    _validate_required_model_slots(request.profile, model_slots, config)
    external_model_snapshots = _external_model_reference_snapshots(model_slots, config)
    runtime_model_slots = _runtime_model_slots_for_request(
        request.profile,
        model_slots,
        model_root=_model_root(config),
        external_roots=_registered_external_model_roots(config),
    )
    styled_prompt, styled_negative_prompt = _compose_with_advanced_styles(
        request.prompt,
        request.negative_prompt,
        request.advanced_styles,
    )
    prompt_expansion = (
        None
        if _spatial_request_enabled(request)
        else _expand_with_pins(styled_prompt, request, config)
    )
    settings, negative_prompt = _settings_from_request(request, config, mode)
    if comparison_metadata is not None:
        settings['prompt_comparison'] = dict(comparison_metadata)
    if request.profile == "anima":
        try:
            anima_bundle = prepare_anima_generation_bundle(
                model_slots,
                adapter_strength=request.anima_qwen35_adapter_strength,
                negative_adapter_strength=(
                    request.anima_qwen35_negative_adapter_strength
                ),
            )
        except AnimaGenerationBundleError as exc:
            code, _, slots = exc.reason_code.partition(":")
            detail: dict[str, Any] = {"code": code}
            if slots:
                detail["slots"] = slots.split(",")
            raise HTTPException(status_code=409, detail=detail) from exc
        settings["anima_generation_bundle"] = anima_bundle.model_dump(
            mode="json",
            exclude_none=True,
        )
        try:
            hires_checkpoint_value = str(
                settings.get("hires_checkpoint") or ""
            ).strip()
            anima_hires_target = None
            if (
                settings.get("hires_enabled")
                and hires_checkpoint_value
                and hires_checkpoint_value != "Use same checkpoint"
            ):
                anima_hires_library = get_model_library(
                    _model_root(config),
                    config=config,
                )
                anima_hires_target = _resolve_anima_hires_diffusion_path(
                    settings,
                    model_root=_model_root(config),
                    library=anima_hires_library,
                )
            if anima_hires_target is not None:
                anima_hires_bundle = prepare_anima_generation_bundle(
                    {
                        **model_slots,
                        "diffusion_model": str(anima_hires_target),
                    },
                    adapter_strength=request.anima_qwen35_adapter_strength,
                    negative_adapter_strength=(
                        request.anima_qwen35_negative_adapter_strength
                    ),
                )
                if (
                    "anima_3_8b_52"
                    in {
                        anima_bundle.variant_id.value,
                        anima_hires_bundle.variant_id.value,
                    }
                    and anima_bundle.variant_id != anima_hires_bundle.variant_id
                ):
                    raise AnimaGenerationBundleError(
                        "anima_3_8b_hires_variant_mismatch"
                    )
                settings["anima_hires_generation_bundle"] = (
                    anima_hires_bundle.model_dump(
                        mode="json",
                        exclude_none=True,
                    )
                )
        except AnimaLoraRemapRegistryError as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": exc.reason_code},
            ) from exc
        except AnimaGenerationBundleError as exc:
            code, _, slots = exc.reason_code.partition(":")
            detail = {"code": code}
            if slots:
                detail["slots"] = slots.split(",")
            raise HTTPException(status_code=409, detail=detail) from exc
        if (
            anima_bundle.variant_id.value == "anima_3_8b_52"
            and mode not in {"txt2img", "img2img", "inpaint"}
        ):
            raise HTTPException(
                status_code=409,
                detail={"code": "anima_3_8b_unsupported_generation_mode"},
            )
    if external_model_snapshots:
        settings["model_reference_snapshots"] = [
            {key: value for key, value in snapshot.items() if key != "execution_path"}
            for snapshot in external_model_snapshots
        ]
    record_merge_xyz_metadata(settings, merge_xyz_metadata)
    negative_prompt = styled_negative_prompt
    if request.advanced_styles:
        settings["raw_prompt"] = request.prompt
        settings["raw_negative_prompt"] = request.negative_prompt
        settings["advanced_styles"] = [style.model_dump() for style in request.advanced_styles]
    known_embeddings = _known_embedding_names(config)
    selected_runtime = _selected_runtime_id(config, request.runtime)
    branch_source = None
    if branch is not None:
        from app.services.generation_branch import validate_capture, branch_request
        try:
            validate_capture(request, mode, selected_runtime)
            if request.branch_source_id:
                branch_source = HistoryRepository(data_dir / 'asteria.db').get(request.branch_source_id)
                if branch_source is None:
                    raise ValueError('branch_not_available')
                canonical = branch_request(branch_source, {'prompt': request.prompt,
                    'negative_prompt': request.negative_prompt, 'cfg_scale': request.cfg_scale})
                if canonical.model_dump() != request.model_dump():
                    raise ValueError('branch_fixed_settings_changed')
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=409, detail={'code': str(exc)}) from exc
    if _spatial_request_enabled(request) and selected_runtime != "comfy_workflow":
        _validate_runtime_feature_requirements(
            config,
            selected_runtime,
            request.profile,
            settings,
            [],
            runtime_model_slots,
        )
    spatial_context = _prepare_spatial_generation(
        request=request,
        settings=settings,
        config=config,
        model_slots=model_slots,
        known_embeddings=known_embeddings,
        output_dir=output_dir,
    )
    expanded_prompt = ""
    expansion_selections: list[object] = []
    spatial_loras: list[dict[str, Any]] | None = None
    if spatial_context is not None:
        settings["spatial_generation"] = spatial_context.history_record
        expanded_prompt = spatial_context.prompt_plan.base.combined_expanded_prompt
        expansion_selections = _spatial_prompt_selections(spatial_context.prompt_plan)
        spatial_loras = [
            item.model_dump(mode="json", exclude_none=True)
            for item in spatial_context.prompt_plan.loras
        ]
    else:
        if prompt_expansion is None:
            raise RuntimeError("prompt expansion was not prepared")
        expanded_prompt = prompt_expansion.expanded
        expansion_selections = list(prompt_expansion.selections)
        settings["prompt_choice_trace"] = {
            "schema_version": "asteria.prompt-choices.v1",
            "prompt_hash": prompt_expansion.prompt_hash,
            "choices": [asdict(item) for item in prompt_expansion.choices],
        }
    loras = _merge_loras(
        spatial_loras if spatial_loras is not None else extract_loras(expanded_prompt),
        [lora.model_dump(exclude_none=True) for lora in request.loras],
    )
    if request.profile == "anima" and loras:
        try:
            anima_library = get_model_library(_model_root(config), config=config)
            anima_fingerprints = ModelFingerprintService(
                config.data_dir / "config" / "model-fingerprints.json"
            )
            loras, anima_lora_receipts = prepare_anima_lora_runtime_entries(
                loras,
                target_diffusion_path=Path(model_slots["diffusion_model"]),
                library=anima_library,
                fingerprints=anima_fingerprints,
            )
            hires_target = _resolve_anima_hires_diffusion_path(
                settings,
                model_root=_model_root(config),
                library=anima_library,
            )
            anima_hires_lora_receipts: list[dict[str, Any]] = []
            if hires_target is not None:
                hires_loras, anima_hires_lora_receipts = (
                    prepare_anima_lora_runtime_entries(
                        loras,
                        target_diffusion_path=hires_target,
                        library=anima_library,
                        fingerprints=anima_fingerprints,
                    )
                )
                for runtime_lora, hires_lora in zip(loras, hires_loras, strict=True):
                    hires_receipt = hires_lora.get("anima_remap")
                    if isinstance(hires_receipt, dict):
                        runtime_lora["hires_anima_remap"] = hires_receipt
        except (AnimaLoraRemapRegistryError, KeyError) as exc:
            reason_code = (
                exc.reason_code
                if isinstance(exc, AnimaLoraRemapRegistryError)
                else "anima_target_model_slot_missing"
            )
            raise HTTPException(
                status_code=409,
                detail={"code": reason_code},
            ) from exc
        if anima_lora_receipts:
            settings["anima_lora_remap_receipts"] = anima_lora_receipts
        if anima_hires_lora_receipts:
            settings["anima_hires_lora_remap_receipts"] = (
                anima_hires_lora_receipts
            )
        settings["anima_remap_lora_count"] = sum(
            1 for lora in loras if isinstance(lora.get("anima_remap"), dict)
        )
        settings["anima_standard_lora_count"] = (
            len(loras) - int(settings["anima_remap_lora_count"])
        )
    has_lora_runtime_extensions = any(
        lora.get("block_weight")
        or lora.get("block_weight_preset")
        or lora.get("hierarchy_mode") not in {None, "", "chain"}
        for lora in loras
    )
    if has_lora_runtime_extensions:
        settings["lora_block_weight_runtime"] = "requested"
    embeddings = sorted(
        set(
            (
                list(spatial_context.prompt_plan.embeddings)
                if spatial_context is not None
                else extract_embeddings(expanded_prompt, known_embeddings)
            )
            + extract_embeddings(negative_prompt, known_embeddings)
        )
    )
    prompt_syntax_record = _prepare_prompt_syntax_record(
        request=request,
        config=config,
        settings=settings,
        expanded_prompt=expanded_prompt,
        expanded_negative_prompt=negative_prompt,
        selected_runtime=selected_runtime,
    )
    settings["prompt_syntax"] = prompt_syntax_record
    _validate_runtime_feature_requirements(
        config,
        selected_runtime,
        request.profile,
        settings,
        loras,
        runtime_model_slots,
    )
    _record_hires_runtime(settings, selected_runtime, request.profile)
    _record_hires_checkpoint_runtime(settings, selected_runtime, request.profile)
    _record_refiner_runtime(settings, selected_runtime, request.profile)
    profile_contract = next((p for p in ModelProfileRepository(config.model_profiles_path).list() if p.id == request.profile), None)
    settings["generation_assets"] = capture_generation_assets(
        model_slots, settings, loras, embeddings,
        get_model_library(_model_root(config), config=config),
        ModelFingerprintService(config.data_dir / "config" / "model-fingerprints.json"),
        slot_types={slot.id: slot.type for slot in profile_contract.model_slots} if profile_contract else {},
        allow_identical_aliases=branch is not None,
    )
    runtime_settings = dict(settings)
    runtime_settings.pop("prompt_input", None)
    runtime_settings.pop("prompt_layout", None)
    runtime_settings.pop("spatial_generation", None)
    runtime_request = {
        "profile": request.profile,
        "prompt": request.prompt,
        "expanded_prompt": expanded_prompt,
        "prompt_expansion_selections": expansion_selections,
        "negative_prompt": negative_prompt,
        "settings": runtime_settings,
        "model_slots": runtime_model_slots,
        "loras": loras,
        "embeddings": embeddings,
        "prompt_syntax": prompt_syntax_record["semantic_plans"],
        "topaz_settings": _topaz_settings_from_request(request).model_dump(),
    }
    if word_maps is not None and request.profile in {'sdxl', 'anima'} and selected_runtime == 'comfy_workflow':
        runtime_request['_word_maps'] = word_maps.begin(check_operation.output_root, request.profile)
    if branch is not None:
        from app.services.generation_branch import check_asset_identity
        try:
            if branch_source:
                check_asset_identity(branch_source['settings']['generation_branch']['generation_assets'], settings['generation_assets'])
            runtime_request['_generation_branch'] = branch.begin(check_operation.output_root, request, branch_source)
            branch.prepare_control_inputs(runtime_settings)
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(status_code=409, detail={'code': 'branch_source_unavailable'}) from exc
    if check_operation.settings.enabled:
        if selected_runtime != "comfy_workflow":
            raise HTTPException(status_code=409, detail={"code": "image_check_runtime_not_supported"})
        runtime_request["_image_check_operation"] = check_operation
        # Live frames retain the existing preview registry and serving path.
        # Completed candidates stay private until their final decision.
        runtime_request['_preview_output_dir'] = str(final_output_dir / '_previews')
        def prepare_final_batch(completed, batch_settings, directory):
            before = completed['output_paths']
            after = _apply_topaz_hires_if_needed(before, batch_settings, request, config, directory)
            primary, mapped = remap_primary_outputs(completed.get('primary_outputs', []), before, after)
            if not mapped:
                raise OutputFinalizationError('Final image manifest does not match completed processing')
            return {**completed, 'output_paths': after, 'primary_outputs': primary}
        runtime_request['_prepare_final_image_batch'] = prepare_final_batch
    if spatial_context is not None:
        runtime_request["_spatial_runtime_context"] = spatial_context.runtime_context
    if merge_runtime_context is not None:
        runtime_request["_merge_runtime_context"] = dict(merge_runtime_context)
    if comfy_admission_lease_token:
        runtime_request["_comfy_admission_lease_token"] = comfy_admission_lease_token
    if job_id is not None:
        def update_job_progress(
            current: int,
            total: int,
            preview_path: str = "",
            step: int | None = None,
            steps: int | None = None,
            phase: str = "",
        ) -> None:
            step_value = max(0, int(step or 0))
            steps_value = max(0, int(steps or 0))
            completed_ratio = current / max(total, 1)
            if steps_value > 0:
                completed_ratio = (current + min(step_value / steps_value, 1)) / max(total, 1)
            patch: dict[str, Any] = {
                "current": current,
                "total": total,
                "progress": 0.05 + (0.9 * completed_ratio),
            }
            if steps is not None:
                patch["step"] = step_value
                patch["steps"] = steps_value
            if phase:
                patch["phase"] = phase
            if preview_path:
                with _jobs_lock:
                    preview_paths = [*_jobs.get(job_id, {}).get("preview_paths", []), preview_path]
                patch["preview_path"] = preview_path
                patch["preview_paths"] = preview_paths[-33:]
                try:
                    preview = Path(preview_path)
                    PreviewRegistry(_data_dir() / "asteria.db").register(
                        job_id,
                        preview,
                        output_root=preview.parent.parent,
                    )
                except (OSError, RuntimeError, ValueError, sqlite3.Error):
                    pass
            _set_job(job_id, **patch)

        runtime_request["progress_callback"] = update_job_progress
        runtime_request["cancel_requested"] = lambda: bool(
            _jobs.get(job_id, {}).get("cancelled")
        )
        runtime_request["runtime_job_callback"] = lambda prompt_id: _set_job(
            job_id,
            runtime_job_id=prompt_id,
            runtime_stop_confirmed=False,
        )
        runtime_request["runtime_stop_confirmed"] = lambda: bool(
            _jobs.get(job_id, {}).get("runtime_stop_confirmed")
        )
    if making is not None:
        runtime_request['progress_callback'] = making.wrap(runtime_request.get('progress_callback'))
    runtime = _runtime(config, selected_runtime)
    _revalidate_external_model_reference_snapshots(external_model_snapshots, config)
    try:
        if mode == "img2img":
            result = runtime.generate_img2img(runtime_request, output_dir)
        elif mode == "inpaint":
            result = runtime.generate_inpaint(runtime_request, output_dir)
        else:
            result = runtime.generate_txt2img(runtime_request, output_dir)
    except RuntimeCancellationConfirmed as exc:
        raise OperationCancelled() from exc
    except (RuntimeErrorResponse, RuntimeAdapterError, TopazError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    runtime_settings_after_execution = runtime_request.get("settings")
    if isinstance(runtime_settings_after_execution, dict):
        settings.update(
            {
                key: value
                for key, value in runtime_settings_after_execution.items()
                if key not in {"prompt_input", "prompt_layout", "spatial_generation"}
            }
        )
    if result.get("runtime_adapter"):
        settings["runtime_adapter"] = result["runtime_adapter"]
        if settings.get("controlnet_units") and request.profile == "sdxl" and result["runtime_adapter"] == "comfy_workflow":
            settings["controlnet_runtime"] = "comfy_workflow"
        if (
            settings.get("detailer_units")
            and request.profile in {"sdxl", "anima"}
            and result["runtime_adapter"] == "comfy_workflow"
        ):
            settings["detailer_runtime"] = "comfy_workflow"
        if settings.get("anima_lllite_units") and request.profile == "anima" and result["runtime_adapter"] == "comfy_workflow":
            settings["anima_lllite_runtime"] = "comfy_workflow"
            _record_anima_lllite_runtime_settings(settings)
        if settings.get("lora_block_weight_runtime") == "requested":
            settings["lora_block_weight_runtime"] = (
                "comfy_workflow_inspire"
                if request.profile == "sdxl" and result["runtime_adapter"] == "comfy_workflow"
                else "metadata_only"
            )
    if result.get("runtime_job_id"):
        settings["runtime_job_id"] = result["runtime_job_id"]
    if result.get("runtime_job_ids"):
        settings["runtime_job_ids"] = result["runtime_job_ids"]
    if result.get("packet_path"):
        settings["packet_path"] = result["packet_path"]
    if result.get("engine_info"):
        settings["engine_info"] = result["engine_info"]
    if result.get("comfy_prompt_id"):
        settings["comfy_prompt_id"] = result["comfy_prompt_id"]
    if result.get("comfy_prompt_ids"):
        settings["comfy_prompt_ids"] = result["comfy_prompt_ids"]
    if result.get("generated_seeds"):
        settings["generated_seeds"] = result["generated_seeds"]
    output_settings = check_operation.output_settings
    settings["output_directory"] = str(final_output_dir)
    settings["output_root"] = str(check_operation.output_root)
    settings["filename_template"] = output_settings.filename_template
    settings["output_image_format"] = output_settings.image_format
    record_hires_metadata(settings)
    record_controlnet_artifacts(settings)
    record_anima_lllite_artifacts(settings)
    record_detailer_metadata(settings)
    output_paths = list(result["output_paths"])
    primary_outputs = [dict(output) for output in result.get("primary_outputs", [])]
    before_paths = output_paths
    if image_session and not result.get('image_check_finalized'):
        raise HTTPException(status_code=502, detail={'code': 'image_check_final_output_missing'})
    output_paths = (output_paths if image_session else _apply_topaz_hires_if_needed(
        output_paths, settings, request, config, output_dir))
    primary_outputs, manifest_mapped = remap_primary_outputs(primary_outputs, before_paths, output_paths)
    if not manifest_mapped:
        settings["primary_output_manifest_warning"] = "topaz_postprocess_path_mapping_failed"
    try:
        before_paths = output_paths
        if image_session is None:
            output_paths = _apply_output_image_format(output_paths, output_settings.image_format)
        remap_checked_image_paths(settings, before_paths, output_paths)
        primary_outputs, manifest_mapped = remap_primary_outputs(primary_outputs, before_paths, output_paths)
        if not manifest_mapped:
            settings["primary_output_manifest_warning"] = "output_format_path_mapping_failed"
    except OutputFinalizationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    settings["generation_duration_seconds"] = round(
        max(0.0, time.perf_counter() - generation_started),
        3,
    )
    generated_tags = metadata_tags(
        profile=request.profile,
        prompt=expanded_prompt or request.prompt,
        model_slots=model_slots,
        embeddings=embeddings,
        loras=loras,
    )
    history = history or HistoryRepository(data_dir / "asteria.db")
    with history.publication() if image_session else nullcontext():
        generation_id = history.add(
            status=result["status"], profile=request.profile, prompt=request.prompt,
            expanded_prompt=expanded_prompt, negative_prompt=negative_prompt,
            settings=settings, model_slots=model_slots, loras=loras, embeddings=embeddings,
            output_paths=[] if image_session else output_paths, generated_tags=generated_tags,
        )
        before_paths = output_paths
        if image_session:
            stage_paths = [s['path'] for s in result.get('enhancement_stages', [])]
            all_paths = list(dict.fromkeys([*output_paths, *stage_paths]))
            published = image_session.publish(all_paths, settings, request.profile, generation_id)
            path_mapping = dict(zip(all_paths, published, strict=True))
            output_paths = [path_mapping[p] for p in output_paths]
            result['enhancement_stages'] = [{**s, 'path': path_mapping[s['path']]}
                                           for s in result.get('enhancement_stages', [])]
        else:
            output_paths = _apply_filename_template(output_paths, settings, request.profile, generation_id)
            remap_checked_image_paths(settings, before_paths, output_paths)
        primary_outputs, manifest_mapped = remap_primary_outputs(primary_outputs, before_paths, output_paths)
        if not manifest_mapped:
            if image_session:
                raise OutputFinalizationError('Final image publication manifest mismatch')
            settings["primary_output_manifest_warning"] = "filename_path_mapping_failed"
        finalize_merge_xyz_metadata(settings, output_paths=output_paths,
            runtime_adapter=str(result.get("runtime_adapter") or selected_runtime), mapping_id=f"{request.profile}_txt2img")
        settings["primary_outputs"] = primary_outputs
        making_output_paths = list(output_paths) if making is not None else []
        from app.services.enhancement_stages import finalize_stages
        finalize_stages(settings, result.get('enhancement_stages', []), primary_outputs, output_paths)
        word_paths = []
        if word_maps is not None:
            settings['word_masks'], word_paths = word_maps.finish(primary_outputs, result['status'])
            output_paths.extend(word_paths)
        branch_paths = []
        if branch is not None:
            settings['generation_branch'], branch_paths = branch.finish(
                request, settings, model_slots, expanded_prompt, negative_prompt, result['status'])
            output_paths.extend(branch_paths)
            record_controlnet_artifacts(settings)
            record_anima_lllite_artifacts(settings)
        if image_session is None and result["status"] != "cancelled":
            _auto_send_primary_outputs_to_eagle(data_dir=data_dir, config=config, primary_outputs=primary_outputs,
                generated_tags=generated_tags, prompt=expanded_prompt or request.prompt,
                negative_prompt=negative_prompt, settings=settings)
        history.update_settings(generation_id, settings)
        metadata_payload = {
            "generation_id": generation_id, "status": result["status"], "profile": request.profile,
            "prompt": request.prompt, "expanded_prompt": expanded_prompt, "negative_prompt": negative_prompt,
            "prompt_expansion_selections": expansion_selections, "settings": settings,
            "model_slots": model_slots, "loras": loras, "embeddings": embeddings, "metadata_tags": generated_tags,
        }
        sidecar_paths = MetadataWriter().write_generation_sidecars(metadata_payload, output_paths)
        if sidecar_paths:
            output_paths = [*output_paths, *sidecar_paths]
        if image_session:
            history.update_output_paths(generation_id, output_paths)
            image_session.seal_history(generation_id)
        else:
            _update_output_paths_safely(history, generation_id, output_paths)
    if image_session:
        image_session.commit()
        if result["status"] != "cancelled":
            _auto_send_primary_outputs_to_eagle(data_dir=data_dir, config=config, primary_outputs=primary_outputs,
                generated_tags=generated_tags, prompt=expanded_prompt or request.prompt,
                negative_prompt=negative_prompt, settings=settings)
            history.update_settings(generation_id, settings)
            MetadataWriter().write_generation_sidecars(metadata_payload,
                [p for p in output_paths if Path(p).suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}])
    remasters: list[RemasterResponse] = []
    if request.remaster.enabled and result["status"] != "cancelled":
        remasters = _fan_out_remasters(
            request,
            generation_id,
            primary_outputs,
            model_slots,
            loras,
            job_id=job_id,
            image_check_operation=check_operation if image_session else None,
        )
        settings["remaster_generation_ids"] = [item.id for item in remasters]
        settings["remaster_statuses"] = [item.status for item in remasters]
        settings["remaster_output_paths"] = [item.output_path for item in remasters]
        settings["remaster_comparison_paths"] = [item.contact_sheet_path for item in remasters]
        if image_session:
            child_checks = [item.settings['image_check'] for item in remasters if 'image_check' in item.settings]
            settings['image_check']['remaster_checks'] = child_checks
            if any(check.get('pause') for check in child_checks):
                settings['image_check']['pause'] = True
                settings['image_check']['status'] = 'unknown'
        if len(remasters) != len(primary_outputs):
            settings["remaster_preparation_warning"] = "primary_output_manifest_incomplete"
        history.update_settings(generation_id, settings)
        MetadataWriter().write_generation_sidecars(
            {
                "generation_id": generation_id,
                "status": result["status"],
                "profile": request.profile,
                "prompt": request.prompt,
                "expanded_prompt": expanded_prompt,
                "negative_prompt": negative_prompt,
                "prompt_expansion_selections": expansion_selections,
                "settings": settings,
                "model_slots": model_slots,
                "loras": loras,
                "embeddings": embeddings,
                "metadata_tags": generated_tags,
            },
            [str(primary["path"]) for primary in primary_outputs],
        )
    if making is not None:
        settings['making_video'] = making.finish(history, generation_id, {
            'status': result['status'], 'profile': request.profile,
            'prompt': request.prompt, 'expanded_prompt': expanded_prompt, 'negative_prompt': negative_prompt,
            'settings': settings, 'model_slots': model_slots, 'loras': loras, 'embeddings': embeddings,
            'output_paths': making_output_paths,
        }, cancelled=runtime_request.get('cancel_requested', lambda: False))
        history.update_settings(generation_id, settings)
        MetadataWriter().write_generation_sidecars({
            'generation_id': generation_id, 'status': result['status'], 'profile': request.profile,
            'prompt': request.prompt, 'expanded_prompt': expanded_prompt, 'negative_prompt': negative_prompt,
            'prompt_expansion_selections': expansion_selections, 'settings': settings,
            'model_slots': model_slots, 'loras': loras, 'embeddings': embeddings, 'metadata_tags': generated_tags,
        }, making_output_paths)
    history_item = history.get(generation_id)
    if branch_paths:
        branch.commit()
    if word_paths:
        word_maps.commit()
    return GenerationResponse(
        id=generation_id,
        status=result["status"],
        profile=request.profile,
        prompt=request.prompt,
        expanded_prompt=expanded_prompt,
        negative_prompt=negative_prompt,
        settings=settings,
        model_slots=model_slots,
        loras=loras,
        embeddings=embeddings,
        output_paths=output_paths,
        media_kind=history_item["media_kind"] if history_item is not None else "image",
        artifacts=history_item["artifacts"] if history_item is not None else [],
        primary_outputs=primary_outputs,
        remasters=remasters,
    )


def _fan_out_remasters(
    request: Txt2ImgRequest,
    generation_id: int,
    primary_outputs: list[dict],
    model_slots: dict[str, str],
    loras: list[dict],
    *,
    job_id: str | None = None,
    orchestrator=None,
    image_check_operation: ImageCheckOperation | None = None,
) -> list[RemasterResponse]:
    from app.api.remaster import _build_orchestrator

    active_orchestrator = orchestrator or _build_orchestrator()
    results: list[RemasterResponse] = []
    for index, primary in enumerate(primary_outputs):
        if job_id is not None:
            _set_job(
                job_id,
                phase="remastering",
                current=index,
                total=max(1, len(primary_outputs)),
                runtime_job_id="",
                runtime_stop_confirmed=False,
            )
        child_request = RemasterRequest(
            source_image_path=str(primary.get("path") or ""),
            source_generation_id=generation_id,
            source_output_index=int(primary.get("output_index", index)),
            profile=request.profile,
            runtime="comfy_workflow",
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            seed=int(primary.get("seed", request.seed)),
            cfg_scale=request.cfg_scale,
            distilled_cfg_scale=request.distilled_cfg_scale,
            model_slot_overrides=model_slots,
            loras=loras,
            remaster=request.remaster,
        )
        try:
            result = active_orchestrator.run(
                child_request,
                **({'image_check_operation': image_check_operation} if image_check_operation else {}),
                cancel_requested=(
                    (lambda: bool(_jobs.get(job_id, {}).get("cancelled"))) if job_id is not None else None
                ),
                runtime_job_callback=(
                    (lambda prompt_id: _set_job(job_id, runtime_job_id=prompt_id)) if job_id is not None else None
                ),
                runtime_stop_confirmed=(
                    (lambda: bool(_jobs.get(job_id, {}).get("runtime_stop_confirmed"))) if job_id is not None else None
                ),
            )
        except RemasterExecutionError as exc:
            status = "cancelled" if exc.code == "remaster_cancelled" else "failed"
            result = RemasterResponse(
                id=exc.remaster_id,
                status=status,
                source_generation_id=generation_id,
                source_image_path=child_request.source_image_path,
                source_output_index=child_request.source_output_index,
                model_slots=model_slots,
                loras=child_request.loras,
                warnings=[exc.code],
            )
        results.append(result)
        if result.settings.get('image_check', {}).get('pause'):
            break
    return results


def _update_output_paths_safely(repository: HistoryRepository, generation_id: int, output_paths: list[str]) -> None:
    try:
        repository.update_output_paths(generation_id, output_paths)
    except (OSError, sqlite3.Error, ValueError):
        return


def _is_referenced_output_path(path: Path, data_dir: Path) -> bool:
    resolved = _resolve_path_safely(path)
    if resolved is None:
        return False
    for item in HistoryRepository(data_dir / "asteria.db").list_all():
        for output_path in item["output_paths"]:
            try:
                if Path(output_path).resolve() == resolved:
                    return True
            except _PATH_ACCESS_ERRORS:
                continue
    return False


def _is_current_output_path(path: Path, data_dir: Path) -> bool:
    settings = UserSettingsRepository(data_dir / "user-settings.json").load().output
    output_dir = _resolve_path_safely(resolve_output_root(data_dir, settings))
    resolved = _resolve_path_safely(path)
    if output_dir is None or resolved is None:
        return False
    return resolved == output_dir or output_dir in resolved.parents


def _known_output_roots(data_dir: Path, items: list[dict] | None = None) -> set[Path]:
    settings = UserSettingsRepository(data_dir / "user-settings.json").load().output
    roots: set[Path] = set()
    if output_root := _resolve_path_safely(resolve_output_root(data_dir, settings)):
        roots.add(output_root)
    history_items = items if items is not None else HistoryRepository(data_dir / "asteria.db").list_all()
    for item in history_items:
        for key in ("output_root", "output_directory"):
            value = item["settings"].get(key)
            if value:
                try:
                    roots.add(Path(str(value)).resolve())
                except _PATH_ACCESS_ERRORS:
                    continue
    return roots


def _media_artifact_roots(
    data_dir: Path,
    item: dict,
    artifact_id: str,
) -> set[Path]:
    roots = _known_output_roots(data_dir, [item])
    for raw in item.get("artifacts", []):
        try:
            artifact = MediaArtifact.model_validate(raw)
        except ValueError:
            continue
        if artifact.artifact_id == artifact_id:
            if artifact.kind == "poster":
                roots.add(data_dir / "cache" / "media-posters")
            break
    return roots


def _model_slots_for_request(request: Txt2ImgRequest, config: AppConfig) -> dict[str, str]:
    model_slots = dict(ModelSelectionRepository(config.model_selections_path).get(request.profile).slots)
    for slot, value in request.model_slot_overrides.items():
        if str(value).strip():
            model_slots[str(slot)] = str(value)
    return model_slots


def _xyz_axis_key(axis: XYZAxis, index: int) -> str:
    return axis.label.strip() or axis.target.strip() or f"axis_{index + 1}"


def _apply_xyz_value(request: Txt2ImgRequest, axis: XYZAxis, value: str) -> Txt2ImgRequest:
    next_request = request.model_copy(deep=True)
    target = axis.target.strip().lower()
    value = str(value)
    if target == "seed":
        next_request.seed = int(value)
    elif target in {"checkpoint", "model", "model_slot:checkpoint"}:
        next_request.model_slot_overrides["checkpoint"] = value
    elif target in {"vae", "model_slot:vae"}:
        next_request.vae = value
    elif target == "sampler":
        next_request.sampler = value
    elif target == "scheduler":
        next_request.scheduler = value
    elif target == "steps":
        next_request.steps = int(value)
    elif target in {"cfg", "cfg_scale"}:
        next_request.cfg_scale = float(value)
    elif target == "alpha":
        next_request.model_slot_overrides["merge.alpha"] = value
    elif target == "beta":
        next_request.model_slot_overrides["merge.beta"] = value
    elif target == "alpha_beta":
        next_request.model_slot_overrides["merge.alpha_beta"] = value
    elif target in {"mbw_preset", "mbw_alpha", "mbw_beta", "pinpoint_blocks", "merge_mode", "calculation_mode", "elemental", "dice", "model_a", "model_b", "model_c"}:
        next_request.model_slot_overrides[f"merge.{target}"] = value
    elif target == "prompt_replace":
        if "=>" in value:
            source, replacement = value.split("=>", 1)
        elif "|" in value:
            source, replacement = value.split("|", 1)
        else:
            source, replacement = "", value
        source = source.strip()
        replacement = replacement.strip()
        next_request.prompt = next_request.prompt.replace(source, replacement) if source else f"{next_request.prompt}, {replacement}".strip(", ")
    elif target.startswith("lora_weight"):
        lora_name = ""
        weight_value = value
        if "=" in value:
            lora_name, weight_value = value.split("=", 1)
        elif ":" in target:
            lora_name = target.split(":", 1)[1]
        weight = float(weight_value)
        if lora_name:
            matched = False
            for lora in next_request.loras:
                if lora.name == lora_name:
                    lora.weight = weight
                    matched = True
            if not matched:
                next_request.loras.append(LoraSelection(name=lora_name, weight=weight))
        elif next_request.loras:
            next_request.loras[0].weight = weight
        else:
            raise HTTPException(
                status_code=400,
                detail={"code": api_errors.XYZ_LORA_WEIGHT_TARGET_REQUIRED, "target": target},
            )
    else:
        raise HTTPException(
            status_code=400,
            detail={"code": api_errors.XYZ_AXIS_UNSUPPORTED, "target": axis.target},
        )
    return next_request


def _xyz_cell_request(base: Txt2ImgRequest, axes: list[XYZAxis], values: tuple[str, ...]) -> tuple[Txt2ImgRequest, dict[str, str]]:
    request = base
    labels: dict[str, str] = {}
    for index, (axis, value) in enumerate(zip(axes, values, strict=True)):
        labels[_xyz_axis_key(axis, index)] = str(value)
        request = _apply_xyz_value(request, axis, str(value))
    return request, labels


def _first_image_output(output_paths: list[str]) -> Path | None:
    for output_path in output_paths:
        path = Path(output_path)
        if (
            path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            and _path_exists_safely(path)
            and _path_is_file_safely(path)
        ):
            return path
    return None


def _xyz_grid_axes(axes: list[XYZAxis]) -> list[XYZGridAxis]:
    return [
        XYZGridAxis(
            key=_xyz_axis_key(axis, index),
            label=axis.label.strip() or axis.target.strip() or f"axis_{index + 1}",
            values=tuple(str(value) for value in axis.values),
        )
        for index, axis in enumerate(axes)
    ]


def _xyz_contact_sheet_result(
    cells: list[XYZCellResult],
    output_dir: Path,
    axes: list[XYZAxis],
) -> tuple[str, str]:
    return create_xyz_contact_sheet_streaming(
        [
            XYZGridCell(
                labels=cell.labels,
                image_path=_first_image_output(cell.generation.output_paths),
                display_id=f"#{cell.index + 1:03d}",
            )
            for cell in cells
        ],
        _xyz_grid_axes(axes),
        output_dir,
        operation_id=uuid4().hex,
    )


def _xyz_contact_sheet(cells: list[XYZCellResult], output_dir: Path, axes: list[XYZAxis]) -> str:
    return _xyz_contact_sheet_result(cells, output_dir, axes)[0]


def _record_xyz_history(
    request: XYZPlotRequest,
    cells: list[XYZCellResult],
    contact_sheet_path: str,
) -> tuple[int, list[str]]:
    data_dir = _data_dir()
    output_paths = [contact_sheet_path] if contact_sheet_path else []
    settings = {
        "mode": "xyz_plot",
        "seed": request.base.seed,
        "width": request.base.width,
        "height": request.base.height,
        "steps": request.base.steps,
        "cfg_scale": request.base.cfg_scale,
        "sampler": request.base.sampler,
        "scheduler": request.base.scheduler,
        "generated_seeds": [
            cell.generation.settings.get("seed")
            for cell in cells
            if cell.generation.settings.get("seed") is not None
        ],
        "axes": [axis.model_dump() for axis in request.axes],
        "cell_count": len(cells),
        "cells": [
            {
                "index": cell.index,
                "labels": cell.labels,
                "generation_id": cell.generation.id,
                "output_paths": cell.generation.output_paths,
                "settings": cell.generation.settings,
            }
            for cell in cells
        ],
    }
    status = str(getattr(request, "status", "completed") or "completed")
    warning_code = str(getattr(request, "warning_code", "") or "")
    if warning_code:
        settings["warning_code"] = warning_code
    source_mode = getattr(request, "mode", None)
    if source_mode:
        settings["source_mode"] = source_mode
    if isinstance(request.base, Img2ImgRequest):
        settings.update(
            {
                "init_image_path": request.base.init_image_path,
                "denoising_strength": request.base.denoising_strength,
            }
        )
        if request.base.image_edit_lineage:
            settings["image_edit_lineage"] = request.base.image_edit_lineage.model_dump(mode="json")
    if isinstance(request.base, InpaintRequest):
        settings.update(
            {
                "mask_image_path": request.base.mask_image_path,
                "mask_blur": request.base.mask_blur,
                "masked_content": request.base.masked_content,
                "inpaint_area": request.base.inpaint_area,
            }
        )
    config = AppConfig()
    model_slots = _model_slots_for_request(request.base, config)
    generated_tags = metadata_tags(
        profile=request.base.profile,
        prompt=request.base.prompt,
        model_slots=model_slots,
        embeddings=[],
        loras=[lora.model_dump(exclude_none=True) for lora in request.base.loras],
    )
    xyz_tags = [*generated_tags, "xyz plot"]
    if contact_sheet_path:
        _auto_send_primary_outputs_to_eagle(
            data_dir=data_dir,
            config=config,
            primary_outputs=[{"path": contact_sheet_path, "output_index": 0}],
            generated_tags=xyz_tags,
            prompt=request.base.prompt,
            negative_prompt=request.base.negative_prompt,
            settings=settings,
        )
    history = HistoryRepository(data_dir / "asteria.db")
    generation_id = history.add(
        status=status,
        profile=request.base.profile,
        prompt=request.base.prompt,
        expanded_prompt=request.base.prompt,
        negative_prompt=request.base.negative_prompt,
        settings=settings,
        model_slots=model_slots,
        loras=[lora.model_dump(exclude_none=True) for lora in request.base.loras],
        embeddings=[],
        output_paths=output_paths,
        generated_tags=xyz_tags,
    )
    sidecar_paths = MetadataWriter().write_generation_sidecars(
        {
            "generation_id": generation_id,
            "status": status,
            "profile": request.base.profile,
            "prompt": request.base.prompt,
            "expanded_prompt": request.base.prompt,
            "negative_prompt": request.base.negative_prompt,
            "settings": settings,
            "model_slots": model_slots,
            "loras": [lora.model_dump(exclude_none=True) for lora in request.base.loras],
            "embeddings": [],
            "metadata_tags": xyz_tags,
        },
        output_paths,
    )
    if sidecar_paths:
        output_paths = [*output_paths, *sidecar_paths]
        _update_output_paths_safely(history, generation_id, output_paths)
    return generation_id, output_paths


@router.post("/txt2img", response_model=GenerationResponse)
def txt2img(request: Txt2ImgRequest) -> GenerationResponse:
    _preflight_controlnet_mode_request(request, "txt2img")
    return _generate(request, "txt2img")


@_artifact_write_operation
def _generate_xyz_plot(request: XYZPlotRequest) -> XYZPlotResponse:
    if _spatial_request_enabled(request.base):
        conflicting_axis = next(
            (
                axis
                for axis in request.axes
                if axis.target.strip().lower() == "prompt_replace"
            ),
            None,
        )
        if conflicting_axis is not None:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": api_errors.XYZ_AXIS_UNSUPPORTED,
                    "target": conflicting_axis.target,
                    "reason": "prompt_replace_conflicts_with_prompt_layout",
                },
            )
    if request.mode == "img2img":
        _ensure_supported_img2img_profile(request.base.profile)
    elif request.mode == "inpaint":
        if not isinstance(request.base, InpaintRequest):
            raise ValueError("inpaint XYZ request did not resolve to InpaintRequest")
        _ensure_supported_inpaint_request(request.base)
    combinations = list(product(*[axis.values for axis in request.axes]))
    if len(combinations) > 64:
        raise HTTPException(
            status_code=400,
            detail={"code": api_errors.XYZ_PLOT_CELL_LIMIT_EXCEEDED, "limit": 64, "cells": len(combinations)},
        )
    _preflight_profile_prompt_request(request.base)
    _preflight_controlnet_mode_request(request.base, request.mode)
    _preflight_seed_resize_request(request.base, request.mode)
    prepared_cells: list[tuple[int, dict[str, str], Txt2ImgRequest]] = []
    for index, values in enumerate(combinations):
        cell_request, labels = _xyz_cell_request(request.base, request.axes, tuple(str(value) for value in values))
        _preflight_profile_prompt_request(cell_request)
        _preflight_controlnet_mode_request(cell_request, request.mode)
        _preflight_seed_resize_request(cell_request, request.mode)
        prepared_cells.append((index, labels, cell_request))
    cells: list[XYZCellResult] = []
    for index, labels, cell_request in prepared_cells:
        generation = _generate(cell_request, request.mode)
        cells.append(XYZCellResult(index=index, labels=labels, generation=generation))
    contact_sheet_path = ""
    contact_sheet_warning = ""
    if request.contact_sheet and cells:
        contact_sheet_path, contact_sheet_warning = _xyz_contact_sheet_result(
            cells,
            _output_dir_for(_data_dir(), request.base.profile),
            request.axes,
        )
    operation_status = "completed-with-warnings" if contact_sheet_warning else "completed"
    record_request = type(
        "_XYZRecord",
        (),
        {
            "base": request.base,
            "axes": request.axes,
            "mode": request.mode,
            "status": operation_status,
            "warning_code": contact_sheet_warning,
        },
    )()
    generation_id, output_paths = _record_xyz_history(record_request, cells, contact_sheet_path)
    return XYZPlotResponse(
        id=generation_id,
        status=operation_status,
        profile=request.base.profile,
        axes=request.axes,
        cells=cells,
        contact_sheet_path=contact_sheet_path,
        output_paths=output_paths,
    )


@router.post("/xyz", response_model=XYZPlotResponse)
def xyz_plot(request: XYZPlotRequest) -> XYZPlotResponse:
    return _generate_xyz_plot(request)


@router.post("/xyz/txt2img", response_model=XYZPlotResponse)
def xyz_txt2img(request: XYZPlotRequest) -> XYZPlotResponse:
    if request.mode != "txt2img":
        raise HTTPException(status_code=400, detail={"code": api_errors.XYZ_AXIS_UNSUPPORTED, "target": "mode"})
    return _generate_xyz_plot(request)


@router.post("/img2img", response_model=GenerationResponse)
def img2img(request: Img2ImgRequest) -> GenerationResponse:
    _ensure_supported_img2img_profile(request.profile)
    _preflight_controlnet_mode_request(request, "img2img")
    return _generate(request, "img2img")


@router.post("/inpaint", response_model=GenerationResponse)
def inpaint(request: InpaintRequest) -> GenerationResponse:
    _ensure_supported_inpaint_request(request)
    _preflight_controlnet_mode_request(request, "inpaint")
    return _generate(request, "inpaint")


def _ensure_supported_img2img_profile(profile: str) -> None:
    supported_profiles = ["sdxl", "anima", "flux1", "flux2"]
    if profile not in supported_profiles:
        raise HTTPException(
            status_code=409,
            detail={
                "code": api_errors.UNSUPPORTED_IMG2IMG_PROFILE,
                "profile": profile,
                "supported_profiles": supported_profiles,
            },
        )


def _ensure_supported_inpaint_profile(profile: str) -> None:
    supported_profiles = ["sdxl", "anima", "flux1", "flux2"]
    if profile not in supported_profiles:
        raise HTTPException(
            status_code=409,
            detail={
                "code": api_errors.UNSUPPORTED_INPAINT_PROFILE,
                "profile": profile,
                "supported_profiles": supported_profiles,
            },
        )


def _ensure_supported_inpaint_request(request: InpaintRequest) -> None:
    _ensure_supported_inpaint_profile(request.profile)
    selected_runtime = _selected_runtime_id(AppConfig(), request.runtime)
    unsupported: list[dict[str, str]] = []
    masked_content = request.masked_content.strip().lower()
    inpaint_area = request.inpaint_area.strip().lower()
    if masked_content != "original":
        unsupported.append(
            {
                "field": "masked_content",
                "reason": f"masked_content_is_not_mapped_to_{request.profile}_inpaint",
            }
        )
    if inpaint_area == "only_masked":
        if request.profile != "sdxl":
            unsupported.append(
                {
                    "field": "inpaint_area",
                    "reason": "only_masked_is_supported_only_for_sdxl",
                }
            )
        if selected_runtime != "comfy_workflow":
            unsupported.append(
                {
                    "field": "inpaint_area",
                    "reason": "only_masked_requires_comfy_workflow",
                }
            )
        if request.hires_enabled:
            unsupported.append(
                {
                    "field": "inpaint_area",
                    "reason": "only_masked_requires_hires_disabled",
                }
            )
        if request.tiling:
            unsupported.append(
                {
                    "field": "inpaint_area",
                    "reason": "only_masked_requires_tiling_disabled",
                }
            )
    elif inpaint_area != "whole_picture":
        unsupported.append(
            {
                "field": "inpaint_area",
                "reason": f"inpaint_area_is_not_mapped_to_{request.profile}_inpaint",
            }
        )
    if unsupported:
        raise HTTPException(
            status_code=409,
            detail={"code": api_errors.UNSUPPORTED_COMFY_RUNTIME_SETTING, "unsupported": unsupported},
        )
    if inpaint_area == "only_masked":
        try:
            validate_only_masked_inputs(
                request.init_image_path,
                request.mask_image_path,
                mask_blur=request.mask_blur,
                requested_size=(request.width, request.height),
            )
        except InpaintRegionError as exc:
            raise HTTPException(status_code=400, detail={"code": exc.code}) from None


def _job_response(job_id: str, job: dict, operation=None) -> GenerationJobResponse:
    if operation is None:
        try:
            operation = _generation_operation_coordinator().get(job_id)
        except OperationNotFoundError:
            operation = None
    status = operation.status if operation is not None else job.get("status", "queued")
    current = operation.progress_current if operation is not None else int(job.get("current", 0))
    total_value = (
        operation.progress_total
        if operation is not None and operation.progress_total is not None
        else int(job.get("total", 1))
    )
    progress = (
        min(1.0, current / total_value)
        if total_value > 0
        else (1.0 if status in {"completed", "completed_with_warnings"} else 0.0)
    )
    if status in {"completed", "completed_with_warnings", "failed", "cancelled", "interrupted"}:
        progress = 1.0
    return GenerationJobResponse(
        id=job_id,
        status=status,
        mode=job.get("mode") or (
            str(operation.request_snapshot.get("mode", "")) if operation is not None else ""
        ),
        progress=progress,
        current=current,
        total=total_value,
        step=int(job.get("step", 0)),
        steps=int(job.get("steps", 0)),
        phase=operation.phase if operation is not None else str(job.get("phase", "")),
        generation=job.get("generation"),
        remaster=job.get("remaster"),
        preview_path=str(job.get("preview_path", "")),
        preview_paths=list(job.get("preview_paths", [])),
        error=(
            str(job.get("error", "")) or operation.error_code or ""
            if operation is not None
            else str(job.get("error", ""))
        ),
        error_target_ids=list(job.get("error_target_ids", [])),
        cancelled=bool(job.get("cancelled", False)) or status in {"cancelling", "cancelled"},
    )


def _set_job(job_id: str, **patch: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(patch)
        comparison_total = _jobs.get(job_id, {}).get('comparison_total')
        comparison_index = int(_jobs.get(job_id, {}).get('comparison_index', 0))
    progress_keys = {"current", "total", "phase", "step", "steps"}
    if not progress_keys.intersection(patch):
        return
    try:
        coordinator = _generation_operation_coordinator()
        current_operation = coordinator.get(job_id)
        if current_operation.status not in {"preparing", "running"}:
            return
        total = patch.get("total", current_operation.progress_total)
        current = patch.get("current")
        if current is None and "progress" in patch and total is not None:
            current = round(float(patch["progress"]) * total)
        if comparison_total:
            current = comparison_index + min(1, max(0, int(current or 0)))
            total = comparison_total
        coordinator.repository.transition(
            job_id,
            expected_revision=current_operation.row_revision,
            target_status="running",
            phase=str(patch.get("phase", current_operation.phase)),
            progress_current=(
                int(current) if current is not None else current_operation.progress_current
            ),
            progress_total=int(total) if total is not None else None,
            progress_unit=current_operation.progress_unit,
        )
    except (OperationNotFoundError, ValueError):
        return


def _run_job(job_id: str, _publisher_lease: ReferenceMutationLease) -> None:
    coordinator = _generation_operation_coordinator()
    while True:
        operation = coordinator.execute_operation(job_id)
        if operation.status != "queued":
            return
        if job_id not in _publisher_leases:
            return
        threading.Event().wait(0.05)


def _fail_scheduled_operation(job_id: str) -> None:
    try:
        coordinator = _generation_operation_coordinator()
        current = coordinator.get(job_id)
        if current.status == "queued":
            coordinator.repository.transition(
                job_id,
                expected_revision=current.row_revision,
                target_status="failed",
                phase="failed",
                error_code="operation_schedule_failed",
                retryable=True,
            )
    except (OperationNotFoundError, ValueError):
        pass
    lease = _publisher_leases.pop(job_id, None)
    if lease is not None:
        lease.release()


def _enqueue_generation_job(mode: str, request: Txt2ImgRequest | RemasterRequest) -> GenerationJobResponse:
    if isinstance(request, Txt2ImgRequest):
        request = _freeze_spatial_request_seed(request)
    coordinator = _generation_operation_coordinator()
    publisher_lease = reference_mutation_coordinator.acquire_publisher()
    registered = False
    operation = None
    try:
        snapshot = {
            "mode": mode,
            "generation_contract_version": CURRENT_SCHEMA_VERSIONS.generation,
            "profile": getattr(request, "profile", ""),
            "runtime": getattr(request, "runtime", None),
            "request": request.model_dump(mode="json"),
            "request_fields_set": sorted(request.model_fields_set),
        }
        if isinstance(request, Txt2ImgRequest):
            snapshot["generation_operation_snapshot"] = _generation_operation_snapshot(request)
        operation = coordinator.enqueue(
            kind=f"generation.{mode}",
            source_actor="local",
            request_snapshot=snapshot,
            request_contract_version=CURRENT_SCHEMA_VERSIONS.generation,
            progress_total=max(1, int(getattr(request, "batch_count", 1))),
            progress_unit="images",
        )
        job_id = operation.operation_id
        with _jobs_lock:
            _jobs[job_id] = _new_job_cache(mode, request)
            _publisher_leases[job_id] = publisher_lease
            registered = True
            response = _job_response(job_id, _jobs[job_id], operation)
        worker = threading.Thread(
            target=_run_job,
            args=(job_id, publisher_lease),
            daemon=True,
        )
        worker.start()
    except BaseException:
        if operation is not None:
            _fail_scheduled_operation(operation.operation_id)
        else:
            publisher_lease.release()
        if registered:
            with _jobs_lock:
                _jobs.pop(job_id, None)
        raise
    return response


def enqueue_discord_txt2img(
    request: Txt2ImgRequest,
    *,
    discord_snapshot: dict[str, Any],
    identity,
    requester_id: str,
    guild_id: str | None,
    channel_id: str | None,
):
    from app.services.discord_receipts import DiscordOperationClaim

    request = _freeze_spatial_request_seed(request)
    _preflight_controlnet_mode_request(request, "txt2img")
    coordinator = _generation_operation_coordinator()
    publisher_lease = reference_mutation_coordinator.acquire_publisher()
    registered = False
    claim: DiscordOperationClaim | None = None
    try:
        snapshot = {
            "mode": "txt2img",
            "generation_contract_version": CURRENT_SCHEMA_VERSIONS.generation,
            "profile": request.profile,
            "runtime": request.runtime,
            "request": request.model_dump(mode="json"),
            "request_fields_set": sorted(request.model_fields_set),
            "discord": dict(discord_snapshot),
            "generation_operation_snapshot": _generation_operation_snapshot(request),
        }
        claim = coordinator.enqueue_with_discord_receipt(
            kind="generation.txt2img",
            source_actor="discord",
            request_snapshot=snapshot,
            identity=identity,
            requester_id=requester_id,
            guild_id=guild_id,
            channel_id=channel_id,
            request_contract_version=CURRENT_SCHEMA_VERSIONS.generation,
            progress_total=max(1, request.batch_count),
            progress_unit="images",
        )
        if not claim.created:
            publisher_lease.release()
            return claim
        job_id = claim.operation.operation_id
        with _jobs_lock:
            _jobs[job_id] = _new_job_cache("txt2img", request)
            _publisher_leases[job_id] = publisher_lease
            registered = True
        worker = threading.Thread(
            target=_run_job,
            args=(job_id, publisher_lease),
            daemon=True,
        )
        worker.start()
        return claim
    except BaseException:
        if claim is not None and claim.created:
            _fail_scheduled_operation(claim.operation.operation_id)
        else:
            publisher_lease.release()
        if registered:
            with _jobs_lock:
                _jobs.pop(claim.operation.operation_id, None)
        raise


@router.post("/jobs/txt2img", response_model=GenerationJobResponse)
def queue_txt2img(request: Txt2ImgRequest) -> GenerationJobResponse:
    _preflight_controlnet_mode_request(request, "txt2img")
    return _enqueue_generation_job("txt2img", request)


@router.post("/jobs/img2img", response_model=GenerationJobResponse)
def queue_img2img(request: Img2ImgRequest) -> GenerationJobResponse:
    _ensure_supported_img2img_profile(request.profile)
    _preflight_controlnet_mode_request(request, "img2img")
    return _enqueue_generation_job("img2img", request)


@router.post("/jobs/inpaint", response_model=GenerationJobResponse)
def queue_inpaint(request: InpaintRequest) -> GenerationJobResponse:
    _ensure_supported_inpaint_request(request)
    _preflight_controlnet_mode_request(request, "inpaint")
    return _enqueue_generation_job("inpaint", request)


@router.get("/jobs/{job_id}", response_model=GenerationJobResponse)
def get_job(job_id: str) -> GenerationJobResponse:
    try:
        operation = _generation_operation_coordinator().get(job_id)
    except OperationNotFoundError:
        raise HTTPException(status_code=404, detail={"code": api_errors.GENERATION_JOB_NOT_FOUND}) from None
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        mode = str(operation.request_snapshot.get("mode", ""))
        request_payload = operation.request_snapshot.get("request", {})
        request_type = {
            "txt2img": Txt2ImgRequest,
            "img2img": Img2ImgRequest,
            "inpaint": InpaintRequest,
            "remaster": RemasterRequest,
        }.get(mode, Txt2ImgRequest)
        try:
            request = request_type.model_validate(request_payload)
            job = _new_job_cache(mode, request)
        except ValueError:
            job = {"mode": mode}
    return _job_response(job_id, job, operation)


@router.websocket("/jobs/{job_id}/events")
async def job_events(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    try:
        while True:
            try:
                response = get_job(job_id).model_dump(mode="json")
            except HTTPException:
                await websocket.send_json({"id": job_id, "status": "missing"})
                return
            await websocket.send_json(response)
            if response["status"] in {
                "completed",
                "completed_with_warnings",
                "failed",
                "cancelled",
                "interrupted",
            }:
                return
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return


@router.post("/jobs/{job_id}/cancel", response_model=GenerationJobResponse)
def cancel_job(job_id: str) -> GenerationJobResponse:
    try:
        cancel_generation_operation(job_id)
    except OperationNotFoundError:
        with _jobs_lock:
            legacy_job = _jobs.get(job_id)
        if legacy_job is not None and legacy_job.get("mode") == "remaster":
            return _cancel_legacy_remaster_job(job_id, legacy_job)
        raise HTTPException(status_code=404, detail={"code": api_errors.GENERATION_JOB_NOT_FOUND}) from None
    return get_job(job_id)


def cancel_generation_operation(job_id: str) -> OperationRecord:
    coordinator = _generation_operation_coordinator()
    operation = coordinator.get(job_id)
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return coordinator.request_cancel(job_id)
    if operation.status in {"completed", "completed_with_warnings", "failed", "cancelled", "interrupted"}:
        return operation
    _set_job(job_id, cancelled=True)
    operation = coordinator.request_cancel(job_id)
    if operation.status != "cancelling":
        return operation
    if job["mode"] == "remaster" and operation.status == "cancelling":
        prompt_id = str(job.get("runtime_job_id") or "")
        if not prompt_id:
            _set_job(job_id, runtime_stop_confirmed=True)
            return coordinator.get(job_id)
        warnings: list[str] = []
        try:
            client = ComfyClient(AppConfig().comfy_base_url)
            queue = client.queue_state()
            if prompt_id in queue["pending"]:
                client.delete_queued_prompt(prompt_id)
                _set_job(job_id, runtime_stop_confirmed=True)
            elif prompt_id in queue["running"]:
                client.interrupt()
                _set_job(job_id, runtime_stop_confirmed=True)
        except RuntimeErrorResponse:
            warnings.append("runtime_cancel_operation_failed")
        if warnings:
            _set_job(job_id, cancel_warnings=warnings)
        return coordinator.get(job_id)
    try:
        ComfyClient(AppConfig().comfy_base_url).interrupt()
        _set_job(job_id, runtime_stop_confirmed=True)
    except RuntimeErrorResponse:
        pass
    return coordinator.get(job_id)


def _cancel_legacy_remaster_job(job_id: str, job: dict[str, Any]) -> GenerationJobResponse:
    if job.get("status") in {"completed", "failed", "cancelled"}:
        return _job_response(job_id, job)
    _set_job(job_id, cancelled=True, status="cancelling", phase="cancelling")
    prompt_id = str(job.get("runtime_job_id") or "")
    if not prompt_id:
        _set_job(job_id, runtime_stop_confirmed=True)
        with _jobs_lock:
            return _job_response(job_id, _jobs[job_id])
    warnings: list[str] = []
    try:
        client = ComfyClient(AppConfig().comfy_base_url)
        queue = client.queue_state()
        if prompt_id in queue["pending"]:
            client.delete_queued_prompt(prompt_id)
            _set_job(job_id, runtime_stop_confirmed=True)
        elif prompt_id in queue["running"]:
            client.interrupt()
            _set_job(job_id, runtime_stop_confirmed=True)
    except RuntimeErrorResponse:
        warnings.append("runtime_cancel_operation_failed")
    if warnings:
        _set_job(job_id, cancel_warnings=warnings)
    with _jobs_lock:
        return _job_response(job_id, _jobs[job_id])


@router.post("/jobs/{job_id}/retry", response_model=GenerationJobResponse)
def retry_job(job_id: str) -> GenerationJobResponse:
    coordinator = _generation_operation_coordinator()
    try:
        source = coordinator.get(job_id)
        mode = str(source.request_snapshot["mode"])
        request_type = {
            "txt2img": Txt2ImgRequest,
            "img2img": Img2ImgRequest,
            "inpaint": InpaintRequest,
            "remaster": RemasterRequest,
        }[mode]
        # Reject retired request contracts before retry creates a new operation.
        request = request_type.model_validate(source.request_snapshot["request"])
        operation = coordinator.retry(job_id, source_actor="local")
    except OperationNotFoundError:
        raise HTTPException(status_code=404, detail={"code": api_errors.GENERATION_JOB_NOT_FOUND}) from None
    except OperationRetryError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc)}) from exc
    except ValueError as exc:
        code = (
            "spatial_prompt_not_in_release"
            if "spatial_prompt_not_in_release" in str(exc)
            else "invalid_generation_snapshot"
        )
        raise HTTPException(status_code=409, detail={"code": code}) from exc
    publisher_lease = reference_mutation_coordinator.acquire_publisher()
    with _jobs_lock:
        _jobs[operation.operation_id] = _new_job_cache(mode, request)
        _publisher_leases[operation.operation_id] = publisher_lease
        response = _job_response(operation.operation_id, _jobs[operation.operation_id], operation)
    try:
        threading.Thread(
            target=_run_job,
            args=(operation.operation_id, publisher_lease),
            daemon=True,
        ).start()
    except BaseException:
        _fail_scheduled_operation(operation.operation_id)
        with _jobs_lock:
            _jobs.pop(operation.operation_id, None)
        raise
    return response


@router.post("/jobs/cleanup", response_model=GenerationJobCleanupResponse)
def cleanup_jobs() -> GenerationJobCleanupResponse:
    data_dir = _data_dir()
    preview_registry = PreviewRegistry(data_dir / "asteria.db")
    before_preview_count = sum(
        len(preview_registry.operation_paths(item.operation_id))
        for item in _generation_operation_coordinator().list(limit=1000)
    )
    result = OperationRetentionService(
        _generation_operation_coordinator().repository
    ).cleanup()
    preview_result = preview_registry.cleanup()
    with _jobs_lock:
        known_operation_ids = {
            item.operation_id
            for item in _generation_operation_coordinator().list(
                limit=1000,
            )
        }
        for cached_id in tuple(_jobs):
            if cached_id not in known_operation_ids:
                _jobs.pop(cached_id, None)
    return GenerationJobCleanupResponse(
        removed_jobs=result.removed_operations,
        remaining_active_jobs=result.remaining_active_operations,
        remaining_terminal_jobs=result.remaining_terminal_operations,
        removed_preview_references=min(
            before_preview_count,
            preview_result.deleted_preview_files,
        ),
    )


@router.get("/history/{generation_id}/reproduction")
def generation_reproduction(generation_id: int) -> dict:
    config = AppConfig()
    item = HistoryRepository(_data_dir() / "asteria.db").get(generation_id)
    if item is None:
        raise HTTPException(status_code=404, detail={"code": "generation_not_found"})
    return compare_generation_assets(
        item.get("settings", {}).get("generation_assets"),
        get_model_library(_model_root(config), config=config),
        ModelFingerprintService(config.data_dir / "config" / "model-fingerprints.json"),
    )


class WordMaskSelection(BaseModel):
    image_index: int = Field(ge=0, strict=True)
    columns: list[int] = Field(min_length=1, max_length=256)
    threshold: float = Field(default=.5, ge=.01, le=.99, allow_inf_nan=False)
    expand: float = Field(default=0, ge=-10, le=10, allow_inf_nan=False)


@router.get('/history/{generation_id}/word-masks')
def generation_word_masks(generation_id: int) -> dict:
    from app.services.word_masks import word_mask_details
    item = HistoryRepository(_data_dir() / 'asteria.db').get(generation_id)
    try:
        return word_mask_details(item or {})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=409, detail={'code': 'word_masks_unavailable'}) from exc


@router.post('/history/{generation_id}/word-masks/preview')
def preview_generation_word_mask(generation_id: int, selection: WordMaskSelection) -> Response:
    from io import BytesIO
    from app.services.word_masks import make_word_mask
    item = HistoryRepository(_data_dir() / 'asteria.db').get(generation_id)
    try:
        mask = make_word_mask(item or {}, selection.image_index, selection.columns, selection.threshold, selection.expand)
        stream = BytesIO(); mask.save(stream, format='PNG')
        return Response(stream.getvalue(), media_type='image/png', headers={'Cache-Control': 'no-store'})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=409, detail={'code': 'word_masks_unavailable'}) from exc


@router.get('/history/{generation_id}/branch')
def generation_branch_details(generation_id: int) -> dict:
    from app.services.generation_branch import branch_request, resolve_source
    item = HistoryRepository(_data_dir() / 'asteria.db').get(generation_id)
    try:
        if item is None:
            raise ValueError('branch_not_available')
        resolve_source(item)
        request = branch_request(item, {})
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(status_code=409, detail={'code': 'branch_source_unavailable'}) from exc
    saved = item['settings']['generation_branch']
    return {'generation_id': generation_id, 'cut_step': saved['cut_step'], 'bytes': saved['bytes'],
            'prompt': request.prompt, 'negative_prompt': request.negative_prompt, 'cfg_scale': request.cfg_scale,
            'profile': request.profile, 'steps': request.steps, 'seed': request.seed,
            'sampler': request.sampler, 'scheduler': request.scheduler, 'width': request.width, 'height': request.height}


@router.post('/history/{generation_id}/branch', response_model=GenerationJobResponse)
def queue_generation_branch(generation_id: int, changes: dict) -> GenerationJobResponse:
    from app.services.generation_branch import branch_request, resolve_source
    item = HistoryRepository(_data_dir() / 'asteria.db').get(generation_id)
    try:
        if item is None:
            raise ValueError('branch_not_available')
        resolve_source(item)
        request = branch_request(item, changes)
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(status_code=409, detail={'code': 'branch_source_unavailable'}) from exc
    return _enqueue_generation_job('txt2img', request)


@router.get("/history", response_model=GenerationHistoryResponse)
def history(
    limit: int = Query(default=72, ge=1, le=100),
    cursor: str = "",
    q: str = "",
    profile: str = "",
    favorite: bool = False,
    tag: str = "",
    folder: str = "all",
    media_kind: Literal["all", "image", "video", "audio"] = "all",
    condition_verdict: Literal["", "match", "mismatch", "unknown", "unchecked"] = "",
) -> GenerationHistoryResponse:
    repository = HistoryRepository(_data_dir() / "asteria.db")
    try:
        page = repository.page(
            limit=limit,
            cursor=cursor,
            query=q,
            profile=profile,
            favorite=favorite,
            tag=tag,
            folder=folder,
            media_kind=media_kind,
            condition_verdict=condition_verdict,
        )
    except ValueError as error:
        if str(error) == "invalid_history_cursor":
            raise HTTPException(status_code=400, detail={"code": "invalid_history_cursor"}) from error
        raise
    history_items = repository.list_all()
    items = VideoSequenceHistoryProjector(
        repository.db_path,
        accepted_roots=_known_output_roots(_data_dir(), history_items),
    ).project_items(
        page.items,
        history_items=history_items,
    )
    return GenerationHistoryResponse(
        items=items,
        next_cursor=page.next_cursor,
        has_more=page.has_more,
    )


@router.api_route(
    "/history/{generation_id}/artifacts/{artifact_id}",
    methods=["GET", "HEAD"],
)
def history_media_artifact(
    generation_id: int,
    artifact_id: str,
    request: Request,
    download: bool = False,
) -> Response:
    data_dir = _data_dir()
    item = HistoryRepository(data_dir / "asteria.db").get(generation_id)
    if item is None:
        raise HTTPException(status_code=404, detail={"code": "media_artifact_not_found"})
    try:
        artifact, path = resolve_registered_media_artifact(
            item,
            artifact_id,
            accepted_roots=_media_artifact_roots(data_dir, item, artifact_id),
            allowed_kinds={"video", "audio", "poster", "image"},
        )
        if artifact.kind == 'image' and (artifact.mime_type != 'image/gif' or artifact.role != 'derived'):
            raise MediaArtifactAccessError('media_artifact_not_found')
    except MediaArtifactAccessError as error:
        raise HTTPException(
            status_code=404,
            detail={"code": "media_artifact_not_found"},
        ) from error
    try:
        path_details = path.stat()
        size = path_details.st_size
    except OSError as error:
        raise HTTPException(
            status_code=404,
            detail={"code": "media_artifact_not_found"},
        ) from error
    byte_range = _single_byte_range(request.headers.get("range", ""), size)
    start, end = byte_range if byte_range is not None else (0, size - 1)
    length = end - start + 1
    status_code = 206 if byte_range is not None else 200
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Disposition": f"{'attachment' if download else 'inline'}; filename*=UTF-8''{quote(path.name, safe='')}",
        "X-Content-Type-Options": "nosniff",
        "Access-Control-Allow-Origin": "*",
    }
    if byte_range is not None:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    if request.method == "HEAD":
        return Response(
            status_code=status_code,
            media_type=artifact.mime_type,
            headers=headers,
        )
    try:
        source = path.open("rb")
        opened_details = os.fstat(source.fileno())
    except OSError as error:
        raise HTTPException(
            status_code=404,
            detail={"code": "media_artifact_not_found"},
        ) from error
    if (
        opened_details.st_size != size
        or opened_details.st_dev != path_details.st_dev
        or opened_details.st_ino != path_details.st_ino
    ):
        source.close()
        raise HTTPException(
            status_code=404,
            detail={"code": "media_artifact_not_found"},
        )
    return StreamingResponse(
        _stream_file_range(source, start=start, length=length),
        status_code=status_code,
        media_type=artifact.mime_type,
        headers=headers,
    )


def _open_containing_folder(path: Path) -> None:
    if os.name != "nt" or not hasattr(os, "startfile"):
        raise OSError("folder reveal is unavailable")
    os.startfile(str(path.parent))


@router.post("/history/{generation_id}/artifacts/{artifact_id}/reveal")
def reveal_history_audio_artifact(
    generation_id: int,
    artifact_id: str,
) -> dict[str, str]:
    data_dir = _data_dir()
    item = HistoryRepository(data_dir / "asteria.db").get(generation_id)
    if item is None:
        raise HTTPException(status_code=404, detail={"code": "media_artifact_not_found"})
    try:
        _artifact, path = resolve_registered_media_artifact(
            item,
            artifact_id,
            accepted_roots=_known_output_roots(data_dir, [item]),
            allowed_kinds={"audio"},
        )
    except MediaArtifactAccessError as error:
        raise HTTPException(
            status_code=404,
            detail={"code": "media_artifact_not_found"},
        ) from error
    try:
        _open_containing_folder(path)
    except OSError as error:
        raise HTTPException(
            status_code=503,
            detail={"code": "media_artifact_reveal_failed"},
        ) from error
    return {"status": "opened"}


def _single_byte_range(header: str, size: int) -> tuple[int, int] | None:
    if not header:
        return None
    invalid_headers = {"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"}
    if size <= 0 or not header.startswith("bytes=") or "," in header:
        raise HTTPException(status_code=416, headers=invalid_headers)
    value = header.removeprefix("bytes=").strip()
    if value.count("-") != 1:
        raise HTTPException(status_code=416, headers=invalid_headers)
    start_text, end_text = (part.strip() for part in value.split("-", 1))
    try:
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length <= 0:
                raise ValueError
            start = max(0, size - suffix_length)
            end = size - 1
        else:
            start = int(start_text)
            end = size - 1 if not end_text else min(int(end_text), size - 1)
            if start < 0 or start >= size or end < start:
                raise ValueError
    except (TypeError, ValueError):
        raise HTTPException(status_code=416, headers=invalid_headers) from None
    return start, end


def _stream_file_range(source, *, start: int, length: int):
    remaining = length
    with source:
        source.seek(start)
        while remaining > 0:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


@router.get("/history/tags", response_model=GenerationTagCatalogResponse)
def history_tags() -> GenerationTagCatalogResponse:
    repository = HistoryRepository(_data_dir() / "asteria.db")
    catalog = repository.tag_catalog()
    return GenerationTagCatalogResponse(**catalog)


@router.get("/history/{generation_id}/thumbnail")
def history_thumbnail(generation_id: int) -> FileResponse:
    data_dir = _data_dir()
    item = HistoryRepository(data_dir / "asteria.db").get(generation_id)
    if item is None:
        raise HTTPException(status_code=404, detail={"code": api_errors.GENERATION_NOT_FOUND})
    accepted_roots = _known_output_roots(data_dir, [item])
    accepted_roots.update(
        {
            data_dir / "init_images",
            data_dir / "mask_images",
            data_dir / "control_images",
            data_dir / "cache" / "media-posters",
        }
    )
    try:
        thumbnail = LibraryThumbnailService(data_dir / "cache" / "library-thumbnails").thumbnail_for(
            item,
            accepted_roots,
        )
    except LibraryThumbnailError as error:
        unavailable = error.code in {"library_thumbnail_cache_unavailable", "library_thumbnail_encode_failed"}
        code = api_errors.LIBRARY_THUMBNAIL_UNAVAILABLE if unavailable else api_errors.LIBRARY_THUMBNAIL_NOT_FOUND
        raise HTTPException(status_code=503 if unavailable else 404, detail={"code": code}) from error
    return FileResponse(
        thumbnail,
        media_type="image/webp",
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


@router.get("/history/folders", response_model=GenerationFolderResponse)
def history_folders() -> GenerationFolderResponse:
    return GenerationFolderResponse(items=HistoryRepository(_data_dir() / "asteria.db").list_folders())


@router.post("/history/folders", response_model=GenerationFolderResponse)
def create_history_folder(request: CreateGenerationFolderRequest) -> GenerationFolderResponse:
    repository = HistoryRepository(_data_dir() / "asteria.db")
    try:
        repository.create_folder(request.name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return GenerationFolderResponse(items=repository.list_folders())


@router.patch("/history/bulk", response_model=GenerationHistoryResponse)
def bulk_update_history_items(request: BulkGenerationCurationRequest) -> GenerationHistoryResponse:
    repository = HistoryRepository(_data_dir() / "asteria.db")
    try:
        items = repository.bulk_update_curation(
            request.ids,
            rating=request.rating,
            favorite=request.favorite,
            tags_add=request.tags_add,
            tags_remove=request.tags_remove,
            folder_id=request.folder_id,
            update_folder=request.update_folder,
        )
    except HistoryTagLimitExceededError as error:
        raise HTTPException(
            status_code=400,
            detail={"code": api_errors.HISTORY_TAG_LIMIT_EXCEEDED, "limit": 160},
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    _sync_curation_sidecars(items)
    return GenerationHistoryResponse(items=items)


@router.post("/history/scan", response_model=LibraryScanResponse)
def scan_history_sidecars() -> LibraryScanResponse:
    data_dir = _data_dir()
    repository = HistoryRepository(data_dir / "asteria.db")
    items = repository.list_all()
    output_roots = _known_output_roots(data_dir, items)
    return _scan_sidecar_metadata(repository, output_roots)


@router.patch("/history/{generation_id}", response_model=GenerationHistoryResponse)
def update_history_item(generation_id: int, request: UpdateGenerationCurationRequest) -> GenerationHistoryResponse:
    repository = HistoryRepository(_data_dir() / "asteria.db")
    item = repository.update_curation(
        generation_id,
        rating=request.rating,
        favorite=request.favorite,
        tags=request.tags,
        notes=request.notes,
        folder_id=request.folder_id,
        update_folder=request.update_folder,
    )
    if item is None:
        raise HTTPException(status_code=404, detail={"code": api_errors.GENERATION_NOT_FOUND})
    _sync_curation_sidecars([item])
    return GenerationHistoryResponse(items=[item])


@router.post("/history/bulk-delete", response_model=BulkDeleteGenerationResponse)
def bulk_delete_history_items(request: BulkDeleteGenerationRequest) -> BulkDeleteGenerationResponse:
    data_dir = _data_dir()
    repository = HistoryRepository(data_dir / "asteria.db")
    try:
        items = repository.bulk_delete(request.ids)
    except HistoryBulkDeleteConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": api_errors.HISTORY_BULK_DELETE_CONFLICT,
                "missing_ids": exc.missing_ids,
            },
        ) from exc
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": api_errors.HISTORY_BULK_DELETE_FAILED},
        ) from exc

    empty_response = BulkDeleteGenerationResponse(
        deleted_ids=[int(item["id"]) for item in items],
        delete_files=request.delete_files,
        deleted_files=[],
        missing_files=[],
        skipped_files=[],
        protected_files=[],
    )
    if not request.delete_files:
        return empty_response

    result = _delete_history_artifacts(items, data_dir, repository)
    return BulkDeleteGenerationResponse(
        deleted_ids=empty_response.deleted_ids,
        delete_files=True,
        deleted_files=result.deleted_files,
        missing_files=result.missing_files,
        skipped_files=result.skipped_files,
        protected_files=result.protected_files,
    )


@router.delete("/history/{generation_id}", response_model=DeleteGenerationResponse)
def delete_history_item(generation_id: int, delete_files: bool = True) -> DeleteGenerationResponse:
    data_dir = _data_dir()
    repository = HistoryRepository(data_dir / "asteria.db")
    item = repository.get(generation_id)
    if item is None:
        raise HTTPException(status_code=404, detail={"code": api_errors.GENERATION_NOT_FOUND})
    deleted_files: list[str] = []
    missing_files: list[str] = []
    skipped_files: list[str] = []
    try:
        deleted = repository.delete(generation_id)
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise HTTPException(status_code=409, detail={"code": api_errors.HISTORY_DELETE_FAILED}) from exc
    if deleted and delete_files:
        result = _delete_history_artifacts([item], data_dir, repository)
        deleted_files = result.deleted_files
        missing_files = result.missing_files
        skipped_files = [*result.skipped_files, *result.protected_files]
    return DeleteGenerationResponse(
        id=generation_id,
        deleted=deleted,
        deleted_files=deleted_files,
        missing_files=missing_files,
        skipped_files=skipped_files,
    )


@router.get("/history/integrity", response_model=HistoryIntegrityResponse)
def history_integrity() -> HistoryIntegrityResponse:
    data_dir = _data_dir()
    items = HistoryRepository(data_dir / "asteria.db").list_all()
    settings = UserSettingsRepository(data_dir / "user-settings.json").load().output
    missing: dict[int, list[str]] = {}
    referenced = set()
    for item in items:
        item_missing = []
        for output_path in item["output_paths"]:
            try:
                path = Path(output_path)
            except TypeError:
                item_missing.append(str(output_path))
                continue
            try:
                resolved = path.resolve()
            except _PATH_ACCESS_ERRORS:
                item_missing.append(str(path))
                continue
            referenced.add(resolved)
            if not _path_exists_safely(resolved) or not _path_is_file_safely(resolved):
                item_missing.append(str(path))
        if item_missing:
            missing[item["id"]] = item_missing
    output_roots = _known_output_roots(data_dir, items)
    orphan_files = []
    for output_root in output_roots:
        if _path_exists_safely(output_root):
            try:
                paths = list(output_root.rglob("*"))
            except _PATH_ACCESS_ERRORS:
                continue
            for path in paths:
                try:
                    resolved = path.resolve()
                except _PATH_ACCESS_ERRORS:
                    continue
                if not is_private_image_candidate(resolved) and _path_is_file_safely(path) and resolved not in referenced:
                    orphan_files.append(str(path))
    return HistoryIntegrityResponse(missing_files=missing, orphan_files=orphan_files)


@router.post("/history/orphans/cleanup", response_model=CleanupResponse)
def cleanup_orphan_outputs() -> CleanupResponse:
    if not _begin_orphan_cleanup():
        raise HTTPException(
            status_code=409,
            detail={"code": api_errors.ORPHAN_CLEANUP_GENERATION_ACTIVE},
        )
    try:
        integrity = history_integrity()
        deleted: list[str] = []
        skipped: list[str] = []
        data_dir = _data_dir()
        output_roots = _known_output_roots(data_dir)
        for orphan in integrity.orphan_files:
            path = Path(orphan)
            try:
                resolved = path.resolve()
            except _PATH_ACCESS_ERRORS:
                skipped.append(orphan)
                continue
            if _is_referenced_output_path(resolved, data_dir):
                skipped.append(str(resolved))
                continue
            if any(resolved == output_root or output_root in resolved.parents for output_root in output_roots):
                if _path_exists_safely(resolved) and _path_is_file_safely(resolved):
                    try:
                        resolved.unlink()
                    except _PATH_ACCESS_ERRORS:
                        skipped.append(str(resolved))
                        continue
                    deleted.append(str(resolved))
                else:
                    skipped.append(orphan)
            else:
                skipped.append(orphan)
        return CleanupResponse(deleted_files=deleted, skipped_files=skipped)
    finally:
        _end_orphan_cleanup()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_export_file(archive: zipfile.ZipFile, path: Path, archive_name: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as source, archive.open(archive_name, "w") as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size_bytes += len(chunk)
            digest.update(chunk)
            target.write(chunk)
    return {
        "source_path": str(path),
        "archive_path": archive_name,
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }


def _export_archive_name(path: Path, used_names: set[str]) -> str:
    base_name = path.name
    archive_name = f"files/{base_name}"
    if archive_name not in used_names:
        used_names.add(archive_name)
        return archive_name
    stem = path.stem
    suffix = path.suffix
    index = 2
    while True:
        archive_name = f"files/{stem}-{index}{suffix}"
        if archive_name not in used_names:
            used_names.add(archive_name)
            return archive_name
        index += 1


def _raise_metadata_export_failed(
    repository: HistoryRepository,
    generation_id: int,
    *,
    stage: str,
    error: Exception,
    bundle_path: Path | None = None,
    cleanup_path: Path | None = None,
) -> None:
    target_cleanup_path = cleanup_path or bundle_path
    if target_cleanup_path is not None:
        try:
            target_cleanup_path.unlink(missing_ok=True)
        except _PATH_ACCESS_ERRORS:
            pass
    _append_metadata_export_event_safely(
        repository,
        generation_id,
        {
            "adapter": "metadata_export",
            "status": "failure",
            "stage": stage,
            "error": str(error),
            **({"path": str(bundle_path)} if bundle_path is not None else {}),
        },
    )
    raise HTTPException(
        status_code=409,
        detail={"code": "metadata_export_failed", "stage": stage},
    ) from error


def _append_metadata_export_event_safely(
    repository: HistoryRepository,
    generation_id: int,
    event: dict[str, Any],
) -> None:
    try:
        repository.append_external_event(generation_id, event)
    except (OSError, sqlite3.Error, ValueError):
        return


@router.post("/history/{generation_id}/export", response_model=ExportBundleResponse)
def export_generation_bundle(generation_id: int) -> ExportBundleResponse:
    data_dir = _data_dir()
    repository = HistoryRepository(data_dir / "asteria.db")
    item = repository.get(generation_id)
    if item is None:
        raise HTTPException(status_code=404, detail={"code": api_errors.GENERATION_NOT_FOUND})
    bundle_dir = data_dir / "exports"
    try:
        bundle_dir.mkdir(parents=True, exist_ok=True)
    except _PATH_ACCESS_ERRORS as exc:
        _raise_metadata_export_failed(repository, generation_id, stage="prepare", error=exc)
    bundle_path = bundle_dir / f"generation-{generation_id}.zip"
    tmp_bundle_path = bundle_dir / f".generation-{generation_id}-{uuid4().hex}.zip.tmp"
    exported_files: list[dict[str, Any]] = []
    missing_files: list[str] = []
    used_archive_names = {"generation.json", "manifest.json"}
    try:
        with zipfile.ZipFile(tmp_bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("generation.json", json.dumps(item, ensure_ascii=False, indent=2))
            for output_path in item["output_paths"]:
                try:
                    path = Path(output_path)
                except TypeError:
                    missing_files.append(str(output_path))
                    continue
                if _path_exists_safely(path) and _path_is_file_safely(path):
                    archive_name = _export_archive_name(path, used_archive_names)
                    try:
                        exported_file = _write_export_file(archive, path, archive_name)
                    except _PATH_ACCESS_ERRORS:
                        missing_files.append(str(path))
                        continue
                    exported_files.append(exported_file)
                else:
                    missing_files.append(str(path))
            archive.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "schema": "asteria.generation.export.v1",
                        "generation_id": generation_id,
                        "generation_archive_path": "generation.json",
                        "files": exported_files,
                        "missing_files": missing_files,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
    except (*_PATH_ACCESS_ERRORS, zipfile.BadZipFile) as exc:
        _raise_metadata_export_failed(
            repository,
            generation_id,
            stage="write",
            error=exc,
            bundle_path=bundle_path,
            cleanup_path=tmp_bundle_path,
        )
    try:
        bundle_size_bytes = tmp_bundle_path.stat().st_size
        bundle_sha256 = _sha256_file(tmp_bundle_path)
        tmp_bundle_path.replace(bundle_path)
    except _PATH_ACCESS_ERRORS as exc:
        _raise_metadata_export_failed(
            repository,
            generation_id,
            stage="finalize",
            error=exc,
            bundle_path=bundle_path,
            cleanup_path=tmp_bundle_path,
        )
    _append_metadata_export_event_safely(
        repository,
        generation_id,
        {
            "adapter": "metadata_export",
            "status": "success",
            "path": str(bundle_path),
            "file_count": len(exported_files),
            "missing_files": missing_files,
            "manifest_schema": "asteria.generation.export.v1",
            "bundle_size_bytes": bundle_size_bytes,
            "bundle_sha256": bundle_sha256,
            "files": exported_files,
        },
    )
    return ExportBundleResponse(
        path=str(bundle_path),
        file_count=len(exported_files),
        missing_files=missing_files,
        bundle_size_bytes=bundle_size_bytes,
        bundle_sha256=bundle_sha256,
        files=[ExportedBundleFile(**file) for file in exported_files],
    )


@router.get("/outputs/{filename:path}")
def output_file(filename: str) -> FileResponse:
    settings = UserSettingsRepository(_data_dir() / "user-settings.json").load().output
    try:
        output_dir = resolve_output_root(_data_dir(), settings).resolve()
        path = (output_dir / Path(filename)).resolve()
    except _PATH_ACCESS_ERRORS:
        raise HTTPException(status_code=404, detail={"code": api_errors.OUTPUT_IMAGE_NOT_FOUND}) from None
    if (
        (output_dir != path and output_dir not in path.parents)
        or is_private_image_candidate(path)
        or not _path_exists_safely(path)
        or not _path_is_file_safely(path)
    ):
        raise HTTPException(status_code=404, detail={"code": api_errors.OUTPUT_IMAGE_NOT_FOUND})
    return FileResponse(path, headers={"Access-Control-Allow-Origin": "*"})


@router.get("/output-by-path")
def output_file_by_path(path: str) -> FileResponse:
    try:
        output_path = Path(path).resolve()
    except _PATH_ACCESS_ERRORS:
        raise HTTPException(status_code=404, detail={"code": api_errors.OUTPUT_IMAGE_NOT_FOUND}) from None
    data_dir = _data_dir()
    if is_private_image_candidate(output_path) or not _path_exists_safely(output_path) or not _path_is_file_safely(output_path):
        raise HTTPException(status_code=404, detail={"code": api_errors.OUTPUT_IMAGE_NOT_FOUND})
    if not _is_current_output_path(output_path, data_dir) and not _is_referenced_output_path(output_path, data_dir):
        raise HTTPException(status_code=403, detail={"code": api_errors.OUTPUT_IMAGE_NOT_REGISTERED})
    return FileResponse(output_path, headers={"Access-Control-Allow-Origin": "*"})
