from __future__ import annotations

import hashlib
import mimetypes
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.routing import APIRoute
from fastapi.responses import FileResponse
from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.generation import _data_dir, _generate, _output_dir_for, _record_xyz_history, _xyz_contact_sheet
from app.services.artifact_lifecycle import artifact_write_operation
from app.schemas.common import LoraSelection
from app.schemas.generation import GenerationResponse, Txt2ImgRequest, XYZAxis, XYZCellResult
from app.services.lora_training import (
    DatasetConfigPreview,
    DatasetMaterializeRequestData,
    DatasetPreviewRequestData,
    LoRACopyMachineMethod,
    LoRACopyMachineADDifTTimestepWindow,
    LoRACopyMachineADDifTSNRGamma,
    LoRACopyMachineAdapterReadiness,
    LoRACopyMachineComparisonPlan,
    LoRACopyMachineComparisonPrepareData,
    LoRACopyMachineSession,
    LoRACopyMachineSourceReference,
    LoRACopyMachineSessionCreateData,
    LoRACopyMachineSessionRepository,
    LoRACopyMachineTrainingRunPlan,
    LoRACopyMachineTrainingRunPrepareData,
    LoRACopyMachineTrainingRun,
    LoRACopyMachineTrainingRunRepository,
    LoRACopyMachineTargetPreparation,
    LoRACopyMachineTargetPrepareData,
    LoRACopyMachineTargetTransform,
    LoRATrainingCaptionAssist,
    LoRATrainingCaptionAssistLanguage,
    LoRATrainingCaptionAssistProviderMode,
    LoRATrainingCaptionAssistRequestData,
    LoRATrainingCaptionAssistStyle,
    LoRATrainingCaptionExportSettings,
    LoRATrainingCharacterCaptionPolicy,
    LoRATrainingCaptionOutputMode,
    LoRATrainingAnimaTaggerRonExportResult,
    LoRATrainingCaptionSourceContentKind,
    LoRATrainingCaptionSourceInput,
    LoRATrainingCaptionSourceToolReadiness,
    LoRATrainingCaptionSourceType,
    LoRATrainingDataset,
    LoRATrainingDatasetCaptionAssistResult,
    LoRATrainingDatasetCaptionCoverageResult,
    LoRATrainingCaptionSourceImportDirectoryResult,
    LoRATrainingCaptionSourceRefreshResult,
    LoRATrainingDatasetDirectoryImportResult,
    LoRATrainingDatasetImageInput,
    LoRATrainingDatasetItemAnalysis,
    LoRATrainingDatasetRepository,
    LoRATrainingPreset,
    LoRATrainingRun,
    LoRATrainingRunModelRegistration,
    LoRATrainingRunRepository,
    LoRATrainingSampleImageScore,
    LoRATrainingStorageError,
    MaterializedDataset,
    TrainingRunPrepareRequestData,
    TrainingProfile,
    TrainingType,
    analyze_lora_training_dataset_item,
    attach_lora_copy_machine_session_target,
    build_lora_copy_machine_adapter_readiness,
    build_lora_copy_machine_source_reference,
    build_lora_training_dataset_caption_assists,
    build_lora_training_dataset_caption_coverage,
    build_lora_training_caption_assist,
    build_lora_training_caption_source_tool_readiness,
    build_dataset_config_preview,
    cancel_lora_training_run,
    create_lora_copy_machine_session,
    create_lora_copy_machine_training_run,
    export_lora_training_anima_tagger_ron_sidecars,
    import_lora_training_dataset_directory,
    import_lora_training_caption_sources_from_directory,
    list_lora_training_presets,
    materialize_lora_training_dataset,
    prepare_lora_copy_machine_comparison,
    prepare_lora_copy_machine_target_image,
    prepare_lora_copy_machine_training_run,
    prepare_lora_training_run,
    register_lora_training_run_models,
    refresh_lora_copy_machine_training_run,
    refresh_lora_training_caption_sources,
    refresh_lora_training_run,
    register_lora_copy_machine_training_run_models,
    resolve_lora_training_dataset_item_image_path,
    score_lora_training_sample_image,
    score_lora_training_run_samples,
    start_lora_copy_machine_training_run,
    start_lora_training_run,
)
from app.services.xyz_grid import XYZGridAxis, XYZGridCell, build_xyz_contact_sheet
from app.core.config import AppConfig
from app.services.model_library import ModelLibraryScanner
from app.services.model_storage import resolve_model_root

class LoRATrainingRoute(APIRoute):
    def get_route_handler(self):
        original_handler = super().get_route_handler()

        async def custom_route_handler(request: Request):
            try:
                return await original_handler(request)
            except LoRATrainingStorageError:
                return JSONResponse(
                    status_code=409,
                    content={"detail": "lora_training_storage_unavailable"},
                )

        return custom_route_handler


router = APIRouter(prefix="/api/lora-training", tags=["lora-training"], route_class=LoRATrainingRoute)


class LoRATrainingPresetListResponse(BaseModel):
    items: list[LoRATrainingPreset]


class DatasetConfigPreviewRequest(BaseModel):
    preset_id: str
    image_dir: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    output_name: str = Field(min_length=1)
    pretrained_model_name_or_path: str = ""
    vae: str = ""
    qwen3: str = ""
    repeats: int | None = Field(default=None, ge=1, le=1000)
    max_train_epochs: int | None = Field(default=None, ge=1, le=1000)
    class_tokens: str = ""
    caption_extension: str | None = None
    optimizer_type: str = ""
    optimizer_args: str = ""
    optimizer_low_vram: bool = False


class LoRATrainingDatasetListResponse(BaseModel):
    items: list[LoRATrainingDataset]


class LoRATrainingDatasetCreateRequest(BaseModel):
    name: str = "Untitled LoRA dataset"
    preset_id: str


class LoRATrainingDatasetAddImagesRequest(BaseModel):
    images: list[LoRATrainingDatasetImageInput] = Field(default_factory=list)


class LoRATrainingDatasetImportDirectoryRequest(BaseModel):
    directory_path: str = Field(min_length=1)
    recursive: bool = False
    caption_extension: str = ".txt"
    default_tags: list[str] = Field(default_factory=list)


class LoRATrainingCaptionSourceImportDirectoryRequest(BaseModel):
    directory_path: str = Field(min_length=1)
    source_type: LoRATrainingCaptionSourceType = "wd14_tagger"
    source_extension: str = ".txt"
    content_kind: LoRATrainingCaptionSourceContentKind = "tags"
    label: str = ""
    confidence: float | None = Field(default=None, ge=0, le=1)
    recursive: bool = False
    replace_existing: bool = True


class LoRATrainingCaptionSourceRefreshRequest(LoRATrainingCaptionSourceImportDirectoryRequest):
    commands: list[str] = Field(default_factory=list)
    working_dir: str = ""
    timeout_seconds: float = Field(default=600.0, ge=1.0, le=3600.0)


class LoRATrainingAnimaTaggerRonExportRequest(BaseModel):
    directory_path: str = Field(min_length=1)
    overwrite: bool = True


class LoRATrainingDatasetCaptionUpdateRequest(BaseModel):
    caption: str = ""
    tags: list[str] = Field(default_factory=list)


class LoRATrainingCaptionExportSettingsRequest(BaseModel):
    suppressed_tags: list[str] = Field(default_factory=list)
    organizational_tags: list[str] = Field(default_factory=list)
    min_source_confidence: float = Field(default=0.0, ge=0, le=1)
    exclude_rating_tags: bool = True
    category_prefixing: bool = False
    output_mode: LoRATrainingCaptionOutputMode = "caption_txt"


class LoRATrainingCharacterCaptionPolicyRequest(BaseModel):
    trigger_token: str = Field(default="", max_length=200)
    excluded_identity_tags: list[str] = Field(default_factory=list, max_length=200)
    variable_detail_guidance: str = Field(default="", max_length=2000)


class LoRATrainingCaptionSourceUpsertRequest(BaseModel):
    id: str | None = None
    source_type: LoRATrainingCaptionSourceType
    label: str = ""
    caption: str = ""
    tags: list[str] = Field(default_factory=list)
    raw_text: str = ""
    confidence: float | None = Field(default=None, ge=0, le=1)


class LoRATrainingCaptionAssistRequest(BaseModel):
    language: LoRATrainingCaptionAssistLanguage = "en"
    caption_style: LoRATrainingCaptionAssistStyle = "tags"
    max_tags: int = Field(default=30, ge=1, le=80)
    extra_instruction: str = ""
    provider_mode: LoRATrainingCaptionAssistProviderMode = "prompt_only"
    provider_endpoint: str = ""
    provider_model: str = ""
    provider_api_key: str = ""
    provider_api_key_env_var: str = ""
    provider_timeout_seconds: float = Field(default=60.0, ge=1.0, le=300.0)
    include_image: bool = False


class LoRATrainingDatasetCaptionAssistRequest(LoRATrainingCaptionAssistRequest):
    only_missing_captions: bool = True
    limit: int = Field(default=50, ge=1, le=100)


class LoRATrainingDatasetMaterializeRequest(BaseModel):
    output_dir: str = "data/training/lora/outputs"
    output_name: str = "asteria_lora"
    pretrained_model_name_or_path: str = ""
    vae: str = ""
    qwen3: str = ""
    repeats: int | None = Field(default=None, ge=1, le=1000)
    max_train_epochs: int | None = Field(default=None, ge=1, le=1000)
    class_tokens: str = ""
    caption_extension: str | None = None
    optimizer_type: str = ""
    optimizer_args: str = ""
    optimizer_low_vram: bool = False


class LoRATrainingRunPrepareRequest(LoRATrainingDatasetMaterializeRequest):
    working_dir: str = ""
    accelerate_command: str = "accelerate"


class LoRATrainingRunListResponse(BaseModel):
    items: list[LoRATrainingRun]


class LoRATrainingRunModelRegistrationResponse(LoRATrainingRunModelRegistration):
    model_library_lora_count: int = 0


class LoRACopyMachineTrainingRunModelRegistrationResponse(LoRACopyMachineTrainingRun):
    model_library_lora_count: int = 0


class LoRACopyMachineSessionListResponse(BaseModel):
    items: list[LoRACopyMachineSession]


class LoRACopyMachineTrainingRunListResponse(BaseModel):
    items: list[LoRACopyMachineTrainingRun]


class LoRACopyMachineSessionCreateRequest(BaseModel):
    name: str = "Untitled copy-machine LoRA"
    profile: TrainingProfile = "sdxl"
    method: LoRACopyMachineMethod = "direct_difference"
    intent: str = "style_delta"
    source_image_path: str = Field(min_length=1)
    target_image_path: str = ""
    target_transform: str = "none"
    trigger_word: str = ""
    training_caption: str = ""


class LoRACopyMachineTargetPrepareRequest(BaseModel):
    source_image_path: str = Field(min_length=1)
    transform: LoRACopyMachineTargetTransform = "grayscale"
    output_name: str = "copy-machine-target"
    output_dir: str = ""


class LoRACopyMachineSessionTargetRequest(BaseModel):
    target_image_path: str = Field(min_length=1)
    target_transform: str = "none"


class LoRACopyMachineTrainingRunPrepareRequest(BaseModel):
    session_id: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    output_name: str = ""
    traintrain_path: str = ""
    python_command: str = "python"
    qwen3_path: str = ""
    vae_path: str = ""
    training_working_dir: str = ""
    accelerate_command: str = "accelerate"
    repeats: int | None = Field(default=None, ge=1, le=1000)
    max_train_epochs: int | None = Field(default=None, ge=1, le=1000)
    addift_timestep_window: LoRACopyMachineADDifTTimestepWindow = "action_500_1000"
    addift_snr_gamma: LoRACopyMachineADDifTSNRGamma = "preset_5_0"


class LoRACopyMachineComparisonPrepareRequest(BaseModel):
    run_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    seed: int = 1
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    positive_strength: float = Field(default=1.0, ge=-2.0, le=2.0)
    model_slot_overrides: dict[str, str] = Field(default_factory=dict)
    expected_model_slot_sha256: dict[str, str] = Field(default_factory=dict)
    expected_base_pixel_sha256: str = Field(
        default="",
        pattern=r"^(?:[0-9a-fA-F]{64})?$",
    )


class LoRACopyMachineComparisonRunResponse(BaseModel):
    run_id: str
    status: Literal["completed", "blocked"]
    comparison_id: int = 0
    profile: TrainingProfile = "sdxl"
    source_reference: LoRACopyMachineSourceReference | None = None
    cells: list[XYZCellResult] = Field(default_factory=list)
    contact_sheet_path: str = ""
    report_path: str = ""
    output_paths: list[str] = Field(default_factory=list)
    artifact_evidence: list[dict] = Field(default_factory=list)
    model_slot_evidence: list[dict] = Field(default_factory=list)
    image_metrics: list[dict] = Field(default_factory=list)
    image_quality_scores: list[LoRATrainingSampleImageScore] = Field(default_factory=list)
    validation: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _dataset_repository() -> LoRATrainingDatasetRepository:
    return LoRATrainingDatasetRepository(AppConfig().data_dir / "lora-training" / "datasets.json")


def _run_repository() -> LoRATrainingRunRepository:
    return LoRATrainingRunRepository(AppConfig().data_dir / "lora-training" / "runs.json")


def _copy_machine_session_repository() -> LoRACopyMachineSessionRepository:
    return LoRACopyMachineSessionRepository(AppConfig().data_dir / "lora-training" / "copy-machine-sessions.json")


def _copy_machine_run_repository() -> LoRACopyMachineTrainingRunRepository:
    return LoRACopyMachineTrainingRunRepository(AppConfig().data_dir / "lora-training" / "copy-machine-runs.json")


def _model_root() -> Path:
    config = AppConfig()
    return resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)


@router.get("/presets", response_model=LoRATrainingPresetListResponse)
def list_presets(
    profile: TrainingProfile | None = Query(default=None),
    training_type: TrainingType | None = Query(default=None),
) -> LoRATrainingPresetListResponse:
    return LoRATrainingPresetListResponse(
        items=list_lora_training_presets(profile=profile, training_type=training_type)
    )


@router.get("/caption-source-tools/readiness", response_model=LoRATrainingCaptionSourceToolReadiness)
def caption_source_tool_readiness() -> LoRATrainingCaptionSourceToolReadiness:
    return build_lora_training_caption_source_tool_readiness(project_root=AppConfig().root_dir)


@router.get("/copy-machine/adapters/readiness", response_model=LoRACopyMachineAdapterReadiness)
def copy_machine_adapter_readiness(traintrain_path: str = Query(default="")) -> LoRACopyMachineAdapterReadiness:
    return build_lora_copy_machine_adapter_readiness(
        project_root=AppConfig().root_dir,
        traintrain_path=traintrain_path,
    )


@router.get("/copy-machine/sessions", response_model=LoRACopyMachineSessionListResponse)
def list_copy_machine_sessions() -> LoRACopyMachineSessionListResponse:
    return LoRACopyMachineSessionListResponse(items=_copy_machine_session_repository().list())


@router.get("/copy-machine/sessions/{session_id}/image")
def get_copy_machine_session_image(
    session_id: str,
    role: Literal["source", "target"] = Query(),
) -> FileResponse:
    session = _copy_machine_session_repository().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="lora_copy_machine_session_not_found")
    evidence = session.source if role == "source" else session.target
    if evidence is None:
        raise HTTPException(status_code=404, detail="lora_copy_machine_session_image_not_found")
    path = Path(evidence.image_path)
    if not path.is_absolute():
        path = AppConfig().root_dir / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise HTTPException(status_code=404, detail="lora_copy_machine_session_image_not_found") from exc
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="lora_copy_machine_session_image_not_found")
    if hashlib.sha256(resolved.read_bytes()).hexdigest() != evidence.sha256:
        raise HTTPException(status_code=409, detail="lora_copy_machine_session_image_modified")
    media_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(resolved.suffix.lower(), mimetypes.guess_type(resolved.name)[0] or "application/octet-stream")
    return FileResponse(resolved, media_type=media_type)


@router.post("/copy-machine/sessions", response_model=LoRACopyMachineSession)
def create_copy_machine_session(request: LoRACopyMachineSessionCreateRequest) -> LoRACopyMachineSession:
    try:
        return create_lora_copy_machine_session(
            LoRACopyMachineSessionCreateData(
                name=request.name,
                profile=request.profile,
                method=request.method,
                intent=request.intent,
                source_image_path=request.source_image_path,
                target_image_path=request.target_image_path,
                target_transform=request.target_transform,
                trigger_word=request.trigger_word,
                training_caption=request.training_caption,
            ),
            _copy_machine_session_repository(),
            project_root=AppConfig().root_dir,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail in {
            "lora_copy_machine_image_not_found",
            "lora_copy_machine_image_invalid",
            "lora_copy_machine_target_image_required",
            "lora_copy_machine_image_size_mismatch",
        }:
            raise HTTPException(status_code=422, detail=detail) from exc
        raise


@router.post("/copy-machine/target-image", response_model=LoRACopyMachineTargetPreparation)
def prepare_copy_machine_target_image(request: LoRACopyMachineTargetPrepareRequest) -> LoRACopyMachineTargetPreparation:
    config = AppConfig()
    output_dir = request.output_dir or str(config.data_dir / "lora-training" / "copy-machine-targets")
    try:
        return prepare_lora_copy_machine_target_image(
            LoRACopyMachineTargetPrepareData(
                source_image_path=request.source_image_path,
                output_dir=output_dir,
                transform=request.transform,
                output_name=request.output_name,
            ),
            project_root=config.root_dir,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail in {
            "lora_copy_machine_image_not_found",
            "lora_copy_machine_image_invalid",
            "lora_copy_machine_image_size_mismatch",
            "lora_copy_machine_target_transform_unsupported",
        }:
            raise HTTPException(status_code=422, detail=detail) from exc
        raise


@router.post("/copy-machine/sessions/{session_id}/target", response_model=LoRACopyMachineSession)
def attach_copy_machine_session_target(
    session_id: str,
    request: LoRACopyMachineSessionTargetRequest,
) -> LoRACopyMachineSession:
    try:
        session = attach_lora_copy_machine_session_target(
            session_id,
            request.target_image_path,
            request.target_transform,
            _copy_machine_session_repository(),
            project_root=AppConfig().root_dir,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail in {
            "lora_copy_machine_image_not_found",
            "lora_copy_machine_image_invalid",
            "lora_copy_machine_image_size_mismatch",
        }:
            raise HTTPException(status_code=422, detail=detail) from exc
        raise
    if session is None:
        raise HTTPException(status_code=404, detail="lora_copy_machine_session_not_found")
    return session


@router.post("/copy-machine/runs/prepare", response_model=LoRACopyMachineTrainingRunPlan)
def prepare_copy_machine_training_run(
    request: LoRACopyMachineTrainingRunPrepareRequest,
) -> LoRACopyMachineTrainingRunPlan:
    plan = prepare_lora_copy_machine_training_run(
        LoRACopyMachineTrainingRunPrepareData(
            session_id=request.session_id,
            model_path=request.model_path,
            output_dir=request.output_dir,
            output_name=request.output_name,
            traintrain_path=request.traintrain_path,
            python_command=request.python_command,
            qwen3_path=request.qwen3_path,
            vae_path=request.vae_path,
            training_working_dir=request.training_working_dir,
            accelerate_command=request.accelerate_command,
            repeats=request.repeats,
            max_train_epochs=request.max_train_epochs,
            addift_timestep_window=request.addift_timestep_window,
            addift_snr_gamma=request.addift_snr_gamma,
        ),
        _copy_machine_session_repository(),
        project_root=AppConfig().root_dir,
    )
    if plan is None:
        raise HTTPException(status_code=404, detail="lora_copy_machine_session_not_found")
    return plan


@router.get("/copy-machine/runs", response_model=LoRACopyMachineTrainingRunListResponse)
def list_copy_machine_training_runs() -> LoRACopyMachineTrainingRunListResponse:
    return LoRACopyMachineTrainingRunListResponse(items=_copy_machine_run_repository().list())


@router.post("/copy-machine/runs", response_model=LoRACopyMachineTrainingRun)
def create_copy_machine_training_run(
    request: LoRACopyMachineTrainingRunPrepareRequest,
) -> LoRACopyMachineTrainingRun:
    run = create_lora_copy_machine_training_run(
        LoRACopyMachineTrainingRunPrepareData(
            session_id=request.session_id,
            model_path=request.model_path,
            output_dir=request.output_dir,
            output_name=request.output_name,
            traintrain_path=request.traintrain_path,
            python_command=request.python_command,
            qwen3_path=request.qwen3_path,
            vae_path=request.vae_path,
            training_working_dir=request.training_working_dir,
            accelerate_command=request.accelerate_command,
            repeats=request.repeats,
            max_train_epochs=request.max_train_epochs,
            addift_timestep_window=request.addift_timestep_window,
            addift_snr_gamma=request.addift_snr_gamma,
        ),
        _copy_machine_session_repository(),
        _copy_machine_run_repository(),
        training_run_repository=_run_repository(),
        data_dir=AppConfig().data_dir,
        project_root=AppConfig().root_dir,
    )
    if run is None:
        raise HTTPException(status_code=404, detail="lora_copy_machine_session_not_found")
    return run


@router.post("/copy-machine/runs/{run_id}/start", response_model=LoRACopyMachineTrainingRun)
def start_copy_machine_training_run(run_id: str) -> LoRACopyMachineTrainingRun:
    try:
        run = start_lora_copy_machine_training_run(
            run_id,
            _copy_machine_run_repository(),
            training_run_repository=_run_repository(),
            data_dir=AppConfig().data_dir,
            project_root=AppConfig().root_dir,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail in {
            "copy_machine_training_run_blocked",
            "copy_machine_training_run_not_startable",
            "copy_machine_training_run_command_empty",
        }:
            raise HTTPException(status_code=422, detail=detail) from exc
        raise
    if run is None:
        raise HTTPException(status_code=404, detail="copy_machine_training_run_not_found")
    return run


@router.post("/copy-machine/runs/{run_id}/refresh", response_model=LoRACopyMachineTrainingRun)
def refresh_copy_machine_training_run(run_id: str) -> LoRACopyMachineTrainingRun:
    run = refresh_lora_copy_machine_training_run(
        run_id,
        _copy_machine_run_repository(),
        training_run_repository=_run_repository(),
        project_root=AppConfig().root_dir,
    )
    if run is None:
        raise HTTPException(status_code=404, detail="copy_machine_training_run_not_found")
    return run


@router.post(
    "/copy-machine/runs/{run_id}/register-models",
    response_model=LoRACopyMachineTrainingRunModelRegistrationResponse,
)
def register_copy_machine_training_run_models(run_id: str) -> LoRACopyMachineTrainingRunModelRegistrationResponse:
    try:
        run = register_lora_copy_machine_training_run_models(
            run_id,
            _copy_machine_run_repository(),
            model_root=_model_root(),
        )
    except ValueError as exc:
        detail = str(exc)
        if detail in {
            "copy_machine_training_run_not_completed",
            "copy_machine_no_trained_models_to_register",
        }:
            raise HTTPException(status_code=422, detail=detail) from exc
        raise
    if run is None:
        raise HTTPException(status_code=404, detail="copy_machine_training_run_not_found")
    library = ModelLibraryScanner(_model_root()).scan()
    return LoRACopyMachineTrainingRunModelRegistrationResponse(
        **run.model_dump(),
        model_library_lora_count=len(library.by_type("lora")),
    )


@router.post("/copy-machine/comparisons/prepare", response_model=LoRACopyMachineComparisonPlan)
def prepare_copy_machine_comparison(
    request: LoRACopyMachineComparisonPrepareRequest,
) -> LoRACopyMachineComparisonPlan:
    plan = prepare_lora_copy_machine_comparison(
        LoRACopyMachineComparisonPrepareData(
            run_id=request.run_id,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            width=request.width,
            height=request.height,
            positive_strength=request.positive_strength,
        ),
        _copy_machine_run_repository(),
    )
    if plan is None:
        raise HTTPException(status_code=404, detail="copy_machine_training_run_not_found")
    return plan


def _generate_txt2img(request: Txt2ImgRequest, mode: str) -> GenerationResponse:
    return _generate(request, mode)


def _copy_machine_comparison_profile(run: LoRACopyMachineTrainingRun) -> TrainingProfile:
    session = _copy_machine_session_repository().get(run.session_id)
    return session.profile if session is not None else "sdxl"


def _copy_machine_txt2img_request(
    request: LoRACopyMachineComparisonPrepareRequest,
    planned_request,
    profile: TrainingProfile,
) -> Txt2ImgRequest:
    return Txt2ImgRequest(
        profile=profile,
        runtime="comfy_workflow",
        prompt=planned_request.prompt,
        negative_prompt=planned_request.negative_prompt,
        seed=planned_request.seed,
        width=planned_request.width,
        height=planned_request.height,
        hires_enabled=False,
        topaz_enabled=False,
        model_slot_overrides=request.model_slot_overrides,
        loras=[
            LoraSelection(name=lora.path, weight=lora.strength)
            for lora in planned_request.loras
        ],
    )


def _copy_machine_comparison_report_path(comparison_id: int) -> Path:
    return AppConfig().data_dir / "reports" / f"copy-machine-comparison-{comparison_id}.json"


def _copy_machine_blocked_comparison_report_path(generation_id: int) -> Path:
    return (
        AppConfig().data_dir
        / "reports"
        / f"copy-machine-comparison-blocked-{generation_id}.json"
    )


def _copy_machine_contact_sheet(
    source_reference: LoRACopyMachineSourceReference,
    cells: list[XYZCellResult],
    output_dir: Path,
) -> str:
    variants = ["source_reference", *[cell.labels.get("variant", "") for cell in cells]]
    generated_paths = [_first_comparison_image_path(cell.generation) for cell in cells]
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        first_generated = next((path for path in generated_paths if path is not None), None)
        if first_generated is None:
            return ""
        with Image.open(first_generated) as generated_image:
            tile_size = generated_image.size
        with TemporaryDirectory(prefix="asteria-copy-machine-") as temporary_dir:
            normalized_source_path = Path(temporary_dir) / "source-reference.png"
            with Image.open(source_reference.image_path) as source_image:
                ImageOps.pad(source_image.convert("RGB"), tile_size, color="white").save(normalized_source_path)
            grid_cells = [
                XYZGridCell(labels={"variant": "source_reference"}, image_path=normalized_source_path),
                *[
                    XYZGridCell(
                        labels={"variant": cell.labels.get("variant", "")},
                        image_path=image_path,
                    )
                    for cell, image_path in zip(cells, generated_paths, strict=True)
                ],
            ]
            sheet = build_xyz_contact_sheet(
                grid_cells,
                [XYZGridAxis(key="variant", label="variant", values=tuple(variants))],
            )
        path = output_dir / f"copy_machine_{uuid4().hex[:10]}.png"
        sheet.save(path)
    except OSError:
        return ""
    return str(path)


def _first_comparison_image_path(generation: GenerationResponse) -> Path | None:
    for output_path in generation.output_paths:
        path = Path(output_path)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


def _copy_machine_image_color_summary(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb_means = ImageStat.Stat(rgb).mean
        mean_luminance = ImageStat.Stat(rgb.convert("L")).mean[0] / 255.0
        mean_saturation = ImageStat.Stat(rgb.convert("HSV")).mean[1] / 255.0
    red, green, blue = (channel / 255.0 for channel in rgb_means)
    return {
        "mean_luminance": round(mean_luminance, 6),
        "mean_saturation": round(mean_saturation, 6),
        "red_green_balance": round(red - green, 6),
        "blue_green_balance": round(blue - green, 6),
    }


def _copy_machine_teacher_fidelity(
    session: LoRACopyMachineSession | None,
    base_path: Path,
    positive_path: Path,
) -> tuple[dict | None, list[str]]:
    if session is None or session.target is None:
        return None, []
    source_path = Path(session.source.image_path)
    target_path = Path(session.target.image_path)
    try:
        summaries = {
            "source": _copy_machine_image_color_summary(source_path),
            "target": _copy_machine_image_color_summary(target_path),
            "base": _copy_machine_image_color_summary(base_path),
            "positive": _copy_machine_image_color_summary(positive_path),
        }
    except (OSError, ValueError):
        return None, ["copy_machine_teacher_reference_invalid"]

    feature_names = tuple(summaries["source"])
    teacher_delta = {
        name: round(summaries["target"][name] - summaries["source"][name], 6)
        for name in feature_names
    }
    generated_delta = {
        name: round(summaries["positive"][name] - summaries["base"][name], 6)
        for name in feature_names
    }
    teacher_vector = [teacher_delta[name] for name in feature_names]
    generated_vector = [generated_delta[name] for name in feature_names]
    teacher_squared = sum(value * value for value in teacher_vector)
    generated_squared = sum(value * value for value in generated_vector)
    warnings: list[str] = []
    if teacher_squared < 0.0001:
        status = "not_available"
        direction_alignment = None
        transfer_progress = None
        warnings.append("copy_machine_teacher_delta_too_small")
    else:
        projection = sum(
            teacher_value * generated_value
            for teacher_value, generated_value in zip(teacher_vector, generated_vector, strict=True)
        )
        direction_alignment = (
            round(projection / math.sqrt(teacher_squared * generated_squared), 6)
            if generated_squared >= 0.000001
            else 0.0
        )
        transfer_progress = round(projection / teacher_squared, 6)
        if projection < 0:
            status = "opposite"
            warnings.append("copy_machine_teacher_direction_opposed")
        elif direction_alignment < 0.5 or transfer_progress < 0.1:
            status = "weak"
            warnings.append("copy_machine_teacher_transfer_weak")
        else:
            status = "aligned"

    return {
        "status": status,
        "method": "global_color_statistics",
        "source_image_path": str(source_path),
        "target_image_path": str(target_path),
        "summaries": summaries,
        "teacher_delta": teacher_delta,
        "generated_delta": generated_delta,
        "direction_alignment": direction_alignment,
        "transfer_progress": transfer_progress,
    }, warnings


def _copy_machine_image_delta_metrics(
    cells: list[XYZCellResult],
    session: LoRACopyMachineSession | None = None,
) -> tuple[list[dict], list[str]]:
    if len(cells) < 2:
        return [], ["copy_machine_comparison_pair_missing"]
    base_path = _first_comparison_image_path(cells[0].generation)
    positive_path = _first_comparison_image_path(cells[1].generation)
    if base_path is None or positive_path is None:
        return [], ["copy_machine_comparison_image_missing"]
    try:
        with Image.open(base_path) as base_image, Image.open(positive_path) as positive_image:
            base_rgb = base_image.convert("RGB")
            positive_rgb = positive_image.convert("RGB")
            if base_rgb.size != positive_rgb.size:
                return [], ["copy_machine_comparison_image_size_mismatch"]
            diff = ImageChops.difference(base_rgb, positive_rgb)
            stat = ImageStat.Stat(diff)
            diff_luma = diff.convert("L")
            diff_histogram = diff_luma.histogram()
            changed_pixels = sum(diff_histogram[1:])
            total_pixels = max(1, diff_luma.width * diff_luma.height)
            edge_diff = ImageChops.difference(
                base_rgb.convert("L").filter(ImageFilter.FIND_EDGES),
                positive_rgb.convert("L").filter(ImageFilter.FIND_EDGES),
            )
            edge_stat = ImageStat.Stat(edge_diff)
    except (OSError, ValueError):
        return [], ["copy_machine_comparison_image_invalid"]
    mean_absolute_delta = round(sum(stat.mean) / len(stat.mean), 4)
    max_channel_delta = int(max(stat.extrema[channel][1] for channel in range(len(stat.extrema))))
    normalized_delta = round(mean_absolute_delta / 255.0, 6)
    changed_pixel_ratio = round(changed_pixels / total_pixels, 6)
    edge_mean_absolute_delta = round(float(edge_stat.mean[0]), 4)
    warnings: list[str] = []
    if mean_absolute_delta < 1.0:
        warnings.append("copy_machine_comparison_delta_too_small")
    teacher_fidelity, teacher_warnings = _copy_machine_teacher_fidelity(session, base_path, positive_path)
    warnings.extend(teacher_warnings)
    return [
        {
            "kind": "base_vs_copy_machine",
            "base_image_path": str(base_path),
            "positive_image_path": str(positive_path),
            "width": base_rgb.width,
            "height": base_rgb.height,
            "mean_absolute_delta": mean_absolute_delta,
            "normalized_delta": normalized_delta,
            "max_channel_delta": max_channel_delta,
            "changed_pixel_ratio": changed_pixel_ratio,
            "edge_mean_absolute_delta": edge_mean_absolute_delta,
            "teacher_fidelity": teacher_fidelity,
        }
    ], warnings


def _copy_machine_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_machine_file_stat_evidence(path: Path | None) -> dict:
    if path is None:
        return {"path": "", "sha256": "", "size_bytes": 0}
    try:
        if not path.is_file():
            return {"path": str(path), "sha256": "", "size_bytes": 0}
        return {
            "path": str(path),
            "sha256": _copy_machine_file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    except OSError:
        return {"path": str(path), "sha256": "", "size_bytes": 0}


def _copy_machine_pixel_sha256(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        with Image.open(path) as image:
            return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
    except (OSError, ValueError):
        return ""


def _copy_machine_comparison_model_slot_evidence(
    request: LoRACopyMachineComparisonPrepareRequest,
    *,
    required_slots: set[str] | None = None,
) -> tuple[list[dict], list[str], dict[str, str]]:
    override_slots = set(request.model_slot_overrides)
    expected_slots = set(request.expected_model_slot_sha256)
    if required_slots is not None and (
        override_slots != required_slots or expected_slots != required_slots
    ):
        return [], ["copy_machine_comparison_required_model_slots_mismatch"], {}
    if not override_slots and not expected_slots:
        return [], [], {}
    if set(request.model_slot_overrides) != set(request.expected_model_slot_sha256):
        return [], ["copy_machine_comparison_model_slot_contract_mismatch"], {}
    evidence: list[dict] = []
    errors: list[str] = []
    normalized_overrides: dict[str, str] = {}
    root = AppConfig().root_dir
    for slot in sorted(request.model_slot_overrides):
        path = Path(request.model_slot_overrides[slot]).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        normalized_overrides[slot] = str(path)
        item = _copy_machine_file_stat_evidence(path)
        expected = request.expected_model_slot_sha256[slot].strip().lower()
        if not item["sha256"]:
            errors.append(f"copy_machine_comparison_model_slot_missing:{slot}")
        elif item["sha256"].lower() != expected:
            errors.append(f"copy_machine_comparison_model_slot_sha256_mismatch:{slot}")
        evidence.append({"slot": slot, **item, "expected_sha256": expected})
    return evidence, errors, normalized_overrides


def _copy_machine_artifact_evidence(cells: list[XYZCellResult]) -> list[dict]:
    evidence: list[dict] = []
    for cell in cells:
        image_path = _first_comparison_image_path(cell.generation)
        sidecar_path = image_path.with_suffix(".asteria.json") if image_path is not None else None
        image_evidence = _copy_machine_file_stat_evidence(image_path)
        sidecar_evidence = _copy_machine_file_stat_evidence(sidecar_path)
        evidence.append(
            {
                "index": cell.index,
                "variant": cell.labels.get("variant", ""),
                "generation_id": cell.generation.id,
                "status": cell.generation.status,
                "image_path": image_evidence["path"],
                "image_sha256": image_evidence["sha256"],
                "pixel_sha256": _copy_machine_pixel_sha256(image_path),
                "image_size_bytes": image_evidence["size_bytes"],
                "sidecar_path": sidecar_evidence["path"] if sidecar_evidence["sha256"] else "",
                "sidecar_sha256": sidecar_evidence["sha256"],
                "sidecar_size_bytes": sidecar_evidence["size_bytes"],
                "seed": cell.generation.settings.get("seed"),
                "width": cell.generation.settings.get("width"),
                "height": cell.generation.settings.get("height"),
                "loras": [lora.model_dump() for lora in cell.generation.loras],
            }
        )
    return evidence


def _copy_machine_image_quality_scores(cells: list[XYZCellResult]) -> list[LoRATrainingSampleImageScore]:
    scores: list[LoRATrainingSampleImageScore] = []
    for cell in cells:
        image_path = _first_comparison_image_path(cell.generation)
        if image_path is None:
            scores.append(
                LoRATrainingSampleImageScore(
                    image_path="",
                    score=0,
                    rating="poor",
                    warnings=["sample_image_unreadable"],
                )
            )
            continue
        scores.append(score_lora_training_sample_image(image_path))
    return scores


def _copy_machine_quality_status(scores: list[LoRATrainingSampleImageScore]) -> str:
    if not scores:
        return "not_available"
    if any(score.rating == "poor" for score in scores):
        return "poor"
    if any(score.rating == "review" or score.warnings for score in scores):
        return "needs_review"
    return "good"


def _copy_machine_comparison_validation(
    image_metrics: list[dict],
    warnings: list[str],
    image_quality_scores: list[LoRATrainingSampleImageScore],
) -> dict:
    if not image_metrics:
        visual_delta_status = "not_available"
    elif "copy_machine_comparison_delta_too_small" in warnings:
        visual_delta_status = "no_visible_delta"
    else:
        visual_delta_status = "difference_found"
    quality_scores = [score.score for score in image_quality_scores]
    teacher_fidelity = image_metrics[0].get("teacher_fidelity") if image_metrics else None
    return {
        "result": "needs_review",
        "visual_delta_status": visual_delta_status,
        "teacher_fidelity_status": teacher_fidelity.get("status") if teacher_fidelity else "not_available",
        "quality_status": _copy_machine_quality_status(image_quality_scores),
        "quality_score_min": min(quality_scores) if quality_scores else None,
        "quality_score_average": round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else None,
        "quality_review_required": True,
    }


def _write_copy_machine_comparison_report(
    *,
    comparison_id: int,
    comparison_request: LoRACopyMachineComparisonPrepareRequest,
    run: LoRACopyMachineTrainingRun,
    session: LoRACopyMachineSession | None,
    source_reference: LoRACopyMachineSourceReference | None,
    profile: TrainingProfile,
    cells: list[XYZCellResult],
    contact_sheet_path: str,
    artifact_evidence: list[dict],
    model_slot_evidence: list[dict],
    image_metrics: list[dict],
    image_quality_scores: list[LoRATrainingSampleImageScore],
    validation: dict,
    warnings: list[str],
) -> str:
    report_path = _copy_machine_comparison_report_path(comparison_id)
    payload = {
        "schema": "asteria.copy_machine_lora_comparison.v1",
        "comparison_id": comparison_id,
        "run_id": run.id,
        "comparison_request": comparison_request.model_dump(mode="json"),
        "profile": profile,
        "session": session.model_dump() if session is not None else None,
        "source_reference": source_reference.model_dump() if source_reference is not None else None,
        "training_run": {
            "id": run.id,
            "session_id": run.session_id,
            "adapter_id": run.adapter_id,
            "method": run.method,
            "status": run.status,
            "output_dir": run.output_dir,
            "output_name": run.output_name,
            "trained_model_paths": run.trained_model_paths,
            "registered_model_paths": run.registered_model_paths,
            "effective_config_path": run.effective_config_path,
            "effective_config_evidence": (
                run.effective_config_evidence.model_dump()
                if run.effective_config_evidence is not None
                else None
            ),
            "started_at": run.started_at,
            "finished_at": run.finished_at,
        },
        "contact_sheet_path": contact_sheet_path,
        "cells": [
            {
                "index": cell.index,
                "labels": cell.labels,
                "generation_id": cell.generation.id,
                "output_paths": cell.generation.output_paths,
                "loras": [lora.model_dump() for lora in cell.generation.loras],
            }
            for cell in cells
        ],
        "artifact_evidence": artifact_evidence,
        "model_slot_evidence": model_slot_evidence,
        "image_metrics": image_metrics,
        "image_quality_scores": [score.model_dump() for score in image_quality_scores],
        "validation": validation,
        "warnings": warnings,
    }
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        return ""
    return str(report_path)


def _write_blocked_copy_machine_comparison_report(
    *,
    comparison_request: LoRACopyMachineComparisonPrepareRequest,
    run: LoRACopyMachineTrainingRun,
    session: LoRACopyMachineSession | None,
    source_reference: LoRACopyMachineSourceReference | None,
    profile: TrainingProfile,
    cells: list[XYZCellResult],
    artifact_evidence: list[dict],
    model_slot_evidence: list[dict],
    validation: dict,
    warnings: list[str],
    errors: list[str],
) -> str:
    if not cells:
        return ""
    report_path = _copy_machine_blocked_comparison_report_path(
        cells[0].generation.id
    )
    payload = {
        "schema": "asteria.copy_machine_lora_comparison.v1",
        "status": "blocked",
        "run_id": run.id,
        "comparison_request": comparison_request.model_dump(mode="json"),
        "profile": profile,
        "session": session.model_dump() if session is not None else None,
        "source_reference": (
            source_reference.model_dump() if source_reference is not None else None
        ),
        "training_run": run.model_dump(),
        "cells": [
            {
                "index": cell.index,
                "labels": cell.labels,
                "generation_id": cell.generation.id,
                "output_paths": cell.generation.output_paths,
                "loras": [lora.model_dump() for lora in cell.generation.loras],
            }
            for cell in cells
        ],
        "artifact_evidence": artifact_evidence,
        "model_slot_evidence": model_slot_evidence,
        "validation": validation,
        "warnings": warnings,
        "errors": errors,
    }
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        return ""
    return str(report_path)


@router.post("/copy-machine/comparisons/run", response_model=LoRACopyMachineComparisonRunResponse)
@artifact_write_operation
def run_copy_machine_comparison(
    request: LoRACopyMachineComparisonPrepareRequest,
) -> LoRACopyMachineComparisonRunResponse:
    run_repository = _copy_machine_run_repository()
    run = run_repository.get(request.run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="copy_machine_training_run_not_found")
    plan = prepare_lora_copy_machine_comparison(
        LoRACopyMachineComparisonPrepareData(
            run_id=request.run_id,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            width=request.width,
            height=request.height,
            positive_strength=request.positive_strength,
        ),
        run_repository,
    )
    if plan is None:
        raise HTTPException(status_code=404, detail="copy_machine_training_run_not_found")
    session = _copy_machine_session_repository().get(run.session_id)
    profile = session.profile if session is not None else "sdxl"
    if plan.status == "blocked":
        return LoRACopyMachineComparisonRunResponse(
            run_id=request.run_id,
            status="blocked",
            profile=profile,
            warnings=plan.warnings,
            errors=plan.errors,
        )
    required_model_slots = (
        {"checkpoint", "vae"}
        if "experiment_override:addift_timestep_window=style_200_400" in run.warnings
        else None
    )
    model_slot_evidence, model_slot_errors, normalized_model_slots = (
        _copy_machine_comparison_model_slot_evidence(
            request,
            required_slots=required_model_slots,
        )
    )
    if model_slot_errors:
        return LoRACopyMachineComparisonRunResponse(
            run_id=request.run_id,
            status="blocked",
            profile=profile,
            model_slot_evidence=model_slot_evidence,
            warnings=plan.warnings,
            errors=model_slot_errors,
        )
    effective_request = request.model_copy(
        update={"model_slot_overrides": normalized_model_slots}
    )
    source_reference: LoRACopyMachineSourceReference | None = None
    if session is not None:
        try:
            source_reference = build_lora_copy_machine_source_reference(session)
        except ValueError as exc:
            return LoRACopyMachineComparisonRunResponse(
                run_id=request.run_id,
                status="blocked",
                profile=profile,
                warnings=plan.warnings,
                errors=[str(exc)],
            )
    cells: list[XYZCellResult] = []
    display_labels = {"base": "base", "copy_machine_positive": "copy-machine"}
    for index, planned_request in enumerate(plan.generation_requests):
        generation = _generate_txt2img(
            _copy_machine_txt2img_request(effective_request, planned_request, profile),
            "txt2img",
        )
        cells.append(
            XYZCellResult(
                index=index,
                labels={"variant": display_labels.get(planned_request.label, planned_request.label)},
                generation=generation,
            )
        )
        if index == 0 and effective_request.expected_base_pixel_sha256:
            base_artifact_evidence = _copy_machine_artifact_evidence(cells)
            expected_base_pixel_sha256 = effective_request.expected_base_pixel_sha256.lower()
            base_artifact_evidence[0]["expected_pixel_sha256"] = expected_base_pixel_sha256
            if base_artifact_evidence[0]["pixel_sha256"] != expected_base_pixel_sha256:
                validation = {
                    "result": "blocked",
                    "reproducibility_status": "base_pixel_sha256_mismatch",
                    "recovery_action": "run_discarded_warmup_comparison_with_different_seed",
                }
                errors = ["copy_machine_comparison_base_pixel_sha256_mismatch"]
                report_path = _write_blocked_copy_machine_comparison_report(
                    comparison_request=effective_request,
                    run=run,
                    session=session,
                    source_reference=source_reference,
                    profile=profile,
                    cells=cells,
                    artifact_evidence=base_artifact_evidence,
                    model_slot_evidence=model_slot_evidence,
                    validation=validation,
                    warnings=plan.warnings,
                    errors=errors,
                )
                return LoRACopyMachineComparisonRunResponse(
                    run_id=request.run_id,
                    status="blocked",
                    profile=profile,
                    source_reference=source_reference,
                    cells=cells,
                    report_path=report_path,
                    output_paths=[
                        *generation.output_paths,
                        *([report_path] if report_path else []),
                    ],
                    artifact_evidence=base_artifact_evidence,
                    model_slot_evidence=model_slot_evidence,
                    validation=validation,
                    warnings=plan.warnings,
                    errors=errors,
                )
    axes = [XYZAxis(target="variant", label="variant", values=[cell.labels["variant"] for cell in cells])]
    if cells and source_reference is not None:
        contact_sheet_path = _copy_machine_contact_sheet(
            source_reference,
            cells,
            _output_dir_for(_data_dir(), profile),
        )
    else:
        contact_sheet_path = _xyz_contact_sheet(cells, _output_dir_for(_data_dir(), profile), axes) if cells else ""
    record_request = type("_CopyMachineComparisonRecord", (), {
        "base": _copy_machine_txt2img_request(effective_request, plan.generation_requests[0], profile),
        "axes": axes,
    })()
    comparison_id, output_paths = _record_xyz_history(record_request, cells, contact_sheet_path)
    image_metrics, metric_warnings = _copy_machine_image_delta_metrics(cells, session)
    warnings = [*plan.warnings, *metric_warnings]
    artifact_evidence = _copy_machine_artifact_evidence(cells)
    if effective_request.expected_base_pixel_sha256 and artifact_evidence:
        artifact_evidence[0]["expected_pixel_sha256"] = (
            effective_request.expected_base_pixel_sha256.lower()
        )
    image_quality_scores = _copy_machine_image_quality_scores(cells)
    validation = _copy_machine_comparison_validation(image_metrics, warnings, image_quality_scores)
    report_path = _write_copy_machine_comparison_report(
        comparison_id=comparison_id,
        comparison_request=effective_request,
        run=run,
        session=session,
        source_reference=source_reference,
        profile=profile,
        cells=cells,
        contact_sheet_path=contact_sheet_path,
        artifact_evidence=artifact_evidence,
        model_slot_evidence=model_slot_evidence,
        image_metrics=image_metrics,
        image_quality_scores=image_quality_scores,
        validation=validation,
        warnings=warnings,
    )
    if report_path:
        output_paths = [*output_paths, report_path]
    return LoRACopyMachineComparisonRunResponse(
        run_id=request.run_id,
        status="completed",
        comparison_id=comparison_id,
        profile=profile,
        source_reference=source_reference,
        cells=cells,
        contact_sheet_path=contact_sheet_path,
        report_path=report_path,
        output_paths=output_paths,
        artifact_evidence=artifact_evidence,
        model_slot_evidence=model_slot_evidence,
        image_metrics=image_metrics,
        image_quality_scores=image_quality_scores,
        validation=validation,
        warnings=warnings,
        errors=plan.errors,
    )


@router.post("/dataset-config/preview", response_model=DatasetConfigPreview)
def preview_dataset_config(request: DatasetConfigPreviewRequest) -> DatasetConfigPreview:
    try:
        return build_dataset_config_preview(
            DatasetPreviewRequestData(
                preset_id=request.preset_id,
                image_dir=request.image_dir,
                output_dir=request.output_dir,
                output_name=request.output_name,
                pretrained_model_name_or_path=request.pretrained_model_name_or_path,
                vae=request.vae,
                qwen3=request.qwen3,
                repeats=request.repeats,
                max_train_epochs=request.max_train_epochs,
                class_tokens=request.class_tokens,
                caption_extension=request.caption_extension,
                optimizer_type=request.optimizer_type,
                optimizer_args=request.optimizer_args,
                optimizer_low_vram=request.optimizer_low_vram,
            )
        )
    except ValueError as exc:
        if str(exc) == "lora_training_preset_not_found":
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise


@router.get("/datasets", response_model=LoRATrainingDatasetListResponse)
def list_datasets() -> LoRATrainingDatasetListResponse:
    return LoRATrainingDatasetListResponse(items=_dataset_repository().list())


@router.get("/runs", response_model=LoRATrainingRunListResponse)
def list_runs() -> LoRATrainingRunListResponse:
    return LoRATrainingRunListResponse(items=_run_repository().list())


@router.delete("/runs/{run_id}")
def delete_run(run_id: str) -> dict[str, str | bool]:
    deleted = _run_repository().delete(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="lora_training_run_not_found")
    return {"deleted": True, "id": run_id}


@router.post("/runs/{run_id}/start", response_model=LoRATrainingRun)
def start_run(run_id: str) -> LoRATrainingRun:
    config = AppConfig()
    try:
        run = start_lora_training_run(
            run_id,
            _run_repository(),
            data_dir=config.data_dir,
            project_root=config.root_dir,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail in {
            "lora_training_run_preflight_not_ready",
            "lora_training_run_not_startable",
            "lora_training_run_command_empty",
            "lora_training_working_dir_not_found",
            "lora_training_command_start_failed",
        }:
            raise HTTPException(status_code=422, detail=detail) from exc
        raise
    if run is None:
        raise HTTPException(status_code=404, detail="lora_training_run_not_found")
    return run


@router.post("/runs/{run_id}/refresh", response_model=LoRATrainingRun)
def refresh_run(run_id: str) -> LoRATrainingRun:
    run = refresh_lora_training_run(run_id, _run_repository(), project_root=AppConfig().root_dir)
    if run is None:
        raise HTTPException(status_code=404, detail="lora_training_run_not_found")
    return run


@router.post("/runs/{run_id}/score-samples", response_model=LoRATrainingRun)
def score_run_samples(run_id: str) -> LoRATrainingRun:
    try:
        run = score_lora_training_run_samples(run_id, _run_repository(), project_root=AppConfig().root_dir)
    except ValueError as exc:
        if str(exc) == "lora_training_run_not_completed":
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raise
    if run is None:
        raise HTTPException(status_code=404, detail="lora_training_run_not_found")
    return run


@router.get("/runs/{run_id}/sample-images/{sample_index}")
def run_sample_image(run_id: str, sample_index: int) -> FileResponse:
    run = _run_repository().get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="lora_training_run_not_found")
    if sample_index < 0 or sample_index >= len(run.sample_image_paths):
        raise HTTPException(status_code=404, detail="lora_training_sample_image_not_found")
    try:
        path = Path(run.sample_image_paths[sample_index]).expanduser().resolve()
        is_file = path.is_file()
    except (OSError, RuntimeError):
        is_file = False
    if not is_file:
        raise HTTPException(status_code=404, detail="lora_training_sample_image_not_found")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, headers={"Access-Control-Allow-Origin": "*"})


@router.post("/runs/{run_id}/cancel", response_model=LoRATrainingRun)
def cancel_run(run_id: str) -> LoRATrainingRun:
    run = cancel_lora_training_run(run_id, _run_repository())
    if run is None:
        raise HTTPException(status_code=404, detail="lora_training_run_not_found")
    return run


@router.post("/runs/{run_id}/register-models", response_model=LoRATrainingRunModelRegistrationResponse)
def register_run_models(run_id: str) -> LoRATrainingRunModelRegistrationResponse:
    config = AppConfig()
    model_root = _model_root()
    try:
        result = register_lora_training_run_models(
            run_id,
            _run_repository(),
            model_root=model_root,
            project_root=config.root_dir,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail in {
            "lora_training_run_not_completed",
            "lora_training_no_trained_models_to_register",
            "lora_training_register_target_invalid",
        }:
            raise HTTPException(status_code=422, detail=detail) from exc
        raise
    if result is None:
        raise HTTPException(status_code=404, detail="lora_training_run_not_found")
    library = ModelLibraryScanner(model_root).scan()
    return LoRATrainingRunModelRegistrationResponse(
        **result.model_dump(),
        model_library_lora_count=len(library.by_type("lora")),
    )


@router.post("/datasets", response_model=LoRATrainingDataset)
def create_dataset(request: LoRATrainingDatasetCreateRequest) -> LoRATrainingDataset:
    try:
        return _dataset_repository().create(name=request.name, preset_id=request.preset_id)
    except ValueError as exc:
        if str(exc) == "lora_training_preset_not_found":
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise


@router.get("/datasets/{dataset_id}", response_model=LoRATrainingDataset)
def get_dataset(dataset_id: str) -> LoRATrainingDataset:
    dataset = _dataset_repository().get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_not_found")
    return dataset


@router.post("/datasets/{dataset_id}/images", response_model=LoRATrainingDataset)
def add_dataset_images(dataset_id: str, request: LoRATrainingDatasetAddImagesRequest) -> LoRATrainingDataset:
    if not request.images:
        raise HTTPException(status_code=422, detail="images_required")
    dataset = _dataset_repository().add_images(dataset_id, request.images)
    if dataset is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_not_found")
    return dataset


@router.post("/datasets/{dataset_id}/images/import-directory", response_model=LoRATrainingDatasetDirectoryImportResult)
def import_dataset_images_from_directory(
    dataset_id: str,
    request: LoRATrainingDatasetImportDirectoryRequest,
) -> LoRATrainingDatasetDirectoryImportResult:
    try:
        result = import_lora_training_dataset_directory(
            dataset_id,
            _dataset_repository(),
            directory_path=request.directory_path,
            project_root=AppConfig().root_dir,
            recursive=request.recursive,
            caption_extension=request.caption_extension,
            default_tags=request.default_tags,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail in {"lora_training_import_directory_not_found", "lora_training_caption_extension_invalid"}:
            raise HTTPException(status_code=422, detail=detail) from exc
        raise
    if result is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_not_found")
    return result


@router.post(
    "/datasets/{dataset_id}/caption-sources/import-directory",
    response_model=LoRATrainingCaptionSourceImportDirectoryResult,
)
def import_dataset_caption_sources_from_directory(
    dataset_id: str,
    request: LoRATrainingCaptionSourceImportDirectoryRequest,
) -> LoRATrainingCaptionSourceImportDirectoryResult:
    try:
        result = import_lora_training_caption_sources_from_directory(
            dataset_id,
            _dataset_repository(),
            directory_path=request.directory_path,
            project_root=AppConfig().root_dir,
            source_type=request.source_type,
            source_extension=request.source_extension,
            content_kind=request.content_kind,
            label=request.label,
            confidence=request.confidence,
            recursive=request.recursive,
            replace_existing=request.replace_existing,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail in {
            "lora_training_caption_source_import_directory_not_found",
            "lora_training_caption_source_extension_invalid",
            "lora_training_caption_source_empty",
        }:
            raise HTTPException(status_code=422, detail=detail) from exc
        raise
    if result is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_not_found")
    return result


@router.post(
    "/datasets/{dataset_id}/caption-sources/refresh",
    response_model=LoRATrainingCaptionSourceRefreshResult,
)
def refresh_dataset_caption_sources(
    dataset_id: str,
    request: LoRATrainingCaptionSourceRefreshRequest,
) -> LoRATrainingCaptionSourceRefreshResult:
    try:
        result = refresh_lora_training_caption_sources(
            dataset_id,
            _dataset_repository(),
            directory_path=request.directory_path,
            project_root=AppConfig().root_dir,
            commands=request.commands,
            source_type=request.source_type,
            source_extension=request.source_extension,
            content_kind=request.content_kind,
            label=request.label,
            confidence=request.confidence,
            recursive=request.recursive,
            replace_existing=request.replace_existing,
            working_dir=request.working_dir,
            timeout_seconds=request.timeout_seconds,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail in {
            "lora_training_caption_source_import_directory_not_found",
            "lora_training_caption_source_extension_invalid",
            "lora_training_caption_source_empty",
            "lora_training_caption_source_refresh_command_required",
            "lora_training_caption_source_refresh_command_failed",
            "lora_training_caption_source_refresh_command_timeout",
            "lora_training_caption_source_refresh_placeholder_invalid",
            "lora_training_caption_source_refresh_working_dir_not_found",
            "lora_training_caption_source_working_dir_not_allowed",
        }:
            raise HTTPException(status_code=422, detail=detail) from exc
        raise
    if result is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_not_found")
    return result


@router.post(
    "/datasets/{dataset_id}/caption-sources/export-anima-tagger-ron",
    response_model=LoRATrainingAnimaTaggerRonExportResult,
)
def export_dataset_caption_sources_to_anima_tagger_ron(
    dataset_id: str,
    request: LoRATrainingAnimaTaggerRonExportRequest,
) -> LoRATrainingAnimaTaggerRonExportResult:
    try:
        result = export_lora_training_anima_tagger_ron_sidecars(
            dataset_id,
            _dataset_repository(),
            directory_path=request.directory_path,
            project_root=AppConfig().root_dir,
            overwrite=request.overwrite,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail == "lora_training_caption_source_import_directory_not_found":
            raise HTTPException(status_code=422, detail=detail) from exc
        raise
    if result is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_not_found")
    return result


@router.put("/datasets/{dataset_id}/caption-export-settings", response_model=LoRATrainingDataset)
def update_dataset_caption_export_settings(
    dataset_id: str,
    request: LoRATrainingCaptionExportSettingsRequest,
) -> LoRATrainingDataset:
    dataset = _dataset_repository().update_caption_export_settings(
        dataset_id,
        LoRATrainingCaptionExportSettings(
            suppressed_tags=request.suppressed_tags,
            organizational_tags=request.organizational_tags,
            min_source_confidence=request.min_source_confidence,
            exclude_rating_tags=request.exclude_rating_tags,
            category_prefixing=request.category_prefixing,
            output_mode=request.output_mode,
        ),
    )
    if dataset is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_not_found")
    return dataset


@router.put("/datasets/{dataset_id}/character-caption-policy", response_model=LoRATrainingDataset)
def update_dataset_character_caption_policy(
    dataset_id: str,
    request: LoRATrainingCharacterCaptionPolicyRequest,
) -> LoRATrainingDataset:
    try:
        dataset = _dataset_repository().update_character_caption_policy(
            dataset_id,
            LoRATrainingCharacterCaptionPolicy(**request.model_dump()),
        )
    except ValueError as exc:
        detail = str(exc)
        if detail == "lora_training_character_caption_policy_not_applicable":
            raise HTTPException(status_code=422, detail=detail) from exc
        raise
    if dataset is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_not_found")
    return dataset


@router.get("/datasets/{dataset_id}/items/{item_id}/image")
def dataset_item_image(dataset_id: str, item_id: str) -> FileResponse:
    dataset = _dataset_repository().get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_not_found")
    try:
        path = resolve_lora_training_dataset_item_image_path(
            dataset,
            item_id,
            project_root=AppConfig().root_dir,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail in {"lora_training_dataset_item_not_found", "lora_training_dataset_image_not_found"}:
            raise HTTPException(status_code=404, detail=detail) from exc
        raise
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, headers={"Access-Control-Allow-Origin": "*"})


@router.put("/datasets/{dataset_id}/items/{item_id}", response_model=LoRATrainingDataset)
def update_dataset_item_caption(
    dataset_id: str,
    item_id: str,
    request: LoRATrainingDatasetCaptionUpdateRequest,
) -> LoRATrainingDataset:
    dataset = _dataset_repository().update_caption(
        dataset_id,
        item_id,
        caption=request.caption,
        tags=request.tags,
    )
    if dataset is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_or_item_not_found")
    return dataset


@router.put("/datasets/{dataset_id}/items/{item_id}/caption-sources", response_model=LoRATrainingDataset)
def upsert_dataset_item_caption_source(
    dataset_id: str,
    item_id: str,
    request: LoRATrainingCaptionSourceUpsertRequest,
) -> LoRATrainingDataset:
    try:
        dataset = _dataset_repository().upsert_caption_source(
            dataset_id,
            item_id,
            LoRATrainingCaptionSourceInput(
                id=request.id,
                source_type=request.source_type,
                label=request.label,
                caption=request.caption,
                tags=request.tags,
                raw_text=request.raw_text,
                confidence=request.confidence,
            ),
        )
    except ValueError as exc:
        if str(exc) == "lora_training_caption_source_empty":
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raise
    if dataset is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_or_item_not_found")
    return dataset


@router.delete("/datasets/{dataset_id}/items/{item_id}/caption-sources/{source_id}", response_model=LoRATrainingDataset)
def delete_dataset_item_caption_source(
    dataset_id: str,
    item_id: str,
    source_id: str,
) -> LoRATrainingDataset:
    dataset = _dataset_repository().delete_caption_source(dataset_id, item_id, source_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_item_or_source_not_found")
    return dataset


@router.delete("/datasets/{dataset_id}/items/{item_id}", response_model=LoRATrainingDataset)
def delete_dataset_item(dataset_id: str, item_id: str) -> LoRATrainingDataset:
    dataset = _dataset_repository().delete_item(dataset_id, item_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_or_item_not_found")
    return dataset


@router.post("/datasets/{dataset_id}/items/{item_id}/analyze", response_model=LoRATrainingDatasetItemAnalysis)
def analyze_dataset_item(dataset_id: str, item_id: str) -> LoRATrainingDatasetItemAnalysis:
    config = AppConfig()
    dataset = _dataset_repository().get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_not_found")
    try:
        analysis = analyze_lora_training_dataset_item(dataset, item_id, project_root=config.root_dir)
    except ValueError as exc:
        detail = str(exc)
        if detail in {
            "lora_training_dataset_image_not_found",
            "lora_training_dataset_image_invalid",
            "lora_caption_assist_image_invalid",
        }:
            raise HTTPException(status_code=422, detail=detail) from exc
        raise
    if analysis is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_or_item_not_found")
    return analysis


@router.post("/datasets/{dataset_id}/items/{item_id}/caption-assist", response_model=LoRATrainingCaptionAssist)
def build_caption_assist(
    dataset_id: str,
    item_id: str,
    request: LoRATrainingCaptionAssistRequest,
) -> LoRATrainingCaptionAssist:
    config = AppConfig()
    dataset = _dataset_repository().get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_not_found")
    try:
        assist = build_lora_training_caption_assist(
            dataset,
            item_id,
            LoRATrainingCaptionAssistRequestData(
                language=request.language,
                caption_style=request.caption_style,
                max_tags=request.max_tags,
                extra_instruction=request.extra_instruction,
                provider_mode=request.provider_mode,
                provider_endpoint=request.provider_endpoint,
                provider_model=request.provider_model,
                provider_api_key=request.provider_api_key,
                provider_api_key_env_var=request.provider_api_key_env_var,
                provider_timeout_seconds=request.provider_timeout_seconds,
                include_image=request.include_image,
            ),
            project_root=config.root_dir,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail in {
            "lora_training_dataset_image_not_found",
            "lora_training_dataset_image_invalid",
            "lora_caption_assist_image_invalid",
        }:
            raise HTTPException(status_code=422, detail=detail) from exc
        if detail.startswith("lora_caption_assist_provider_"):
            raise HTTPException(status_code=422, detail=detail) from exc
        raise
    if assist is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_or_item_not_found")
    return assist


@router.post("/datasets/{dataset_id}/caption-assist", response_model=LoRATrainingDatasetCaptionAssistResult)
def build_dataset_caption_assists(
    dataset_id: str,
    request: LoRATrainingDatasetCaptionAssistRequest,
) -> LoRATrainingDatasetCaptionAssistResult:
    config = AppConfig()
    dataset = _dataset_repository().get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_not_found")
    try:
        return build_lora_training_dataset_caption_assists(
            dataset,
            LoRATrainingCaptionAssistRequestData(
                language=request.language,
                caption_style=request.caption_style,
                max_tags=request.max_tags,
                extra_instruction=request.extra_instruction,
                provider_mode=request.provider_mode,
                provider_endpoint=request.provider_endpoint,
                provider_model=request.provider_model,
                provider_api_key=request.provider_api_key,
                provider_api_key_env_var=request.provider_api_key_env_var,
                provider_timeout_seconds=request.provider_timeout_seconds,
                include_image=request.include_image,
            ),
            project_root=config.root_dir,
            only_missing_captions=request.only_missing_captions,
            limit=request.limit,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail.startswith("lora_caption_assist_provider_"):
            raise HTTPException(status_code=422, detail=detail) from exc
        raise


@router.get("/datasets/{dataset_id}/caption-coverage", response_model=LoRATrainingDatasetCaptionCoverageResult)
def get_dataset_caption_coverage(dataset_id: str) -> LoRATrainingDatasetCaptionCoverageResult:
    dataset = _dataset_repository().get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_not_found")
    return build_lora_training_dataset_caption_coverage(dataset)


@router.post("/datasets/{dataset_id}/materialize", response_model=MaterializedDataset)
def materialize_dataset(
    dataset_id: str,
    request: LoRATrainingDatasetMaterializeRequest,
) -> MaterializedDataset:
    config = AppConfig()
    dataset = _dataset_repository().get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_not_found")
    try:
        return materialize_lora_training_dataset(
            dataset,
            DatasetMaterializeRequestData(
                data_dir=config.data_dir,
                project_root=config.root_dir,
                output_dir=request.output_dir,
                output_name=request.output_name,
                pretrained_model_name_or_path=request.pretrained_model_name_or_path,
                vae=request.vae,
                qwen3=request.qwen3,
                repeats=request.repeats,
                max_train_epochs=request.max_train_epochs,
                class_tokens=request.class_tokens,
                caption_extension=request.caption_extension,
                optimizer_type=request.optimizer_type,
                optimizer_args=request.optimizer_args,
                optimizer_low_vram=request.optimizer_low_vram,
            ),
        )
    except ValueError as exc:
        detail = str(exc)
        if detail == "lora_training_preset_not_found":
            raise HTTPException(status_code=404, detail=detail) from exc
        if detail in {
            "lora_training_dataset_empty",
            "lora_training_dataset_has_no_materializable_images",
            "lora_training_caption_extension_invalid",
        }:
            raise HTTPException(status_code=422, detail=detail) from exc
        raise


@router.post("/datasets/{dataset_id}/runs/prepare", response_model=LoRATrainingRun)
def prepare_training_run(
    dataset_id: str,
    request: LoRATrainingRunPrepareRequest,
) -> LoRATrainingRun:
    config = AppConfig()
    dataset = _dataset_repository().get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="lora_training_dataset_not_found")
    try:
        return prepare_lora_training_run(
            dataset,
            TrainingRunPrepareRequestData(
                data_dir=config.data_dir,
                project_root=config.root_dir,
                output_dir=request.output_dir,
                output_name=request.output_name,
                pretrained_model_name_or_path=request.pretrained_model_name_or_path,
                vae=request.vae,
                qwen3=request.qwen3,
                repeats=request.repeats,
                max_train_epochs=request.max_train_epochs,
                class_tokens=request.class_tokens,
                caption_extension=request.caption_extension,
                working_dir=request.working_dir,
                accelerate_command=request.accelerate_command,
                optimizer_type=request.optimizer_type,
                optimizer_args=request.optimizer_args,
                optimizer_low_vram=request.optimizer_low_vram,
            ),
            _run_repository(),
        )
    except ValueError as exc:
        detail = str(exc)
        if detail == "lora_training_preset_not_found":
            raise HTTPException(status_code=404, detail=detail) from exc
        if detail in {
            "lora_training_dataset_empty",
            "lora_training_dataset_has_no_materializable_images",
            "lora_training_caption_extension_invalid",
        }:
            raise HTTPException(status_code=422, detail=detail) from exc
        raise


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict[str, str | bool]:
    deleted = _dataset_repository().delete_dataset(dataset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="lora_training_dataset_not_found")
    return {"deleted": True, "id": dataset_id}
