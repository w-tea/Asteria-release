from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, model_validator

from app.api.operations import get_operation_coordinator
from app.api.video import VideoOperationService
from app.core.config import AppConfig
from app.schemas.resource_advisory import ResourceAdvisoryAuthorization
from app.schemas.video_upscale import VideoUpscaleRequest
from app.services.history import HistoryRepository
from app.services.operation_coordinator import OperationCoordinator
from app.services.operation_repository import OperationNotFoundError, OperationRetryError
from app.services.runtime import ComfyClient
from app.services.runtime_dependencies import RuntimeDependencyRepository
from app.services.user_settings import resolve_current_output_root
from app.services.video_h3_upscale_adapter import ComfyH3VideoUpscaleAdapter
from app.services.video_local_adapter import (
    HistoryVideoSourceResolver,
    LocalH3VideoExecutionAdapter,
    VideoLocalAdapterError,
)
from app.services.video_model_resolver import production_video_model_resolver
from app.services.video_packet import VideoPacketBuilder
from app.services.video_upscale_jobs import (
    VIDEO_UPSCALE_OPERATION_KIND,
    VideoUpscaleJobError,
    VideoUpscaleOperationExecutor,
    VideoUpscaleOperationService,
    VideoUpscaleSourceResolver,
    resolve_video_upscale_target,
)


router = APIRouter(prefix="/api/video/upscale", tags=["video-upscale"])
_video_upscale_service: VideoUpscaleOperationService | None = None


class VideoUpscaleJobSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request: VideoUpscaleRequest
    confirmed_after_preview: Literal[True]
    resource_authorization: ResourceAdvisoryAuthorization | None = None
    run_anyway: Literal[True] | None = None

    @model_validator(mode="after")
    def validate_authorization_choice(self) -> "VideoUpscaleJobSubmission":
        if self.resource_authorization is not None and self.run_anyway is not None:
            raise ValueError("video upscale authorization choices are mutually exclusive")
        return self


class VideoUpscaleRetrySubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resource_authorization: ResourceAdvisoryAuthorization | None = None
    run_anyway: Literal[True] | None = None

    @model_validator(mode="after")
    def validate_authorization_choice(self) -> "VideoUpscaleRetrySubmission":
        if self.resource_authorization is not None and self.run_anyway is not None:
            raise ValueError("video upscale authorization choices are mutually exclusive")
        return self


def configure_video_upscale_operation_coordinator(
    coordinator: OperationCoordinator,
    *,
    video_service: VideoOperationService,
    config: AppConfig | None = None,
    history: HistoryRepository | None = None,
    executor: VideoUpscaleOperationExecutor | None = None,
    output_root_provider: Callable[[], Path] | None = None,
    readiness_check: Callable[[], bool] | None = None,
) -> VideoUpscaleOperationService:
    global _video_upscale_service
    active_config = config or AppConfig()
    active_history = history or HistoryRepository(active_config.data_dir / "asteria.db")
    source_resolver = VideoUpscaleSourceResolver(active_history)
    if executor is None:
        model_resolver = production_video_model_resolver(active_config)
        base_adapter = LocalH3VideoExecutionAdapter(
            client=ComfyClient(active_config.comfy_base_url),
            workflow_root=active_config.root_dir
            / "workflows"
            / "video"
            / "minimax-h3",
            source_resolver=HistoryVideoSourceResolver(active_history, active_config),
            output_root=lambda: resolve_current_output_root(active_config.data_dir),
            comfy_output_root=(
                active_config.comfy_output_dir
                or active_config.root_dir / "runtime" / "ComfyUI" / "output"
            ),
            model_resolver=model_resolver,
        )
        executor = VideoUpscaleOperationExecutor(
            history=active_history,
            source_resolver=source_resolver,
            adapter=ComfyH3VideoUpscaleAdapter(
                base_adapter=base_adapter,
                model_resolver=model_resolver,
                packet_builder=VideoPacketBuilder(model_preflight=model_resolver),
            ),
            poster_root=active_config.data_dir / "cache" / "media-posters",
        )
    coordinator.register_executor(VIDEO_UPSCALE_OPERATION_KIND, executor)
    if readiness_check is None and config is not None:
        dependencies = RuntimeDependencyRepository(
            active_config.runtime_dependencies_path,
            active_config.comfy_dir,
            python_path=active_config.comfy_python_path,
        )

        def runtime_ready() -> bool:
            return any(
                item.id == "comfyui-mmh3-ultimate-upscale"
                and item.status == "installed"
                for item in dependencies.list_status()
            )

        readiness_check = runtime_ready
    _video_upscale_service = VideoUpscaleOperationService(
        coordinator=coordinator,
        history=active_history,
        source_resolver=source_resolver,
        video_service=video_service,
        output_root_provider=(
            output_root_provider
            or (lambda: resolve_current_output_root(active_config.data_dir))
        ),
        readiness_check=readiness_check,
    )
    return _video_upscale_service


def _service(request: Request | None = None) -> VideoUpscaleOperationService:
    global _video_upscale_service
    if _video_upscale_service is None:
        from app.api.video import configure_video_operation_coordinator

        coordinator = get_operation_coordinator()
        config = getattr(request.app.state, "config", AppConfig()) if request is not None else AppConfig()
        video_service = configure_video_operation_coordinator(coordinator, config=config)
        configure_video_upscale_operation_coordinator(
            coordinator,
            video_service=video_service,
            config=config,
        )
    assert _video_upscale_service is not None
    return _video_upscale_service


def _summary(operation) -> dict[str, Any]:
    payload = operation.model_dump(mode="json", exclude={"request_snapshot"})
    generation_id = operation.request_snapshot.get("generation_id")
    if isinstance(generation_id, int):
        payload["generation_id"] = generation_id
    source_request = operation.request_snapshot.get("request")
    if isinstance(source_request, dict):
        source_generation_id = source_request.get("source_generation_id")
        source_artifact_id = source_request.get("source_artifact_id")
        if isinstance(source_generation_id, int):
            payload["source_generation_id"] = source_generation_id
        if isinstance(source_artifact_id, str):
            payload["source_artifact_id"] = source_artifact_id
    payload["resource_authorization"] = operation.request_snapshot.get(
        "resource_authorization"
    )
    return payload


def _status_for(error: VideoUpscaleJobError) -> int:
    return 409 if error.code in {
        "video_upscale_setup_required",
        "video_upscale_source_unavailable",
        "video_upscale_work_bundle_unavailable",
        "video_upscale_work_bundle_changed",
    } else 422


@router.get("/sources/{generation_id}")
def get_video_upscale_source(generation_id: int, artifact_id: str, request: Request) -> dict[str, Any]:
    service = _service(request)
    try:
        source = service.source_resolver.resolve(
            VideoUpscaleRequest.model_validate(
                {
                    "source_generation_id": generation_id,
                    "source_artifact_id": artifact_id,
                    "mode": "model_free",
                    "target": {"scale": 2},
                    "interpolation": "bicubic",
                    "sampling": {
                        "steps": 24,
                        "sampler": "res_multistep",
                        "scheduler": "beta",
                    },
                }
            )
        )
        target_width, target_height = resolve_video_upscale_target(
            source.request, source.bundle
        )
    except (VideoUpscaleJobError, ValueError) as error:
        code = error.code if isinstance(error, VideoUpscaleJobError) else "video_upscale_source_unavailable"
        raise HTTPException(status_code=409, detail={"code": code}) from error
    return {
        "schema_version": "asteria.video-upscale.source.v1",
        "generation_id": generation_id,
        "artifact_id": artifact_id,
        "available": True,
        "source_width": source.bundle.actual_width,
        "source_height": source.bundle.actual_height,
        "frame_count": source.bundle.actual_frame_count,
        "fps_numerator": source.bundle.fps_numerator,
        "fps_denominator": source.bundle.fps_denominator,
        "has_audio": source.bundle.audio_sample_rate is not None,
        "default_target_width": target_width,
        "default_target_height": target_height,
        "supported_modes": ["h3_3d_model", "model_free"],
        "cfg_supported": False,
        "cfg_fixed_value": 1,
        "setup_ready": service.readiness_check is None or service.readiness_check(),
    }


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
def submit_video_upscale_job(
    submission: VideoUpscaleJobSubmission,
    request: Request,
) -> dict[str, Any]:
    service = _service(request)
    try:
        submitted = service.submit(
            submission.request,
            source_actor="local",
            resource_authorization=submission.resource_authorization,
            run_anyway=submission.run_anyway is True,
        )
    except VideoLocalAdapterError as error:
        raise HTTPException(status_code=422, detail={"code": error.code}) from error
    except VideoUpscaleJobError as error:
        raise HTTPException(status_code=_status_for(error), detail={"code": error.code}) from error
    return {**_summary(submitted.operation), "generation_id": submitted.generation_id}


@router.get("/jobs/{operation_id}")
def get_video_upscale_job(operation_id: str, request: Request) -> dict[str, Any]:
    service = _service(request)
    try:
        operation = service.coordinator.get(operation_id)
    except OperationNotFoundError as error:
        raise HTTPException(status_code=404, detail={"code": "video_upscale_job_not_found"}) from error
    if operation.kind != VIDEO_UPSCALE_OPERATION_KIND:
        raise HTTPException(status_code=404, detail={"code": "video_upscale_job_not_found"})
    return _summary(operation)


@router.post("/jobs/{operation_id}/cancel")
def cancel_video_upscale_job(operation_id: str, request: Request) -> dict[str, Any]:
    service = _service(request)
    try:
        operation = service.coordinator.get(operation_id)
        if operation.kind != VIDEO_UPSCALE_OPERATION_KIND:
            raise OperationNotFoundError("operation_not_found")
        return _summary(service.coordinator.request_cancel(operation_id))
    except OperationNotFoundError as error:
        raise HTTPException(status_code=404, detail={"code": "video_upscale_job_not_found"}) from error


@router.post("/jobs/{operation_id}/retry", status_code=status.HTTP_202_ACCEPTED)
def retry_video_upscale_job(
    operation_id: str,
    request: Request,
    submission: VideoUpscaleRetrySubmission | None = None,
) -> dict[str, Any]:
    service = _service(request)
    try:
        operation = service.retry(
            operation_id,
            source_actor="local",
            resource_authorization=(
                submission.resource_authorization if submission is not None else None
            ),
            run_anyway=submission is not None and submission.run_anyway is True,
        )
        return _summary(operation)
    except OperationNotFoundError as error:
        raise HTTPException(status_code=404, detail={"code": "video_upscale_job_not_found"}) from error
    except OperationRetryError as error:
        raise HTTPException(status_code=409, detail={"code": str(error)}) from error
    except VideoLocalAdapterError as error:
        raise HTTPException(status_code=422, detail={"code": error.code}) from error
    except VideoUpscaleJobError as error:
        raise HTTPException(status_code=_status_for(error), detail={"code": error.code}) from error
