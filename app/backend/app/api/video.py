from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, model_validator

from app.api.operations import get_operation_coordinator
from app.core.config import AppConfig
from app.schemas.provider_capabilities import (
    PROVIDER_CAPABILITY_SCHEMA_VERSION,
    ProviderCapabilityDescriptor,
)
from app.services.video_runtime_adapters import (
    default_video_adapters,
    video_runtime_dependencies_ready,
)
from app.schemas.resource_advisory import ResourceAdvisoryAuthorization
from app.schemas.resource_advisory import ResourceObservationSnapshot
from app.schemas.video import VideoRequest
from app.services.history import HistoryRepository
from app.services.operation_coordinator import OperationCoordinator
from app.services.operation_executor import OperationExecutor
from app.services.operation_repository import OperationNotFoundError, OperationRetryError
from app.services.resource_advisor import ResourceAdvisor
from app.services.runtime import ComfyClient
from app.services.video_executor import VideoOperationExecutor, VideoOperationService
from app.services.video_local_adapter import (
    HistoryVideoSourceResolver,
    LocalH3VideoExecutionAdapter,
    VideoLocalAdapterError,
)
from app.schemas.generation_working_artifact import GenerationWorkingArtifactSummary
from app.services.video_model_resolver import production_video_model_resolver
from app.services.video_packet import VideoPacketBuilder, VideoPacketError
from app.services.user_settings import resolve_current_output_root
from app.services.runtime_dependencies import RuntimeDependencyRepository


router = APIRouter(prefix="/api/video", tags=["video"])
_video_service: VideoOperationService | None = None


class VideoCapabilityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    providers: tuple[ProviderCapabilityDescriptor, ...]


class VideoParameterRestoreResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["asteria.video.parameter-restore.v1"] = (
        "asteria.video.parameter-restore.v1"
    )
    generation_id: int
    seed_decimal: str
    request: VideoRequest
    sources: tuple["VideoParameterRestoreSource", ...] = ()


class VideoParameterRestoreSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generation_id: int
    artifact_id: str
    kind: Literal["image", "video", "audio"]
    label: str


class VideoJobSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request: VideoRequest
    resource_authorization: ResourceAdvisoryAuthorization | None = None
    run_anyway: Literal[True] | None = None

    @model_validator(mode="after")
    def validate_authorization_choice(self) -> "VideoJobSubmission":
        if self.resource_authorization is not None and self.run_anyway is not None:
            raise ValueError("video submit authorization choices are mutually exclusive")
        return self


class VideoJobRetrySubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resource_authorization: ResourceAdvisoryAuthorization | None = None
    run_anyway: Literal[True] | None = None

    @model_validator(mode="after")
    def validate_authorization_choice(self) -> "VideoJobRetrySubmission":
        if self.resource_authorization is not None and self.run_anyway is not None:
            raise ValueError("video retry authorization choices are mutually exclusive")
        return self


def configure_video_operation_coordinator(
    coordinator: OperationCoordinator,
    *,
    config: AppConfig | None = None,
    executor: OperationExecutor | None = None,
    history: HistoryRepository | None = None,
    resource_advisor_provider: Callable[[], ResourceAdvisor] | None = None,
    resource_observation_provider: Callable[[], ResourceObservationSnapshot] | None = None,
) -> VideoOperationService:
    global _video_service
    active_config = config or AppConfig()
    active_history = history or HistoryRepository(active_config.data_dir / "asteria.db")
    model_resolver = production_video_model_resolver(active_config)
    packet_builder = VideoPacketBuilder(model_preflight=model_resolver)
    if executor is None:
        adapter = LocalH3VideoExecutionAdapter(
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
        executor = VideoOperationExecutor(
            history=active_history,
            local_adapter=adapter,
            poster_root=active_config.data_dir / "cache" / "media-posters",
        )
    coordinator.register_executor("video.generate", executor)
    readiness_check = None
    if config is not None and executor is not None:
        from app.api.video_setup import get_video_setup_service

        setup = get_video_setup_service(active_config)
        runtime_dependencies = RuntimeDependencyRepository(
            active_config.runtime_dependencies_path,
            active_config.comfy_dir,
            python_path=active_config.comfy_python_path,
        )
        readiness_check = lambda video_request: (
            setup.plan(
                "reference" if video_request.mode in {"reference", "sequence"} else "base"
            ).state
            == "ready"
            and video_runtime_dependencies_ready(
                video_request,
                runtime_dependencies.list_status(),
            )
        )
    if resource_advisor_provider is None or resource_observation_provider is None:
        from app.api import resource_advisory as resource_advisory_api

        resource_advisor_provider = resource_advisor_provider or resource_advisory_api.get_resource_advisor
        resource_observation_provider = (
            resource_observation_provider or resource_advisory_api.observe_advisory_capacity
        )
    _video_service = VideoOperationService(
        coordinator=coordinator,
        history=active_history,
        packet_builder=packet_builder,
        readiness_check=readiness_check,
        resource_advisor_provider=resource_advisor_provider,
        resource_observation_provider=resource_observation_provider,
        output_root_provider=(
            (lambda: resolve_current_output_root(active_config.data_dir))
            if config is not None
            else None
        ),
    )
    return _video_service


def _service(request: Request | None = None) -> VideoOperationService:
    global _video_service
    if _video_service is None:
        coordinator = get_operation_coordinator()
        config = getattr(request.app.state, "config", AppConfig()) if request is not None else AppConfig()
        configure_video_operation_coordinator(coordinator, config=config)
    assert _video_service is not None
    return _video_service


def _summary(operation) -> dict[str, Any]:
    return operation.model_dump(mode="json", exclude={"request_snapshot"})


def _video_summary(operation, service: VideoOperationService | None = None) -> dict[str, Any]:
    payload = _summary(operation)
    generation_id = operation.request_snapshot.get("generation_id")
    if isinstance(generation_id, int) and generation_id > 0:
        payload["generation_id"] = generation_id
    payload["resource_authorization"] = operation.request_snapshot.get(
        "resource_authorization"
    )
    batch = operation.request_snapshot.get("batch")
    if service is not None and isinstance(batch, dict):
        candidates = []
        for candidate in batch.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            seed = candidate.get("seed")
            generation_id = candidate.get("generation_id")
            item = service.history.get(generation_id) if isinstance(generation_id, int) else None
            artifacts = []
            working_artifacts = []
            if item is not None:
                artifacts = [
                    value.get("artifact_id")
                    for value in item.get("artifacts", [])
                    if isinstance(value, dict) and value.get("kind") == "video" and value.get("role") == "primary"
                ]
                working_artifacts = _public_working_artifacts(item)
            candidates.append(
                {
                    "index": candidate.get("index"),
                    "generation_id": generation_id,
                    # JSON numbers above 2^53 are rounded by browsers. Keep the
                    # exact generation receipt as a decimal string in the UI API.
                    "seed": str(seed) if isinstance(seed, int) and not isinstance(seed, bool) else None,
                    "status": item.get("status") if item is not None else "missing",
                    "artifact_refs": artifacts,
                    "working_artifacts": working_artifacts,
                }
            )
        active = next((item["index"] for item in candidates if item["status"] in {"queued", "running"}), None)
        payload["batch"] = {
            "batch_id": batch.get("batch_id"),
            "count": batch.get("count"),
            "seed_policy": batch.get("seed_policy"),
            "completed": sum(item["status"] == "completed" for item in candidates),
            "active_index": active,
            "candidates": candidates,
        }
    elif service is not None and isinstance(generation_id, int):
        item = service.history.get(generation_id)
        if item is not None:
            working_artifacts = _public_working_artifacts(item)
            if working_artifacts:
                payload["working_artifacts"] = working_artifacts
    return payload


def _public_working_artifacts(item: dict[str, Any]) -> list[dict[str, Any]]:
    settings = item.get("settings")
    if not isinstance(settings, dict):
        return []
    result: list[dict[str, Any]] = []
    for value in settings.get("working_artifacts", []):
        if not isinstance(value, dict):
            continue
        try:
            summary = GenerationWorkingArtifactSummary.model_validate(value)
        except ValueError:
            continue
        result.append(summary.model_dump(mode="json"))
    return result


@router.get("/capabilities", response_model=VideoCapabilityResponse)
def video_capabilities() -> VideoCapabilityResponse:
    return VideoCapabilityResponse(
        schema_version=PROVIDER_CAPABILITY_SCHEMA_VERSION,
        providers=tuple(adapter.capabilities() for adapter in default_video_adapters()),
    )


@router.get(
    "/parameters/{generation_id}",
    response_model=VideoParameterRestoreResponse,
)
def video_parameters(generation_id: int, request: Request) -> VideoParameterRestoreResponse:
    item = _service(request).history.get(generation_id)
    if (
        item is None
        or item.get("media_kind") != "video"
        or item.get("status") not in {"completed", "completed_with_warnings"}
    ):
        raise HTTPException(
            status_code=404,
            detail={"code": "video_parameters_not_found"},
        )
    settings = item.get("settings")
    snapshot = settings.get("request_snapshot") if isinstance(settings, dict) else None
    try:
        restored = VideoRequest.model_validate(snapshot)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=409,
            detail={"code": "video_parameters_unavailable"},
        ) from error
    if restored.provider != "local_h3":
        raise HTTPException(
            status_code=409,
            detail={"code": "video_parameters_unavailable"},
        )
    requested_artifact_ids = {
        value
        for value in (
            restored.first_frame_artifact_id,
            restored.last_frame_artifact_id,
            *(reference.artifact_id for reference in restored.references),
        )
        if value is not None
    }
    sources: list[VideoParameterRestoreSource] = []
    if requested_artifact_ids:
        for source_item in _service(request).history.list_all():
            source_generation_id = source_item.get("id")
            if not isinstance(source_generation_id, int):
                continue
            for artifact in source_item.get("artifacts", []):
                if not isinstance(artifact, dict):
                    continue
                artifact_id = artifact.get("artifact_id")
                kind = artifact.get("kind")
                if artifact_id not in requested_artifact_ids or kind not in {"image", "video", "audio"}:
                    continue
                sources.append(
                    VideoParameterRestoreSource(
                        generation_id=source_generation_id,
                        artifact_id=artifact_id,
                        kind=kind,
                        label=Path(str(artifact.get("path") or artifact_id)).name,
                    )
                )
    return VideoParameterRestoreResponse(
        generation_id=generation_id,
        seed_decimal=str(restored.seed),
        request=restored,
        sources=tuple(sources),
    )


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
def submit_video_job(submission: VideoJobSubmission, request: Request) -> dict[str, Any]:
    service = _service(request)
    try:
        submitted = service.submit(
            submission.request,
            source_actor="local",
            resource_authorization=submission.resource_authorization,
            run_anyway=submission.run_anyway is True,
        )
    except (VideoLocalAdapterError, VideoPacketError) as error:
        status_code = 409 if error.code == "video_local_setup_required" else 422
        raise HTTPException(status_code=status_code, detail={"code": error.code}) from error
    return {**_video_summary(submitted.operation, service), "generation_id": submitted.generation_id}


@router.get("/jobs/{operation_id}")
def get_video_job(operation_id: str, request: Request) -> dict[str, Any]:
    service = _service(request)
    try:
        operation = service.coordinator.get(operation_id)
    except OperationNotFoundError as error:
        raise HTTPException(status_code=404, detail={"code": "video_job_not_found"}) from error
    if operation.kind != "video.generate":
        raise HTTPException(status_code=404, detail={"code": "video_job_not_found"})
    return _video_summary(operation, service)


@router.post("/jobs/{operation_id}/cancel")
def cancel_video_job(operation_id: str, request: Request) -> dict[str, Any]:
    service = _service(request)
    try:
        operation = service.coordinator.get(operation_id)
        if operation.kind != "video.generate":
            raise OperationNotFoundError("operation_not_found")
        return _video_summary(service.coordinator.request_cancel(operation_id), service)
    except OperationNotFoundError as error:
        raise HTTPException(status_code=404, detail={"code": "video_job_not_found"}) from error


@router.post("/jobs/{operation_id}/retry", status_code=status.HTTP_202_ACCEPTED)
def retry_video_job(
    operation_id: str,
    request: Request,
    submission: VideoJobRetrySubmission | None = None,
) -> dict[str, Any]:
    service = _service(request)
    try:
        retry = service.retry(
            operation_id,
            source_actor="local",
            resource_authorization=(
                submission.resource_authorization if submission is not None else None
            ),
            run_anyway=submission is not None and submission.run_anyway is True,
        )
        return _video_summary(retry, service)
    except OperationNotFoundError as error:
        raise HTTPException(status_code=404, detail={"code": "video_job_not_found"}) from error
    except OperationRetryError as error:
        raise HTTPException(status_code=409, detail={"code": str(error)}) from error
    except VideoLocalAdapterError as error:
        status_code = 409 if error.code == "video_local_setup_required" else 422
        raise HTTPException(status_code=status_code, detail={"code": error.code}) from error
