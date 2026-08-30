from __future__ import annotations

from typing import Any, Callable, Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.api.operations import get_operation_coordinator
from app.core.config import AppConfig
from app.schemas.resource_advisory import (
    ResourceAdvisoryAuthorization,
    ResourceObservationSnapshot,
)
from app.schemas.video import VideoRequest
from app.schemas.video_sequence import VideoSequence
from app.services.history import HistoryRepository
from app.services.operation_coordinator import OperationCoordinator
from app.services.operation_executor import OperationExecutor
from app.services.operation_repository import OperationNotFoundError
from app.services.resource_advisor import ResourceAdvisor
from app.services.runtime import ComfyClient
from app.services.video_executor import VideoOperationExecutor
from app.services.video_local_adapter import (
    HistoryVideoSourceResolver,
    HistoryVideoWorkingArtifactResolver,
    LocalH3VideoExecutionAdapter,
    VideoLocalAdapterError,
)
from app.services.video_model_resolver import production_video_model_resolver
from app.services.video_packet import VideoPacketBuilder, VideoPacketError
from app.services.runtime_dependencies import RuntimeDependencyRepository
from app.services.video_runtime_adapters import video_runtime_dependencies_ready
from app.services.video_sequence_context import VideoSequenceContextError
from app.services.video_sequence_jobs import (
    VIDEO_SEQUENCE_ASSEMBLE_OPERATION_KIND,
    VIDEO_SEQUENCE_CLIP_OPERATION_KIND,
    VideoSequenceAssemblyExecutor,
    VideoSequenceClipExecutor,
    VideoSequenceJobService,
)
from app.services.video_sequence_assembly import (
    ComfyH3LatentAssemblyAdapter,
    VideoSequenceAssemblyService,
)
from app.services.video_sequence_repository import (
    VideoSequenceConflictError,
    VideoSequenceNotFoundError,
    VideoSequenceRepository,
    VideoSequenceTransitionError,
)
from app.services.user_settings import resolve_current_output_root


router = APIRouter(prefix="/api/video/sequences", tags=["video-sequences"])
_sequence_service: VideoSequenceJobService | None = None


class SequenceCreateSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request: VideoRequest
    receipt_id: str = Field(min_length=1, max_length=192)


class SequenceMutation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_revision: int = Field(ge=1)
    expected_segment_revision: int = Field(ge=1)
    receipt_id: str = Field(min_length=1, max_length=192)


class SequenceGenerateSubmission(SequenceMutation):
    resource_authorization: ResourceAdvisoryAuthorization | None = None
    run_anyway: Literal[True] | None = None

    @model_validator(mode="after")
    def validate_authorization_choice(self) -> "SequenceGenerateSubmission":
        if self.resource_authorization is not None and self.run_anyway is not None:
            raise ValueError("video sequence authorization choices are mutually exclusive")
        return self


class SequenceAcceptContinueSubmission(SequenceMutation):
    candidate_artifact_id: str
    next_request: VideoRequest


class SequenceCandidateSubmission(SequenceMutation):
    candidate_artifact_id: str


class SequencePrepareRegenerationSubmission(SequenceMutation):
    request: VideoRequest
    parent_segment_revision: int = Field(ge=1)


class SequenceAssemblyRetrySubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_revision: int = Field(ge=1)
    receipt_id: str = Field(min_length=1, max_length=192)


def configure_video_sequence_operation_coordinator(
    coordinator: OperationCoordinator,
    *,
    config: AppConfig | None = None,
    executor: OperationExecutor | None = None,
    history: HistoryRepository | None = None,
    resource_advisor_provider: Callable[[], ResourceAdvisor] | None = None,
    resource_observation_provider: Callable[[], ResourceObservationSnapshot] | None = None,
) -> VideoSequenceJobService:
    global _sequence_service
    active_config = config or AppConfig()
    active_history = history or HistoryRepository(active_config.data_dir / "asteria.db")
    sequences = VideoSequenceRepository(active_config.data_dir / "asteria.db")
    model_resolver = production_video_model_resolver(active_config)
    comfy_client = ComfyClient(active_config.comfy_base_url)
    packet_builder = VideoPacketBuilder(model_preflight=model_resolver)
    if executor is None:
        local_adapter = LocalH3VideoExecutionAdapter(
            client=comfy_client,
            workflow_root=active_config.root_dir
            / "workflows"
            / "video"
            / "minimax-h3",
            source_resolver=HistoryVideoSourceResolver(active_history, active_config),
            output_root=lambda: resolve_current_output_root(active_config.data_dir),
            working_artifact_resolver=HistoryVideoWorkingArtifactResolver(active_history),
            comfy_output_root=(
                active_config.comfy_output_dir
                or active_config.root_dir / "runtime" / "ComfyUI" / "output"
            ),
            model_resolver=model_resolver,
        )
        delegate = VideoOperationExecutor(
            history=active_history,
            local_adapter=local_adapter,
            poster_root=active_config.data_dir / "cache" / "media-posters",
        )
        executor = VideoSequenceClipExecutor(
            delegate=delegate,
            sequences=sequences,
            history=active_history,
        )
    coordinator.register_executor(VIDEO_SEQUENCE_CLIP_OPERATION_KIND, executor)
    coordinator.register_executor(
        VIDEO_SEQUENCE_ASSEMBLE_OPERATION_KIND,
        VideoSequenceAssemblyExecutor(
            sequences=sequences,
            history=active_history,
            assembler=VideoSequenceAssemblyService(
                output_root=lambda: resolve_current_output_root(active_config.data_dir),
                latent_adapter=ComfyH3LatentAssemblyAdapter(
                    client=comfy_client,
                    workflow_path=(
                        active_config.root_dir
                        / "workflows"
                        / "video"
                        / "minimax-h3"
                        / "sequence-latent-assembly.json"
                    ),
                    model_resolver=model_resolver,
                ),
            ),
            poster_root=active_config.data_dir / "cache" / "media-posters",
            accepted_roots=lambda: {
                active_config.data_dir / "outputs",
                resolve_current_output_root(active_config.data_dir),
            },
        ),
    )
    readiness_check = None
    if config is not None:
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

        resource_advisor_provider = (
            resource_advisor_provider or resource_advisory_api.get_resource_advisor
        )
        resource_observation_provider = (
            resource_observation_provider
            or resource_advisory_api.observe_advisory_capacity
        )
    _sequence_service = VideoSequenceJobService(
        sequences=sequences,
        coordinator=coordinator,
        history=active_history,
        packet_builder=packet_builder,
        readiness_check=readiness_check,
        resource_advisor_provider=resource_advisor_provider,
        resource_observation_provider=resource_observation_provider,
        output_root_provider=lambda: resolve_current_output_root(active_config.data_dir),
    )
    return _sequence_service


def _service(request: Request | None = None) -> VideoSequenceJobService:
    global _sequence_service
    if _sequence_service is None:
        coordinator = get_operation_coordinator()
        config = (
            getattr(request.app.state, "config", AppConfig())
            if request is not None
            else AppConfig()
        )
        configure_video_sequence_operation_coordinator(coordinator, config=config)
    assert _sequence_service is not None
    return _sequence_service


def _sequence_payload(value: VideoSequence) -> dict[str, Any]:
    return value.model_dump(mode="json")


def _operation_payload(value) -> dict[str, Any]:
    return value.model_dump(mode="json", exclude={"request_snapshot"})


def _raise_api_error(error: Exception) -> None:
    if isinstance(error, OperationNotFoundError):
        raise HTTPException(404, detail={"code": "video_sequence_job_not_found"}) from error
    if isinstance(error, VideoSequenceNotFoundError):
        raise HTTPException(404, detail={"code": error.code}) from error
    if isinstance(error, VideoSequenceConflictError):
        raise HTTPException(409, detail={"code": error.code}) from error
    if isinstance(error, VideoSequenceTransitionError):
        raise HTTPException(409, detail={"code": error.code}) from error
    code = getattr(error, "code", "video_sequence_request_invalid")
    status_code = 409 if code == "video_local_setup_required" else 422
    raise HTTPException(status_code, detail={"code": code}) from error


@router.post("", status_code=status.HTTP_201_CREATED)
def create_sequence(submission: SequenceCreateSubmission, request: Request) -> dict[str, Any]:
    try:
        return _sequence_payload(
            _service(request).create(submission.request, receipt_id=submission.receipt_id)
        )
    except (
        VideoLocalAdapterError,
        VideoPacketError,
        VideoSequenceConflictError,
        VideoSequenceTransitionError,
    ) as error:
        _raise_api_error(error)


@router.get("")
def list_sequences(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    values = _service(request).list(limit=limit)
    return {"items": [_sequence_payload(value) for value in values]}


@router.get("/{sequence_id}")
def get_sequence(sequence_id: str, request: Request) -> dict[str, Any]:
    try:
        return _sequence_payload(_service(request).get(sequence_id))
    except VideoSequenceNotFoundError as error:
        _raise_api_error(error)


@router.post("/{sequence_id}/segments/{segment_id}/generate", status_code=202)
def generate_sequence_clip(
    sequence_id: str,
    segment_id: str,
    submission: SequenceGenerateSubmission,
    request: Request,
) -> dict[str, Any]:
    try:
        value = _service(request).generate(
            sequence_id,
            segment_id=segment_id,
            expected_revision=submission.expected_revision,
            expected_segment_revision=submission.expected_segment_revision,
            receipt_id=submission.receipt_id,
            source_actor="local",
            resource_authorization=submission.resource_authorization,
            run_anyway=submission.run_anyway is True,
        )
        return {
            "sequence": _sequence_payload(value.sequence),
            "operation": _operation_payload(value.operation),
            "generation_id": value.generation_id,
        }
    except (
        VideoLocalAdapterError,
        VideoPacketError,
        VideoSequenceContextError,
        VideoSequenceNotFoundError,
        VideoSequenceConflictError,
        VideoSequenceTransitionError,
    ) as error:
        _raise_api_error(error)


@router.get("/{sequence_id}/jobs/{operation_id}")
def get_sequence_job(sequence_id: str, operation_id: str, request: Request) -> dict[str, Any]:
    try:
        operation = _service(request).coordinator.get(operation_id)
        owner_sequence_id = (
            operation.request_snapshot.get("sequence", {}).get("sequence_id")
            if operation.kind == VIDEO_SEQUENCE_CLIP_OPERATION_KIND
            else operation.request_snapshot.get("sequence_id")
            if operation.kind == VIDEO_SEQUENCE_ASSEMBLE_OPERATION_KIND
            else None
        )
        if owner_sequence_id != sequence_id:
            raise OperationNotFoundError("operation_not_found")
        return _operation_payload(operation)
    except OperationNotFoundError as error:
        raise HTTPException(404, detail={"code": "video_sequence_job_not_found"}) from error


@router.post("/{sequence_id}/segments/{segment_id}/jobs/{operation_id}/cancel")
def cancel_sequence_clip(
    sequence_id: str,
    segment_id: str,
    operation_id: str,
    submission: SequenceMutation,
    request: Request,
) -> dict[str, Any]:
    try:
        return _operation_payload(
            _service(request).cancel(
                sequence_id,
                segment_id=segment_id,
                operation_id=operation_id,
                expected_revision=submission.expected_revision,
                expected_segment_revision=submission.expected_segment_revision,
                receipt_id=submission.receipt_id,
            )
        )
    except (
        OperationNotFoundError,
        VideoSequenceNotFoundError,
        VideoSequenceConflictError,
        VideoSequenceTransitionError,
    ) as error:
        _raise_api_error(error)


@router.post("/{sequence_id}/segments/{segment_id}/retry", status_code=202)
def retry_sequence_clip(
    sequence_id: str,
    segment_id: str,
    submission: SequenceGenerateSubmission,
    request: Request,
) -> dict[str, Any]:
    try:
        value = _service(request).generate(
            sequence_id,
            segment_id=segment_id,
            expected_revision=submission.expected_revision,
            expected_segment_revision=submission.expected_segment_revision,
            receipt_id=submission.receipt_id,
            source_actor="local",
            resource_authorization=submission.resource_authorization,
            run_anyway=submission.run_anyway is True,
            retry=True,
        )
        return {
            "sequence": _sequence_payload(value.sequence),
            "operation": _operation_payload(value.operation),
            "generation_id": value.generation_id,
        }
    except Exception as error:
        if isinstance(
            error,
            (
                VideoLocalAdapterError,
                VideoPacketError,
                VideoSequenceContextError,
                VideoSequenceNotFoundError,
                VideoSequenceConflictError,
                VideoSequenceTransitionError,
            ),
        ):
            _raise_api_error(error)
        raise


@router.post("/{sequence_id}/segments/{segment_id}/accept-continue")
def accept_continue(
    sequence_id: str,
    segment_id: str,
    submission: SequenceAcceptContinueSubmission,
    request: Request,
) -> dict[str, Any]:
    try:
        return _sequence_payload(
            _service(request).accept_and_continue(
                sequence_id,
                segment_id=segment_id,
                candidate_artifact_id=submission.candidate_artifact_id,
                next_request=submission.next_request,
                expected_revision=submission.expected_revision,
                expected_segment_revision=submission.expected_segment_revision,
                receipt_id=submission.receipt_id,
            )
        )
    except Exception as error:
        if isinstance(
            error,
            (
                VideoLocalAdapterError,
                VideoSequenceNotFoundError,
                VideoSequenceConflictError,
                VideoSequenceTransitionError,
            ),
        ):
            _raise_api_error(error)
        raise


@router.post(
    "/{sequence_id}/segments/{segment_id}/accept-finish",
    status_code=status.HTTP_202_ACCEPTED,
)
def accept_finish(
    sequence_id: str,
    segment_id: str,
    submission: SequenceCandidateSubmission,
    request: Request,
) -> dict[str, Any]:
    try:
        value = _service(request).accept_and_finish(
            sequence_id,
            segment_id=segment_id,
            candidate_artifact_id=submission.candidate_artifact_id,
            expected_revision=submission.expected_revision,
            expected_segment_revision=submission.expected_segment_revision,
            receipt_id=submission.receipt_id,
        )
        return {
            "sequence": _sequence_payload(value.sequence),
            "operation": _operation_payload(value.operation),
            "generation_id": value.generation_id,
        }
    except Exception as error:
        if isinstance(
            error,
            (
                VideoLocalAdapterError,
                VideoSequenceNotFoundError,
                VideoSequenceConflictError,
                VideoSequenceTransitionError,
            ),
        ):
            _raise_api_error(error)
        raise


@router.post("/{sequence_id}/assembly/retry", status_code=status.HTTP_202_ACCEPTED)
def retry_assembly(
    sequence_id: str,
    submission: SequenceAssemblyRetrySubmission,
    request: Request,
) -> dict[str, Any]:
    try:
        value = _service(request).retry_assembly(
            sequence_id,
            expected_revision=submission.expected_revision,
            receipt_id=submission.receipt_id,
            source_actor="local",
        )
        return {
            "sequence": _sequence_payload(value.sequence),
            "operation": _operation_payload(value.operation),
            "generation_id": value.generation_id,
        }
    except Exception as error:
        if isinstance(
            error,
            (
                VideoSequenceNotFoundError,
                VideoSequenceConflictError,
                VideoSequenceTransitionError,
            ),
        ):
            _raise_api_error(error)
        raise


@router.post("/{sequence_id}/segments/{segment_id}/replace")
def replace_candidate(
    sequence_id: str,
    segment_id: str,
    submission: SequenceCandidateSubmission,
    request: Request,
) -> dict[str, Any]:
    try:
        return _sequence_payload(
            _service(request).sequences.replace_accepted(
                sequence_id,
                segment_id=segment_id,
                candidate_artifact_id=submission.candidate_artifact_id,
                expected_revision=submission.expected_revision,
                expected_segment_revision=submission.expected_segment_revision,
                receipt_id=submission.receipt_id,
            )
        )
    except Exception as error:
        if isinstance(
            error,
            (
                VideoSequenceNotFoundError,
                VideoSequenceConflictError,
                VideoSequenceTransitionError,
            ),
        ):
            _raise_api_error(error)
        raise


@router.post("/{sequence_id}/segments/{segment_id}/resume")
def resume_sequence(
    sequence_id: str,
    segment_id: str,
    submission: SequenceMutation,
    request: Request,
) -> dict[str, Any]:
    try:
        return _sequence_payload(
            _service(request).sequences.resume(
                sequence_id,
                segment_id=segment_id,
                expected_revision=submission.expected_revision,
                expected_segment_revision=submission.expected_segment_revision,
                receipt_id=submission.receipt_id,
            )
        )
    except Exception as error:
        if isinstance(
            error,
            (
                VideoSequenceNotFoundError,
                VideoSequenceConflictError,
                VideoSequenceTransitionError,
            ),
        ):
            _raise_api_error(error)
        raise


@router.post("/{sequence_id}/segments/{segment_id}/prepare-regeneration")
def prepare_sequence_regeneration(
    sequence_id: str,
    segment_id: str,
    submission: SequencePrepareRegenerationSubmission,
    request: Request,
) -> dict[str, Any]:
    try:
        return _sequence_payload(
            _service(request).sequences.prepare_regeneration(
                sequence_id,
                segment_id=segment_id,
                request_snapshot=submission.request,
                parent_segment_revision=submission.parent_segment_revision,
                expected_revision=submission.expected_revision,
                expected_segment_revision=submission.expected_segment_revision,
                receipt_id=submission.receipt_id,
            )
        )
    except Exception as error:
        if isinstance(
            error,
            (
                VideoSequenceNotFoundError,
                VideoSequenceConflictError,
                VideoSequenceTransitionError,
            ),
        ):
            _raise_api_error(error)
        raise


@router.delete("/{sequence_id}/segments/{segment_id}", status_code=204)
def delete_draft_sequence(
    sequence_id: str,
    segment_id: str,
    request: Request,
    expected_revision: int = Query(ge=1),
    expected_segment_revision: int = Query(ge=1),
    receipt_id: str = Query(min_length=1, max_length=192),
) -> Response:
    try:
        _service(request).sequences.delete_draft(
            sequence_id,
            segment_id=segment_id,
            expected_revision=expected_revision,
            expected_segment_revision=expected_segment_revision,
            receipt_id=receipt_id,
        )
        return Response(status_code=204)
    except Exception as error:
        if isinstance(
            error,
            (
                VideoSequenceNotFoundError,
                VideoSequenceConflictError,
                VideoSequenceTransitionError,
            ),
        ):
            _raise_api_error(error)
        raise
