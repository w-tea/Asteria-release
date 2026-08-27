from __future__ import annotations

from typing import Any, Callable, Literal

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, model_validator

from app.api.operations import get_operation_coordinator
from app.core.config import AppConfig
from app.schemas.music import MusicModelComponentSelection, MusicRequest
from app.schemas.resource_advisory import ResourceAdvisoryAuthorization, ResourceObservationSnapshot
from app.schemas.provider_capabilities import (
    PROVIDER_CAPABILITY_SCHEMA_VERSION,
    ProviderCapabilityDescriptor,
)
from app.services.history import HistoryRepository
from app.services.music_jobs import MusicOperationExecutor, MusicOperationService
from app.services.music_local_adapter import LocalMusic3ExecutionAdapter, MusicLocalAdapterError
from app.services.music_model_bundle_resolver import MusicModelBundlePreflightError
from app.services.music_model_registry import production_music_model_registry
from app.services.music_packet import MusicPacketError
from app.services.music_runtime_adapters import default_music_adapters
from app.services.operation_coordinator import OperationCoordinator
from app.services.operation_executor import OperationExecutor
from app.services.operation_repository import OperationNotFoundError, OperationRetryError
from app.services.resource_advisor import ResourceAdvisor
from app.services.runtime import ComfyClient
from app.services.user_settings import resolve_current_output_root


router = APIRouter(prefix="/api/music", tags=["music"])
_music_service: MusicOperationService | None = None


class MusicCapabilityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str
    providers: tuple[ProviderCapabilityDescriptor, ...]


class MusicModelBundleCandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_id: str
    label: str
    profile_id: str
    architecture: str
    managed: bool
    components: tuple[MusicModelComponentSelection, ...]
    compatible: bool
    reason_code: str | None = None


class MusicModelBundleListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["asteria.music.model-bundles.v1"] = (
        "asteria.music.model-bundles.v1"
    )
    bundles: tuple[MusicModelBundleCandidateResponse, ...]


class MusicJobSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request: MusicRequest
    resource_authorization: ResourceAdvisoryAuthorization | None = None
    run_anyway: Literal[True] | None = None

    @model_validator(mode="after")
    def validate_authorization_choice(self) -> "MusicJobSubmission":
        if self.resource_authorization is not None and self.run_anyway is not None:
            raise ValueError("music submit authorization choices are mutually exclusive")
        return self


class MusicJobRetrySubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resource_authorization: ResourceAdvisoryAuthorization | None = None
    run_anyway: Literal[True] | None = None

    @model_validator(mode="after")
    def validate_authorization_choice(self) -> "MusicJobRetrySubmission":
        if self.resource_authorization is not None and self.run_anyway is not None:
            raise ValueError("music retry authorization choices are mutually exclusive")
        return self


def configure_music_operation_coordinator(
    coordinator: OperationCoordinator,
    *,
    config: AppConfig | None = None,
    executor: OperationExecutor | None = None,
    history: HistoryRepository | None = None,
    resource_advisor_provider: Callable[[], ResourceAdvisor] | None = None,
    resource_observation_provider: Callable[[], ResourceObservationSnapshot] | None = None,
) -> None:
    global _music_service
    active_config = config or AppConfig()
    active_history = history or HistoryRepository(active_config.data_dir / "asteria.db")
    readiness_check = None
    execution_profile_provider: Callable[[], str] = lambda: "layer_streaming"
    if executor is None:
        from app.api.music_setup import get_music_setup_service

        setup = get_music_setup_service(active_config)
        model_registry = production_music_model_registry(active_config)
        selected = setup.selected_execution_profile
        execution_profile_provider = selected

        def resolve_model_bundle(request: MusicRequest):
            try:
                return model_registry.resolve(request)
            except MusicModelBundlePreflightError as error:
                raise MusicLocalAdapterError(error.code) from error

        executor = MusicOperationExecutor(
            history=active_history,
            local_adapter=LocalMusic3ExecutionAdapter(
                client=ComfyClient(active_config.comfy_base_url),
                workflow_root=active_config.root_dir
                / "workflows"
                / "music"
                / "minimax-music3",
                output_root=lambda: resolve_current_output_root(active_config.data_dir),
                model_profile_path=active_config.root_dir
                / "profiles"
                / "music"
                / "minimax-music3-local.v1.json",
                execution_profile=selected,
                model_bundle_resolver=resolve_model_bundle,
            ),
            poster_root=active_config.data_dir / "cache" / "media-posters",
        )
        def readiness_check(request: MusicRequest) -> bool:
            plan = setup.plan(selected())
            if request.schema_version == "asteria.music.request.v2":
                return plan.runtime.state == "current" and plan.license.accepted
            return plan.state == "ready"

        model_preflight = model_registry.resolve
    else:
        model_preflight = None
    coordinator.register_executor("music.generate", executor)
    if resource_advisor_provider is None or resource_observation_provider is None:
        from app.api import resource_advisory as resource_advisory_api

        resource_advisor_provider = resource_advisor_provider or resource_advisory_api.get_resource_advisor
        resource_observation_provider = resource_observation_provider or resource_advisory_api.observe_advisory_capacity
    _music_service = MusicOperationService(
        coordinator=coordinator,
        history=active_history,
        readiness_check=readiness_check,
        resource_advisor_provider=resource_advisor_provider,
        resource_observation_provider=resource_observation_provider,
        execution_profile_provider=execution_profile_provider,
        model_preflight=model_preflight,
    )


def _service(request: Request | None = None) -> MusicOperationService:
    global _music_service
    if _music_service is None:
        coordinator = get_operation_coordinator()
        config = getattr(request.app.state, "config", AppConfig()) if request else AppConfig()
        configure_music_operation_coordinator(coordinator, config=config)
    assert _music_service is not None
    return _music_service


def _summary(operation) -> dict[str, Any]:
    return operation.model_dump(mode="json", exclude={"request_snapshot"})


def _music_summary(operation) -> dict[str, Any]:
    payload = _summary(operation)
    generation_id = operation.request_snapshot.get("generation_id")
    if isinstance(generation_id, int) and generation_id > 0:
        payload["generation_id"] = generation_id
    payload["resource_authorization"] = operation.request_snapshot.get(
        "resource_authorization"
    )
    return payload


@router.get("/capabilities", response_model=MusicCapabilityResponse)
def music_capabilities() -> MusicCapabilityResponse:
    return MusicCapabilityResponse(
        schema_version=PROVIDER_CAPABILITY_SCHEMA_VERSION,
        providers=tuple(adapter.capabilities() for adapter in default_music_adapters()),
    )


@router.get("/model-bundles", response_model=MusicModelBundleListResponse)
def music_model_bundles(request: Request) -> MusicModelBundleListResponse:
    try:
        candidates = production_music_model_registry(
            request.app.state.config
        ).managed_bundle_candidates()
    except MusicModelBundlePreflightError as error:
        raise HTTPException(status_code=409, detail={"code": error.code}) from error
    return MusicModelBundleListResponse(
        bundles=tuple(
            MusicModelBundleCandidateResponse(
                bundle_id=candidate.bundle_id,
                label=candidate.label,
                profile_id=candidate.profile_id,
                architecture=candidate.architecture,
                managed=candidate.managed,
                components=tuple(
                    MusicModelComponentSelection(
                        role=component.role.value,
                        stable_asset_id=component.stable_asset_id,
                        fingerprint_algorithm=component.fingerprint_algorithm,
                        fingerprint_version=component.fingerprint_version,
                        fingerprint_value=component.fingerprint_value,
                    )
                    for component in candidate.components
                ),
                compatible=candidate.compatible,
                reason_code=candidate.reason_code,
            )
            for candidate in candidates
        )
    )


@router.post("/draft")
def draft_music_job(submission: MusicJobSubmission, request: Request) -> dict[str, Any]:
    try:
        return _service(request).draft(submission.request)
    except (MusicLocalAdapterError, MusicPacketError) as error:
        raise HTTPException(status_code=422, detail={"code": error.code}) from error


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
def submit_music_job(submission: MusicJobSubmission, request: Request) -> dict[str, Any]:
    try:
        submitted = _service(request).submit(
            submission.request,
            source_actor="local",
            resource_authorization=submission.resource_authorization,
            run_anyway=submission.run_anyway is True,
        )
    except (MusicLocalAdapterError, MusicPacketError) as error:
        code = error.code
        raise HTTPException(
            status_code=409 if code == "music_local_setup_required" else 422,
            detail={"code": code},
        ) from error
    return {**_music_summary(submitted.operation), "generation_id": submitted.generation_id}


@router.get("/jobs/{operation_id}")
def get_music_job(operation_id: str, request: Request) -> dict[str, Any]:
    try:
        operation = _service(request).coordinator.get(operation_id)
    except OperationNotFoundError as error:
        raise HTTPException(status_code=404, detail={"code": "music_job_not_found"}) from error
    if operation.kind != "music.generate":
        raise HTTPException(status_code=404, detail={"code": "music_job_not_found"})
    return _music_summary(operation)


@router.post("/jobs/{operation_id}/cancel")
def cancel_music_job(operation_id: str, request: Request) -> dict[str, Any]:
    try:
        service = _service(request)
        operation = service.coordinator.get(operation_id)
        if operation.kind != "music.generate":
            raise OperationNotFoundError("operation_not_found")
        return _music_summary(service.coordinator.request_cancel(operation_id))
    except OperationNotFoundError as error:
        raise HTTPException(status_code=404, detail={"code": "music_job_not_found"}) from error


@router.post("/jobs/{operation_id}/retry", status_code=status.HTTP_202_ACCEPTED)
def retry_music_job(
    operation_id: str,
    request: Request,
    submission: MusicJobRetrySubmission | None = None,
) -> dict[str, Any]:
    try:
        return _music_summary(_service(request).retry(
            operation_id,
            source_actor="local",
            resource_authorization=submission.resource_authorization if submission else None,
            run_anyway=submission is not None and submission.run_anyway is True,
        ))
    except OperationNotFoundError as error:
        raise HTTPException(status_code=404, detail={"code": "music_job_not_found"}) from error
    except OperationRetryError as error:
        raise HTTPException(status_code=409, detail={"code": str(error)}) from error
    except MusicLocalAdapterError as error:
        raise HTTPException(status_code=409, detail={"code": error.code}) from error
