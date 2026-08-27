from fastapi import APIRouter, HTTPException

from app.core.config import AppConfig
from app.schemas.remaster import RemasterRequest, RemasterResponse
from app.services import api_errors
from app.services.artifact_lifecycle import artifact_write_operation
from app.services.history import HistoryRepository
from app.services.remaster import (
    RemasterExecutionError,
    RemasterOrchestrator,
    RemasterValidationError,
    validate_remaster_scheduler_contract,
)
from app.services.runtime import ComfyClient, ComfyGenerationRuntime


router = APIRouter(prefix="/api/generations", tags=["generations"])


def _preflight_remaster_scheduler(request: RemasterRequest) -> None:
    try:
        validate_remaster_scheduler_contract(request)
    except RemasterValidationError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code, **exc.details}) from exc


def _build_orchestrator() -> RemasterOrchestrator:
    config = AppConfig()
    client = ComfyClient(config.comfy_base_url)
    return RemasterOrchestrator(
        config=config,
        history=HistoryRepository(config.data_dir / "asteria.db"),
        runtime=ComfyGenerationRuntime(
            client=client,
            model_root=config.model_dir,
            comfy_output_dir=config.comfy_output_dir,
        ),
        object_info_provider=client.object_info,
    )


@artifact_write_operation
def execute_remaster(
    request: RemasterRequest,
    *,
    cancel_requested=None,
    runtime_job_callback=None,
    runtime_stop_confirmed=None,
) -> RemasterResponse:
    _preflight_remaster_scheduler(request)
    try:
        return _build_orchestrator().run(
            request,
            cancel_requested=cancel_requested,
            runtime_job_callback=runtime_job_callback,
            runtime_stop_confirmed=runtime_stop_confirmed,
        )
    except RemasterExecutionError as exc:
        status_code = 502 if exc.code in {
            api_errors.GENERATION_RUNTIME_UNREACHABLE,
            "remaster_execution_failed",
            "remaster_primary_output_invalid",
        } else 409
        raise HTTPException(
            status_code=status_code,
            detail={"code": exc.code, "remaster_id": exc.remaster_id, **exc.details},
        ) from exc


@router.post("/remaster", response_model=RemasterResponse)
def remaster(request: RemasterRequest) -> RemasterResponse:
    return execute_remaster(request)


@router.post("/remaster/{remaster_id}/retry", response_model=RemasterResponse)
def retry_remaster(remaster_id: int) -> RemasterResponse:
    config = AppConfig()
    item = HistoryRepository(config.data_dir / "asteria.db").get(remaster_id)
    if item is None or item.get("settings", {}).get("mode") != "remaster":
        raise HTTPException(status_code=404, detail={"code": api_errors.REMASTER_NOT_FOUND})
    if item["status"] not in {"failed", "cancelled"}:
        raise HTTPException(status_code=409, detail={"code": api_errors.REMASTER_RETRY_UNAVAILABLE})
    snapshot = item.get("settings", {}).get("remaster_request_snapshot")
    if not isinstance(snapshot, dict):
        raise HTTPException(status_code=409, detail={"code": api_errors.REMASTER_RETRY_UNAVAILABLE})
    request = RemasterRequest.model_validate(snapshot)
    _preflight_remaster_scheduler(request)
    return execute_remaster(request)


@router.post("/jobs/remaster")
def queue_remaster(request: RemasterRequest):
    from app.api.generation import _enqueue_generation_job

    _preflight_remaster_scheduler(request)
    return _enqueue_generation_job("remaster", request)
