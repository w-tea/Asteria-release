from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import AppConfig
from app.services.managed_artifacts import ManagedArtifactService
from app.services.managed_downloads import (
    consume_partial_discard,
    ManagedDownloadOperationExecutor,
    ManagedDownloadTelemetryRepository,
    request_partial_discard,
)
from app.services.operation_coordinator import OperationCoordinator
from app.services.operation_repository import OperationNotFoundError
from app.services.runtime_process import RuntimeProcessManager


router = APIRouter(prefix="/api/downloads", tags=["downloads"])
_coordinator: OperationCoordinator | None = None


class ManagedDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_version: int = Field(ge=1)
    component_id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    source_url: str
    destination: str
    expected_size: int = Field(gt=0)
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: str = Field(pattern=r"^(file|zip)$")
    strip_single_root: bool = False
    idempotency_key: str | None = None

    @field_validator("source_url")
    @classmethod
    def require_https(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("managed download URL must use HTTPS")
        return value


def configure_download_operation_coordinator(
    coordinator: OperationCoordinator,
    *,
    config: AppConfig | None = None,
) -> OperationCoordinator:
    global _coordinator
    config = config or AppConfig()
    coordinator.register_executor(
        "managed.download",
        ManagedDownloadOperationExecutor(
            temp_root=config.data_dir / "temp",
            db_path=coordinator.repository.db_path,
            runtime_stopper=lambda: RuntimeProcessManager(config).stop(),
        ),
    )
    _coordinator = coordinator
    return coordinator


def _operations() -> OperationCoordinator:
    if _coordinator is None:
        raise RuntimeError("managed_download_coordinator_unavailable")
    return _coordinator


def _download_payload(operation_id: str) -> dict:
    try:
        operation = _operations().get(operation_id)
    except OperationNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "operation_not_found"}) from exc
    if operation.kind != "managed.download":
        raise HTTPException(status_code=404, detail={"code": "operation_not_found"})
    payload = operation.model_dump(mode="json", exclude={"request_snapshot"})
    telemetry = ManagedDownloadTelemetryRepository(
        _operations().repository.db_path
    ).get(operation_id)
    payload["download"] = (
        {
            "bytes": telemetry.bytes_downloaded,
            "total": telemetry.total_bytes,
            "percent": telemetry.percent,
            "speed_bytes_per_second": telemetry.speed_bytes_per_second,
            "eta_seconds": telemetry.eta_seconds,
            "state": telemetry.state,
        }
        if telemetry is not None
        else None
    )
    return payload


@router.post("")
def enqueue_download(request: ManagedDownloadRequest) -> dict:
    config = AppConfig()
    destination = Path(request.destination).absolute()
    runtime_root = (config.root_dir / "runtime").absolute()
    if destination != runtime_root and destination.is_relative_to(runtime_root):
        raise HTTPException(
            status_code=400,
            detail={"code": "managed_download_runtime_destination_requires_setup"},
        )
    allowed_roots = tuple(
        path.absolute()
        for path in (
            config.data_dir,
            config.model_dir,
        )
    )
    if not any(
        destination != root and destination.is_relative_to(root)
        for root in allowed_roots
    ):
        raise HTTPException(
            status_code=400,
            detail={"code": "managed_download_destination_not_allowed"},
        )
    snapshot = request.model_dump(exclude={"idempotency_key"})
    snapshot["destination"] = str(destination)
    operation = _operations().enqueue(
        kind="managed.download",
        source_actor="local",
        request_snapshot=snapshot,
        idempotency_key=request.idempotency_key,
        progress_total=request.expected_size,
        progress_unit="bytes",
    )
    return _download_payload(operation.operation_id)


@router.get("/{operation_id}")
def get_download(operation_id: str) -> dict:
    return _download_payload(operation_id)


@router.post("/{operation_id}/cancel")
def cancel_download(operation_id: str, discard_partial: bool = False) -> dict:
    if discard_partial:
        request_partial_discard(operation_id)
    try:
        operation = _operations().request_cancel(operation_id)
    except OperationNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "operation_not_found"}) from exc
    if operation.kind != "managed.download":
        raise HTTPException(status_code=404, detail={"code": "operation_not_found"})
    if discard_partial and operation.status == "cancelled":
        spec = ManagedDownloadOperationExecutor._spec(operation.request_snapshot)
        ManagedArtifactService(AppConfig().data_dir / "temp").discard_partial(spec)
        consume_partial_discard(operation_id)
    return _download_payload(operation_id)
