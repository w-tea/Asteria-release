from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from app.core.config import AppConfig
from app.schemas.maintenance import (
    MaintenanceBackupListResponse,
    MaintenanceBackupSummaryResponse,
    MaintenanceRestorePreviewResponse,
    MaintenanceRestoreRequest,
    MaintenanceSchemaVersions,
    MaintenanceStatusResponse,
)
from app.services.maintenance_backup import MaintenanceBackupError, MaintenanceBackupService


router = APIRouter(prefix="/api/maintenance", tags=["maintenance"])


def _service() -> MaintenanceBackupService:
    data_dir = AppConfig().data_dir
    return MaintenanceBackupService(data_dir, data_dir / "maintenance-backups")


def _http_error(exc: MaintenanceBackupError) -> HTTPException:
    code = str(exc)
    status = 404 if code == "maintenance_backup_not_found" else 409
    return HTTPException(status_code=status, detail={"code": code})


@router.get("/status", response_model=MaintenanceStatusResponse)
def maintenance_status() -> MaintenanceStatusResponse:
    journal = _service().journal.load()
    last_operation = asdict(journal) if journal is not None else None
    return MaintenanceStatusResponse(
        schemas=MaintenanceSchemaVersions(),
        last_operation=last_operation,
    )


@router.post("/backups", response_model=MaintenanceBackupSummaryResponse)
def create_maintenance_backup() -> MaintenanceBackupSummaryResponse:
    try:
        return MaintenanceBackupSummaryResponse(**asdict(_service().create_backup()))
    except MaintenanceBackupError as exc:
        raise _http_error(exc) from exc


@router.get("/backups", response_model=MaintenanceBackupListResponse)
def list_maintenance_backups() -> MaintenanceBackupListResponse:
    return MaintenanceBackupListResponse(
        items=[
            MaintenanceBackupSummaryResponse(**asdict(item))
            for item in _service().list_backups()
        ]
    )


@router.post("/restore/preview", response_model=MaintenanceRestorePreviewResponse)
def preview_maintenance_restore(request: MaintenanceRestoreRequest) -> MaintenanceRestorePreviewResponse:
    try:
        return MaintenanceRestorePreviewResponse(**asdict(_service().preview_restore(request.backup_id)))
    except MaintenanceBackupError as exc:
        raise _http_error(exc) from exc


@router.post("/restore", response_model=MaintenanceBackupSummaryResponse)
def restore_maintenance_backup(request: MaintenanceRestoreRequest) -> MaintenanceBackupSummaryResponse:
    if not request.confirmed:
        raise HTTPException(status_code=409, detail={"code": "maintenance_restore_confirmation_required"})
    try:
        return MaintenanceBackupSummaryResponse(**asdict(_service().restore_backup(request.backup_id)))
    except MaintenanceBackupError as exc:
        raise _http_error(exc) from exc
