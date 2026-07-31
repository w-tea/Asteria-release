from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.api.operations import get_operation_coordinator
from app.core.config import AppConfig
from app.services.diagnostics import DiagnosticsError, DiagnosticsService


router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])
_service_factory: Callable[[], DiagnosticsService] | None = None


def configure_diagnostics_service_factory(
    factory: Callable[[], DiagnosticsService] | None,
) -> None:
    global _service_factory
    _service_factory = factory


def _service() -> DiagnosticsService:
    if _service_factory is not None:
        return _service_factory()
    return DiagnosticsService(AppConfig(), get_operation_coordinator())


def _raise_http(error: DiagnosticsError) -> None:
    code = str(error)
    if code == "diagnostics_export_id_invalid":
        status = 400
    elif code == "diagnostics_export_not_found":
        status = 404
    elif code in {
        "runtime_recovery_active_operations",
        "runtime_ownership_unverified",
        "runtime_stop_failed",
    }:
        status = 409
    else:
        status = 503
    raise HTTPException(status_code=status, detail={"code": code}) from error


@router.get("")
def diagnostics_status() -> dict:
    return _service().status()


@router.post("/exports")
def create_diagnostics_export() -> dict:
    try:
        return _service().export()
    except DiagnosticsError as error:
        _raise_http(error)


@router.get("/exports/{export_id}")
def download_diagnostics_export(export_id: str) -> FileResponse:
    try:
        path = _service().export_path(export_id)
    except DiagnosticsError as error:
        _raise_http(error)
    return FileResponse(
        path,
        media_type="application/json",
        filename=f"Asteria-diagnostics-{export_id}.json",
    )


@router.post("/recovery/cleanup")
def cleanup_diagnostics_metadata() -> dict:
    try:
        return _service().cleanup_owned_metadata()
    except DiagnosticsError as error:
        _raise_http(error)


@router.post("/recovery/restart-runtime")
def restart_runtime_clean() -> dict:
    try:
        return _service().restart_runtime_clean()
    except DiagnosticsError as error:
        _raise_http(error)
