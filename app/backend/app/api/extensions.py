from __future__ import annotations

from base64 import b64decode, b64encode
import binascii
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.extension_catalog import ExtensionCatalogProjectedEntry
from app.schemas.extensions import validate_extension_id
from app.services.extension_assets import (
    ExtensionAssetError,
    resolve_active_manifest,
    resolve_extension_asset,
)
from app.services.extension_standard_frame import (
    StandardExtensionFrameError,
    build_standard_extension_frame,
)
from app.services.extension_standard_network import (
    StandardExtensionNetworkBroker,
    StandardExtensionNetworkError,
)
from app.services.extension_catalog import ExtensionCatalogError, ExtensionCatalogService
from app.services.extension_installer import ExtensionInstallError, ExtensionInstaller
from app.services.extension_registry import ExtensionRegistry
from app.services.extension_repository import ExtensionRepository, ExtensionRepositoryError
from app.services.extension_sources import ExtensionSourceError, SourceKind, describe_extension_source


router = APIRouter(prefix="/api/extensions", tags=["extensions"])
_DEFAULT_STANDARD_NETWORK_BROKER = StandardExtensionNetworkBroker()


class _StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CatalogExtensionSourceRequest(_StrictRequest):
    mode: Literal["catalog"]
    extension_id: str
    version: str


class DirectExtensionSourceRequest(_StrictRequest):
    mode: Literal["direct"]
    reference: str = Field(min_length=1, max_length=2048)
    expected_sha256: str | None = None
    expected_size_bytes: int | None = Field(default=None, gt=0)
    confirm_unverified: bool = False


ExtensionSourceRequest = Annotated[
    CatalogExtensionSourceRequest | DirectExtensionSourceRequest,
    Field(discriminator="mode"),
]


class ExtensionMutationRequest(_StrictRequest):
    source: ExtensionSourceRequest
    confirm: Literal[True]
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=160)


class ExtensionConfirmRequest(_StrictRequest):
    confirm: Literal[True]
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=160)


class ExtensionEnabledRequest(_StrictRequest):
    enabled: bool


class ExtensionOperationResponse(_StrictRequest):
    operation_id: str
    kind: str
    status: str
    phase: str
    restart_required: Literal[True] = True


class ExtensionEnabledResponse(_StrictRequest):
    extension_id: str
    enabled: bool
    restart_required: Literal[True] = True


class StandardExtensionNetworkRequest(_StrictRequest):
    url: str = Field(min_length=1, max_length=4096)
    method: Literal["GET", "POST"] = "GET"
    headers: dict[str, str] = Field(default_factory=dict)
    body_base64: str = Field(default="", max_length=350_000)


class StandardExtensionNetworkResponse(_StrictRequest):
    status_code: int
    headers: dict[str, str]
    body_base64: str


class ExtensionCatalogResponse(_StrictRequest):
    source: Literal["network", "cache", "empty"]
    stale: bool
    cache_age_seconds: int | None
    error_code: str | None
    rejected_count: int
    entries: list[ExtensionCatalogProjectedEntry]


class ExtensionStatusItem(_StrictRequest):
    extension_id: str
    name: str
    active_version: str | None
    loaded_version: str | None
    previous_version: str | None
    enabled: bool
    source_kind: str | None
    registry_status: str
    recovery_status: str | None
    pending_action: str | None
    load_status: str
    error_code: str | None
    execution_level: Literal["standard", "full_access"] | None
    permissions: dict[str, Any] | None
    capabilities: list[str]
    contributions: dict[str, Any]
    frontend_bundle_url: str | None
    frontend_element: str | None
    frontend_frame_url: str | None


class ExtensionStatusResponse(_StrictRequest):
    safe_mode: bool
    previous_incomplete: bool
    journal_error: bool
    items: list[ExtensionStatusItem]


def _repository(request: Request) -> ExtensionRepository:
    return ExtensionRepository(request.app.state.config.extension_root)


def _catalog(request: Request) -> ExtensionCatalogService:
    return request.app.state.extension_catalog


def _coordinator(request: Request):
    return request.app.state.operation_coordinator


def _standard_network_broker(request: Request) -> StandardExtensionNetworkBroker:
    return getattr(
        request.app.state,
        "extension_standard_network_broker",
        _DEFAULT_STANDARD_NETWORK_BROKER,
    )


@router.get("/status", response_model=ExtensionStatusResponse)
def extension_status(request: Request) -> dict:
    repository = _repository(request)
    startup = request.app.state.extension_startup
    startup_records = {
        record.extension_id: record
        for record in startup.records
    }
    items = []
    for record in ExtensionRegistry(repository).scan():
        state = record.state
        if state is None:
            items.append(
                {
                    "extension_id": record.extension_id,
                    "name": record.extension_id,
                    "active_version": None,
                    "loaded_version": None,
                    "previous_version": None,
                    "enabled": False,
                    "source_kind": None,
                    "registry_status": "error",
                    "recovery_status": None,
                    "pending_action": None,
                    "load_status": "failed",
                    "error_code": record.error_code,
                    "execution_level": None,
                    "permissions": None,
                    "capabilities": [],
                    "contributions": {"workspaces": [], "panels": [], "actions": []},
                    "frontend_bundle_url": None,
                    "frontend_element": None,
                    "frontend_frame_url": None,
                }
            )
            continue
        manifest = record.manifest
        if manifest is None and state.recovery.status != "missing_install":
            try:
                manifest, _root = resolve_active_manifest(
                    repository,
                    record.extension_id,
                    state,
                )
            except ExtensionRepositoryError:
                manifest = None
        startup_record = startup_records.get(record.extension_id)
        loaded_version = (
            startup_record.version
            if startup_record is not None and startup_record.status == "loaded"
            else None
        )
        can_serve = (
            not startup.safe_mode
            and state.enabled
            and state.recovery.status == "ready"
            and state.pending_action is None
            and loaded_version == state.active_version
            and manifest is not None
        )
        items.append(
            {
                "extension_id": record.extension_id,
                "name": manifest.name if manifest is not None else record.extension_id,
                "active_version": state.active_version,
                "loaded_version": loaded_version,
                "previous_version": state.previous_version,
                "enabled": state.enabled,
                "source_kind": state.source.kind,
                "registry_status": record.status,
                "recovery_status": state.recovery.status,
                "pending_action": state.pending_action,
                "load_status": startup_record.status if startup_record is not None else "not_loaded",
                "error_code": (
                    startup_record.error_code
                    if startup_record is not None
                    else state.last_load.error_code
                ),
                "execution_level": manifest.execution_level if manifest is not None else None,
                "permissions": (
                    manifest.permissions.model_dump(mode="json")
                    if manifest is not None
                    else None
                ),
                "capabilities": list(manifest.capabilities) if manifest is not None else [],
                "contributions": (
                    manifest.contributions.model_dump(mode="json")
                    if manifest is not None
                    else {"workspaces": [], "panels": [], "actions": []}
                ),
                "frontend_bundle_url": (
                    f"/api/extensions/assets/{record.extension_id}/{manifest.entrypoints.frontend}"
                    if can_serve and manifest is not None and manifest.execution_level == "full_access"
                    else None
                ),
                "frontend_element": (
                    manifest.entrypoints.frontend_element
                    if can_serve and manifest is not None and manifest.execution_level == "full_access"
                    else None
                ),
                "frontend_frame_url": (
                    f"/api/extensions/standard-frame/{record.extension_id}/{state.active_version}"
                    if can_serve and manifest is not None and manifest.execution_level == "standard"
                    else None
                ),
            }
        )
    return {
        "safe_mode": startup.safe_mode,
        "previous_incomplete": startup.previous_incomplete,
        "journal_error": startup.journal_error,
        "items": items,
    }


@router.get("/catalog", response_model=ExtensionCatalogResponse)
def extension_catalog(request: Request) -> dict:
    catalog = _catalog(request)
    result = catalog.refresh()
    return {
        "source": result.source,
        "stale": result.stale,
        "cache_age_seconds": result.cache_age_seconds,
        "error_code": result.error_code,
        "rejected_count": result.rejected_count,
        "entries": [
            ExtensionCatalogProjectedEntry(
                **entry.model_dump(),
                runtime_compatibility=catalog.runtime_compatibility(entry),
            )
            for entry in result.entries
        ],
    }


@router.post("/install", status_code=202, response_model=ExtensionOperationResponse)
def install_extension(payload: ExtensionMutationRequest, request: Request) -> dict:
    source, expected_id, expected_version, expected_level, expected_permissions = _resolve_source(payload.source, request)
    snapshot = _install_snapshot(source, confirm=True)
    if expected_id is not None:
        snapshot["expected_extension_id"] = expected_id
    if expected_version is not None:
        snapshot["expected_version"] = expected_version
    if expected_level is not None:
        snapshot["expected_execution_level"] = expected_level
    if expected_permissions is not None:
        snapshot["expected_permissions"] = expected_permissions
    operation = _coordinator(request).enqueue(
        kind="extension.install",
        source_actor="local",
        request_snapshot=snapshot,
        idempotency_key=payload.idempotency_key,
        progress_total=2,
        progress_unit="steps",
    )
    return _operation_response(operation)


@router.post(
    "/{extension_id}/update",
    status_code=202,
    response_model=ExtensionOperationResponse,
)
def update_extension(
    extension_id: str,
    payload: ExtensionMutationRequest,
    request: Request,
) -> dict:
    _valid_id(extension_id)
    source, catalog_id, expected_version, expected_level, expected_permissions = _resolve_source(payload.source, request)
    if catalog_id is not None and catalog_id != extension_id:
        raise HTTPException(status_code=409, detail={"code": "extension_source_mismatch"})
    snapshot = _install_snapshot(source, confirm=True)
    snapshot["expected_extension_id"] = extension_id
    if expected_version is not None:
        snapshot["expected_version"] = expected_version
    if expected_level is not None:
        snapshot["expected_execution_level"] = expected_level
    if expected_permissions is not None:
        snapshot["expected_permissions"] = expected_permissions
    operation = _coordinator(request).enqueue(
        kind="extension.update",
        source_actor="local",
        request_snapshot=snapshot,
        idempotency_key=payload.idempotency_key,
        progress_total=2,
        progress_unit="steps",
    )
    return _operation_response(operation)


@router.post(
    "/{extension_id}/rollback",
    status_code=202,
    response_model=ExtensionOperationResponse,
)
def rollback_extension(
    extension_id: str,
    payload: ExtensionConfirmRequest,
    request: Request,
) -> dict:
    _valid_id(extension_id)
    operation = _coordinator(request).enqueue(
        kind="extension.rollback",
        source_actor="local",
        request_snapshot={"extension_id": extension_id, "confirm": True},
        idempotency_key=payload.idempotency_key,
        progress_total=2,
        progress_unit="steps",
    )
    return _operation_response(operation)


@router.patch("/{extension_id}", response_model=ExtensionEnabledResponse)
def set_extension_enabled(
    extension_id: str,
    payload: ExtensionEnabledRequest,
    request: Request,
) -> dict:
    _valid_id(extension_id)
    installer = ExtensionInstaller(_repository(request))
    try:
        state = installer.set_enabled(extension_id, enabled=payload.enabled)
    except (ExtensionInstallError, ExtensionRepositoryError) as exc:
        _raise_extension_error(getattr(exc, "code", "extension_state_invalid"))
    return {
        "extension_id": extension_id,
        "enabled": state.enabled,
        "restart_required": True,
    }


@router.delete(
    "/{extension_id}",
    status_code=202,
    response_model=ExtensionOperationResponse,
)
def uninstall_extension(
    extension_id: str,
    payload: ExtensionConfirmRequest,
    request: Request,
) -> dict:
    _valid_id(extension_id)
    operation = _coordinator(request).enqueue(
        kind="extension.uninstall",
        source_actor="local",
        request_snapshot={"extension_id": extension_id, "confirm": True},
        idempotency_key=payload.idempotency_key,
        progress_total=2,
        progress_unit="steps",
    )
    return _operation_response(operation)


@router.get("/assets/{extension_id}/{asset_path:path}")
def extension_asset(extension_id: str, asset_path: str, request: Request) -> FileResponse:
    try:
        path, media_type = resolve_extension_asset(
            _repository(request),
            extension_id=extension_id,
            relative_path=asset_path,
            startup_result=request.app.state.extension_startup,
        )
    except (ExtensionAssetError, ExtensionRepositoryError, ValueError):
        raise HTTPException(
            status_code=404,
            detail={"code": "extension_asset_not_found"},
        ) from None
    return FileResponse(
        path,
        media_type=media_type,
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.get(
    "/standard-frame/{extension_id}/{version}/{contribution_id}",
    response_class=HTMLResponse,
)
def standard_extension_frame(
    request: Request,
    extension_id: str,
    version: str,
    contribution_id: str,
    nonce: str,
) -> HTMLResponse:
    try:
        frame = build_standard_extension_frame(
            _repository(request),
            extension_id=extension_id,
            version=version,
            contribution_id=contribution_id,
            frame_nonce=nonce,
            startup_result=request.app.state.extension_startup,
        )
    except (StandardExtensionFrameError, ExtensionRepositoryError, OSError, ValueError):
        raise HTTPException(
            status_code=404,
            detail={"code": "extension_standard_frame_not_found"},
        ) from None
    return HTMLResponse(
        frame.content,
        headers={
            "Content-Security-Policy": frame.content_security_policy,
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post(
    "/standard-broker/{extension_id}/network",
    response_model=StandardExtensionNetworkResponse,
)
def standard_extension_network(
    extension_id: str,
    payload: StandardExtensionNetworkRequest,
    request: Request,
) -> dict:
    _valid_id(extension_id)
    manifest = _resolve_active_standard_manifest(request, extension_id)
    try:
        body = b64decode(payload.body_base64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=422,
            detail={"code": "extension_standard_network_request_invalid"},
        ) from None
    try:
        response = _standard_network_broker(request).fetch(
            manifest,
            url=payload.url,
            method=payload.method,
            headers=payload.headers,
            body=body,
        )
    except StandardExtensionNetworkError as exc:
        code = str(exc)
        status = 403 if code in {
            "extension_standard_network_denied",
            "extension_standard_network_private_target",
            "extension_standard_network_reserved_target",
        } else 422 if code in {
            "extension_standard_network_request_invalid",
            "extension_standard_network_target_invalid",
        } else 429 if code == "extension_standard_network_busy" else 502
        raise HTTPException(status_code=status, detail={"code": code}) from None
    return {
        "status_code": response.status_code,
        "headers": response.headers,
        "body_base64": b64encode(response.body).decode("ascii"),
    }


def _resolve_active_standard_manifest(request: Request, extension_id: str):
    repository = _repository(request)
    try:
        state = repository.read_state(extension_id)
        manifest, _root = resolve_active_manifest(repository, extension_id, state)
    except (ExtensionRepositoryError, OSError, ValueError):
        raise HTTPException(
            status_code=404,
            detail={"code": "extension_standard_runtime_unavailable"},
        ) from None
    loaded = any(
        record.extension_id == extension_id
        and record.status == "loaded"
        and record.version == state.active_version
        for record in request.app.state.extension_startup.records
    )
    if (
        request.app.state.extension_startup.safe_mode
        or not state.enabled
        or state.recovery.status != "ready"
        or state.pending_action is not None
        or manifest.execution_level != "standard"
        or not loaded
    ):
        raise HTTPException(
            status_code=404,
            detail={"code": "extension_standard_runtime_unavailable"},
        )
    return manifest


def _resolve_source(source_request: ExtensionSourceRequest, request: Request):
    try:
        if isinstance(source_request, CatalogExtensionSourceRequest):
            entry, source = _catalog(request).resolve_source(
                source_request.extension_id,
                source_request.version,
            )
            return (
                source,
                entry.id,
                entry.version,
                entry.execution_level,
                entry.permissions.model_dump(mode="json"),
            )
        source = describe_extension_source(
            source_request.reference,
            expected_sha256=source_request.expected_sha256,
            expected_size_bytes=source_request.expected_size_bytes,
            confirmed_unverified=source_request.confirm_unverified,
        )
        return source, None, None, None, None
    except ExtensionCatalogError as exc:
        status = 404 if exc.code == "extension_catalog_entry_not_found" else 409
        raise HTTPException(status_code=status, detail={"code": exc.code}) from None
    except ExtensionSourceError:
        raise HTTPException(
            status_code=422,
            detail={"code": "extension_source_invalid"},
        ) from None


def _install_snapshot(source, *, confirm: bool) -> dict:
    snapshot = {"source": source.to_snapshot(), "confirm": confirm}
    if source.kind is SourceKind.LOCAL_ZIP:
        snapshot["package_path"] = source.reference
    return snapshot


def _operation_response(operation) -> dict:
    return {
        "operation_id": operation.operation_id,
        "kind": operation.kind,
        "status": operation.status,
        "phase": operation.phase,
        "restart_required": True,
    }


def _valid_id(extension_id: str) -> None:
    try:
        validate_extension_id(extension_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=404, detail={"code": "extension_not_found"}) from None


def _raise_extension_error(code: str) -> None:
    status = 404 if code in {"extension_state_invalid", "extension_install_missing"} else 409
    raise HTTPException(status_code=status, detail={"code": code})
