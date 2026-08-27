from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Response, status

from app.api.operations import get_operation_coordinator
from app.core.config import AppConfig
from app.schemas.external_models import (
    ExternalModelCollisionResponse,
    ExternalModelLibraryItemResponse,
    ExternalModelOwnedLinkRequest,
    ExternalModelRegistryResponse,
    ExternalModelRootCreate,
    ExternalModelRootPreviewRequest,
    ExternalModelRootPreviewResponse,
    ExternalModelRootRebind,
    ExternalModelRootResponse,
    ExternalModelRootUpdate,
)
from app.schemas.operations import ACTIVE_OPERATION_STATUSES
from app.services.external_model_libraries import (
    ExternalModelLibraryError,
    ExternalModelLibraryRepository,
    ExternalModelLibraryService,
    ExternalModelRoot,
)
from app.services.model_storage import resolve_model_root
from app.services.model_library_cache import (
    clear_model_library_cache,
    refresh_model_library,
)
from app.services.model_reference_migration import (
    migrate_persisted_model_references,
    model_library_reference_candidates,
)


router = APIRouter(prefix="/api/external-model-libraries", tags=["external-model-libraries"])


def _snapshot_root_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"root_id", "resolved_root_id"} and isinstance(child, str):
                found.add(child)
            else:
                found.update(_snapshot_root_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_snapshot_root_ids(child))
    return found


def _active_root_ids() -> set[str]:
    found: set[str] = set()
    for operation in get_operation_coordinator().list(limit=1000):
        if operation.status in ACTIVE_OPERATION_STATUSES:
            found.update(_snapshot_root_ids(operation.request_snapshot))
    return found


def _service() -> ExternalModelLibraryService:
    config = AppConfig()
    return ExternalModelLibraryService(
        ExternalModelLibraryRepository(config.data_dir / "config" / "external-model-libraries.json"),
        protected_roots=(config.root_dir, config.data_dir),
        active_root_ids=_active_root_ids,
    )


def _status(root: ExternalModelRoot) -> str:
    if not root.enabled:
        return "disabled"
    if not root.path:
        return "offline"
    try:
        return "online" if Path(root.path).is_dir() else "offline"
    except OSError:
        return "offline"


def _root_response(root: ExternalModelRoot, *, root_status: str | None = None) -> ExternalModelRootResponse:
    return ExternalModelRootResponse(
        root_id=root.root_id,
        display_name=root.display_name,
        layout=root.layout,
        mappings={key: list(value) for key, value in root.mappings.items()},
        priority=root.priority,
        enabled=root.enabled,
        status=root_status or _status(root),
        path_hint=root.path_hint,
        revision=root.revision,
    )


def _error(exc: ExternalModelLibraryError) -> HTTPException:
    code = str(exc)
    return HTTPException(
        status_code=404 if code == "root_not_found" else 409,
        detail={"code": code},
    )


@router.get("", response_model=ExternalModelRegistryResponse)
def list_external_model_libraries() -> ExternalModelRegistryResponse:
    return ExternalModelRegistryResponse(roots=[_root_response(root) for root in _service().list_roots()])


@router.get("/export")
def export_external_model_libraries(
    include_machine_paths: bool = Query(default=False),
) -> dict[str, object]:
    return _service().export_registry(include_machine_paths=include_machine_paths)


@router.post("/import", response_model=ExternalModelRegistryResponse)
def import_external_model_libraries(
    payload: dict[str, object] = Body(...),
) -> ExternalModelRegistryResponse:
    try:
        _service().import_registry(payload)
    except ExternalModelLibraryError as exc:
        raise _error(exc) from exc
    clear_model_library_cache()
    return list_external_model_libraries()


@router.post("", response_model=ExternalModelRootResponse, status_code=status.HTTP_201_CREATED)
def add_external_model_library(request: ExternalModelRootCreate) -> ExternalModelRootResponse:
    try:
        root = _service().add_root(
            display_name=request.display_name,
            path=Path(request.path),
            layout=request.layout,
            mappings=request.mappings,
            priority=request.priority,
        )
    except ExternalModelLibraryError as exc:
        raise _error(exc) from exc
    clear_model_library_cache()
    return _root_response(root)


@router.post("/preview", response_model=ExternalModelRootPreviewResponse)
def preview_external_model_library(
    request: ExternalModelRootPreviewRequest,
) -> ExternalModelRootPreviewResponse:
    try:
        preview = _service().preview_root(
            Path(request.path),
            exclude_root_id=request.exclude_root_id,
        )
    except ExternalModelLibraryError as exc:
        raise _error(exc) from exc
    return ExternalModelRootPreviewResponse(
        path=preview.path,
        path_hint=preview.path_hint,
        suggested_name=preview.suggested_name,
        detected_layout=preview.detected_layout,
        counts=preview.counts,
        total_items=preview.total_items,
        truncated=preview.truncated,
    )


@router.patch("/{root_id}", response_model=ExternalModelRootResponse)
def update_external_model_library(
    root_id: str,
    request: ExternalModelRootUpdate,
) -> ExternalModelRootResponse:
    try:
        root = _service().update_root(root_id, **request.model_dump(exclude_none=True))
    except ExternalModelLibraryError as exc:
        raise _error(exc) from exc
    clear_model_library_cache()
    return _root_response(root)


@router.post("/{root_id}/rebind", response_model=ExternalModelRootResponse)
def rebind_external_model_library(
    root_id: str,
    request: ExternalModelRootRebind,
) -> ExternalModelRootResponse:
    try:
        root = _service().rebind_root(root_id, Path(request.path))
    except ExternalModelLibraryError as exc:
        raise _error(exc) from exc
    clear_model_library_cache()
    return _root_response(root)


@router.delete("/{root_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_external_model_library(root_id: str) -> Response:
    try:
        _service().remove_root(root_id)
    except ExternalModelLibraryError as exc:
        raise _error(exc) from exc
    clear_model_library_cache()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{root_id}/links")
def create_external_model_link(root_id: str, request: ExternalModelOwnedLinkRequest) -> dict[str, str]:
    config = AppConfig()
    model_root = resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)
    try:
        _service().create_owned_link(
            root_id,
            model_type=request.model_type,
            local_models_root=model_root,
            link_name=request.link_name,
        )
    except ExternalModelLibraryError as exc:
        raise _error(exc) from exc
    clear_model_library_cache()
    return {"status": "linked", "link_name": request.link_name}


@router.delete("/{root_id}/links/{model_type}/{link_name}", status_code=status.HTTP_204_NO_CONTENT)
def remove_external_model_link(root_id: str, model_type: str, link_name: str) -> Response:
    config = AppConfig()
    model_root = resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)
    try:
        _service().remove_owned_link(
            root_id,
            model_type=model_type,
            local_models_root=model_root,
            link_name=link_name,
        )
    except ExternalModelLibraryError as exc:
        raise _error(exc) from exc
    clear_model_library_cache()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/rescan", response_model=ExternalModelRegistryResponse)
def rescan_external_model_libraries() -> ExternalModelRegistryResponse:
    service = _service()
    try:
        result = service.scan()
    except ExternalModelLibraryError as exc:
        raise _error(exc) from exc
    config = AppConfig()
    model_root = resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)
    library = refresh_model_library(model_root)
    migrate_persisted_model_references(
        config.data_dir,
        model_library_reference_candidates(library.items),
    )
    roots = [
        _root_response(root, root_status=result.root_statuses.get(root.root_id))
        for root in service.list_roots()
    ]
    items = [
        ExternalModelLibraryItemResponse(
            stable_model_id=item.stable_model_id,
            root_id=item.root_id,
            model_type=item.model_type,
            logical_name=item.logical_name,
            normalized_relative_path=item.normalized_relative_path,
            size_bytes=item.size_bytes,
            fingerprint_algorithm=item.fingerprint_algorithm,
            fingerprint_version=item.fingerprint_version,
            fingerprint_value=item.fingerprint_value,
        )
        for item in result.items
    ]
    collisions = [
        ExternalModelCollisionResponse(
            model_type=item.model_type,
            logical_name=item.logical_name,
            winner_root_id=item.winner_root_id,
            shadowed_root_ids=list(item.shadowed_root_ids),
        )
        for item in result.collisions
    ]
    return ExternalModelRegistryResponse(
        roots=roots,
        items=items,
        collisions=collisions,
        truncated=result.truncated,
    )
