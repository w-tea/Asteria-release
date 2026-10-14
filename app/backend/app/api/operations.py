from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from app.core.config import AppConfig
from app.schemas.operations import OperationRecord
from app.services.operation_coordinator import OperationCoordinator
from app.services.operation_repository import (
    OperationDismissError,
    OperationNotFoundError,
    OperationRepository,
    OperationRetryError,
)


router = APIRouter(prefix="/api/operations", tags=["operations"])
_coordinator: OperationCoordinator | None = None


class OperationRetryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_actor: str = "local"
    idempotency_key: str | None = None


def get_operation_coordinator() -> OperationCoordinator:
    global _coordinator
    if _coordinator is None:
        _coordinator = OperationCoordinator(
            OperationRepository(AppConfig().data_dir / "asteria.db"),
            auto_start=False,
        )
    return _coordinator


def _not_found(exc: OperationNotFoundError) -> HTTPException:
    return HTTPException(status_code=404, detail={"code": str(exc)})


def _summary(record: OperationRecord) -> dict[str, Any]:
    return record.model_dump(mode="json", exclude={"request_snapshot"})


@router.get("")
def list_operations(limit: int = Query(default=200, ge=1, le=1000)) -> dict[str, Any]:
    return {"items": [_summary(item) for item in get_operation_coordinator().list(limit=limit)]}


@router.get("/{operation_id}")
def get_operation(operation_id: str) -> dict[str, Any]:
    try:
        operation = get_operation_coordinator().get(operation_id)
        excluded = {"request_snapshot"} if operation.kind.startswith("extension.") else None
        return operation.model_dump(mode="json", exclude=excluded)
    except OperationNotFoundError as exc:
        raise _not_found(exc) from exc


@router.get("/{operation_id}/events")
def get_operation_events(
    operation_id: str,
    after_revision: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    try:
        events = get_operation_coordinator().events(
            operation_id,
            after_revision=after_revision,
            limit=limit,
        )
    except OperationNotFoundError as exc:
        raise _not_found(exc) from exc
    return {"items": [_summary(event) for event in events]}


@router.post("/{operation_id}/cancel")
def cancel_operation(operation_id: str) -> dict[str, Any]:
    try:
        coordinator = get_operation_coordinator()
        operation = coordinator.get(operation_id)
        if operation.kind.startswith("generation."):
            from app.api.generation import cancel_generation_operation

            operation = cancel_generation_operation(operation_id)
        else:
            operation = coordinator.request_cancel(operation_id)
        return _summary(operation)
    except OperationNotFoundError as exc:
        raise _not_found(exc) from exc


@router.post("/{operation_id}/retry")
def retry_operation(operation_id: str, request: OperationRetryRequest) -> dict[str, Any]:
    try:
        coordinator = get_operation_coordinator()
        original = coordinator.get(operation_id)
        if original.kind == "video.generate":
            raise OperationRetryError("operation_retry_requires_video_studio")
        if original.kind == "music.generate":
            raise OperationRetryError("operation_retry_requires_music_studio")
        if original.kind == "generation.prompt_comparison":
            raise OperationRetryError("comparison_prepare_new_run")
        operation = coordinator.retry(
            operation_id,
            source_actor=request.source_actor,
            idempotency_key=request.idempotency_key,
        )
        return _summary(operation)
    except OperationNotFoundError as exc:
        raise _not_found(exc) from exc
    except OperationRetryError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc)}) from exc


@router.post("/{operation_id}/dismiss")
def dismiss_operation(operation_id: str) -> dict[str, Any]:
    try:
        return _summary(get_operation_coordinator().dismiss(operation_id))
    except OperationNotFoundError as exc:
        raise _not_found(exc) from exc
    except OperationDismissError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc)}) from exc
