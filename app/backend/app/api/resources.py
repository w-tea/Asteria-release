from __future__ import annotations

from fastapi import APIRouter

from app.api.operations import get_operation_coordinator
from app.core.config import AppConfig
from app.services.resource_observation import ResourceObserver


router = APIRouter(prefix="/api/runtime/resources", tags=["runtime-resources"])


@router.get("")
def observe_resources() -> dict:
    return ResourceObserver(
        AppConfig(),
        get_operation_coordinator(),
    ).snapshot()
