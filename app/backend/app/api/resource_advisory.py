from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.operations import get_operation_coordinator
from app.core.config import AppConfig
from app.schemas.provider_capabilities import ProviderCapabilityDescriptor
from app.schemas.resource_advisory import ResourceAdvisory, ResourceObservationSnapshot, ResourceWorkload
from app.services.resource_advisor import (
    ResourceAdvisoryError,
    ResourceCapabilityRegistry,
    get_resource_advisor,
)
from app.services.resource_observation import ResourceObserver


router = APIRouter(prefix="/api/resource-advisory", tags=["resource-advisory"])


def observe_advisory_capacity() -> ResourceObservationSnapshot:
    return ResourceObserver(AppConfig(), get_operation_coordinator()).advisory_capacity()


def _capability_for(
    workload: ResourceWorkload,
    registry: ResourceCapabilityRegistry,
) -> ProviderCapabilityDescriptor:
    capability = next(
        (
            item
            for item in registry.snapshot()
            if item.domain == workload.domain
            and item.provider_id == workload.provider_id
            and item.adapter_id == workload.adapter_id
            and item.model_profile == workload.model_profile
        ),
        None,
    )
    if capability is not None:
        return capability
    raise ResourceAdvisoryError("resource_advisory_provider_unavailable")


@router.post("/estimate", response_model=ResourceAdvisory)
def estimate_resource_advisory(workload: ResourceWorkload, request: Request) -> ResourceAdvisory:
    try:
        registry: ResourceCapabilityRegistry = request.app.state.resource_capability_registry
        return get_resource_advisor().advise(
            workload,
            _capability_for(workload, registry),
            observe_advisory_capacity(),
        )
    except ResourceAdvisoryError as error:
        raise HTTPException(status_code=422, detail=error.code) from error
