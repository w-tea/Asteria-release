from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.workflow_mapping import (
    WorkflowMappingNotFound,
    WorkflowMappingRepository,
    WorkflowNodeContract,
    workflow_node_contract_issues,
)


@dataclass(frozen=True)
class SeedResizeNodeContract:
    logical_name: str
    class_type: str
    contract: WorkflowNodeContract


@dataclass(frozen=True)
class SeedResizeContract:
    nodes: tuple[SeedResizeNodeContract, ...]


@dataclass(frozen=True)
class SeedResizeContractStatus:
    ready: bool
    issues: tuple[str, ...]


def resolve_seed_resize_contract(
    repository: WorkflowMappingRepository,
    profile: str = "sdxl",
) -> SeedResizeContract:
    try:
        mapping = repository.get_txt2img(profile)
    except WorkflowMappingNotFound as exc:
        raise ValueError(f"seed resize contract is not mapped for profile: {profile}") from exc
    nodes: list[SeedResizeNodeContract] = []
    for logical_name in ("seed_explorer", "composite_noise", "ksampler_advanced_inspire"):
        contract = mapping.node_contract(logical_name)
        class_type = mapping.class_type(logical_name, "")
        if contract is None or not class_type:
            raise ValueError(f"seed resize contract is not mapped for profile: {profile}")
        nodes.append(SeedResizeNodeContract(logical_name, class_type, contract))
    return SeedResizeContract(tuple(nodes))


def seed_resize_contract_status(
    object_info: dict[str, Any],
    contract: SeedResizeContract,
) -> SeedResizeContractStatus:
    issues = [
        issue
        for node in contract.nodes
        for issue in workflow_node_contract_issues(
            object_info,
            node.logical_name,
            node.class_type,
            node.contract,
        )
    ]
    return SeedResizeContractStatus(ready=not issues, issues=tuple(issues))


def seed_resize_requested(settings: dict[str, Any]) -> bool:
    return bool(
        int(settings.get("seed_resize_from_w") or 0)
        or int(settings.get("seed_resize_from_h") or 0)
    )


def seed_resize_active(settings: dict[str, Any]) -> bool:
    if not seed_resize_requested(settings):
        return False
    source = (
        int(settings.get("seed_resize_from_w") or 0),
        int(settings.get("seed_resize_from_h") or 0),
    )
    target = (int(settings.get("width") or 0), int(settings.get("height") or 0))
    return not all(target) or source != target


def seed_variation_requested(settings: dict[str, Any]) -> bool:
    subseed = int(settings.get("subseed") if settings.get("subseed") is not None else -1)
    return subseed != -1 or float(settings.get("subseed_strength") or 0) != 0


def seed_resize_request_issues(
    profile: str,
    mode: str,
    settings: dict[str, Any],
) -> list[dict[str, str]]:
    if not seed_resize_requested(settings):
        return []
    width = int(settings.get("seed_resize_from_w") or 0)
    height = int(settings.get("seed_resize_from_h") or 0)
    if profile != "sdxl" or mode != "txt2img":
        reason = "seed_resize_is_mapped_only_for_sdxl_txt2img"
    elif not width or not height:
        reason = "seed_resize_requires_both_dimensions"
    elif width < 8 or height < 8 or width > 4096 or height > 4096:
        reason = "seed_resize_dimensions_out_of_range"
    elif width % 8 or height % 8:
        reason = "seed_resize_dimensions_must_be_multiples_of_8"
    elif seed_variation_requested(settings):
        reason = "seed_resize_with_seed_variation_is_not_supported"
    else:
        return []
    return [{"field": "seed_resize", "reason": reason}]
