from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, get_args

from app.services.workflow_mapping import WorkflowMappingNotFound, WorkflowMappingRepository, WorkflowNodeContract


ControlNetMode = Literal[
    "Balanced",
    "My prompt is more important",
    "ControlNet is more important",
]
CONTROLNET_MODE_LABELS = tuple(str(mode) for mode in get_args(ControlNetMode))
BALANCED_CONTROLNET_MODE = CONTROLNET_MODE_LABELS[0]


@dataclass(frozen=True)
class ControlNetModeContract:
    soft_weights_class: str
    soft_weights: WorkflowNodeContract
    advanced_apply_class: str
    advanced_apply: WorkflowNodeContract


@dataclass(frozen=True)
class ControlNetModeContractStatus:
    ready: bool
    issues: tuple[str, ...]


def enabled_priority_controlnet_units(units: Any) -> list[Any]:
    if not isinstance(units, list):
        return []
    priority: list[Any] = []
    for unit in units:
        enabled = unit.get("enabled", True) if isinstance(unit, dict) else getattr(unit, "enabled", True)
        mode = unit.get("control_mode", BALANCED_CONTROLNET_MODE) if isinstance(unit, dict) else getattr(
            unit,
            "control_mode",
            BALANCED_CONTROLNET_MODE,
        )
        if enabled and str(mode) != BALANCED_CONTROLNET_MODE:
            priority.append(unit)
    return priority


def resolve_controlnet_mode_contract(
    repository: WorkflowMappingRepository,
    profile: str = "sdxl",
) -> ControlNetModeContract:
    try:
        mapping = repository.get_txt2img(profile)
    except WorkflowMappingNotFound as exc:
        raise ValueError(f"controlnet mode contract is not mapped for profile: {profile}") from exc
    soft_weights = mapping.node_contract("controlnet_soft_weights")
    advanced_apply = mapping.node_contract("controlnet_advanced_apply")
    if soft_weights is None or advanced_apply is None:
        raise ValueError(f"controlnet mode contract is not mapped for profile: {profile}")
    return ControlNetModeContract(
        soft_weights_class=mapping.class_type("controlnet_soft_weights", ""),
        soft_weights=soft_weights,
        advanced_apply_class=mapping.class_type("controlnet_advanced_apply", ""),
        advanced_apply=advanced_apply,
    )


def controlnet_mode_contract_status(
    object_info: dict[str, Any],
    contract: ControlNetModeContract,
) -> ControlNetModeContractStatus:
    issues = [
        *_node_contract_issues(
            object_info,
            "controlnet_soft_weights",
            contract.soft_weights_class,
            contract.soft_weights,
        ),
        *_node_contract_issues(
            object_info,
            "controlnet_advanced_apply",
            contract.advanced_apply_class,
            contract.advanced_apply,
        ),
    ]
    return ControlNetModeContractStatus(ready=not issues, issues=tuple(issues))


def _node_contract_issues(
    object_info: dict[str, Any],
    logical_name: str,
    class_type: str,
    contract: WorkflowNodeContract,
) -> list[str]:
    node_info = object_info.get(class_type)
    if not class_type or not isinstance(node_info, dict):
        return [f"{logical_name}:class"]
    inputs = node_info.get("input")
    if not isinstance(inputs, dict):
        inputs = {}
    required = inputs.get("required")
    optional = inputs.get("optional")
    required_names = set(required) if isinstance(required, dict) else set()
    optional_names = set(optional) if isinstance(optional, dict) else set()
    available_names = required_names | optional_names
    issues = [
        f"{logical_name}:input:{name}"
        for name in contract.required_inputs
        if name not in required_names
    ]
    issues.extend(
        f"{logical_name}:input:{name}"
        for name in contract.accepted_inputs
        if name not in available_names
    )
    outputs = node_info.get("output")
    output_values = outputs if isinstance(outputs, list) else []
    for index, expected in contract.outputs_by_index.items():
        if index >= len(output_values) or output_values[index] != expected:
            issues.append(f"{logical_name}:output:{index}")
    return issues
