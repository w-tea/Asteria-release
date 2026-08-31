from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SPATIAL_PROMPT_CONTRACT_VERSION = "asteria.spatial-prompt.node.v2"
SPATIAL_PROMPT_PACK_VERSION = "3.0.0"
SPATIAL_PROMPT_PROFILE_COVERAGE = {
    "sdxl": "supported",
    "illustrious": "supported",
    "anima_28": "supported",
    "anima_40": "supported",
    "anima_52": "manual_selection",
}

_NODE_CONTRACTS = {
    "mask_compiler_adapter": (
        "AsteriaSpatialMaskCollector",
        ["ASTERIA_SPATIAL_MASK_BUNDLE", "STRING"],
    ),
    "token_binding": (
        "AsteriaSpatialTokenBinding",
        ["CONDITIONING", "ASTERIA_SPATIAL_TOKEN_BINDING", "STRING"],
    ),
    "guidance_adapter": (
        "AsteriaSpatialGuidanceAdapter",
        [
            "MODEL",
            "CONDITIONING",
            "CONDITIONING",
            "ASTERIA_SPATIAL_GUIDANCE_STATE",
        ],
    ),
    "guidance_sampler": (
        "AsteriaSpatialGuidanceSampler",
        ["LATENT", "ASTERIA_SPATIAL_GUIDANCE_STATE"],
    ),
    "execution_receipt": (
        "AsteriaSpatialExecutionReceipt",
        ["STRING"],
    ),
}


@dataclass(frozen=True, slots=True)
class SpatialRuntimeContractStatus:
    available: bool
    missing_components: tuple[str, ...]


def _contract_versions(node: dict[str, Any]) -> list[str]:
    inputs = node.get("input")
    if not isinstance(inputs, dict):
        return []
    required = inputs.get("required")
    if not isinstance(required, dict):
        return []
    raw = required.get("contract_version")
    if not isinstance(raw, list) or not raw:
        return []
    values = raw[0]
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def spatial_runtime_contract_status(
    object_info: dict[str, Any],
) -> SpatialRuntimeContractStatus:
    missing: list[str] = []
    for logical_name, (node_name, expected_outputs) in _NODE_CONTRACTS.items():
        node = object_info.get(node_name)
        if (
            not isinstance(node, dict)
            or node.get("output") != expected_outputs
            or _contract_versions(node) != [SPATIAL_PROMPT_CONTRACT_VERSION]
        ):
            missing.append(logical_name)
    return SpatialRuntimeContractStatus(
        available=not missing,
        missing_components=tuple(sorted(missing)),
    )


def uses_spatial_prompt_layout(settings: dict[str, Any]) -> bool:
    layout = settings.get("prompt_layout")
    return isinstance(layout, dict) and layout.get("mode") == "enabled"
