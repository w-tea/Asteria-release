from __future__ import annotations

from typing import Any, Literal


AnimaLLLiteRuntimeContract = Literal["model_patch", "legacy_direct"]
ANIMA_LLLITE_CURRENT_REQUIRED_NODES = frozenset(
    {"LoadImage", "ModelPatchLoader", "AnimaLLLiteApply"}
)


def anima_lllite_runtime_contract(
    object_info: dict[str, Any],
) -> AnimaLLLiteRuntimeContract | None:
    if _node_has_input(object_info, "AnimaLLLiteApply", "model_patch") and _node_has_input(
        object_info,
        "ModelPatchLoader",
        "name",
    ):
        return "model_patch"
    if _node_has_input(object_info, "AnimaLLLiteApply", "lllite_name"):
        return "legacy_direct"
    return None


def anima_lllite_model_values(object_info: dict[str, Any]) -> set[str]:
    contract = anima_lllite_runtime_contract(object_info)
    if contract == "model_patch":
        values = _node_input_values(object_info, "ModelPatchLoader", "name")
    elif contract == "legacy_direct":
        values = _node_input_values(object_info, "AnimaLLLiteApply", "lllite_name")
    else:
        return set()
    return {
        value
        for value in values
        if is_anima_lllite_model_name(value)
    }


def anima_lllite_missing_nodes(
    object_info: dict[str, Any],
    nodes: set[str],
) -> list[str]:
    if "AnimaLLLiteApply" not in nodes:
        return ["AnimaLLLiteApply"]
    apply_has_model_patch = _node_has_input(object_info, "AnimaLLLiteApply", "model_patch")
    if apply_has_model_patch and "ModelPatchLoader" not in nodes:
        return ["ModelPatchLoader"]
    if anima_lllite_runtime_contract(object_info) is None:
        return ["AnimaLLLiteApply"]
    return []


def anima_lllite_model_directory(contract: AnimaLLLiteRuntimeContract) -> str:
    return "model_patches" if contract == "model_patch" else "controlnet"


def is_anima_lllite_model_name(value: object) -> bool:
    normalized = str(value).casefold().replace("\\", "/")
    path_parts = {part for part in normalized.split("/") if part}
    return "lllite" in normalized or "anima" in path_parts


def is_standard_controlnet_model_name(value: object) -> bool:
    return not is_anima_lllite_model_name(value)


def _node_has_input(object_info: dict[str, Any], node_name: str, input_name: str) -> bool:
    return _node_input_definition(object_info, node_name, input_name) is not None


def _node_input_values(object_info: dict[str, Any], node_name: str, input_name: str) -> list[str]:
    raw = _node_input_definition(object_info, node_name, input_name)
    if not isinstance(raw, list) or not raw:
        return []
    values: object = raw[0]
    if values == "COMBO" and len(raw) > 1 and isinstance(raw[1], dict):
        values = raw[1].get("options")
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def _node_input_definition(
    object_info: dict[str, Any],
    node_name: str,
    input_name: str,
) -> object | None:
    node = object_info.get(node_name)
    if not isinstance(node, dict):
        return None
    inputs = node.get("input")
    if not isinstance(inputs, dict):
        return None
    for section_name in ("required", "optional"):
        section = inputs.get(section_name)
        if isinstance(section, dict) and input_name in section:
            return section[input_name]
    return None
