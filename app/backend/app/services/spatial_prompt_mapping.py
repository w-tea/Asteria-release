from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.schemas.spatial_prompt_runtime import (
    SpatialConditioningBundle,
    SpatialMaskBundle,
)
from app.services.spatial_conditioning_builder import parse_workflow_ref
from app.services.workflow_mapping import WorkflowMapping


ASTERIA_SPATIAL_NODE_CONTRACT_VERSION = "asteria.spatial-prompt.node.v1"
_AUTO_STRENGTH = 100.0
_AUTO_BLENDING = 25.0


class SpatialPromptMappingError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class SpatialWorkflowRefs:
    model_ref: list[str | int]
    positive_ref: list[str | int]
    negative_ref: list[str | int]
    execution_receipt_ref: list[str | int]
    next_id: int


def _workflow_ref(value: Sequence[str | int]) -> list[str | int]:
    if (
        isinstance(value, (str, bytes))
        or len(value) != 2
        or not isinstance(value[0], str)
        or not value[0]
        or isinstance(value[1], bool)
        or not isinstance(value[1], int)
        or value[1] < 0
    ):
        raise SpatialPromptMappingError("spatial_mapping_ref_invalid")
    return [value[0], value[1]]


def _profile_identity(profile_json: str) -> tuple[str, str, tuple[str, ...]]:
    try:
        payload = json.loads(profile_json)
        receipt = payload["profile_receipt"]
        stage = receipt["stage"]
        receipt_sha256 = receipt["receipt_sha256"]
        target_ids = tuple(receipt["target_ids"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SpatialPromptMappingError("spatial_mapping_profile_invalid") from error
    if (
        payload.get("schema_version") != "asteria.spatial-node-profile.v1"
        or payload.get("contract_version") != ASTERIA_SPATIAL_NODE_CONTRACT_VERSION
        or not isinstance(stage, str)
        or not isinstance(receipt_sha256, str)
        or any(not isinstance(target_id, str) for target_id in target_ids)
        or len(target_ids) != len(set(target_ids))
    ):
        raise SpatialPromptMappingError("spatial_mapping_profile_invalid")
    return stage, receipt_sha256, target_ids


def append_spatial_prompt_mapping(
    *,
    workflow: MutableMapping[str, dict[str, Any]],
    next_id: int,
    mapping: WorkflowMapping,
    model_ref: Sequence[str | int],
    conditioning_bundle: SpatialConditioningBundle,
    mask_bundle: SpatialMaskBundle,
    mask_refs: Mapping[str, Sequence[str | int]],
    profile_json: str,
) -> SpatialWorkflowRefs:
    if next_id < 0:
        raise SpatialPromptMappingError("spatial_mapping_node_id_invalid")
    normalized_model_ref = _workflow_ref(model_ref)
    conditioning_target_ids = tuple(
        item.target_id for item in conditioning_bundle.targets
    )
    mask_by_id = {item.target_id: item for item in mask_bundle.targets}
    expected = set(conditioning_target_ids)
    if (
        not expected
        or set(mask_by_id) != expected
        or set(mask_refs) != expected
        or mask_bundle.stage != conditioning_bundle.stage
    ):
        raise SpatialPromptMappingError("spatial_mapping_target_ids_mismatch")
    profile_stage, profile_receipt_sha256, profile_target_ids = _profile_identity(
        profile_json
    )
    ordered_masks = sorted(mask_bundle.targets, key=lambda item: item.depth_index)
    if [item.depth_index for item in ordered_masks] != list(range(len(ordered_masks))):
        raise SpatialPromptMappingError("spatial_mapping_depth_order_invalid")
    target_ids = tuple(item.target_id for item in ordered_masks)
    if (
        profile_stage != conditioning_bundle.stage
        or profile_receipt_sha256 != conditioning_bundle.profile_receipt_sha256
        or set(profile_target_ids) != expected
        or profile_target_ids != target_ids
    ):
        raise SpatialPromptMappingError("spatial_mapping_profile_mismatch")

    class_types = {
        logical_name: mapping.class_type(logical_name, "")
        for logical_name in (
            "spatial_mask_collector",
            "spatial_prompt_binder",
            "spatial_model_adapter",
        )
    }
    if any(not value for value in class_types.values()):
        raise SpatialPromptMappingError("spatial_mapping_contract_missing")
    global_ref = parse_workflow_ref(conditioning_bundle.global_conditioning_ref)
    negative_ref = parse_workflow_ref(conditioning_bundle.negative_conditioning_ref)
    conditioning_refs = {
        item.target_id: parse_workflow_ref(item.target_conditioning_ref)
        for item in conditioning_bundle.targets
    }
    normalized_mask_refs = {
        target_id: _workflow_ref(value) for target_id, value in mask_refs.items()
    }

    fragment: dict[str, dict[str, Any]] = {}
    previous_mask: list[str | int] | None = None
    previous_conditioning: list[str | int] | None = None
    cursor = next_id
    for target_id in target_ids:
        mask_receipt = mask_by_id[target_id]
        mask_node = str(cursor)
        binder_node = str(cursor + 1)
        cursor += 2
        mask_inputs: dict[str, Any] = {
            "mask": normalized_mask_refs[target_id],
            "target_id": target_id,
            "depth_index": mask_receipt.depth_index,
            "overlap_weight": mask_receipt.overlap_weight,
            "strength": (
                _AUTO_STRENGTH
                if mask_receipt.strength_mode == "auto"
                else float(mask_receipt.strength)
            ),
            "blending": (
                _AUTO_BLENDING
                if mask_receipt.blending_mode == "auto"
                else float(mask_receipt.blending)
            ),
            "mask_sha256": mask_receipt.mask_sha256,
            "contract_version": ASTERIA_SPATIAL_NODE_CONTRACT_VERSION,
        }
        binder_inputs: dict[str, Any] = {
            "conditioning": conditioning_refs[target_id],
            "target_id": target_id,
            "contract_version": ASTERIA_SPATIAL_NODE_CONTRACT_VERSION,
        }
        if previous_mask is not None:
            mask_inputs["previous"] = previous_mask
        if previous_conditioning is not None:
            binder_inputs["previous"] = previous_conditioning
        fragment[mask_node] = {
            "class_type": class_types["spatial_mask_collector"],
            "inputs": mask_inputs,
        }
        fragment[binder_node] = {
            "class_type": class_types["spatial_prompt_binder"],
            "inputs": binder_inputs,
        }
        previous_mask = [mask_node, 0]
        previous_conditioning = [binder_node, 0]

    adapter_node = str(cursor)
    fragment[adapter_node] = {
        "class_type": class_types["spatial_model_adapter"],
        "inputs": {
            "model": normalized_model_ref,
            "global_positive": global_ref,
            "negative": negative_ref,
            "conditions": previous_conditioning,
            "masks": previous_mask,
            "profile_json": profile_json,
            "stage": conditioning_bundle.stage,
            "contract_version": ASTERIA_SPATIAL_NODE_CONTRACT_VERSION,
        },
    }
    collisions = set(fragment).intersection(workflow)
    if collisions:
        raise SpatialPromptMappingError("spatial_mapping_node_id_collision")
    workflow.update(fragment)
    return SpatialWorkflowRefs(
        model_ref=[adapter_node, 0],
        positive_ref=[adapter_node, 1],
        negative_ref=[adapter_node, 2],
        execution_receipt_ref=[adapter_node, 3],
        next_id=cursor + 1,
    )
