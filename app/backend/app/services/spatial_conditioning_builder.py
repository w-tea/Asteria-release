from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Literal

from app.schemas.spatial_prompt_runtime import (
    SpatialConditioningBundle,
    SpatialMaskBundle,
    SpatialModelProfileReceipt,
    SpatialPromptPlan,
    SpatialStagePromptPlan,
    canonical_runtime_sha256,
)


SpatialStage = Literal["base", "hires", "refiner"]
WorkflowRef = Sequence[str | int]


class SpatialConditioningBuilderError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def spatial_prompt_stage(
    prompt_plan: SpatialPromptPlan,
    stage: SpatialStage,
) -> SpatialStagePromptPlan:
    selected = getattr(prompt_plan, stage, None)
    if selected is None or selected.stage != stage:
        raise SpatialConditioningBuilderError("spatial_prompt_stage_unavailable")
    return selected


def canonical_workflow_ref(value: WorkflowRef) -> str:
    if (
        isinstance(value, (str, bytes))
        or len(value) != 2
        or not isinstance(value[0], str)
        or not value[0]
        or isinstance(value[1], bool)
        or not isinstance(value[1], int)
        or value[1] < 0
    ):
        raise SpatialConditioningBuilderError("spatial_conditioning_ref_invalid")
    return json.dumps([value[0], value[1]], ensure_ascii=False, separators=(",", ":"))


def parse_workflow_ref(value: str) -> list[str | int]:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise SpatialConditioningBuilderError("spatial_conditioning_ref_invalid") from error
    canonical_workflow_ref(parsed)
    return [parsed[0], parsed[1]]


def build_spatial_conditioning_bundle(
    *,
    prompt_plan: SpatialPromptPlan,
    mask_bundle: SpatialMaskBundle,
    profile_receipt: SpatialModelProfileReceipt,
    stage: SpatialStage,
    encoder_id: str,
    global_conditioning_ref: WorkflowRef,
    negative_conditioning_ref: WorkflowRef,
    target_conditioning_refs: Mapping[str, WorkflowRef],
) -> SpatialConditioningBundle:
    if not encoder_id.strip():
        raise SpatialConditioningBuilderError("spatial_conditioning_encoder_invalid")
    stage_plan = spatial_prompt_stage(prompt_plan, stage)
    if mask_bundle.stage != stage or profile_receipt.stage != stage:
        raise SpatialConditioningBuilderError("spatial_conditioning_stage_mismatch")
    if profile_receipt.support_status != "supported":
        raise SpatialConditioningBuilderError("spatial_conditioning_profile_unsupported")

    prompt_ids = tuple(item.target_id for item in stage_plan.target_fragments)
    mask_ids = tuple(item.target_id for item in mask_bundle.targets)
    profile_ids = tuple(profile_receipt.target_ids)
    ref_ids = tuple(target_conditioning_refs)
    expected = set(prompt_ids)
    if not expected:
        raise SpatialConditioningBuilderError("spatial_conditioning_targets_empty")
    if any(set(values) != expected for values in (mask_ids, profile_ids, ref_ids)):
        raise SpatialConditioningBuilderError("spatial_conditioning_target_ids_mismatch")

    targets = [
        {
            "target_id": target_id,
            "target_conditioning_ref": canonical_workflow_ref(
                target_conditioning_refs[target_id]
            ),
        }
        for target_id in prompt_ids
    ]
    payload = {
        "schema_version": "asteria.spatial-conditioning-bundle.v1",
        "stage": stage,
        "encoder_id": encoder_id,
        "profile_receipt_sha256": profile_receipt.receipt_sha256,
        "global_conditioning_ref": canonical_workflow_ref(global_conditioning_ref),
        "negative_conditioning_ref": canonical_workflow_ref(negative_conditioning_ref),
        "targets": targets,
    }
    return SpatialConditioningBundle.model_validate(
        {**payload, "bundle_sha256": canonical_runtime_sha256(payload)}
    )
