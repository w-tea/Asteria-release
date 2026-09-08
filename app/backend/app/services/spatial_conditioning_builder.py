from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal

from app.schemas.prompt_input import join_prompt_contract_fragments
from app.schemas.spatial_prompt_runtime import (
    SpatialConditioningBundle,
    SpatialMaskBundle,
    SpatialModelProfileReceipt,
    SpatialPromptPlan,
    SpatialStagePromptPlan,
    canonical_runtime_sha256,
    runtime_text_sha256,
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
    text_encoder_ref: WorkflowRef,
    full_positive_conditioning_ref: WorkflowRef,
    negative_conditioning_ref: WorkflowRef,
) -> SpatialConditioningBundle:
    if not encoder_id.strip():
        raise SpatialConditioningBuilderError("spatial_conditioning_encoder_invalid")
    stage_plan = spatial_prompt_stage(prompt_plan, stage)
    if mask_bundle.stage != stage or profile_receipt.stage != stage:
        raise SpatialConditioningBuilderError("spatial_conditioning_stage_mismatch")
    if profile_receipt.support_status != "supported":
        raise SpatialConditioningBuilderError("spatial_conditioning_profile_unsupported")
    if stage_plan.schema_version != "asteria.spatial-stage-prompt.v2":
        raise SpatialConditioningBuilderError("spatial_conditioning_prompt_v2_required")
    if (
        stage_plan.full_text is None
        or stage_plan.full_text_sha256 is None
        or stage_plan.character_bindings is None
    ):
        raise SpatialConditioningBuilderError("spatial_conditioning_prompt_v2_required")
    prompt_ids = tuple(item.target_id for item in stage_plan.target_fragments)
    mask_ids = tuple(item.target_id for item in mask_bundle.targets)
    profile_ids = tuple(profile_receipt.target_ids)
    binding_ids = tuple(item.target_id for item in stage_plan.character_bindings)
    expected = set(prompt_ids)
    if not expected:
        raise SpatialConditioningBuilderError("spatial_conditioning_targets_empty")
    if any(set(values) != expected for values in (mask_ids, profile_ids, binding_ids)):
        raise SpatialConditioningBuilderError("spatial_conditioning_target_ids_mismatch")
    encoder_contracts = {
        "clip": (("l", "g"), "identity"),
        "native_qwen": (
            ("qwen3_06b", "t5xxl"),
            "anima_native_t5_query_v1",
        ),
        "qwen35_progressive": (
            ("qwen35_4b", "t5xxl"),
            "anima_progressive_qwen35_t5_query_v1",
        ),
    }
    encoder_contract = encoder_contracts.get(profile_receipt.encoder_kind)
    if encoder_contract is None:
        raise SpatialConditioningBuilderError("spatial_conditioning_encoder_invalid")
    required_encoders, adapter_kind = encoder_contract
    fragment_payload = {
        "schema_version": "asteria.spatial-fragments.v6",
        "global_fragment": {
            "fragment_id": stage_plan.global_fragment.fragment_id,
            "text": stage_plan.global_fragment.expanded_prompt,
        },
        "target_fragments": [
            {
                "fragment_id": item.fragment_id,
                "target_id": item.target_id,
                "text": item.expanded_prompt,
            }
            for item in stage_plan.target_fragments
        ],
    }
    if stage_plan.scheduled_bindings:
        fragment_by_target = {
            item.target_id: item for item in stage_plan.target_fragments
        }
        fragment_payload["schedules"] = [
            {
                "start_percent": scheduled.start_percent,
                "end_percent": scheduled.end_percent,
                "global_fragment": {
                    "fragment_id": stage_plan.global_fragment.fragment_id,
                    "text": join_prompt_contract_fragments(
                        scheduled.full_text[start:end]
                        for start, end in scheduled.global_character_ranges
                    ),
                },
                "target_fragments": [
                    {
                        "fragment_id": fragment_by_target[binding.target_id].fragment_id,
                        "target_id": binding.target_id,
                        "text": scheduled.full_text[binding.start : binding.end],
                    }
                    for binding in scheduled.character_bindings
                ],
            }
            for scheduled in stage_plan.scheduled_bindings
        ]
    fragments_json = json.dumps(
        fragment_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    payload = {
        "schema_version": "asteria.spatial-conditioning-bundle.v3",
        "stage": stage,
        "encoder_id": encoder_id,
        "profile_receipt_sha256": profile_receipt.receipt_sha256,
        "text_encoder_ref": canonical_workflow_ref(text_encoder_ref),
        "full_positive_conditioning_ref": canonical_workflow_ref(
            full_positive_conditioning_ref
        ),
        "negative_conditioning_ref": canonical_workflow_ref(negative_conditioning_ref),
        "full_text": stage_plan.full_text,
        "full_text_sha256": stage_plan.full_text_sha256,
        "character_bindings": [
            item.model_dump(mode="json") for item in stage_plan.character_bindings
        ],
        "scheduled_bindings": [
            item.model_dump(mode="json")
            for item in (stage_plan.scheduled_bindings or ())
        ],
        "required_encoders": list(required_encoders),
        "adapter_kind": adapter_kind,
        "fragments_json": fragments_json,
        "fragments_sha256": runtime_text_sha256(fragments_json),
        "step_start_percent": 0.0,
        "step_end_percent": 1.0,
    }
    return SpatialConditioningBundle.model_validate(
        {**payload, "bundle_sha256": canonical_runtime_sha256(payload)}
    )
