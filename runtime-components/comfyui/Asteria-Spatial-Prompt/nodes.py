from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from .mask_runtime import collect_mask_target, validate_mask_bundle
from .profile_runtime import SpatialPhaseProfile, load_phase_profile
from .spatial_contract import (
    CONTRACT_VERSION,
    SpatialConditioningBundle,
    SpatialConditioningTarget,
    SpatialMaskBundle,
    canonical_json_sha256,
    clone_conditioning,
    conditioning_bundle_payload,
    validate_bundle_target_ids,
    validate_contract_version,
    validate_stage,
    validate_target_id,
)


def _report(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class AsteriaSpatialMaskCollector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "target_id": ("STRING", {"default": ""}),
                "depth_index": ("INT", {"default": 0, "min": 0}),
                "overlap_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "strength": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0}),
                "blending": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 100.0}),
                "mask_sha256": ("STRING", {"default": ""}),
                "contract_version": ([CONTRACT_VERSION],),
            },
            "optional": {"previous": ("ASTERIA_SPATIAL_MASK_BUNDLE",)},
        }

    RETURN_TYPES = ("ASTERIA_SPATIAL_MASK_BUNDLE", "STRING")
    RETURN_NAMES = ("masks", "receipt")
    FUNCTION = "collect"
    CATEGORY = "Asteria/spatial"

    @staticmethod
    def collect(
        mask,
        target_id,
        depth_index,
        overlap_weight,
        strength,
        blending,
        mask_sha256,
        contract_version,
        previous=None,
    ):
        bundle = collect_mask_target(
            mask,
            target_id,
            depth_index,
            overlap_weight,
            strength,
            blending,
            mask_sha256,
            contract_version,
            previous,
        )
        return bundle, _report(
            {
                "schema_version": "asteria.spatial-node-mask-receipt.v1",
                "target_ids": [target.target_id for target in bundle.targets],
                "receipt_sha256": bundle.receipt_sha256,
            }
        )


class AsteriaSpatialPromptBinder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "target_id": ("STRING", {"default": ""}),
                "contract_version": ([CONTRACT_VERSION],),
            },
            "optional": {"previous": ("ASTERIA_SPATIAL_CONDITIONING",)},
        }

    RETURN_TYPES = ("ASTERIA_SPATIAL_CONDITIONING", "STRING")
    RETURN_NAMES = ("conditions", "receipt")
    FUNCTION = "bind"
    CATEGORY = "Asteria/spatial"

    @staticmethod
    def bind(conditioning, target_id, contract_version, previous=None):
        validate_contract_version(contract_version)
        normalized_id = validate_target_id(target_id)
        existing = previous.targets if previous is not None else ()
        if normalized_id in {target.target_id for target in existing}:
            raise ValueError("Asteria spatial bundle contains a duplicate target")
        cloned = clone_conditioning(conditioning)
        targets = existing + (
            SpatialConditioningTarget(
                target_id=normalized_id,
                conditioning=tuple((entry[0], entry[1]) for entry in cloned),
            ),
        )
        validate_bundle_target_ids([target.target_id for target in targets])
        receipt = canonical_json_sha256(conditioning_bundle_payload(targets))
        bundle = SpatialConditioningBundle(targets=targets, receipt_sha256=receipt)
        return bundle, _report(
            {
                "schema_version": "asteria.spatial-node-conditioning-receipt.v1",
                "target_ids": [target.target_id for target in targets],
                "receipt_sha256": receipt,
            }
        )


def _validate_conditioning_bundle(bundle: SpatialConditioningBundle) -> None:
    target_ids = [target.target_id for target in bundle.targets]
    validate_bundle_target_ids(target_ids)
    for target in bundle.targets:
        clone_conditioning(target.conditioning)
    expected = canonical_json_sha256(conditioning_bundle_payload(bundle.targets))
    if expected != bundle.receipt_sha256:
        raise ValueError("Asteria spatial conditioning bundle receipt does not match")


def _phase_conditioning(
    target,
    mask_target,
    profile: SpatialPhaseProfile,
) -> list[list[Any]]:
    # Layout Compiler has already normalized overlap and applied the soft depth
    # priority to each mask. Applying overlap_weight or depth_index again here
    # would distort the compiler-owned effective mask.
    phases = (
        ("early_guidance", 0.0, profile.early_guidance_end, mask_target.strength / 100.0),
        ("attention_isolation", profile.isolation_start, profile.isolation_end, mask_target.strength / 100.0),
        (
            "late_blending",
            profile.late_blending_start,
            1.0,
            mask_target.strength / 100.0 * mask_target.blending / 100.0,
        ),
    )
    output: list[list[Any]] = []
    for phase, start, end, mask_strength in phases:
        for tensor, source_metadata in target.conditioning:
            metadata = dict(source_metadata)
            metadata.update(
                {
                    "mask": mask_target.mask,
                    "mask_strength": mask_strength,
                    "set_area_to_bounds": False,
                    "start_percent": start,
                    "end_percent": end,
                    "asteria_spatial": {
                        "contract_version": CONTRACT_VERSION,
                        "target_id": target.target_id,
                        "phase": phase,
                        "depth_index": mask_target.depth_index,
                        "overlap_weight": mask_target.overlap_weight,
                        "mask_sha256": mask_target.mask_sha256,
                        "profile_id": profile.profile_id,
                        "profile_revision": profile.profile_revision,
                    },
                }
            )
            output.append([tensor, metadata])
    return output


class AsteriaSpatialModelAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "global_positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "conditions": ("ASTERIA_SPATIAL_CONDITIONING",),
                "masks": ("ASTERIA_SPATIAL_MASK_BUNDLE",),
                "profile_json": ("STRING", {"multiline": True, "default": ""}),
                "stage": (["base", "hires", "refiner"],),
                "contract_version": ([CONTRACT_VERSION],),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "execution_receipt")
    FUNCTION = "apply"
    CATEGORY = "Asteria/spatial"

    @staticmethod
    def apply(
        model,
        global_positive,
        negative,
        conditions,
        masks,
        profile_json,
        stage,
        contract_version,
    ):
        validate_contract_version(contract_version)
        normalized_stage = validate_stage(stage)
        mask_ids = [target.target_id for target in masks.targets]
        condition_ids = [target.target_id for target in conditions.targets]
        validate_bundle_target_ids(mask_ids)
        validate_bundle_target_ids(condition_ids)
        if mask_ids != condition_ids:
            raise ValueError("Asteria spatial mask and conditioning target ids do not match")
        validate_mask_bundle(masks)
        _validate_conditioning_bundle(conditions)
        profile = load_phase_profile(profile_json)
        if profile.stage != normalized_stage:
            raise ValueError("Asteria spatial profile stage does not match")
        if list(profile.target_ids) != mask_ids:
            raise ValueError("Asteria spatial profile target ids do not match")
        positive = clone_conditioning(global_positive)
        final_negative = clone_conditioning(negative)
        for condition_target, mask_target in zip(conditions.targets, masks.targets, strict=True):
            positive.extend(_phase_conditioning(condition_target, mask_target, profile))
        if not hasattr(model, "clone"):
            raise ValueError("Asteria spatial model cannot be cloned")
        source_options = getattr(model, "model_options", None)
        if not isinstance(source_options, dict):
            raise ValueError("Asteria spatial model options are invalid")
        source_transformer_options = source_options.get("transformer_options", {})
        if not isinstance(source_transformer_options, dict):
            raise ValueError("Asteria spatial model options are invalid")
        if "asteria_spatial_prompt" in source_transformer_options:
            raise ValueError("Asteria spatial prompt is already applied")
        execution_body = {
            "schema_version": "asteria.spatial-node-execution.v1",
            "contract_version": CONTRACT_VERSION,
            "stage": normalized_stage,
            "profile_receipt_sha256": profile.profile_receipt_sha256,
            "profile_payload_sha256": profile.profile_payload_sha256,
            "mask_bundle_sha256": masks.receipt_sha256,
            "conditioning_bundle_sha256": conditions.receipt_sha256,
            "target_ids": mask_ids,
            "phase_order": ["early_guidance", "attention_isolation", "late_blending"],
        }
        execution = {
            **execution_body,
            "execution_sha256": canonical_json_sha256(execution_body),
        }
        patched = model.clone()
        if not isinstance(getattr(patched, "model_options", None), dict):
            raise ValueError("Asteria spatial model options are invalid")
        transformer_options = deepcopy(
            patched.model_options.get("transformer_options", {})
        )
        transformer_options["asteria_spatial_prompt"] = execution
        patched.model_options = dict(patched.model_options)
        patched.model_options["transformer_options"] = transformer_options
        return patched, positive, final_negative, _report(execution)


NODE_CLASS_MAPPINGS = {
    "AsteriaSpatialMaskCollector": AsteriaSpatialMaskCollector,
    "AsteriaSpatialPromptBinder": AsteriaSpatialPromptBinder,
    "AsteriaSpatialModelAdapter": AsteriaSpatialModelAdapter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AsteriaSpatialMaskCollector": "Asteria Spatial Mask Collector",
    "AsteriaSpatialPromptBinder": "Asteria Spatial Prompt Binder",
    "AsteriaSpatialModelAdapter": "Asteria Spatial Model Adapter",
}
