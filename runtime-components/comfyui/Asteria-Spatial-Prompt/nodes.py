from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

import torch

from .attention_runtime import SpatialAttentionRouter
from .mask_runtime import collect_mask_target, validate_mask_bundle
from .profile_runtime import load_phase_profile
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


@dataclass(frozen=True, slots=True)
class SpatialExecutionState:
    execution_body: dict[str, Any]
    router: SpatialAttentionRouter

    def receipt(self, samples: Any) -> str:
        if not isinstance(samples, dict):
            raise ValueError("Asteria spatial sampler result is invalid")
        latent = samples.get("samples")
        if latent is None or not hasattr(latent, "shape"):
            raise ValueError("Asteria spatial sampler result is invalid")
        body = {
            **self.execution_body,
            "runtime": self.router.snapshot(),
            "latent_shape": [int(value) for value in latent.shape],
        }
        return _report({**body, "execution_sha256": canonical_json_sha256(body)})


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


def _anima_text_preprocessor(model: Any, bundle: SpatialConditioningBundle):
    for target in bundle.targets:
        for _, metadata in target.conditioning:
            if not isinstance(metadata.get("t5xxl_ids"), torch.Tensor):
                raise ValueError("Asteria Anima spatial conditioning requires t5xxl_ids")
    base_model = getattr(model, "model", None)
    diffusion_model = getattr(base_model, "diffusion_model", None)
    preprocessor = getattr(diffusion_model, "preprocess_text_embeds", None)
    if not callable(preprocessor):
        raise ValueError("Asteria Anima spatial text preprocessor is unavailable")
    return preprocessor


def _validate_progressive_qwen_conditioning(bundle: SpatialConditioningBundle) -> None:
    expected = "anima_progressive_qwen35_cross_adapter_v1"
    for target in bundle.targets:
        for _, metadata in target.conditioning:
            if (
                metadata.get("qwen35_expanded_architecture") != expected
                or not isinstance(metadata.get("qwen35_expanded_adapter"), str)
                or not metadata["qwen35_expanded_adapter"].strip()
            ):
                raise ValueError(
                    "Asteria Anima progressive Qwen3.5 conditioning provenance is invalid"
                )


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

    RETURN_TYPES = (
        "MODEL",
        "CONDITIONING",
        "CONDITIONING",
        "ASTERIA_SPATIAL_EXECUTION_STATE",
    )
    RETURN_NAMES = ("model", "positive", "negative", "execution_state")
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
        is_anima = profile.model_family == "anima"
        uses_native_anima_preprocessor = is_anima and profile.encoder_kind == "native_qwen"
        uses_progressive_qwen = is_anima and profile.encoder_kind == "qwen35_progressive"
        if is_anima and not (uses_native_anima_preprocessor or uses_progressive_qwen):
            raise ValueError("Asteria Anima spatial encoder kind is unsupported")
        if uses_progressive_qwen:
            _validate_progressive_qwen_conditioning(conditions)
        anima_text_preprocessor = (
            _anima_text_preprocessor(model, conditions)
            if uses_native_anima_preprocessor
            else None
        )
        registered_patches = [
            "attn1_patch",
            "attn2_patch",
            "optimized_attention_override",
        ]
        if uses_native_anima_preprocessor:
            registered_patches.insert(0, "post_input")
        preprocess_mode = "direct"
        if uses_native_anima_preprocessor:
            preprocess_mode = "native_anima_llm_adapter"
        elif uses_progressive_qwen:
            preprocess_mode = "progressive_qwen35_preprocessed"
        execution_body = {
            "schema_version": "asteria.spatial-node-execution.v2",
            "contract_version": CONTRACT_VERSION,
            "adapter_revision": "asteria-layout-attention-v2",
            "routing_mode": "cross_and_self_attention",
            "conditioning_preprocess_mode": preprocess_mode,
            "stage": normalized_stage,
            "profile_receipt_sha256": profile.profile_receipt_sha256,
            "profile_payload_sha256": profile.profile_payload_sha256,
            "mask_bundle_sha256": masks.receipt_sha256,
            "conditioning_bundle_sha256": conditions.receipt_sha256,
            "target_ids": mask_ids,
            "registered_patches": registered_patches,
            "coefficients": {
                "early_guidance_end": profile.early_guidance_end,
                "isolation_start": profile.isolation_start,
                "isolation_end": profile.isolation_end,
                "late_blending_start": profile.late_blending_start,
                "schedule_smoothing": profile.schedule_smoothing,
                "cross_inside_bias": profile.cross_inside_bias,
                "cross_outside_penalty": profile.cross_outside_penalty,
                "global_inside_penalty": profile.global_inside_penalty,
                "self_leakage_penalty": profile.self_leakage_penalty,
                "self_attention_token_limit": profile.self_attention_token_limit,
            },
        }
        patched = model.clone()
        if not isinstance(getattr(patched, "model_options", None), dict):
            raise ValueError("Asteria spatial model options are invalid")
        transformer_options = deepcopy(
            patched.model_options.get("transformer_options", {})
        )
        transformer_options["asteria_spatial_prompt"] = execution_body
        patched.model_options = dict(patched.model_options)
        patched.model_options["transformer_options"] = transformer_options
        for method_name in (
            "set_model_attn1_patch",
            "set_model_attn2_patch",
        ):
            if not hasattr(patched, method_name):
                raise ValueError("Asteria spatial model patch API is unavailable")
        router = SpatialAttentionRouter(
            conditions,
            masks,
            profile,
            prior_attention_override=source_transformer_options.get(
                "optimized_attention_override"
            ),
            anima_text_preprocessor=anima_text_preprocessor,
        )
        if uses_native_anima_preprocessor:
            if not hasattr(patched, "set_model_post_input_patch"):
                raise ValueError("Asteria Anima spatial post-input patch API is unavailable")
            patched.set_model_post_input_patch(router.prepare_anima_contexts)
        patched.set_model_attn1_patch(router.self_patch)
        patched.set_model_attn2_patch(router.cross_patch)
        patched.model_options["transformer_options"][
            "optimized_attention_override"
        ] = router.attention_override
        return (
            patched,
            positive,
            final_negative,
            SpatialExecutionState(execution_body=execution_body, router=router),
        )


class AsteriaSpatialExecutionReceipt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "execution_state": ("ASTERIA_SPATIAL_EXECUTION_STATE",),
                "samples": ("LATENT",),
                "contract_version": ([CONTRACT_VERSION],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("execution_receipt",)
    FUNCTION = "finalize"
    CATEGORY = "Asteria/spatial"
    OUTPUT_NODE = True

    @staticmethod
    def finalize(execution_state, samples, contract_version=CONTRACT_VERSION):
        validate_contract_version(contract_version)
        if not isinstance(execution_state, SpatialExecutionState):
            raise ValueError("Asteria spatial execution state is invalid")
        receipt = execution_state.receipt(samples)
        return {
            "ui": {"asteria_spatial_receipts": [receipt]},
            "result": (receipt,),
        }


NODE_CLASS_MAPPINGS = {
    "AsteriaSpatialMaskCollector": AsteriaSpatialMaskCollector,
    "AsteriaSpatialPromptBinder": AsteriaSpatialPromptBinder,
    "AsteriaSpatialModelAdapter": AsteriaSpatialModelAdapter,
    "AsteriaSpatialExecutionReceipt": AsteriaSpatialExecutionReceipt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AsteriaSpatialMaskCollector": "Asteria Spatial Mask Collector",
    "AsteriaSpatialPromptBinder": "Asteria Spatial Prompt Binder",
    "AsteriaSpatialModelAdapter": "Asteria Spatial Model Adapter",
    "AsteriaSpatialExecutionReceipt": "Asteria Spatial Execution Receipt",
}
