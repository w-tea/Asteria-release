from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from typing import Any

import torch

from .attention_capture_runtime import SpatialAttentionCollector
from .attention_runtime import SpatialAttentionRouter
from .mask_runtime import collect_mask_target, validate_mask_bundle
from .profile_runtime import load_phase_profile
from .spatial_contract import (
    CONTRACT_VERSION,
    SpatialConditioningBundle,
    SpatialConditioningTarget,
    SpatialMaskBundle,
    SpatialTokenBindingBundle,
    SpatialScheduledTokenBindingBundle,
    canonical_json_sha256,
    clone_conditioning,
    conditioning_bundle_payload,
    validate_bundle_target_ids,
    validate_contract_version,
    validate_stage,
    validate_target_id,
)
from .token_binding_runtime import (
    prepare_spatial_token_schedule,
    prepare_spatial_tokens,
    scheduled_token_binding_payload,
    token_binding_payload,
)
from .sampler_runtime import sample_comfy_spatial_guidance


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


@dataclass(slots=True)
class SpatialGuidanceState:
    execution_body: dict[str, Any]
    collector: SpatialAttentionCollector
    profile: Any
    masks: SpatialMaskBundle
    token_binding: SpatialTokenBindingBundle | SpatialScheduledTokenBindingBundle
    solver_receipts: tuple[Any, ...] = ()
    loss_receipts: tuple[dict[str, Any], ...] = ()
    global_attention_baseline: Any | None = None

    def receipt(self, samples: Any) -> str:
        if not isinstance(samples, dict):
            raise ValueError("Asteria spatial sampler result is invalid")
        latent = samples.get("samples")
        if latent is None or not hasattr(latent, "shape"):
            raise ValueError("Asteria spatial sampler result is invalid")
        if not self.solver_receipts:
            raise ValueError("Asteria spatial sampler did not record its trajectory")
        body = {
            **self.execution_body,
            "solver_steps": [asdict(item) for item in self.solver_receipts],
            "loss_steps": list(self.loss_receipts),
            "sampler_step_count": len(self.solver_receipts),
            "sampler_consumption_count": sum(
                int(item.sampler_consumption_count) for item in self.solver_receipts
            ),
            "latent_shape": [int(value) for value in latent.shape],
        }
        return _report({**body, "execution_sha256": canonical_json_sha256(body)})


def _validate_token_binding_targets(
    token_binding: SpatialTokenBindingBundle | SpatialScheduledTokenBindingBundle,
    target_ids: tuple[str, ...],
) -> None:
    if isinstance(token_binding, SpatialScheduledTokenBindingBundle):
        if not token_binding.bindings:
            raise ValueError("Asteria spatial token binding schedule is empty")
        for binding in token_binding.bindings:
            _validate_token_binding_targets(binding, target_ids)
        return
    if not isinstance(token_binding, SpatialTokenBindingBundle):
        raise ValueError("Asteria spatial token binding is invalid")
    if not token_binding.encoders:
        raise ValueError("Asteria spatial token binding has no encoders")
    for encoder in token_binding.encoders:
        if tuple(target.target_id for target in encoder.targets) != target_ids:
            raise ValueError("Asteria spatial token binding target ids do not match")


class AsteriaSpatialGuidanceAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "full_positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "token_binding": ("ASTERIA_SPATIAL_TOKEN_BINDING",),
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
        "ASTERIA_SPATIAL_GUIDANCE_STATE",
    )
    RETURN_NAMES = ("model", "positive", "negative", "guidance_state")
    FUNCTION = "apply"
    CATEGORY = "Asteria/spatial"

    @staticmethod
    def apply(
        model,
        full_positive,
        negative,
        token_binding,
        masks,
        profile_json,
        stage,
        contract_version,
    ):
        validate_contract_version(contract_version)
        normalized_stage = validate_stage(stage)
        validate_mask_bundle(masks)
        target_ids = tuple(target.target_id for target in masks.targets)
        _validate_token_binding_targets(token_binding, target_ids)
        profile = load_phase_profile(profile_json)
        if profile.stage != normalized_stage:
            raise ValueError("Asteria spatial profile stage does not match")
        if tuple(profile.target_ids) != target_ids:
            raise ValueError("Asteria spatial profile target ids do not match")
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
        positive = clone_conditioning(full_positive)
        final_negative = clone_conditioning(negative)
        layout_masks = []
        for target in masks.targets:
            layout_mask = target.mask.detach().to(device="cpu", dtype=torch.float32)
            if layout_mask.ndim == 3:
                if int(layout_mask.shape[0]) != 1:
                    raise ValueError(
                        "Asteria spatial layout mask batch must contain one field"
                    )
                layout_mask = layout_mask[0]
            layout_masks.append(layout_mask)
        layout_fields = torch.stack(layout_masks)
        collector = SpatialAttentionCollector(
            token_binding=token_binding,
            target_ids=target_ids,
            layout_fields=layout_fields,
            prior_attention_override=source_transformer_options.get(
                "optimized_attention_override"
            ),
            attention_layers=profile.attention_layers,
            self_relation_layers=profile.self_relation_layers,
            attention_prior_scale=profile.attention_prior_scale,
        )
        execution_body = {
            "schema_version": "asteria.spatial-node-execution.v3",
            "contract_version": CONTRACT_VERSION,
            "adapter_revision": "asteria-layout-latent-guidance-v1",
            "routing_mode": "single_full_text_latent_guidance",
            "stage": normalized_stage,
            "profile_receipt_sha256": profile.profile_receipt_sha256,
            "profile_payload_sha256": profile.profile_payload_sha256,
            "mask_bundle_sha256": masks.receipt_sha256,
            "token_binding_sha256": token_binding.receipt_sha256,
            "full_text_sha256": (
                [item.full_text_sha256 for item in token_binding.bindings]
                if isinstance(token_binding, SpatialScheduledTokenBindingBundle)
                else token_binding.full_text_sha256
            ),
            "scheduled_binding_count": (
                len(token_binding.bindings)
                if isinstance(token_binding, SpatialScheduledTokenBindingBundle)
                else 0
            ),
            "target_ids": list(target_ids),
            "registered_patches": [
                "attn1_patch",
                "attn2_patch",
                "optimized_attention_override",
            ],
            "target_latent_count": 0,
            "target_noise_prediction_count": 0,
            "target_denoised_output_count": 0,
            "target_only_attention_forward_count": 0,
            "global_only_attention_forward_count": 0,
            "spatial_output_blend_count": 0,
            "background_substitution_count": 0,
        }
        patched = model.clone()
        if not isinstance(getattr(patched, "model_options", None), dict):
            raise ValueError("Asteria spatial model options are invalid")
        patched.model_options = dict(patched.model_options)
        transformer_options = deepcopy(
            patched.model_options.get("transformer_options", {})
        )
        transformer_options["asteria_spatial_prompt"] = execution_body
        patched.model_options["transformer_options"] = transformer_options
        for method_name in ("set_model_attn1_patch", "set_model_attn2_patch"):
            if not hasattr(patched, method_name):
                raise ValueError("Asteria spatial model patch API is unavailable")
        patched.set_model_attn1_patch(collector.self_patch)
        patched.set_model_attn2_patch(collector.cross_patch)
        patched.model_options["transformer_options"][
            "optimized_attention_override"
        ] = collector.attention_override
        return (
            patched,
            positive,
            final_negative,
            SpatialGuidanceState(
                execution_body=execution_body,
                collector=collector,
                profile=profile,
                masks=masks,
                token_binding=token_binding,
            ),
        )


class AsteriaSpatialGuidanceSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (["euler_ancestral", "dpmpp_2m", "er_sde"],),
                "scheduler": (
                    [
                        "normal",
                        "karras",
                        "exponential",
                        "polyexponential",
                        "simple",
                        "uniform",
                        "sgm_uniform",
                        "linear_quadratic",
                        "kl_optimal",
                        "ddim",
                        "beta",
                        "turbo",
                    ],
                ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "guidance_state": ("ASTERIA_SPATIAL_GUIDANCE_STATE",),
                "contract_version": ([CONTRACT_VERSION],),
            },
            "optional": {
                "noise_opt": ("NOISE_IMAGE",),
                "add_noise": (["enable", "disable"],),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
            },
        }

    RETURN_TYPES = ("LATENT", "ASTERIA_SPATIAL_GUIDANCE_STATE")
    RETURN_NAMES = ("samples", "guidance_state")
    FUNCTION = "sample"
    CATEGORY = "Asteria/spatial"

    @staticmethod
    def sample(
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise,
        guidance_state,
        contract_version=CONTRACT_VERSION,
        noise_opt=None,
        add_noise="enable",
        start_at_step=0,
        end_at_step=10000,
        return_with_leftover_noise="disable",
    ):
        validate_contract_version(contract_version)
        if not isinstance(guidance_state, SpatialGuidanceState):
            raise ValueError("Asteria spatial guidance state is invalid")
        return sample_comfy_spatial_guidance(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent=latent_image,
            denoise=denoise,
            guidance_state=guidance_state,
            noise_opt=noise_opt,
            add_noise=add_noise,
            start_at_step=start_at_step,
            end_at_step=end_at_step,
            return_with_leftover_noise=return_with_leftover_noise,
        )


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


class AsteriaSpatialFullPromptEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "full_text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "character_ranges_json": ("STRING", {"multiline": True, "default": "[]"}),
                "required_encoders_json": ("STRING", {"default": '["l","g"]'}),
                "adapter_kind": (
                    ["identity", "anima_native_t5_query_v1"],
                ),
                "step_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "step_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "contract_version": ([CONTRACT_VERSION],),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "ASTERIA_SPATIAL_TOKEN_BINDING", "STRING")
    RETURN_NAMES = ("conditioning", "token_binding", "receipt")
    FUNCTION = "encode"
    CATEGORY = "Asteria/spatial"

    @staticmethod
    def encode(
        clip,
        full_text,
        character_ranges_json,
        required_encoders_json,
        adapter_kind,
        step_start_percent,
        step_end_percent,
        contract_version,
    ):
        validate_contract_version(contract_version)
        try:
            character_ranges = json.loads(character_ranges_json)
            required_encoders = json.loads(required_encoders_json)
        except json.JSONDecodeError as error:
            raise ValueError("Asteria spatial token binding JSON is invalid") from error
        if not isinstance(character_ranges, list) or not isinstance(required_encoders, list):
            raise ValueError("Asteria spatial token binding JSON is invalid")
        prepared = prepare_spatial_tokens(
            clip,
            full_text=full_text,
            character_ranges=character_ranges,
            required_encoders=tuple(required_encoders),
            adapter_kind=adapter_kind,
            step_start_percent=step_start_percent,
            step_end_percent=step_end_percent,
        )
        encode = getattr(clip, "encode_from_tokens_scheduled", None)
        if not callable(encode):
            raise ValueError("Asteria spatial text encoder is unavailable")
        conditioning = encode(prepared.tokens)
        payload = token_binding_payload(prepared.binding)
        return conditioning, prepared.binding, _report(payload)


class AsteriaSpatialTokenBinding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "clip": ("CLIP",),
                "full_text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "character_ranges_json": ("STRING", {"multiline": True, "default": "[]"}),
                "required_encoders_json": ("STRING", {"default": '["l","g"]'}),
                "adapter_kind": (
                    [
                        "identity",
                        "anima_native_t5_query_v1",
                        "anima_progressive_qwen35_t5_query_v1",
                    ],
                ),
                "step_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "step_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "contract_version": ([CONTRACT_VERSION],),
            },
            "optional": {
                "scheduled_bindings_json": ("STRING", {"multiline": True, "default": "[]"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "ASTERIA_SPATIAL_TOKEN_BINDING", "STRING")
    RETURN_NAMES = ("conditioning", "token_binding", "receipt")
    FUNCTION = "bind"
    CATEGORY = "Asteria/spatial"

    @staticmethod
    def bind(
        conditioning,
        clip,
        full_text,
        character_ranges_json,
        required_encoders_json,
        adapter_kind,
        step_start_percent,
        step_end_percent,
        contract_version,
        scheduled_bindings_json="[]",
    ):
        validate_contract_version(contract_version)
        try:
            character_ranges = json.loads(character_ranges_json)
            required_encoders = json.loads(required_encoders_json)
            scheduled_bindings = json.loads(scheduled_bindings_json)
        except json.JSONDecodeError as error:
            raise ValueError("Asteria spatial token binding JSON is invalid") from error
        if (
            not isinstance(character_ranges, list)
            or not isinstance(required_encoders, list)
            or not isinstance(scheduled_bindings, list)
        ):
            raise ValueError("Asteria spatial token binding JSON is invalid")
        if scheduled_bindings:
            schedule = prepare_spatial_token_schedule(
                clip,
                scheduled_bindings=scheduled_bindings,
                required_encoders=tuple(required_encoders),
                adapter_kind=adapter_kind,
            )
            return (
                clone_conditioning(conditioning),
                schedule,
                _report(scheduled_token_binding_payload(schedule)),
            )
        prepared = prepare_spatial_tokens(
            clip,
            full_text=full_text,
            character_ranges=character_ranges,
            required_encoders=tuple(required_encoders),
            adapter_kind=adapter_kind,
            step_start_percent=step_start_percent,
            step_end_percent=step_end_percent,
        )
        return (
            clone_conditioning(conditioning),
            prepared.binding,
            _report(token_binding_payload(prepared.binding)),
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
            "adapter_revision": "asteria-layout-attention-v3",
            "routing_mode": "cross_and_self_attention",
            "instance_formation_mode": "mask_anchored_early_cohesion_and_isolation",
            "regional_binding_mode": "unconditional_background_then_global_refinement",
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
                "execution_state": ("ASTERIA_SPATIAL_GUIDANCE_STATE",),
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
        if not isinstance(execution_state, (SpatialExecutionState, SpatialGuidanceState)):
            raise ValueError("Asteria spatial execution state is invalid")
        receipt = execution_state.receipt(samples)
        return {
            "ui": {"asteria_spatial_receipts": [receipt]},
            "result": (receipt,),
        }


NODE_CLASS_MAPPINGS = {
    "AsteriaSpatialMaskCollector": AsteriaSpatialMaskCollector,
    "AsteriaSpatialPromptBinder": AsteriaSpatialPromptBinder,
    "AsteriaSpatialFullPromptEncoder": AsteriaSpatialFullPromptEncoder,
    "AsteriaSpatialTokenBinding": AsteriaSpatialTokenBinding,
    "AsteriaSpatialModelAdapter": AsteriaSpatialModelAdapter,
    "AsteriaSpatialGuidanceAdapter": AsteriaSpatialGuidanceAdapter,
    "AsteriaSpatialGuidanceSampler": AsteriaSpatialGuidanceSampler,
    "AsteriaSpatialExecutionReceipt": AsteriaSpatialExecutionReceipt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AsteriaSpatialMaskCollector": "Asteria Spatial Mask Collector",
    "AsteriaSpatialPromptBinder": "Asteria Spatial Prompt Binder",
    "AsteriaSpatialFullPromptEncoder": "Asteria Spatial Full Prompt Encoder",
    "AsteriaSpatialTokenBinding": "Asteria Spatial Token Binding",
    "AsteriaSpatialModelAdapter": "Asteria Spatial Model Adapter",
    "AsteriaSpatialGuidanceAdapter": "Asteria Spatial Guidance Adapter",
    "AsteriaSpatialGuidanceSampler": "Asteria Spatial Guidance Sampler",
    "AsteriaSpatialExecutionReceipt": "Asteria Spatial Execution Receipt",
}
