from __future__ import annotations

from dataclasses import dataclass
import json

import torch

from .mask_runtime import collect_mask_target, validate_mask_bundle
from .native_attention_runtime import AnimaCrossCall, NativeAttentionRouter, NativeWindow, shared_columns
from .profile_runtime import load_phase_profile
from .spatial_contract import CONTRACT_VERSION, canonical_json_sha256, validate_contract_version
from .token_binding_runtime import prepare_spatial_tokens, token_binding_payload


def _report(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class FullTextBinding:
    windows: tuple[NativeWindow, ...]
    target_ids: tuple[str, ...]
    full_text_sha256: object
    receipt_sha256: str


@dataclass(frozen=True)
class SpatialExecutionState:
    execution_body: dict
    router: NativeAttentionRouter

    def receipt(self, samples):
        body = {**self.execution_body, "runtime": self.router.snapshot(),
                "latent_shape": [int(v) for v in samples["samples"].shape]}
        self.router.clear()
        return _report({**body, "execution_sha256": canonical_json_sha256(body)})


class AsteriaSpatialMaskCollector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "target_id": ("STRING", {"default": ""}),
                "depth_index": ("INT", {"default": 0, "min": 0, "max": 4096}),
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


class AsteriaSpatialFullTextBinding:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",), "source_positive": ("CONDITIONING",),
            "binding_json": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            "family": (["sdxl", "anima"],),
            "contract_version": ([CONTRACT_VERSION],),
        }}

    RETURN_TYPES = ("CONDITIONING", "ASTERIA_SPATIAL_FULL_TEXT", "STRING")
    FUNCTION = "bind"
    CATEGORY = "Asteria/spatial"

    @staticmethod
    def bind(clip, source_positive, binding_json, family, contract_version):
        validate_contract_version(contract_version)
        payload = json.loads(binding_json)
        windows, evidence = [], []
        target_ids = None
        for item in payload["schedules"]:
            ranges = [{"target_id": r["target_id"], "character_start": r["start"],
                       "character_end": r["end"]} for r in item["character_bindings"]]
            current_ids = tuple(r["target_id"] for r in ranges)
            if target_ids is not None and current_ids != target_ids:
                raise ValueError("Spatial scheduled target order changed")
            target_ids = current_ids
            prepared = prepare_spatial_tokens(clip, full_text=item["full_text"], character_ranges=ranges,
                required_encoders=("l", "g") if family == "sdxl" else ("t5xxl",),
                adapter_kind="identity", step_start_percent=item["start_percent"],
                step_end_percent=item["end_percent"])
            bound = prepared.binding
            columns = shared_columns(bound.encoders)
            length = max(len(e.token_ids) for e in bound.encoders) if family == "sdxl" else 512
            windows.append(NativeWindow(item["start_percent"], item["end_percent"],
                                        bound.full_text_sha256, length, columns))
            evidence.append(token_binding_payload(bound))
        if not windows or not target_ids:
            raise ValueError("Spatial full-text binding is empty")
        hashes = [w.full_text_sha256 for w in windows]
        binding = FullTextBinding(tuple(windows), target_ids,
            hashes if payload["scheduled"] else hashes[0], canonical_json_sha256(evidence))
        # Tokenize only. Native full conditioning, pooled output and weights are untouched.
        return source_positive, binding, _report({"bindings": evidence, "receipt_sha256": binding.receipt_sha256})


class AsteriaSpatialNativeGuidance:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
            "binding": ("ASTERIA_SPATIAL_FULL_TEXT",), "masks": ("ASTERIA_SPATIAL_MASK_BUNDLE",),
            "geometry_json": ("STRING", {"multiline": True}),
            "profile_json": ("STRING", {"multiline": True}),
            "stage": (["base"],),
            "initial_noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            "contract_version": ([CONTRACT_VERSION],),
        }}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "ASTERIA_SPATIAL_GUIDANCE_STATE")
    FUNCTION = "apply"
    CATEGORY = "Asteria/spatial"

    @staticmethod
    def apply(model, positive, negative, binding, masks, geometry_json, profile_json,
              stage, initial_noise_seed, contract_version):
        validate_contract_version(contract_version)
        validate_mask_bundle(masks)
        profile = load_phase_profile(profile_json)
        if stage != "base" or profile.stage != stage or not isinstance(binding, FullTextBinding):
            raise ValueError("Invalid native placement stage")
        target_ids = tuple(t.target_id for t in masks.targets)
        if target_ids != binding.target_ids or target_ids != profile.target_ids:
            raise ValueError("Native placement target order mismatch")
        geometry = json.loads(geometry_json)
        by_id = {r["target_id"]: r for r in geometry}
        if set(by_id) != set(target_ids) or len(by_id) != len(geometry):
            raise ValueError("Native placement geometry targets mismatch")
        regions = []
        for target in masks.targets:
            region = dict(by_id[target.target_id], strength=target.strength/100.,
                          blending=target.blending, depth_index=target.depth_index)
            if region["shape"] == "mask":
                region["mask"] = target.mask.detach().cpu()
            regions.append(region)
        family = "sdxl" if profile.model_family == "illustrious" else profile.model_family
        router = NativeAttentionRouter(windows=binding.windows, regions=regions,
            family=family, model_sampling=model.get_model_object("model_sampling"))
        patched = model.clone()
        options = patched.model_options.get("transformer_options", {})
        if options.get("asteria_spatial_prompt"):
            raise ValueError("Placement is already applied to this model")
        patched.model_options["transformer_options"] = dict(options, asteria_spatial_prompt="native-full-text-v9")
        count = {"sdxl_unet_v1": 24, "anima_base_28": 28,
                 "anima_2_9b_40": 40, "anima_3_8b_52": 52}[profile.verified_structure_id]
        registered = []
        if profile.model_family == "anima":
            diffusion = model.get_model_object("diffusion_model")
            if len(diffusion.blocks) != count:
                raise ValueError("Native Anima block count changed")
            for index in range(count):
                path = f"diffusion_model.blocks.{index}.cross_attn.attn_op"
                original = model.get_model_object(path)
                patched.add_object_patch(path, AnimaCrossCall(router, index, original))
            patched.set_model_patch(router.post_input, "post_input")
            registered = ["native_cross_attn", "post_input"]
        else:
            existing = options.get("patches_replace", {}).get("attn2", {})
            for block, number in (("input",7),("input",8),("middle",0),("output",0),("output",1),("output",2)):
                for index in range(10):
                    key = (block,number,index)
                    if key in existing or (block,number) in existing:
                        raise ValueError("Another extension already replaces this attention operation")
                    patched.set_model_attn2_replace(router.sdxl,block,number,index)
            registered = ["attn2_replace"]
        body = dict(schema_version="asteria.spatial-node-execution.v9", contract_version=CONTRACT_VERSION,
            adapter_revision="asteria-native-full-text-v9", stage=stage, hires_mode="base_only",
            model_family=family, model_slot_id=profile.model_slot_id,
            encoder_kind=profile.encoder_kind, verified_structure_id=profile.verified_structure_id,
            diffusion_block_count=count, layout_fingerprint=profile.layout_fingerprint,
            structure_fingerprint=profile.structure_fingerprint, profile_id=profile.profile_id,
            profile_revision=profile.profile_revision, profile_receipt_sha256=profile.profile_receipt_sha256,
            profile_payload_sha256=profile.profile_payload_sha256, target_ids=list(target_ids),
            mask_bundle_sha256=masks.receipt_sha256, field_sha256=[t.mask_sha256 for t in masks.targets],
            full_text_sha256=binding.full_text_sha256, token_binding_sha256=binding.receipt_sha256,
            geometry_sha256=canonical_json_sha256(geometry), registered_patches=registered,
            initial_noise_seed=int(initial_noise_seed), experimental=True)
        return patched, positive, negative, SpatialExecutionState(body,router)


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
        if not isinstance(execution_state, SpatialExecutionState):
            raise ValueError("Asteria spatial execution state is invalid")
        receipt = execution_state.receipt(samples)
        return {
            "ui": {"asteria_spatial_receipts": [receipt]},
            "result": (receipt,),
        }


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (
    AsteriaSpatialMaskCollector, AsteriaSpatialFullTextBinding,
    AsteriaSpatialNativeGuidance, AsteriaSpatialExecutionReceipt,
)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AsteriaSpatialMaskCollector": "Asteria Spatial Mask Collector",
    "AsteriaSpatialFullTextBinding": "Asteria Spatial Full Text Binding",
    "AsteriaSpatialNativeGuidance": "Asteria Spatial Placement (Experimental)",
    "AsteriaSpatialExecutionReceipt": "Asteria Spatial Execution Receipt",
}
