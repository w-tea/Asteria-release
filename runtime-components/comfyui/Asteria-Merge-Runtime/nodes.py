from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import torch

try:
    from .merge_math import (
        MergeRuntimeMathError,
        cosine_context_for_pairs,
        merge_single_tensor,
        validate_payload,
    )
except ImportError:
    _math_path = Path(__file__).with_name("merge_math.py")
    _math_spec = importlib.util.spec_from_file_location("asteria_merge_runtime_math", _math_path)
    if _math_spec is None or _math_spec.loader is None:
        raise
    _math_module = importlib.util.module_from_spec(_math_spec)
    _math_spec.loader.exec_module(_math_module)
    MergeRuntimeMathError = _math_module.MergeRuntimeMathError
    cosine_context_for_pairs = _math_module.cosine_context_for_pairs
    merge_single_tensor = _math_module.merge_single_tensor
    validate_payload = _math_module.validate_payload

CONTRACT_VERSION = "asteria.merge.runtime.v1"


def _contract_input() -> tuple[list[str]]:
    return ([CONTRACT_VERSION],)


class AsteriaSDXLMergeRuntime:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_a": ("MODEL",),
                "model_b": ("MODEL",),
                "clip_a": ("CLIP",),
                "clip_b": ("CLIP",),
                "vae_a": ("VAE",),
                "vae_b": ("VAE",),
                "recipe_payload": ("STRING", {"multiline": False}),
                "contract_version": _contract_input(),
            },
            "optional": {
                "model_c": ("MODEL",),
                "clip_c": ("CLIP",),
                "vae_c": ("VAE",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "merge"
    CATEGORY = "Asteria/runtime"

    def merge(
        self,
        model_a,
        model_b,
        clip_a,
        clip_b,
        vae_a,
        vae_b,
        recipe_payload,
        contract_version,
        model_c=None,
        clip_c=None,
        vae_c=None,
    ):
        payload = _payload(recipe_payload, contract_version, "sdxl")
        return (
            _merge_patcher(model_a, model_b, model_c, payload, "model"),
            _merge_patcher(clip_a, clip_b, clip_c, payload, "clip"),
            _merge_vae(vae_a, vae_b, vae_c, payload),
        )


class AsteriaAnimaMergeRuntime:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_a": ("MODEL",),
                "model_b": ("MODEL",),
                "recipe_payload": ("STRING", {"multiline": False}),
                "contract_version": _contract_input(),
            },
            "optional": {"model_c": ("MODEL",)},
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "merge"
    CATEGORY = "Asteria/runtime"

    def merge(self, model_a, model_b, recipe_payload, contract_version, model_c=None):
        payload = _payload(recipe_payload, contract_version, "anima")
        return (_merge_patcher(model_a, model_b, model_c, payload, "model"),)


class AsteriaCleanupFence:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"contract_version": _contract_input()}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("acknowledgement",)
    FUNCTION = "acknowledge"
    OUTPUT_NODE = True
    CATEGORY = "Asteria/runtime"

    def acknowledge(self, contract_version):
        return (CONTRACT_VERSION,)


def _payload(recipe_payload: str, contract_version: str, expected_family: str) -> dict:
    if contract_version != CONTRACT_VERSION:
        raise MergeRuntimeMathError("incompatible merge runtime contract")
    try:
        payload = json.loads(recipe_payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise MergeRuntimeMathError("invalid merge runtime recipe") from exc
    if not isinstance(payload, dict):
        raise MergeRuntimeMathError("invalid merge runtime recipe")
    validate_payload(payload)
    if payload.get("model_family") != expected_family:
        raise MergeRuntimeMathError("merge runtime family does not match node")
    return payload


def _patch_entries(owner, prefix: str | None = None) -> dict:
    return owner.get_key_patches(prefix) if prefix is not None else owner.get_key_patches()


def _materialize_entry(key: str, entry) -> torch.Tensor:
    if not isinstance(entry, list) or not entry or not isinstance(entry[0], tuple):
        raise MergeRuntimeMathError("unsupported source patch format")
    base, convert = entry[0]
    if not isinstance(base, torch.Tensor):
        raise MergeRuntimeMathError("source patch does not contain a tensor")
    value = convert(base.detach().clone(), inplace=True)
    if len(entry) > 1:
        import comfy.lora

        value = comfy.lora.calculate_weight(entry[1:], value, key)
    return value.detach().clone().cpu()


def _merge_patcher(source_a, source_b, source_c, payload: dict, domain: str):
    prefix = "diffusion_model." if domain == "model" else None
    entries_a = _patch_entries(source_a, prefix)
    entries_b = _patch_entries(source_b, prefix)
    entries_c = _patch_entries(source_c, prefix) if source_c is not None else None
    context = cosine_context_for_pairs(
        payload,
        (
            (key, _materialize_entry(key, entry_a), _materialize_entry(key, entries_b[key]))
            for key, entry_a in entries_a.items()
            if key in entries_b
        ),
        domain=domain,
    )
    candidate = source_a.clone()
    for tensor_index, (key, entry_a) in enumerate(entries_a.items()):
        tensor_a = _materialize_entry(key, entry_a)
        tensor_b = _materialize_entry(key, entries_b[key]) if key in entries_b else None
        tensor_c = _materialize_entry(key, entries_c[key]) if entries_c is not None and key in entries_c else None
        merged = merge_single_tensor(
            payload,
            key,
            tensor_a,
            tensor_b,
            tensor_c,
            domain=domain,
            tensor_index=tensor_index,
            cosine_context=context,
        )
        if not torch.equal(merged, tensor_a):
            candidate.add_patches({key: ("set", (merged,))}, 1.0, 1.0)
    return candidate


def _merge_vae(source_a, source_b, source_c, payload: dict):
    candidate = copy.copy(source_a)
    candidate.patcher = _merge_patcher(source_a.patcher, source_b.patcher, source_c.patcher if source_c is not None else None, payload, "vae")
    return candidate


NODE_CLASS_MAPPINGS = {
    "AsteriaSDXLMergeRuntime": AsteriaSDXLMergeRuntime,
    "AsteriaAnimaMergeRuntime": AsteriaAnimaMergeRuntime,
    "AsteriaCleanupFence": AsteriaCleanupFence,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AsteriaSDXLMergeRuntime": "Asteria SDXL Merge Runtime",
    "AsteriaAnimaMergeRuntime": "Asteria Anima Merge Runtime",
    "AsteriaCleanupFence": "Asteria Cleanup Fence",
}
