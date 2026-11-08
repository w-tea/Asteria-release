from __future__ import annotations

import re

from app.schemas.spatial_prompt_runtime import canonical_runtime_sha256


def validate_native_spatial_receipt(payload: dict) -> None:
    """Validate the new execution path without treating provisional quality as accepted."""
    def digest(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    def integer(value, minimum=0):
        return type(value) is int and value >= minimum

    body = {key: value for key, value in payload.items() if key != "execution_sha256"}
    runtime = payload["runtime"]
    count = {"sdxl_unet_v1": 24, "anima_base_28": 28,
             "anima_2_9b_40": 40, "anima_3_8b_52": 52}.get(payload["verified_structure_id"])
    targets = payload["target_ids"]
    full_hash = payload["full_text_sha256"]
    hashes = full_hash if isinstance(full_hash, list) else [full_hash]
    if (
        payload["schema_version"] != "asteria.spatial-node-execution.v9"
        or payload["contract_version"] != "asteria.spatial-prompt.node.v4"
        or payload["adapter_revision"] != "asteria-native-full-text-v9"
        or payload["stage"] != "base" or payload["hires_mode"] != "base_only"
        or payload["experimental"] is not True
        or count is None or payload["diffusion_block_count"] != count
        or payload["model_family"] not in ("sdxl", "anima")
        or (payload["model_family"] == "sdxl") != (count == 24)
        or not targets or len(set(targets)) != len(targets)
        or not hashes or not all(digest(h) for h in hashes)
        or not all(digest(payload[name]) for name in (
            "layout_fingerprint", "structure_fingerprint", "profile_receipt_sha256",
            "profile_payload_sha256", "mask_bundle_sha256", "token_binding_sha256", "geometry_sha256"))
        or len(payload["field_sha256"]) != len(targets)
        or not all(digest(h) for h in payload["field_sha256"])
        or canonical_runtime_sha256(body) != payload["execution_sha256"]
        or not integer(payload["initial_noise_seed"]) or payload["initial_noise_seed"] > 0xFFFFFFFFFFFFFFFF
        or not payload["latent_shape"] or not all(integer(n,1) for n in payload["latent_shape"])
        or runtime["router_revision"] != "asteria-native-full-text-v9"
        or not integer(runtime["attention_calls"],1)
        or runtime["full_attention_calls"] != runtime["attention_calls"]
        or not integer(runtime["routed_calls"]) or runtime["routed_calls"] > runtime["attention_calls"]
        or not all(integer(runtime[name]) for name in ("mask_cache_bytes", "peak_mask_cache_bytes", "cache_limit_bytes"))
        or not runtime["mask_cache_bytes"] <= runtime["peak_mask_cache_bytes"] <= runtime["cache_limit_bytes"] <= 16*1024*1024
        or not runtime["windows"]
        or payload["registered_patches"] != (["attn2_replace"] if count == 24 else ["native_cross_attn", "post_input"])
    ):
        raise ValueError("Native placement receipt mismatch")
    for window in runtime["windows"]:
        if len(window["columns"]) != len(targets) or not integer(window["context_length"],1):
            raise ValueError("Native placement token windows mismatch")
        indices = [i for span in window["columns"] for i in span]
        if len(indices) != len(set(indices)) or any(not integer(i) or i >= window["context_length"] for i in indices):
            raise ValueError("Native placement token columns mismatch")
