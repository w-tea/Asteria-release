from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import comfy.sd
import comfy.utils
import folder_paths

try:
    from .remap_contract import load_registry, remap_lora_state_dict
except ImportError:
    _contract_path = Path(__file__).with_name("remap_contract.py")
    _contract_spec = importlib.util.spec_from_file_location(
        "asteria_anima_lora_remap_contract", _contract_path
    )
    if _contract_spec is None or _contract_spec.loader is None:
        raise
    _contract = importlib.util.module_from_spec(_contract_spec)
    _contract_spec.loader.exec_module(_contract)
    load_registry = _contract.load_registry
    remap_lora_state_dict = _contract.remap_lora_state_dict


_REGISTRY_PATH = Path(__file__).with_name("lora-block-remap.v1.json")
_LAYOUT_IDS = ["anima_base_28", "anima_2_9b_40", "anima_3_8b_52"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


class AsteriaAnimaLoraRemap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "expected_lora_sha256": ("STRING", {"multiline": False}),
                "source_layout_id": (_LAYOUT_IDS,),
                "target_layout_id": (_LAYOUT_IDS,),
                "mode": (["preserve"],),
                "mapping_revision": ("INT", {"default": 1, "min": 1, "max": 1}),
                "mapping_hash": ("STRING", {"multiline": False}),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "remap_receipt")
    FUNCTION = "apply"
    CATEGORY = "Asteria/runtime"

    def apply(
        self,
        model,
        clip,
        lora_name,
        expected_lora_sha256,
        source_layout_id,
        target_layout_id,
        mode,
        mapping_revision,
        mapping_hash,
        strength_model,
        strength_clip,
    ):
        lora_path = Path(
            folder_paths.get_full_path_or_raise("loras", lora_name)
        )
        actual_sha256 = _sha256(lora_path)
        if actual_sha256 != expected_lora_sha256:
            raise ValueError("anima_lora_source_fingerprint_changed")
        registry, bundled_hash = load_registry(_REGISTRY_PATH)
        if bundled_hash != mapping_hash:
            raise ValueError("anima_lora_mapping_hash_mismatch")
        state_dict, metadata = comfy.utils.load_torch_file(
            str(lora_path),
            safe_load=True,
            return_metadata=True,
        )
        remapped, receipt = remap_lora_state_dict(
            state_dict,
            source_layout_id=source_layout_id,
            target_layout_id=target_layout_id,
            mode=mode,
            expected_revision=int(mapping_revision),
            expected_mapping_hash=mapping_hash,
            registry=registry,
        )
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model,
            clip,
            remapped,
            float(strength_model),
            float(strength_clip),
            lora_metadata=metadata,
        )
        receipt["source_fingerprint"] = actual_sha256
        return (
            model_lora,
            clip_lora,
            json.dumps(
                receipt,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )


NODE_CLASS_MAPPINGS = {"AsteriaAnimaLoraRemap": AsteriaAnimaLoraRemap}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AsteriaAnimaLoraRemap": "Asteria Anima LoRA Compatibility"
}
