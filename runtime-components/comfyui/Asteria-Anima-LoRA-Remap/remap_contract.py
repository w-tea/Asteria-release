from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


_DIFFUSION_BLOCK = re.compile(r"^diffusion_model\.blocks\.(\d+)\.(.+)$")
_LLM_ADAPTER_BLOCK = re.compile(
    r"^diffusion_model\.llm_adapter\.blocks\.(\d+)\."
)


class AnimaLoraRemapError(ValueError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def load_registry(path: Path) -> tuple[dict, str]:
    try:
        raw = path.read_bytes()
        registry = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnimaLoraRemapError(
            "anima_lora_mapping_registry_unreadable"
        ) from exc
    canonical = _canonical_bytes(registry)
    if raw != canonical:
        raise AnimaLoraRemapError("anima_lora_mapping_registry_not_canonical")
    return registry, f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _layout(registry: dict, layout_id: str) -> dict:
    layout = next(
        (
            value
            for value in registry.get("layouts", [])
            if value.get("id") == layout_id
        ),
        None,
    )
    if not isinstance(layout, dict):
        raise AnimaLoraRemapError("anima_lora_layout_unknown")
    return layout


def _validate_source_state(
    state_dict: dict[str, object],
    registry: dict,
    source_layout: dict,
) -> None:
    allowed = tuple(registry.get("allowed_key_prefixes", ()))
    if not state_dict:
        raise AnimaLoraRemapError("anima_lora_state_dict_empty")
    if not allowed or any(
        not isinstance(key, str) or not key.startswith(allowed)
        for key in state_dict
    ):
        raise AnimaLoraRemapError("anima_lora_key_prefix_not_allowed")
    diffusion_ordinals = {
        int(match.group(1))
        for key in state_dict
        if (match := _DIFFUSION_BLOCK.match(key)) is not None
    }
    expected_diffusion = int(source_layout["diffusion_block_count"])
    if diffusion_ordinals != set(range(expected_diffusion)):
        raise AnimaLoraRemapError("anima_lora_source_layout_mismatch")
    llm_ordinals = {
        int(match.group(1))
        for key in state_dict
        if (match := _LLM_ADAPTER_BLOCK.match(key)) is not None
    }
    expected_llm = int(source_layout["llm_adapter_block_count"])
    if llm_ordinals != set(range(expected_llm)):
        raise AnimaLoraRemapError("anima_lora_llm_adapter_layout_mismatch")


def remap_lora_state_dict(
    state_dict: dict[str, object],
    *,
    source_layout_id: str,
    target_layout_id: str,
    mode: str,
    expected_revision: int,
    expected_mapping_hash: str,
    registry: dict,
) -> tuple[dict[str, object], dict[str, object]]:
    canonical_hash = f"sha256:{hashlib.sha256(_canonical_bytes(registry)).hexdigest()}"
    if int(registry.get("revision", 0)) != int(expected_revision):
        raise AnimaLoraRemapError("anima_lora_mapping_revision_mismatch")
    if canonical_hash != expected_mapping_hash:
        raise AnimaLoraRemapError("anima_lora_mapping_hash_mismatch")
    if mode not in {"preserve", "expand"}:
        raise AnimaLoraRemapError("anima_lora_mapping_mode_unknown")

    source_layout = _layout(registry, source_layout_id)
    target_layout = _layout(registry, target_layout_id)
    _validate_source_state(state_dict, registry, source_layout)
    source_count = int(source_layout["diffusion_block_count"])
    target_count = int(target_layout["diffusion_block_count"])

    if source_layout_id == target_layout_id:
        if mode == "expand":
            raise AnimaLoraRemapError("anima_lora_expand_disabled")
        return dict(state_dict), {
            "source_layout_id": source_layout_id,
            "target_layout_id": target_layout_id,
            "mode": mode,
            "mapping_revision": expected_revision,
            "mapping_hash": canonical_hash,
            "applied_source_blocks": source_count,
            "unapplied_target_blocks": 0,
        }

    direction = next(
        (
            value
            for value in registry.get("directions", [])
            if value.get("source_layout_id") == source_layout_id
            and value.get("target_layout_id") == target_layout_id
        ),
        None,
    )
    if not isinstance(direction, dict):
        raise AnimaLoraRemapError("anima_lora_mapping_direction_unknown")
    if direction.get("status") != "enabled":
        raise AnimaLoraRemapError("anima_lora_mapping_direction_disabled")
    if mode == "preserve" and not direction.get("preserve_enabled"):
        raise AnimaLoraRemapError("anima_lora_preserve_disabled")
    if mode == "expand" and not direction.get("expand_enabled"):
        raise AnimaLoraRemapError("anima_lora_expand_disabled")

    retained = tuple(int(value) for value in direction["retained_target_by_source"])
    if len(retained) != source_count or len(set(retained)) != source_count:
        raise AnimaLoraRemapError("anima_lora_mapping_registry_invalid")
    output: dict[str, object] = {}
    for key, value in state_dict.items():
        match = _DIFFUSION_BLOCK.match(key)
        if match is None:
            output[key] = value
            continue
        source_ordinal = int(match.group(1))
        target_ordinal = retained[source_ordinal]
        remapped_key = (
            f"diffusion_model.blocks.{target_ordinal}.{match.group(2)}"
        )
        if remapped_key in output:
            raise AnimaLoraRemapError("anima_lora_mapping_key_collision")
        output[remapped_key] = value

    if mode == "expand":
        for mapping in direction.get("inserted_source_by_target", []):
            source_ordinal = int(mapping["source"])
            target_ordinal = int(mapping["target"])
            prefix = f"diffusion_model.blocks.{source_ordinal}."
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    expanded_key = (
                        f"diffusion_model.blocks.{target_ordinal}."
                        f"{key[len(prefix):]}"
                    )
                    if expanded_key in output:
                        raise AnimaLoraRemapError(
                            "anima_lora_mapping_key_collision"
                        )
                    output[expanded_key] = value

    unapplied = len(direction.get("inserted_targets", [])) if mode == "preserve" else 0
    target_ordinals = {
        int(match.group(1))
        for key in output
        if (match := _DIFFUSION_BLOCK.match(key)) is not None
    }
    if len(target_ordinals) != target_count - unapplied:
        raise AnimaLoraRemapError("anima_lora_mapping_target_incomplete")
    return output, {
        "source_layout_id": source_layout_id,
        "target_layout_id": target_layout_id,
        "mode": mode,
        "mapping_revision": expected_revision,
        "mapping_hash": canonical_hash,
        "applied_source_blocks": source_count,
        "unapplied_target_blocks": unapplied,
    }
