from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from typing import Any

import torch


CONTRACT_VERSION = "asteria.spatial-prompt.node.v4"
_TARGET_ID = re.compile(r"^target_[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_sha256(payload: Any) -> str:
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def validate_contract_version(value: Any) -> str:
    normalized = str(value or "")
    if normalized != CONTRACT_VERSION:
        raise ValueError("Asteria spatial contract version is unsupported")
    return normalized


def validate_stage(value: Any) -> str:
    normalized = str(value or "")
    if normalized not in {"base", "hires", "refiner"}:
        raise ValueError("Asteria spatial stage is invalid")
    return normalized


def validate_target_id(value: Any) -> str:
    normalized = str(value or "")
    if _TARGET_ID.fullmatch(normalized) is None:
        raise ValueError("Asteria spatial target id is invalid")
    return normalized


def validate_sha256(value: Any, label: str) -> str:
    normalized = str(value or "").lower()
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError(f"Asteria spatial {label} is invalid")
    return normalized


def finite_number(value: Any, label: str, *, minimum: float, maximum: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Asteria spatial {label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"Asteria spatial {label} must be finite")
    if result < minimum or result > maximum:
        raise ValueError(f"Asteria spatial {label} is outside its range")
    return result


def canonical_mask(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor) or mask.ndim not in {2, 3}:
        raise ValueError("Asteria spatial mask shape is invalid")
    if mask.numel() < 1:
        raise ValueError("Asteria spatial mask shape is invalid")
    normalized = mask.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError("Asteria spatial mask values must be finite")
    if float(normalized.min()) < 0.0 or float(normalized.max()) > 1.0:
        raise ValueError("Asteria spatial mask values are outside their range")
    return normalized


def canonical_mask_sha256(mask: torch.Tensor) -> str:
    normalized = canonical_mask(mask)
    quantized = torch.round(normalized * 255.0).to(dtype=torch.uint8).contiguous()
    digest = sha256()
    digest.update(
        json.dumps(
            {"dtype": "uint8", "shape": list(quantized.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(quantized.numpy().tobytes(order="C"))
    return digest.hexdigest()


def clone_conditioning(conditioning: Any) -> list[list[Any]]:
    if not isinstance(conditioning, (list, tuple)) or not conditioning:
        raise ValueError("Asteria spatial conditioning is empty")
    cloned: list[list[Any]] = []
    for entry in conditioning:
        if (
            not isinstance(entry, (list, tuple))
            or len(entry) != 2
            or not isinstance(entry[0], torch.Tensor)
            or not isinstance(entry[1], dict)
        ):
            raise ValueError("Asteria spatial conditioning entry is invalid")
        cloned.append([entry[0], dict(entry[1])])
    return cloned


def _tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    if tensor.layout != torch.strided or tensor.is_quantized or tensor.device.type == "meta":
        raise ValueError("Asteria spatial conditioning tensor is unsupported")
    normalized = tensor.detach().to(device="cpu").contiguous()
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError("Asteria spatial conditioning tensor must be finite")
    byte_view = normalized.view(torch.uint8).numpy()
    return {
        "type": "tensor",
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "device_type": tensor.device.type,
        "sha256": sha256(memoryview(byte_view)).hexdigest(),
    }


def _conditioning_value_payload(value: Any, active: set[int]) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_fingerprint(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Asteria spatial conditioning metadata must be finite")
        return value
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValueError("Asteria spatial conditioning metadata is recursive")
        active.add(identity)
        try:
            return {
                "type": "tuple" if isinstance(value, tuple) else "list",
                "items": [_conditioning_value_payload(item, active) for item in value],
            }
        finally:
            active.remove(identity)
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Asteria spatial conditioning metadata key is unsupported")
        identity = id(value)
        if identity in active:
            raise ValueError("Asteria spatial conditioning metadata is recursive")
        active.add(identity)
        try:
            return {
                "type": "dict",
                "items": {
                    key: _conditioning_value_payload(value[key], active)
                    for key in sorted(value)
                },
            }
        finally:
            active.remove(identity)
    raise ValueError("Asteria spatial conditioning metadata value is unsupported")


def conditioning_entry_payload(entry: tuple[torch.Tensor, dict[str, Any]]) -> dict[str, Any]:
    tensor, metadata = entry
    return {
        "conditioning": _tensor_fingerprint(tensor),
        "metadata": _conditioning_value_payload(metadata, set()),
    }


@dataclass(frozen=True, slots=True)
class SpatialMaskTarget:
    target_id: str
    mask: torch.Tensor
    mask_sha256: str
    depth_index: int
    overlap_weight: float
    strength: float
    blending: float


@dataclass(frozen=True, slots=True)
class SpatialMaskBundle:
    targets: tuple[SpatialMaskTarget, ...]
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class SpatialConditioningTarget:
    target_id: str
    conditioning: tuple[tuple[torch.Tensor, dict[str, Any]], ...]


@dataclass(frozen=True, slots=True)
class SpatialConditioningBundle:
    targets: tuple[SpatialConditioningTarget, ...]
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class SpatialTokenChunkRange:
    chunk_index: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class SpatialTargetTokenBinding:
    target_id: str
    character_start: int
    character_end: int
    semantic_start: int
    semantic_end: int
    context_indices: tuple[int, ...]
    adapter_context_indices: tuple[int, ...]
    chunk_ranges: tuple[SpatialTokenChunkRange, ...]
    token_weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SpatialEncoderTokenBinding:
    encoder_id: str
    tokenizer_fingerprint: str
    full_text_sha256: str
    token_ids: tuple[int, ...]
    special_token_indices: tuple[int, ...]
    padding_token_indices: tuple[int, ...]
    overflow_chunk_indices: tuple[int, ...]
    truncated: bool
    truncation_character_start: int | None
    adapter_role: str
    targets: tuple[SpatialTargetTokenBinding, ...]


@dataclass(frozen=True, slots=True)
class SpatialTokenBindingBundle:
    schema_version: str
    full_text: str
    full_text_sha256: str
    step_start_percent: float
    step_end_percent: float
    pooled_output_scope: str
    encoders: tuple[SpatialEncoderTokenBinding, ...]
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class SpatialScheduledTokenBindingBundle:
    schema_version: str
    bindings: tuple[SpatialTokenBindingBundle, ...]
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class SpatialIsolatedTargetRange:
    target_id: str
    fragment_id: str
    start: int
    end: int
    destination_indices: tuple[int, ...] = ()
    source_start: int | None = None
    source_end: int | None = None


@dataclass(frozen=True, slots=True)
class SpatialIsolatedContextBundle:
    schema_version: str
    model_family: str
    adapter_kind: str
    global_fragment_id: str
    global_range: tuple[int, int]
    targets: tuple[SpatialIsolatedTargetRange, ...]
    context_length: int
    source_fragment_lengths: tuple[int, ...]
    adapter_fragment_lengths: tuple[int, ...]
    adapter_context_length: int
    fragments_sha256: str
    receipt_sha256: str
    full_context_length: int | None = None
    aligned_target_values: tuple[torch.Tensor, ...] = ()


def mask_bundle_payload(targets: tuple[SpatialMaskTarget, ...]) -> dict[str, Any]:
    return {
        "schema_version": "asteria.spatial-node-mask-bundle.v1",
        "contract_version": CONTRACT_VERSION,
        "targets": [
            {
                "target_id": target.target_id,
                "mask_sha256": target.mask_sha256,
                "shape": list(target.mask.shape),
                "depth_index": target.depth_index,
                "overlap_weight": target.overlap_weight,
                "strength": target.strength,
                "blending": target.blending,
            }
            for target in targets
        ],
    }


def conditioning_bundle_payload(
    targets: tuple[SpatialConditioningTarget, ...],
) -> dict[str, Any]:
    return {
        "schema_version": "asteria.spatial-node-conditioning-bundle.v1",
        "contract_version": CONTRACT_VERSION,
        "targets": [
            {
                "target_id": target.target_id,
                "entries": [
                    conditioning_entry_payload(entry)
                    for entry in target.conditioning
                ],
            }
            for target in targets
        ],
    }


def validate_bundle_target_ids(target_ids: list[str]) -> None:
    if not target_ids:
        raise ValueError("Asteria spatial target count must be positive")
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("Asteria spatial bundle contains a duplicate target")
