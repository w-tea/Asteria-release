from __future__ import annotations

from typing import Any

import torch

from .spatial_contract import (
    SpatialMaskBundle,
    SpatialMaskTarget,
    canonical_json_sha256,
    canonical_mask,
    canonical_mask_sha256,
    finite_number,
    mask_bundle_payload,
    validate_bundle_target_ids,
    validate_contract_version,
    validate_sha256,
    validate_target_id,
)


def collect_mask_target(
    mask: torch.Tensor,
    target_id: Any,
    depth_index: Any,
    overlap_weight: Any,
    strength: Any,
    blending: Any,
    mask_sha256: Any,
    contract_version: Any,
    previous: SpatialMaskBundle | None = None,
) -> SpatialMaskBundle:
    validate_contract_version(contract_version)
    normalized_id = validate_target_id(target_id)
    try:
        normalized_depth = int(depth_index)
    except (TypeError, ValueError) as error:
        raise ValueError("Asteria spatial depth is invalid") from error
    existing = previous.targets if previous is not None else ()
    if normalized_id in {target.target_id for target in existing}:
        raise ValueError("Asteria spatial bundle contains a duplicate target")
    if normalized_depth != len(existing):
        raise ValueError("Asteria spatial depth must be contiguous")
    normalized_overlap = finite_number(
        overlap_weight,
        "overlap",
        minimum=0.0,
        maximum=1.0,
    )
    normalized_strength = finite_number(
        strength,
        "strength",
        minimum=0.0,
        maximum=100.0,
    )
    normalized_blending = finite_number(
        blending,
        "blending",
        minimum=0.0,
        maximum=100.0,
    )
    normalized_mask = canonical_mask(mask)
    expected_digest = validate_sha256(mask_sha256, "mask digest")
    actual_digest = canonical_mask_sha256(normalized_mask)
    if actual_digest != expected_digest:
        raise ValueError("Asteria spatial mask digest does not match")
    targets = existing + (
        SpatialMaskTarget(
            target_id=normalized_id,
            mask=normalized_mask,
            mask_sha256=actual_digest,
            depth_index=normalized_depth,
            overlap_weight=normalized_overlap,
            strength=normalized_strength,
            blending=normalized_blending,
        ),
    )
    validate_bundle_target_ids([target.target_id for target in targets])
    return SpatialMaskBundle(
        targets=targets,
        receipt_sha256=canonical_json_sha256(mask_bundle_payload(targets)),
    )


def validate_mask_bundle(bundle: SpatialMaskBundle) -> None:
    target_ids = [target.target_id for target in bundle.targets]
    validate_bundle_target_ids(target_ids)
    if [target.depth_index for target in bundle.targets] != list(range(len(bundle.targets))):
        raise ValueError("Asteria spatial depth must be contiguous")
    shapes = {tuple(target.mask.shape) for target in bundle.targets}
    if len(shapes) != 1:
        raise ValueError("Asteria spatial mask shapes do not match")
    for target in bundle.targets:
        if float(target.mask.max()) <= 0.0:
            raise ValueError("Asteria spatial mask is empty")
        if canonical_mask_sha256(target.mask) != target.mask_sha256:
            raise ValueError("Asteria spatial mask digest does not match")
        finite_number(target.overlap_weight, "overlap", minimum=0.0, maximum=1.0)
        finite_number(target.strength, "strength", minimum=0.0, maximum=100.0)
        finite_number(target.blending, "blending", minimum=0.0, maximum=100.0)
    expected = canonical_json_sha256(mask_bundle_payload(bundle.targets))
    if expected != bundle.receipt_sha256:
        raise ValueError("Asteria spatial mask bundle receipt does not match")
