from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.schemas.prompt_layout import PromptLayout, SpatialLayoutTarget
from app.schemas.spatial_prompt_guidance import SpatialLayoutFieldBundle
from app.schemas.spatial_prompt_runtime import canonical_runtime_sha256


SpatialFieldStage = Literal["base", "hires", "refiner"]
_FIELD_EPSILON = np.float32(np.exp(-60.0))
_OVERLAP_FLOOR = np.float32(0.05)


class SpatialLayoutFieldError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class CompiledSpatialLayoutFields:
    receipt: SpatialLayoutFieldBundle
    raw_fields: dict[str, np.ndarray]
    normalized_fields: dict[str, np.ndarray]


def spatial_field_sha256(values: np.ndarray) -> str:
    normalized = np.asarray(values, dtype="<f4", order="C")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": "float32", "shape": list(normalized.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _blending(target: SpatialLayoutTarget) -> float:
    if target.blending_mode == "auto":
        return 0.18
    return 0.06 + 0.32 * float(target.blending or 0) / 100.0


def _strength(target: SpatialLayoutTarget) -> float:
    if target.strength_mode == "auto":
        return 1.0
    return max(0.05, float(target.strength or 0) / 100.0)


def _coordinate_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(width, dtype=np.float32) + 0.5) / np.float32(width)
    y = (np.arange(height, dtype=np.float32) + 0.5) / np.float32(height)
    return np.meshgrid(x, y)


def _soft_tail(distance: np.ndarray, sigma: float) -> np.ndarray:
    outside = np.maximum(distance - np.float32(0.72), np.float32(0.0))
    exponent = np.minimum(
        np.square(outside / np.float32(max(sigma, 1e-4))) * np.float32(0.5),
        np.float32(60.0),
    )
    return np.maximum(np.exp(-exponent).astype(np.float32), _FIELD_EPSILON)


def _ellipse_field(
    target: SpatialLayoutTarget,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    radius_x = np.float32(max(target.width / 2.0, 1e-6))
    radius_y = np.float32(max(target.height / 2.0, 1e-6))
    distance = np.sqrt(
        np.square((x - np.float32(target.center_x)) / radius_x)
        + np.square((y - np.float32(target.center_y)) / radius_y)
    )
    return _soft_tail(distance, _blending(target))


def _rectangle_field(
    target: SpatialLayoutTarget,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    radius_x = np.float32(max(target.width / 2.0, 1e-6))
    radius_y = np.float32(max(target.height / 2.0, 1e-6))
    dx = np.abs(x - np.float32(target.center_x)) / radius_x
    dy = np.abs(y - np.float32(target.center_y)) / radius_y
    smoothness = np.float32(12.0)
    distance = np.logaddexp(dx * smoothness, dy * smoothness) / smoothness
    return _soft_tail(distance, _blending(target))


def _raw_field(
    target: SpatialLayoutTarget,
    x: np.ndarray,
    y: np.ndarray,
    *,
    source_masks: Mapping[str, np.ndarray] | None,
) -> np.ndarray:
    if target.shape == "ellipse":
        return _ellipse_field(target, x, y)
    if target.shape == "rectangle":
        return _rectangle_field(target, x, y)
    if source_masks is not None and target.target_id in source_masks:
        values = np.asarray(source_masks[target.target_id], dtype=np.float32, order="C")
        return np.maximum(np.clip(values, 0.0, 1.0), _FIELD_EPSILON)
    raise SpatialLayoutFieldError("spatial_layout_field_shape_unsupported")


def _depth_order(layout: PromptLayout, target_ids: tuple[str, ...]) -> tuple[str, ...]:
    if layout.depth_mode == "manual":
        return tuple(target_id for target_id in layout.depth_order if target_id in target_ids)
    return target_ids


def _normalize_overlap(
    raw_fields: dict[str, np.ndarray],
    depth_order: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, float]]:
    stack = np.stack([raw_fields[target_id] for target_id in depth_order])
    overlap = np.sum(stack >= _OVERLAP_FLOOR, axis=0) >= 2
    count = len(depth_order)
    priors = {
        target_id: 1.0 - (0.20 * index / max(1, count - 1))
        for index, target_id in enumerate(depth_order)
    }
    prior_values = np.asarray([priors[target_id] for target_id in depth_order], dtype=np.float32)
    weighted = stack * prior_values[:, None, None]
    prioritized = np.where(overlap[None, :, :], weighted, stack)
    normalizer = np.maximum(prioritized.sum(axis=0), np.float32(1.0))
    normalized_stack = prioritized / normalizer
    return (
        {
            target_id: np.asarray(normalized_stack[index], dtype=np.float32, order="C")
            for index, target_id in enumerate(depth_order)
        },
        overlap,
        priors,
    )


def compile_spatial_layout_fields(
    layout: PromptLayout,
    *,
    width: int,
    height: int,
    stage: SpatialFieldStage,
    crop_rect: tuple[int, int, int, int] | None = None,
    source_masks: Mapping[str, np.ndarray] | None = None,
) -> CompiledSpatialLayoutFields:
    if layout.mode != "enabled":
        raise SpatialLayoutFieldError("spatial_prompt_layout_not_enabled")
    if width <= 0 or height <= 0 or width > 8192 or height > 8192:
        raise SpatialLayoutFieldError("spatial_prompt_canvas_invalid")
    if crop_rect is not None:
        left, top, right, bottom = crop_rect
        if left < 0 or top < 0 or right <= left or bottom <= top or right > width or bottom > height:
            raise SpatialLayoutFieldError("spatial_prompt_crop_invalid")
    excluded = set(layout.excluded_target_ids)
    targets = tuple(target for target in layout.targets if target.target_id not in excluded)
    if not targets:
        raise SpatialLayoutFieldError("spatial_prompt_no_active_targets")
    x, y = _coordinate_grid(width, height)
    raw_fields: dict[str, np.ndarray] = {}
    for target in targets:
        values = _raw_field(target, x, y, source_masks=source_masks)
        if target.shape == "mask":
            expected_height = crop_rect[3] - crop_rect[1] if crop_rect is not None else height
            expected_width = crop_rect[2] - crop_rect[0] if crop_rect is not None else width
            if values.shape != (expected_height, expected_width):
                raise SpatialLayoutFieldError("spatial_layout_field_mask_shape_mismatch")
        elif crop_rect is not None:
            left, top, right, bottom = crop_rect
            values = values[top:bottom, left:right]
        raw_fields[target.target_id] = np.asarray(values, dtype=np.float32, order="C")
    target_ids = tuple(target.target_id for target in targets)
    depth_order = _depth_order(layout, target_ids)
    normalized_fields, overlap, priors = _normalize_overlap(raw_fields, depth_order)
    depth_indexes = {target_id: index for index, target_id in enumerate(depth_order)}
    target_by_id = {target.target_id: target for target in targets}
    receipts = []
    for target_id in depth_order:
        target = target_by_id[target_id]
        raw = raw_fields[target_id]
        normalized = normalized_fields[target_id]
        receipts.append(
            {
                "target_id": target_id,
                "shape": target.shape,
                "center_x": target.center_x,
                "center_y": target.center_y,
                "width": target.width,
                "height": target.height,
                "variance_x": (target.width / 2.0) ** 2,
                "variance_y": (target.height / 2.0) ** 2,
                "depth_index": depth_indexes[target_id],
                "depth_prior": priors[target_id],
                "strength": _strength(target),
                "blending": _blending(target),
                "raw_field_sha256": spatial_field_sha256(raw),
                "normalized_field_sha256": spatial_field_sha256(normalized),
                "overlap_pixel_count": int(np.count_nonzero(overlap & (raw >= _OVERLAP_FLOOR))),
            }
        )
    raw_bundle_sha = canonical_runtime_sha256(
        {target_id: spatial_field_sha256(raw_fields[target_id]) for target_id in depth_order}
    )
    normalized_bundle_sha = canonical_runtime_sha256(
        {
            target_id: spatial_field_sha256(normalized_fields[target_id])
            for target_id in depth_order
        }
    )
    execution_width = crop_rect[2] - crop_rect[0] if crop_rect is not None else width
    execution_height = crop_rect[3] - crop_rect[1] if crop_rect is not None else height
    payload = {
        "schema_version": "asteria.spatial-layout-field.v1",
        "stage": stage,
        "width": execution_width,
        "height": execution_height,
        "targets": receipts,
        "raw_bundle_sha256": raw_bundle_sha,
        "normalized_bundle_sha256": normalized_bundle_sha,
    }
    receipt = SpatialLayoutFieldBundle.model_validate(
        {**payload, "bundle_sha256": canonical_runtime_sha256(payload)}
    )
    return CompiledSpatialLayoutFields(
        receipt=receipt,
        raw_fields=raw_fields,
        normalized_fields=normalized_fields,
    )
