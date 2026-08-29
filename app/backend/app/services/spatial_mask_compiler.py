from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from app.schemas.prompt_layout import PromptLayout, SpatialLayoutTarget
from app.schemas.spatial_prompt_runtime import (
    SpatialMaskBundle,
    canonical_runtime_sha256,
)
from app.services.spatial_asset_resolver import resolve_spatial_mask_source


_MANUAL_DEPTH_DECAY_BY_STAGE = {
    "base": 0.60,
    "hires": 0.70,
    "refiner": 0.80,
}


class SpatialMaskCompilerError(ValueError):
    def __init__(self, code: str, target_ids: tuple[str, ...] = ()) -> None:
        super().__init__(code)
        self.code = code
        self.target_ids = target_ids


@dataclass(frozen=True, slots=True)
class SpatialCompiledMask:
    target_id: str
    width: int
    height: int
    mask_bytes: bytes
    mask_sha256: str

    def open_image(self) -> Image.Image:
        return Image.frombytes("L", (self.width, self.height), self.mask_bytes)


@dataclass(frozen=True, slots=True)
class CompiledSpatialMaskBundle:
    receipt: SpatialMaskBundle
    masks: tuple[SpatialCompiledMask, ...]


def runtime_mask_sha256(values: np.ndarray) -> str:
    normalized = np.expand_dims(np.asarray(values, dtype=np.uint8, order="C"), axis=0)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": "uint8", "shape": list(normalized.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def intersect_spatial_mask_once(spatial: Image.Image, inpaint: Image.Image) -> Image.Image:
    if spatial.size != inpaint.size:
        raise SpatialMaskCompilerError("spatial_prompt_inpaint_size_mismatch")
    spatial_values = np.asarray(spatial.convert("L"), dtype=np.uint16)
    inpaint_values = np.asarray(inpaint.convert("L"), dtype=np.uint16)
    result = ((spatial_values * inpaint_values + 127) // 255).astype(np.uint8)
    return Image.fromarray(result, mode="L")


def _target_box(
    target: SpatialLayoutTarget,
    *,
    reference_width: int,
    reference_height: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if (
        not math.isfinite(target.width)
        or not math.isfinite(target.height)
        or target.width <= 0
        or target.height <= 0
        or target.width > 16
        or target.height > 16
    ):
        raise SpatialMaskCompilerError("spatial_prompt_geometry_invalid")
    scale = min(width, height) / min(reference_width, reference_height)
    raw_width = target.width * reference_width * scale
    raw_height = target.height * reference_height * scale
    if (
        not math.isfinite(raw_width)
        or not math.isfinite(raw_height)
        or raw_width > 2**30
        or raw_height > 2**30
    ):
        raise SpatialMaskCompilerError("spatial_prompt_geometry_invalid")

    def round_half_up(value: float) -> int:
        return math.floor(value + 0.5)

    target_width = max(1, round_half_up(raw_width))
    target_height = max(1, round_half_up(raw_height))
    center_x = target.center_x * width
    center_y = target.center_y * height
    left = round_half_up(center_x - target_width / 2)
    top = round_half_up(center_y - target_height / 2)
    return left, top, left + target_width, top + target_height


def _blur_radius(target: SpatialLayoutTarget, box: tuple[int, int, int, int]) -> float:
    edge = min(8192, max(1, min(box[2] - box[0], box[3] - box[1])))
    blending = 24 if target.blending_mode == "auto" else int(target.blending or 0)
    return edge * (blending / 100.0) * 0.12


def _geometric_mask(
    target: SpatialLayoutTarget,
    *,
    reference_width: int,
    reference_height: int,
    width: int,
    height: int,
) -> Image.Image:
    box = _target_box(
        target,
        reference_width=reference_width,
        reference_height=reference_height,
        width=width,
        height=height,
    )
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw_box = (box[0], box[1], box[2] - 1, box[3] - 1)
    if target.shape == "ellipse":
        draw.ellipse(draw_box, fill=255)
    else:
        draw.rectangle(draw_box, fill=255)
    radius = _blur_radius(target, box)
    return mask.filter(ImageFilter.GaussianBlur(radius)) if radius > 0 else mask


def _imported_mask(
    target: SpatialLayoutTarget,
    *,
    data_dir: Path,
    reference_width: int,
    reference_height: int,
    width: int,
    height: int,
) -> Image.Image:
    if target.mask_source is None:
        raise SpatialMaskCompilerError("spatial_prompt_mask_source_required")
    resolved = resolve_spatial_mask_source(target.mask_source, data_dir)
    source = Image.frombytes(
        "L",
        (resolved.width, resolved.height),
        bytes(255 if value else 0 for value in resolved.binary_mask),
    )
    box = _target_box(
        target,
        reference_width=reference_width,
        reference_height=reference_height,
        width=width,
        height=height,
    )
    mask = Image.new("L", (width, height), 0)
    visible = (
        max(0, box[0]),
        max(0, box[1]),
        min(width, box[2]),
        min(height, box[3]),
    )
    if visible[0] >= visible[2] or visible[1] >= visible[3]:
        raise SpatialMaskCompilerError("spatial_prompt_geometry_invalid")
    target_width = box[2] - box[0]
    target_height = box[3] - box[1]
    source_extent = (
        (visible[0] - box[0]) / target_width * source.width,
        (visible[1] - box[1]) / target_height * source.height,
        (visible[2] - box[0]) / target_width * source.width,
        (visible[3] - box[1]) / target_height * source.height,
    )
    resized = source.transform(
        (visible[2] - visible[0], visible[3] - visible[1]),
        Image.Transform.EXTENT,
        source_extent,
        resample=Image.Resampling.BILINEAR,
    )
    mask.paste(resized, (visible[0], visible[1]))
    radius = _blur_radius(target, box)
    return mask.filter(ImageFilter.GaussianBlur(radius)) if radius > 0 else mask


def _apply_depth(
    masks: dict[str, np.ndarray],
    layout: PromptLayout,
    *,
    stage: Literal["base", "hires", "refiner"],
) -> dict[str, np.ndarray]:
    if layout.depth_mode == "auto":
        total = np.sum(tuple(masks.values()), axis=0)
        denominator = np.maximum(total, 1.0)
        return {
            target_id: np.where(total > 1.0, values / denominator, values)
            for target_id, values in masks.items()
        }
    active_order = [target_id for target_id in layout.depth_order if target_id in masks]
    # Earlier stages need a clearer composition bias. Later stages retain more
    # of the rear target so refinement does not turn a soft priority into a cutout.
    priority_decay = _MANUAL_DEPTH_DECAY_BY_STAGE[stage]
    priorities = {
        target_id: priority_decay**depth_index
        for depth_index, target_id in enumerate(active_order)
    }
    raw_total = np.sum(tuple(masks.values()), axis=0)
    weighted = {
        target_id: masks[target_id] * priorities[target_id]
        for target_id in active_order
    }
    weighted_total = np.maximum(np.sum(tuple(weighted.values()), axis=0), 1e-12)
    return {
        target_id: np.where(
            raw_total > 1.0,
            weighted[target_id] / weighted_total,
            masks[target_id],
        )
        for target_id in active_order
    }


def compile_spatial_masks(
    layout: PromptLayout,
    *,
    width: int,
    height: int,
    stage: Literal["base", "hires", "refiner"],
    data_dir: str | Path | None = None,
    inpaint_mask: Image.Image | None = None,
    crop_rect: tuple[int, int, int, int] | None = None,
) -> CompiledSpatialMaskBundle:
    if layout.mode != "enabled":
        raise SpatialMaskCompilerError("spatial_prompt_layout_not_enabled")
    if width <= 0 or height <= 0 or width > 8192 or height > 8192:
        raise SpatialMaskCompilerError("spatial_prompt_canvas_invalid")
    if (
        layout.reference_canvas.width <= 0
        or layout.reference_canvas.height <= 0
        or layout.reference_canvas.width > 8192
        or layout.reference_canvas.height > 8192
    ):
        raise SpatialMaskCompilerError("spatial_prompt_canvas_invalid")
    if inpaint_mask is not None and inpaint_mask.size != (width, height):
        raise SpatialMaskCompilerError("spatial_prompt_inpaint_size_mismatch")
    if crop_rect is not None:
        left, top, right, bottom = crop_rect
        if (
            left < 0
            or top < 0
            or right <= left
            or bottom <= top
            or right > width
            or bottom > height
        ):
            raise SpatialMaskCompilerError("spatial_prompt_crop_invalid")
    excluded = set(layout.excluded_target_ids)
    active_targets = [target for target in layout.targets if target.target_id not in excluded]
    if not active_targets:
        raise SpatialMaskCompilerError("spatial_prompt_no_active_targets")

    images: dict[str, Image.Image] = {}
    for target in active_targets:
        if target.shape == "mask":
            if data_dir is None:
                raise SpatialMaskCompilerError("spatial_prompt_mask_source_required")
            mask = _imported_mask(
                target,
                data_dir=Path(data_dir),
                reference_width=layout.reference_canvas.width,
                reference_height=layout.reference_canvas.height,
                width=width,
                height=height,
            )
        else:
            mask = _geometric_mask(
                target,
                reference_width=layout.reference_canvas.width,
                reference_height=layout.reference_canvas.height,
                width=width,
                height=height,
            )
        if inpaint_mask is not None:
            mask = intersect_spatial_mask_once(mask, inpaint_mask)
        if crop_rect is not None:
            mask = mask.crop(crop_rect)
        images[target.target_id] = mask

    nonintersecting = tuple(
        target.target_id
        for target in active_targets
        if images[target.target_id].getbbox() is None
    )
    if nonintersecting:
        raise SpatialMaskCompilerError("spatial_prompt_inpaint_no_overlap", nonintersecting)

    execution_width = crop_rect[2] - crop_rect[0] if crop_rect is not None else width
    execution_height = crop_rect[3] - crop_rect[1] if crop_rect is not None else height
    arrays = {
        target_id: np.asarray(mask, dtype=np.float32) / 255.0
        for target_id, mask in images.items()
    }
    weighted = _apply_depth(arrays, layout, stage=stage)
    depth_order = (
        tuple(layout.depth_order)
        if layout.depth_mode == "manual"
        else tuple(target.target_id for target in active_targets)
    )
    active_depth_order = tuple(target_id for target_id in depth_order if target_id in weighted)
    depth_index = {target_id: index for index, target_id in enumerate(active_depth_order)}
    compiled: list[SpatialCompiledMask] = []
    target_receipts: list[dict[str, object]] = []
    for target in active_targets:
        values = np.rint(np.clip(weighted[target.target_id], 0.0, 1.0) * 255.0).astype(np.uint8)
        mask_bytes = values.tobytes(order="C")
        mask_sha = runtime_mask_sha256(values)
        compiled.append(
            SpatialCompiledMask(
                target_id=target.target_id,
                width=execution_width,
                height=execution_height,
                mask_bytes=mask_bytes,
                mask_sha256=mask_sha,
            )
        )
        target_receipts.append(
            {
                "target_id": target.target_id,
                "mask_sha256": mask_sha,
                "overlap_weight": float(values.mean() / 255.0),
                "depth_index": depth_index[target.target_id],
                "strength_mode": target.strength_mode,
                "strength": target.strength,
                "blending_mode": target.blending_mode,
                "blending": target.blending,
            }
        )
    payload: dict[str, object] = {
        "schema_version": "asteria.spatial-mask-bundle.v1",
        "stage": stage,
        "width": execution_width,
        "height": execution_height,
        "targets": target_receipts,
    }
    receipt = SpatialMaskBundle.model_validate(
        {**payload, "bundle_sha256": canonical_runtime_sha256(payload)}
    )
    return CompiledSpatialMaskBundle(receipt=receipt, masks=tuple(compiled))
