from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

from app.schemas.prompt_input import PromptInput
from app.schemas.prompt_layout import PromptLayout, SpatialAssetReceipt
from app.services.spatial_asset_resolver import (
    SpatialAssetResolutionError,
    decode_spatial_asset_rgba,
    extract_spatial_binary_mask,
    resolve_spatial_asset,
)


LegacySpatialImportStatus = Literal["ready", "needs_source", "invalid", "unsupported"]


@dataclass(frozen=True, slots=True)
class LegacySpatialImportResult:
    status: LegacySpatialImportStatus
    reason: str | None
    prompt: str
    prompt_input: PromptInput | None
    prompt_layout: PromptLayout | None
    legacy_snapshot: object
    legacy_mappings: tuple["LegacySpatialMappingReceipt", ...] = ()


@dataclass(frozen=True, slots=True)
class LegacySpatialMappingReceipt:
    legacy_mapping_key: str
    target_id: str
    source_identity: str
    unit_index: int
    region_index: int
    source_color: str


def _result(
    *,
    status: LegacySpatialImportStatus,
    reason: str | None,
    snapshot: object,
    prompt: str = "",
    prompt_input: PromptInput | None = None,
    prompt_layout: PromptLayout | None = None,
    legacy_mappings: tuple[LegacySpatialMappingReceipt, ...] = (),
) -> LegacySpatialImportResult:
    return LegacySpatialImportResult(
        status=status,
        reason=reason,
        prompt=prompt,
        prompt_input=prompt_input,
        prompt_layout=prompt_layout,
        legacy_snapshot=deepcopy(snapshot),
        legacy_mappings=legacy_mappings,
    )


def _legacy_mapping_key(
    source_identity: str,
    unit_index: int,
    region_index: int,
    color: str,
) -> str:
    digest = hashlib.sha256(
        f"{source_identity}\0{unit_index}\0{region_index}\0{color.upper()}".encode("utf-8")
    ).hexdigest()
    return f"legacy-region:sha256:{digest}"


def _target_id(mapping_key: str) -> str:
    payload = hashlib.sha256(mapping_key.encode("utf-8")).digest()
    value = bytearray(payload[:16])
    value[6] = (value[6] & 0x0F) | 0x70
    value[8] = (value[8] & 0x3F) | 0x80
    return f"target_{UUID(bytes=bytes(value))}"


def _parse_regions(snapshot: object) -> list[tuple[str, str]] | None:
    if not isinstance(snapshot, dict):
        return None
    raw_regions = snapshot.get("regional_prompts")
    if not isinstance(raw_regions, list):
        return None
    regions: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in raw_regions:
        if not isinstance(item, dict):
            return None
        color = item.get("color")
        prompt = item.get("prompt")
        if (
            not isinstance(color, str)
            or re.fullmatch(r"#[0-9A-Fa-f]{6}", color) is None
            or not isinstance(prompt, str)
            or not prompt.strip()
        ):
            return None
        normalized_color = color.upper()
        if normalized_color in seen:
            return None
        seen.add(normalized_color)
        regions.append((normalized_color, prompt.replace("\r\n", "\n").replace("\r", "\n").strip()))
    return regions


def import_legacy_anima_layout(
    legacy_unit: object,
    *,
    asset_receipt: SpatialAssetReceipt | None = None,
    data_dir: str | Path | None = None,
    source_identity: str | None = None,
    unit_index: int = 0,
) -> LegacySpatialImportResult:
    snapshot = deepcopy(legacy_unit)
    regions = _parse_regions(snapshot)
    if regions is None:
        return _result(
            status="invalid",
            reason="legacy_spatial_data_invalid",
            snapshot=snapshot,
        )
    if not regions:
        return _result(
            status="unsupported",
            reason="legacy_spatial_regions_missing",
            snapshot=snapshot,
        )
    if source_identity is not None and (not source_identity.strip() or unit_index < 0):
        return _result(
            status="invalid",
            reason="legacy_spatial_provenance_invalid",
            snapshot=snapshot,
        )
    legacy_mappings = tuple(
        LegacySpatialMappingReceipt(
            legacy_mapping_key=(key := _legacy_mapping_key(source_identity, unit_index, index, color)),
            target_id=_target_id(key),
            source_identity=source_identity,
            unit_index=unit_index,
            region_index=index,
            source_color=color,
        )
        for index, (color, _prompt) in enumerate(regions)
    ) if source_identity is not None else ()
    if asset_receipt is None or data_dir is None:
        return _result(
            status="needs_source",
            reason="legacy_spatial_source_required",
            snapshot=snapshot,
            legacy_mappings=legacy_mappings,
        )
    if source_identity is None:
        return _result(
            status="needs_source",
            reason="legacy_spatial_provenance_required",
            snapshot=snapshot,
        )
    try:
        asset = resolve_spatial_asset(asset_receipt, data_dir)
        rgba = decode_spatial_asset_rgba(asset)
    except (SpatialAssetResolutionError, OSError, ValueError):
        return _result(
            status="invalid",
            reason="legacy_spatial_source_invalid",
            snapshot=snapshot,
        )

    prompt_targets: list[dict[str, object]] = []
    layout_targets: list[dict[str, object]] = []
    for index, (color, prompt) in enumerate(regions):
        try:
            bounds, compact_binary = extract_spatial_binary_mask(
                rgba,
                source_rect=(0, 0, asset.width, asset.height),
                source_color=color,
            )
        except SpatialAssetResolutionError:
            return _result(
                status="invalid",
                reason="legacy_spatial_color_not_found",
                snapshot=snapshot,
                legacy_mappings=legacy_mappings,
            )
        left, top, right, bottom = bounds
        target_id = legacy_mappings[index].target_id
        prompt_targets.append(
            {
                "id": target_id,
                "name": f"対象{index + 1}",
                "text": prompt,
                "order": index,
            }
        )
        source_bounds = {
            "x": left,
            "y": top,
            "width": right - left,
            "height": bottom - top,
        }
        layout_targets.append(
            {
                "target_id": target_id,
                "shape": "mask",
                "center_x": ((left + right) / 2) / asset.width,
                "center_y": ((top + bottom) / 2) / asset.height,
                "width": (right - left) / asset.width,
                "height": (bottom - top) / asset.height,
                "size_mode": "custom",
                "strength_mode": "auto",
                "strength": None,
                "blending_mode": "auto",
                "blending": None,
                "mask_source": {
                    "asset": asset_receipt.model_dump(mode="json"),
                    "source_color": color,
                    "source_rect_px": {
                        "x": 0,
                        "y": 0,
                        "width": asset.width,
                        "height": asset.height,
                    },
                    "source_bounds_px": source_bounds,
                    "binary_mask_sha256": hashlib.sha256(compact_binary).hexdigest(),
                    "normalization_version": "asteria.spatial-mask.v1",
                },
            }
        )

    prompt_input = PromptInput.model_validate(
        {
            "schema_version": "asteria.prompt-input.v2",
            "mode": "sections",
            "sections": [
                {
                    "id": "base",
                    "label": "ベースプロンプト",
                    "order": 0,
                    "content": {"mode": "global", "text": ""},
                },
                {
                    "id": "character",
                    "label": "キャラクタープロンプト",
                    "order": 1,
                    "content": {"mode": "targets", "targets": prompt_targets},
                },
                {
                    "id": "detail",
                    "label": "その他詳細",
                    "order": 2,
                    "content": {"mode": "global", "text": ""},
                },
            ],
        }
    )
    prompt_layout = PromptLayout.model_validate(
        {
            "schema_version": "asteria.prompt-layout.v1",
            "mode": "enabled",
            "reference_canvas": {"width": asset.width, "height": asset.height},
            "depth_mode": "auto",
            "depth_order": [],
            "excluded_target_ids": [],
            "targets": layout_targets,
        }
    )
    prompt = prompt_input.canonical_prompt or ""
    return _result(
        status="ready",
        reason=None,
        snapshot=snapshot,
        prompt=prompt,
        prompt_input=prompt_input,
        prompt_layout=prompt_layout,
        legacy_mappings=legacy_mappings,
    )
