from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.services.media_model_catalog import SafetensorsHeader
from app.services.media_model_catalog import read_safetensors_header
from app.schemas.model_architecture import (
    AnimaArchitectureComponent,
    ArchitectureDetectionStatus,
)
from app.services.model_architecture_inspector import AnimaArchitectureInspector
from app.services.model_fingerprints import ModelFingerprintService
from app.services.model_library import ModelLibrary, ModelLibraryItem


_LORA_DIFFUSION_BLOCK = re.compile(r"^diffusion_model\.blocks\.(\d+)\.")
_LORA_LLM_ADAPTER_BLOCK = re.compile(
    r"^diffusion_model\.llm_adapter\.blocks\.(\d+)\."
)
_LORA_FILE_SUFFIXES = frozenset(
    {".safetensors", ".ckpt", ".pt", ".pt2", ".bin", ".pth", ".pkl", ".sft"}
)


class AnimaLoraRemapRegistryError(ValueError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class AnimaLoraLayoutDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    layout_id: str = Field(min_length=1)
    diffusion_block_count: int = Field(gt=0)
    llm_adapter_block_count: int = Field(ge=0)
    key_count: int = Field(gt=0)


class AnimaLoraRemapReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_layout_id: str = Field(min_length=1)
    target_layout_id: str = Field(min_length=1)
    mode: Literal["preserve", "expand"]
    source_fingerprint: str = Field(min_length=1)
    mapping_revision: int = Field(gt=0)
    mapping_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    direction_id: str | None = None
    pass_through: bool
    applied_source_blocks: int = Field(ge=0)
    unapplied_target_blocks: int = Field(ge=0)
    warnings: tuple[str, ...] = ()


class AnimaLoraRemapLayout(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    diffusion_block_count: int = Field(gt=0)
    llm_adapter_block_count: int = Field(ge=0)
    evidence_status: Literal["verified_header"]


class AnimaLoraInsertedTarget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target: int = Field(ge=0)
    source: int = Field(ge=0)


class AnimaLoraRemapDirection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    source_layout_id: str = Field(min_length=1)
    target_layout_id: str = Field(min_length=1)
    source_block_count: int = Field(gt=0)
    target_block_count: int = Field(gt=0)
    status: Literal["enabled", "disabled"]
    preserve_enabled: bool
    expand_enabled: bool
    retained_target_by_source: tuple[int, ...]
    inserted_targets: tuple[int, ...]
    inserted_source_by_target: tuple[AnimaLoraInsertedTarget, ...]
    evidence: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_ordinals(self) -> "AnimaLoraRemapDirection":
        if len(self.retained_target_by_source) != self.source_block_count:
            raise ValueError("anima_lora_mapping_source_incomplete")
        retained = set(self.retained_target_by_source)
        if len(retained) != len(self.retained_target_by_source):
            raise ValueError("anima_lora_mapping_target_duplicate")
        inserted = set(self.inserted_targets)
        if len(inserted) != len(self.inserted_targets):
            raise ValueError("anima_lora_mapping_target_duplicate")
        if retained & inserted:
            raise ValueError("anima_lora_mapping_target_duplicate")
        if retained | inserted != set(range(self.target_block_count)):
            raise ValueError("anima_lora_mapping_target_incomplete")
        inserted_mapping_targets = {
            mapping.target for mapping in self.inserted_source_by_target
        }
        if len(inserted_mapping_targets) != len(self.inserted_source_by_target):
            raise ValueError("anima_lora_mapping_inserted_target_duplicate")
        if any(target not in inserted for target in inserted_mapping_targets):
            raise ValueError("anima_lora_mapping_inserted_target_unknown")
        if any(
            mapping.source >= self.source_block_count
            for mapping in self.inserted_source_by_target
        ):
            raise ValueError("anima_lora_mapping_source_out_of_range")
        if self.status == "enabled" and not self.preserve_enabled:
            raise ValueError("anima_lora_mapping_enabled_without_preserve")
        if self.expand_enabled and inserted_mapping_targets != inserted:
            raise ValueError("anima_lora_mapping_expand_incomplete")
        return self


class AnimaLoraRemapRegistry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_id: Literal["asteria.anima_lora_block_remap.v1"]
    revision: int = Field(gt=0)
    allowed_key_prefixes: tuple[str, ...]
    layouts: tuple[AnimaLoraRemapLayout, ...]
    directions: tuple[AnimaLoraRemapDirection, ...]
    canonical_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_registry(self) -> "AnimaLoraRemapRegistry":
        layout_ids = [layout.id for layout in self.layouts]
        if len(layout_ids) != len(set(layout_ids)):
            raise ValueError("anima_lora_mapping_layout_duplicate")
        direction_ids = [direction.id for direction in self.directions]
        if len(direction_ids) != len(set(direction_ids)):
            raise ValueError("anima_lora_mapping_direction_duplicate")
        known_layouts = set(layout_ids)
        if any(
            direction.source_layout_id not in known_layouts
            or direction.target_layout_id not in known_layouts
            for direction in self.directions
        ):
            raise ValueError("anima_lora_mapping_layout_unknown")
        return self

    def require_enabled_direction(
        self,
        source_layout_id: str,
        target_layout_id: str,
        *,
        mode: Literal["preserve", "expand"],
    ) -> AnimaLoraRemapDirection:
        direction = next(
            (
                item
                for item in self.directions
                if item.source_layout_id == source_layout_id
                and item.target_layout_id == target_layout_id
            ),
            None,
        )
        if direction is None:
            raise AnimaLoraRemapRegistryError(
                "anima_lora_mapping_direction_unknown"
            )
        if direction.status != "enabled":
            raise AnimaLoraRemapRegistryError(
                "anima_lora_mapping_direction_disabled"
            )
        if mode == "preserve" and not direction.preserve_enabled:
            raise AnimaLoraRemapRegistryError("anima_lora_preserve_disabled")
        if mode == "expand" and not direction.expand_enabled:
            raise AnimaLoraRemapRegistryError("anima_lora_expand_disabled")
        return direction


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


def load_anima_lora_remap_registry(
    path: Path | None = None,
) -> AnimaLoraRemapRegistry:
    registry_path = path or (
        Path(__file__).resolve().parents[3]
        / "mappings"
        / "anima"
        / "lora-block-remap.v1.json"
    )
    try:
        raw = registry_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnimaLoraRemapRegistryError(
            "anima_lora_mapping_registry_unreadable"
        ) from exc
    canonical = _canonical_bytes(payload)
    if raw != canonical:
        raise AnimaLoraRemapRegistryError(
            "anima_lora_mapping_registry_not_canonical"
        )
    digest = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    try:
        return AnimaLoraRemapRegistry(
            **payload,
            canonical_sha256=digest,
        )
    except ValidationError as exc:
        reason = next(
            (
                str(error.get("ctx", {}).get("error"))
                for error in exc.errors()
                if error.get("ctx", {}).get("error") is not None
            ),
            "anima_lora_mapping_registry_invalid",
        )
        raise AnimaLoraRemapRegistryError(reason) from exc


def inspect_anima_lora_header(
    header: SafetensorsHeader,
    *,
    registry: AnimaLoraRemapRegistry | None = None,
) -> AnimaLoraLayoutDescriptor:
    active_registry = registry or load_anima_lora_remap_registry()
    keys = tuple(header.tensors)
    if not keys:
        raise AnimaLoraRemapRegistryError("anima_lora_header_empty")
    if any(
        not any(key.startswith(prefix) for prefix in active_registry.allowed_key_prefixes)
        for key in keys
    ):
        raise AnimaLoraRemapRegistryError("anima_lora_key_prefix_not_allowed")

    diffusion_ordinals = {
        int(match.group(1))
        for key in keys
        if (match := _LORA_DIFFUSION_BLOCK.match(key)) is not None
    }
    if not diffusion_ordinals:
        raise AnimaLoraRemapRegistryError("anima_lora_diffusion_blocks_missing")
    diffusion_count = max(diffusion_ordinals) + 1
    if diffusion_ordinals != set(range(diffusion_count)):
        raise AnimaLoraRemapRegistryError(
            "anima_lora_diffusion_blocks_not_contiguous"
        )

    llm_ordinals = {
        int(match.group(1))
        for key in keys
        if (match := _LORA_LLM_ADAPTER_BLOCK.match(key)) is not None
    }
    llm_count = max(llm_ordinals) + 1 if llm_ordinals else 0
    if llm_ordinals != set(range(llm_count)):
        raise AnimaLoraRemapRegistryError(
            "anima_lora_llm_adapter_blocks_not_contiguous"
        )
    layout = next(
        (
            item
            for item in active_registry.layouts
            if item.diffusion_block_count == diffusion_count
            and item.llm_adapter_block_count == llm_count
        ),
        None,
    )
    if layout is None:
        raise AnimaLoraRemapRegistryError("anima_lora_layout_unverified")
    return AnimaLoraLayoutDescriptor(
        layout_id=layout.id,
        diffusion_block_count=diffusion_count,
        llm_adapter_block_count=llm_count,
        key_count=len(keys),
    )


def prepare_anima_lora_remap(
    header: SafetensorsHeader,
    *,
    target_layout_id: str,
    mode: Literal["preserve", "expand"],
    source_fingerprint: str,
    expected_mapping_revision: int,
    expected_mapping_hash: str,
    registry: AnimaLoraRemapRegistry | None = None,
) -> AnimaLoraRemapReceipt:
    active_registry = registry or load_anima_lora_remap_registry()
    if expected_mapping_revision != active_registry.revision:
        raise AnimaLoraRemapRegistryError(
            "anima_lora_mapping_revision_mismatch"
        )
    if expected_mapping_hash != active_registry.canonical_sha256:
        raise AnimaLoraRemapRegistryError("anima_lora_mapping_hash_mismatch")

    source = inspect_anima_lora_header(header, registry=active_registry)
    target = next(
        (layout for layout in active_registry.layouts if layout.id == target_layout_id),
        None,
    )
    if target is None:
        raise AnimaLoraRemapRegistryError("anima_lora_target_layout_unknown")

    if source.layout_id == target_layout_id:
        if mode == "expand":
            raise AnimaLoraRemapRegistryError("anima_lora_expand_disabled")
        return AnimaLoraRemapReceipt(
            source_layout_id=source.layout_id,
            target_layout_id=target_layout_id,
            mode=mode,
            source_fingerprint=source_fingerprint,
            mapping_revision=active_registry.revision,
            mapping_hash=active_registry.canonical_sha256,
            pass_through=True,
            applied_source_blocks=source.diffusion_block_count,
            unapplied_target_blocks=0,
        )

    direction = active_registry.require_enabled_direction(
        source.layout_id,
        target_layout_id,
        mode=mode,
    )
    unapplied = len(direction.inserted_targets) if mode == "preserve" else 0
    return AnimaLoraRemapReceipt(
        source_layout_id=source.layout_id,
        target_layout_id=target_layout_id,
        mode=mode,
        source_fingerprint=source_fingerprint,
        mapping_revision=active_registry.revision,
        mapping_hash=active_registry.canonical_sha256,
        direction_id=direction.id,
        pass_through=False,
        applied_source_blocks=source.diffusion_block_count,
        unapplied_target_blocks=unapplied,
        warnings=("inserted_target_blocks_left_unmodified",)
        if unapplied
        else (),
    )


def _normalize_lora_reference(value: str) -> str:
    return value.strip().replace("\\", "/").removeprefix("./").strip("/").casefold()


def _without_lora_suffix(value: str) -> str:
    path = Path(value)
    return value[: -len(path.suffix)] if path.suffix.casefold() in _LORA_FILE_SUFFIXES else value


def _runtime_relative_path(item: ModelLibraryItem) -> str:
    return str(
        item.metadata.get("external_relative_path")
        or item.relative_path.as_posix()
    )


def _legacy_lora_path_keys(item: ModelLibraryItem) -> set[str]:
    values = {
        _normalize_lora_reference(_runtime_relative_path(item)),
        _normalize_lora_reference(item.relative_path.as_posix()),
    }
    expanded = set(values)
    for value in values:
        if value.startswith("loras/"):
            expanded.add(value[len("loras/") :])
        elif value.startswith("lora/"):
            expanded.add(value[len("lora/") :])
        elif value.startswith("merged/lora/"):
            expanded.add(value[len("merged/lora/") :])
    return expanded | {_without_lora_suffix(value) for value in expanded}


def _legacy_lora_basename_keys(item: ModelLibraryItem) -> set[str]:
    basename = _normalize_lora_reference(item.name)
    return {basename, _without_lora_suffix(basename)}


def _resolve_legacy_lora_item(
    name: str,
    library: ModelLibrary,
) -> ModelLibraryItem:
    reference = _normalize_lora_reference(name)
    references = {reference, _without_lora_suffix(reference)}
    items = [
        item
        for item in library.items
        if item.type == "lora" and str(item.stable_model_id or "").strip()
    ]
    path_matches = [
        item for item in items if references & _legacy_lora_path_keys(item)
    ]
    if len(path_matches) == 1:
        return path_matches[0]
    if len(path_matches) > 1:
        raise AnimaLoraRemapRegistryError("anima_lora_identity_ambiguous")
    basename_matches = [
        item for item in items if references & _legacy_lora_basename_keys(item)
    ]
    if len(basename_matches) == 1:
        return basename_matches[0]
    if len(basename_matches) > 1:
        raise AnimaLoraRemapRegistryError("anima_lora_identity_ambiguous")
    raise AnimaLoraRemapRegistryError("anima_lora_identity_unresolved")


def prepare_anima_lora_runtime_entries(
    loras: list[dict],
    *,
    target_diffusion_path: Path,
    library: ModelLibrary,
    fingerprints: ModelFingerprintService,
    registry: AnimaLoraRemapRegistry | None = None,
) -> tuple[list[dict], list[dict]]:
    if not loras:
        return [], []
    active_registry = registry or load_anima_lora_remap_registry()
    try:
        target_header = read_safetensors_header(target_diffusion_path)
    except (OSError, ValueError) as exc:
        raise AnimaLoraRemapRegistryError(
            "anima_target_model_header_unavailable"
        ) from exc
    target_descriptor = AnimaArchitectureInspector().inspect(target_header)
    if (
        target_descriptor.status != ArchitectureDetectionStatus.SUPPORTED
        or target_descriptor.component
        != AnimaArchitectureComponent.DIFFUSION_MODEL
        or target_descriptor.variant_id is None
    ):
        raise AnimaLoraRemapRegistryError("anima_target_model_layout_unverified")
    target_layout_id = target_descriptor.variant_id.value

    by_stable_id: dict[str, object] = {}
    duplicate_ids: set[str] = set()
    for item in library.items:
        stable_id = str(item.stable_model_id or "").strip()
        if not stable_id:
            continue
        if stable_id in by_stable_id:
            duplicate_ids.add(stable_id)
        else:
            by_stable_id[stable_id] = item

    runtime_loras: list[dict] = []
    receipts: list[dict] = []
    for raw_lora in loras:
        lora = dict(raw_lora)
        stable_asset_id = str(lora.get("stable_asset_id") or "").strip()
        legacy_resolved = False
        if not stable_asset_id:
            if target_layout_id == "anima_base_28":
                runtime_loras.append(lora)
                continue
            item = _resolve_legacy_lora_item(str(lora.get("name") or ""), library)
            stable_asset_id = str(item.stable_model_id or "").strip()
            legacy_resolved = True
        else:
            if stable_asset_id in duplicate_ids:
                raise AnimaLoraRemapRegistryError("anima_lora_identity_ambiguous")
            item = by_stable_id.get(stable_asset_id)
        if item is None or getattr(item, "type", None) != "lora":
            raise AnimaLoraRemapRegistryError("anima_lora_identity_unresolved")
        if lora.get("block_weight") or lora.get("block_weight_preset"):
            raise AnimaLoraRemapRegistryError(
                "anima_lora_remap_block_weight_not_supported"
            )

        relative_path = _runtime_relative_path(item)
        try:
            current = fingerprints.fast_fingerprint(
                item.path,
                root_id=str(item.root_id),
                model_type="lora",
                relative_path=relative_path,
            )
        except OSError as exc:
            raise AnimaLoraRemapRegistryError("anima_lora_source_offline") from exc
        if (
            item.fingerprint_algorithm != current.algorithm
            or item.fingerprint_version != current.version
            or item.fingerprint_value != current.value
        ):
            raise AnimaLoraRemapRegistryError("anima_lora_fingerprint_changed")
        if legacy_resolved:
            lora.update(
                {
                    "stable_asset_id": stable_asset_id,
                    "fingerprint_algorithm": current.algorithm,
                    "fingerprint_version": current.version,
                    "fingerprint_value": current.value,
                    "compatibility_mode": "preserve",
                }
            )
        elif (
            lora.get("fingerprint_algorithm") != current.algorithm
            or lora.get("fingerprint_version") != current.version
            or lora.get("fingerprint_value") != current.value
        ):
            raise AnimaLoraRemapRegistryError("anima_lora_fingerprint_changed")

        try:
            source_header = read_safetensors_header(item.path)
            source_component = AnimaArchitectureInspector().inspect(
                source_header
            )
            if (
                source_component.status
                == ArchitectureDetectionStatus.SUPPORTED
                and source_component.component
                == AnimaArchitectureComponent.PROGRESSIVE_ADAPTER
            ):
                raise AnimaLoraRemapRegistryError(
                    "anima_progressive_adapter_cannot_be_used_as_lora"
                )
            source_sha256 = f"sha256:{fingerprints.strong_hash(item.path)}"
        except AnimaLoraRemapRegistryError:
            raise
        except (OSError, ValueError) as exc:
            raise AnimaLoraRemapRegistryError(
                "anima_lora_source_header_unavailable"
            ) from exc
        mode = str(lora.get("compatibility_mode") or "preserve")
        if mode not in {"preserve", "expand"}:
            raise AnimaLoraRemapRegistryError(
                "anima_lora_mapping_mode_unknown"
            )
        receipt = prepare_anima_lora_remap(
            source_header,
            target_layout_id=target_layout_id,
            mode=mode,
            source_fingerprint=source_sha256,
            expected_mapping_revision=active_registry.revision,
            expected_mapping_hash=active_registry.canonical_sha256,
            registry=active_registry,
        ).model_dump(mode="json")
        receipt.update(
            {
                "stable_asset_id": stable_asset_id,
                "source_sha256": source_sha256,
                "fingerprint_algorithm": current.algorithm,
                "fingerprint_version": current.version,
                "fingerprint_value": current.value,
            }
        )
        runtime_lora = dict(lora)
        runtime_lora["name"] = relative_path
        runtime_lora["anima_remap"] = receipt
        runtime_loras.append(runtime_lora)
        receipts.append(receipt)
    return runtime_loras, receipts
