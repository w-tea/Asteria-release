from __future__ import annotations

import os
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable

from app.services.external_model_libraries import ExternalModelItem
from app.services.model_library import ModelLibraryItem
from app.services.model_fingerprints import ModelFingerprintService


class ModelReferenceError(ValueError):
    pass


def _normalized(value: str) -> str:
    return PurePosixPath(value.replace("\\", "/")).as_posix().casefold()


def migrate_legacy_model_reference(
    value: str | dict[str, Any],
    candidates: Iterable[ExternalModelItem],
    model_type: str,
) -> dict[str, Any]:
    if isinstance(value, dict):
        if value.get("stable_model_id"):
            return {"reference_schema_version": 1, **value}
        legacy = str(
            value.get("unresolved_legacy_reference")
            or value.get("path")
            or value.get("name")
            or ""
        )
    else:
        legacy = str(value)
    needle = _normalized(legacy)
    matches: list[ExternalModelItem] = []
    for item in candidates:
        if item.model_type != model_type:
            continue
        aliases = {
            _normalized(item.normalized_relative_path),
            _normalized(item.logical_name),
            _normalized(item.execution_path.name),
            os.path.normcase(os.path.abspath(os.fspath(item.execution_path))).replace("\\", "/"),
        }
        normalized_absolute = os.path.normcase(os.path.abspath(legacy)).replace("\\", "/")
        if needle in aliases or normalized_absolute in aliases:
            matches.append(item)
    if len(matches) != 1:
        return {
            "reference_schema_version": 1,
            "model_type": model_type,
            "unresolved_legacy_reference": legacy,
            "reason": "ambiguous" if len(matches) > 1 else "offline_or_missing",
        }
    item = matches[0]
    return {
        "reference_schema_version": 1,
        "stable_model_id": item.stable_model_id,
        "root_id": item.root_id,
        "model_type": item.model_type,
        "normalized_relative_path": item.normalized_relative_path,
        "fingerprint_algorithm": item.fingerprint_algorithm,
        "fingerprint_version": item.fingerprint_version,
        "fingerprint_value": item.fingerprint_value,
    }


def migrate_generation_reference_metadata(
    payload: dict[str, Any],
    candidates: Iterable[ExternalModelItem],
) -> dict[str, Any]:
    migrated = dict(payload)
    materialized = list(candidates)
    existing_slots = _snapshots_by_key(payload.get("model_reference_snapshots"), "slot_id")
    model_slots = payload.get("model_slots")
    if isinstance(model_slots, dict):
        snapshots: list[dict[str, Any]] = []
        for slot_id, value in sorted(model_slots.items()):
            if not isinstance(value, (str, dict)) or not value:
                continue
            existing = existing_slots.get(str(slot_id))
            reference = migrate_legacy_model_reference(
                existing or value,
                materialized,
                _model_type_for_slot(str(slot_id)),
            )
            snapshots.append({"slot_id": str(slot_id), **reference})
        if snapshots:
            migrated["model_reference_snapshots"] = snapshots

    loras = payload.get("loras")
    if isinstance(loras, list):
        existing_loras = _snapshots_by_key(payload.get("lora_reference_snapshots"), "index")
        snapshots = []
        for index, lora in enumerate(loras):
            if isinstance(lora, dict):
                value = lora.get("model_reference") or lora.get("path") or lora.get("name")
                display_name = str(lora.get("name") or lora.get("path") or "")
            else:
                value = lora
                display_name = str(lora)
            if not isinstance(value, (str, dict)) or not value:
                continue
            reference = migrate_legacy_model_reference(
                existing_loras.get(index) or value,
                materialized,
                "lora",
            )
            snapshots.append({"index": index, "logical_name": display_name, **reference})
        if snapshots:
            migrated["lora_reference_snapshots"] = snapshots
    return migrated


def migrate_merge_reference_metadata(
    payload: dict[str, Any],
    candidates: Iterable[ExternalModelItem],
) -> dict[str, Any]:
    migrated = dict(payload)
    materialized = list(candidates)
    merge_type = str(payload.get("merge_type") or "checkpoint")
    source_type = "lora" if merge_type == "lora" else "checkpoint"
    existing_sources = _snapshots_by_key(payload.get("model_reference_snapshots"), "source_index")
    snapshots: list[dict[str, Any]] = []
    sources = payload.get("sources")
    if isinstance(sources, list):
        for index, source in enumerate(sources):
            if not isinstance(source, dict):
                continue
            value = source.get("model_reference") or source.get("path")
            if not isinstance(value, (str, dict)) or not value:
                continue
            reference = migrate_legacy_model_reference(
                existing_sources.get(index) or value,
                materialized,
                source_type,
            )
            snapshots.append(
                {
                    "source_index": index,
                    "role": str(source.get("role") or ""),
                    **reference,
                }
            )
    bake_in_vae = payload.get("bake_in_vae")
    if isinstance(bake_in_vae, (str, dict)) and bake_in_vae:
        snapshots.append(
            {
                "source_index": -1,
                "role": "bake_in_vae",
                **migrate_legacy_model_reference(
                    existing_sources.get(-1) or bake_in_vae,
                    materialized,
                    "vae",
                ),
            }
        )
    if snapshots:
        migrated["model_reference_snapshots"] = snapshots
    return migrated


def migrate_persisted_model_references(
    data_dir: Path,
    candidates: Iterable[ExternalModelItem],
) -> dict[str, int]:
    from app.services.history import HistoryRepository
    from app.services.merge_recipes import MergeRecipeRepository
    from app.services.presets import PresetRepository

    materialized = list(candidates)
    return {
        "history": HistoryRepository(data_dir / "asteria.db").migrate_model_references(materialized),
        "presets": PresetRepository(data_dir / "presets.json").migrate_model_references(materialized),
        "merge_recipes": MergeRecipeRepository(
            data_dir / "merge-recipes.json"
        ).migrate_model_references(materialized),
    }


def model_library_reference_candidates(
    items: Iterable[ModelLibraryItem],
) -> list[ExternalModelItem]:
    candidates: list[ExternalModelItem] = []
    for item in items:
        if (
            not item.stable_model_id
            or not item.fingerprint_algorithm
            or item.fingerprint_version is None
            or not item.fingerprint_value
        ):
            continue
        relative_path = str(
            item.metadata.get("external_relative_path") or item.relative_path.as_posix()
        )
        candidates.append(
            ExternalModelItem(
                stable_model_id=item.stable_model_id,
                root_id=item.root_id,
                model_type=item.type,
                logical_name=item.name,
                normalized_relative_path=relative_path,
                execution_path=item.path,
                size_bytes=item.size_bytes,
                fingerprint_algorithm=item.fingerprint_algorithm,
                fingerprint_version=item.fingerprint_version,
                fingerprint_value=item.fingerprint_value,
            )
        )
    return candidates


def _model_type_for_slot(slot_id: str) -> str:
    normalized = slot_id.strip().casefold().replace("-", "_")
    if normalized in {"vae"} or normalized.endswith("_vae"):
        return "vae"
    if normalized in {"clip", "text_encoder", "text_encoders"} or normalized.endswith(
        "_text_encoder"
    ):
        return "text_encoder"
    if "lora" in normalized:
        return "lora"
    if "embedding" in normalized:
        return "embedding"
    if "control" in normalized:
        return "controlnet"
    if "upscal" in normalized:
        return "upscaler"
    return "checkpoint"


def _snapshots_by_key(value: object, key: str) -> dict[object, dict[str, Any]]:
    if not isinstance(value, list):
        return {}
    return {
        snapshot[key]: snapshot
        for snapshot in value
        if isinstance(snapshot, dict) and key in snapshot
    }


def revalidate_model_reference(
    reference: dict[str, Any],
    resolved: ExternalModelItem,
    fingerprints: ModelFingerprintService,
) -> None:
    if reference.get("stable_model_id") != resolved.stable_model_id:
        raise ModelReferenceError("model_reference_rebound")
    if reference.get("root_id") != resolved.root_id:
        raise ModelReferenceError("model_root_rebound")
    current = fingerprints.fast_fingerprint(
        resolved.execution_path,
        root_id=resolved.root_id,
        model_type=resolved.model_type,
        relative_path=resolved.normalized_relative_path,
    )
    if (
        reference.get("fingerprint_algorithm") != current.algorithm
        or int(reference.get("fingerprint_version", -1)) != current.version
        or reference.get("fingerprint_value") != current.value
    ):
        raise ModelReferenceError("model_fingerprint_changed")
