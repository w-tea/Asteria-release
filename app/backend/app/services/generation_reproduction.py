"""Content identities for selected generation assets; no arbitrary path reads."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.detailer_catalog import detailer_detector_model
from app.services.model_fingerprints import ModelFingerprintService
from app.services.model_library import MODEL_TYPE_DIRECTORIES, ModelLibrary, ModelLibraryItem
from app.services.upscaler_catalog import upscaler_model_name

_SLOT_TYPES = {"checkpoint": "checkpoint", "diffusion_model": "checkpoint", "vae": "vae", "text_encoder": "text_encoder",
               "clip_l": "text_encoder", "t5xxl": "text_encoder", "qwen35_text_encoder": "text_encoder",
               "llm_adapter": "text_encoder", "progressive_adapter": "text_encoder"}
_SPECIAL = {"", "automatic", "none", "use same checkpoint", "use same vae"}


def _normalize(value: str) -> str:
    return value.replace("\\", "/").casefold()


def _candidates(reference: str, kind: str, library: ModelLibrary) -> list[ModelLibraryItem]:
    target = _normalize(reference)
    items = library.by_type(kind)
    exact_items = [*items, *(item for item in library.aliases if item.type == kind)]
    exact = [item for item in exact_items if target in {_normalize(str(item.path)), _normalize(str(item.relative_path))}]
    if kind == 'detailer' and target.startswith(('bbox/', 'segm/')):
        # Impact uses a family prefix plus a path relative to its registered roots.
        # Do not prefer one of several different files with the same runtime name.
        family, name = target.split('/', 1)
        exact = []
        for item in exact_items:
            relative = _normalize(str(item.relative_path))
            for directory in ('adetailer', family, f'ultralytics/{family}'):
                if relative == f'{directory}/{name}':
                    exact.append(item)
                    break
    if not exact:
        exact = []
        for item in items:
            relative = _normalize(str(item.relative_path))
            aliases = {relative, item.name.casefold()}
            for directory in MODEL_TYPE_DIRECTORIES.get(kind, ()):
                prefix = directory.casefold() + "/"
                if relative.startswith(prefix):
                    aliases.add(relative[len(prefix):])
            if relative.startswith("external/"):
                aliases.add("/".join(relative.split("/")[3:]))
            if kind == "embedding":
                aliases |= {str(Path(name).with_suffix("")) for name in list(aliases)}
            if target in aliases:
                exact.append(item)
    # A local link and a shared-library registration may refer to the same file.
    unique = {}
    for item in exact:
        try:
            key = _normalize(str(item.path.resolve()))
        except OSError:
            key = _normalize(str(item.path))
        unique.setdefault(key, item)
    return list(unique.values())


def _hash(item: ModelLibraryItem, hashes: ModelFingerprintService, *, refresh: bool) -> dict:
    before = item.path.stat()
    digest = hashes.strong_hash(item.path, refresh=refresh)
    after = item.path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if identity(before) != identity(after):
        raise OSError("asset_changed_while_hashing")
    return {"path": str(item.path), "relative_path": str(item.relative_path), "root_id": item.root_id,
            "stable_model_id": item.stable_model_id, "size_bytes": after.st_size, "sha256": digest}


def _references(slots, settings, loras, embeddings, slot_types):
    refs = [(role, slot_types.get(role, _SLOT_TYPES.get(role, "checkpoint")), value) for role, value in slots.items()]
    for index, lora in enumerate(loras):
        if isinstance(lora, dict) and lora.get("enabled", True):
            refs.append((f"lora:{index}", "lora", lora.get("name", "")))
    refs.extend((f"embedding:{index}", "embedding", name) for index, name in enumerate(embeddings))
    if settings.get("hires_enabled"):
        refs.append(("hires_checkpoint", "checkpoint", settings.get("hires_checkpoint", "")))
        refs.append(("hires_upscaler", "upscaler", upscaler_model_name(settings.get("hires_upscaler"), settings.get("hires_upscaler_model")) or ""))
    if settings.get("refiner_enabled"):
        refs.append(("refiner_checkpoint", "checkpoint", settings.get("refiner_checkpoint", "")))
    for group, kind, key in (("controlnet_units", "controlnet", "model"), ("anima_lllite_units", "controlnet", "model"), ("detailer_units", "detailer", "detector")):
        for index, unit in enumerate(settings.get(group) or []):
            if not isinstance(unit, dict) or not unit.get("enabled", True):
                continue
            value = unit.get(key, "")
            if kind == "detailer":
                value = detailer_detector_model(value)
            refs.append((f"{group}:{index}", kind, value))
    return [(role, kind, value) for role, kind, value in refs if isinstance(value, str) and value.casefold() not in _SPECIAL]


def capture_generation_assets(slots: dict, settings: dict, loras: list, embeddings: list,
                              library: ModelLibrary, hashes: ModelFingerprintService, *, slot_types: dict | None = None,
                              allow_identical_aliases: bool = False) -> dict:
    assets = []
    for role, kind, reference in _references(slots, settings, loras, embeddings, slot_types or {}):
        entry: dict[str, Any] = {"role": role, "type": kind, "reference": reference}
        candidates = _candidates(reference, kind, library)
        if allow_identical_aliases and len(candidates) > 1:
            try:
                identities = [_hash(item, hashes, refresh=False) for item in candidates]
            except OSError:
                identities = []
            if identities and len({item['sha256'] for item in identities}) == 1:
                # Freeze every candidate, without claiming which runtime path was selected.
                assets.extend({**entry, **item, 'status': 'recorded', 'resolution': 'identical_content_candidates'} for item in identities)
                continue
        if len(candidates) != 1:
            entry["status"] = "ambiguous" if candidates else "unavailable"
        else:
            try:
                entry.update(_hash(candidates[0], hashes, refresh=False), status="recorded")
            except OSError:
                entry["status"] = "unavailable"
        assets.append(entry)
    return {"version": 1, "basis": "selected_assets_at_operation_start", "assets": assets}


def compare_generation_assets(snapshot: object, library: ModelLibrary, hashes: ModelFingerprintService) -> dict:
    if not isinstance(snapshot, dict) or snapshot.get("version") != 1 or not isinstance(snapshot.get("assets"), list):
        return {"status": "unrecorded", "assets": []}
    assets = []
    current_files: dict[str, dict] = {}
    for saved in snapshot["assets"]:
        if not isinstance(saved, dict):
            continue
        reference = saved.get("reference", "")
        kind = saved.get("type", "")
        if not isinstance(reference, str) or not isinstance(kind, str):
            continue
        result = {"role": saved.get("role", ""), "type": kind, "reference": reference,
                  "recorded_sha256": saved.get("sha256"), "status": "unknown"}
        candidates = _candidates(reference, kind, library)
        if len(candidates) != 1:
            result["status"] = "ambiguous" if candidates else "missing" if saved.get("sha256") else "unknown"
        else:
            try:
                key = str(candidates[0].path.resolve()).casefold()
                if key not in current_files:
                    current_files[key] = _hash(candidates[0], hashes, refresh=True)
                current = current_files[key]
                result.update(current_sha256=current["sha256"], current_path=current["path"])
                if isinstance(saved.get("sha256"), str):
                    result["status"] = "same" if saved["sha256"] == current["sha256"] else "changed"
            except OSError:
                result["status"] = "missing"
        assets.append(result)
    return {"status": "checked", "assets": assets}
