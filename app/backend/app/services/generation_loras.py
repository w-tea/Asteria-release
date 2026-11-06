from pathlib import Path
from typing import Any

from app.services.comfy_model_paths import comfy_model_name
from app.services.external_model_libraries import ExternalModelRoot
from app.services.model_library import COMFY_WEIGHT_EXTENSIONS, ModelLibrary


def _name_key(value: str) -> str:
    return value.replace("\\", "/").casefold()


def _without_weight_extension(value: str) -> str:
    suffix = Path(value).suffix
    return value[:-len(suffix)] if suffix in COMFY_WEIGHT_EXTENSIONS else value


def merge_generation_loras(
    prompt_loras: list[dict],
    request_loras: list[dict],
    *,
    library: ModelLibrary | None = None,
    model_root: Path | None = None,
    external_roots: tuple[ExternalModelRoot, ...] = (),
    weight_overrides: dict[str, float] | None = None,
    applied_prompt_overrides: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    def runtime_name(name: str) -> str:
        return comfy_model_name(name, "loras", model_root=model_root, external_roots=external_roots)

    catalogue = library.by_type("lora") if library and prompt_loras else []
    exact = {_name_key(str(path)): item for item in catalogue for path in (item.path, item.relative_path)}
    aliases: dict[str, dict[Path, str]] = {}
    for item in catalogue:
        path = _name_key(runtime_name(str(item.relative_path)))
        for alias in {path, _without_weight_extension(path), path.rsplit("/", 1)[-1],
                      _without_weight_extension(path.rsplit("/", 1)[-1])}:
            aliases.setdefault(alias, {})[item.path] = str(item.relative_path)

    def resolve(name: str) -> tuple[str, str]:
        runtime = runtime_name(name)
        key = _name_key(runtime)
        item = exact.get(_name_key(name))
        matches = aliases.get(key, {})
        registered = str(item.relative_path) if item else next(iter(matches.values())) if len(matches) == 1 else name
        registered_runtime = runtime_name(registered)
        # Keep the supplied spelling if it already selects the same loader input.
        return _name_key(registered_runtime), name if runtime == registered_runtime else registered

    selected: dict[str, dict[str, Any]] = {}
    for lora in request_loras:
        key, name = resolve(str(lora["name"]))
        selected[key] = {**lora, "name": name}
    merged: dict[str, dict[str, Any]] = {}
    explicit_clip: set[str] = set()
    prompt_names: dict[str, set[str]] = {}
    for tag in prompt_loras:
        key, name = resolve(str(tag["name"]))
        if applied_prompt_overrides is not None:
            prompt_names.setdefault(key, set()).add(str(tag["name"]))
        current = merged.setdefault(key, selected.get(key, {"name": name}))
        if "weight" in tag:
            if key not in explicit_clip and current.get("clip_weight") == current.get("weight", 1.0):
                current["clip_weight"] = tag["weight"]
            current["weight"] = tag["weight"]
        if "clip_weight" in tag:
            current["clip_weight"] = tag["clip_weight"]
            explicit_clip.add(key)
        if "block_weight" in tag or "block_weight_preset" in tag:
            current.update(
                block_weight=tag.get("block_weight", ""),
                block_weight_preset=tag.get("block_weight_preset", ""),
                hierarchy_mode="chain",
            )
        current.setdefault("weight", 1.0)
    # Preserve the established chain order: prompt references, then GUI-only selections.
    merged.update((key, row) for key, row in selected.items() if key not in merged)
    for name, weight in (weight_overrides or {}).items():
        key, _ = resolve(name)
        if key in merged:
            row = merged[key]
            if applied_prompt_overrides is not None and key in prompt_names and row["weight"] != weight:
                clip_weight = row.get("clip_weight")
                applied_prompt_overrides.append({
                    "name": row["name"],
                    "prompt_names": sorted(prompt_names[key]),
                    "weight": weight,
                    "clip_weight": weight if clip_weight is None else clip_weight,
                })
            row["weight"] = weight
    return list(merged.values())
