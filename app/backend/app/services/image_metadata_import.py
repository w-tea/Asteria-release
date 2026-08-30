from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
from math import isfinite
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from app.schemas.image_metadata import (
    CompatibleTxt2ImgSettings,
    HiresCheckpointReference,
    ImageMetadataPreview,
    ImageMetadataSource,
    InformationalReference,
    LoRAReference,
    MetadataTextAbsent,
    MetadataTextPresent,
    ProfileModelSlotReference,
    RefinerCheckpointReference,
    ResourceReference,
)
from app.schemas.prompt_input import PromptInput
from app.schemas.prompt_layout import PromptLayout
from app.services.image_metadata_text import (
    first_generation_parameters_text,
    metadata_value_to_text,
    read_image_metadata_values,
)
from app.services.metadata import parse_generation_parameters_text
from app.services.model_library import ModelLibrary, ModelLibraryItem
from app.services.spatial_asset_resolver import (
    SpatialAssetResolutionError,
    resolve_spatial_mask_source,
)
from app.services.spatial_layout_import import import_legacy_anima_layout
from app.services.spatial_prompt_contract import recover_saved_spatial_prompt_contract


MAX_SIDECAR_BYTES = 4 * 1024 * 1024
MAX_EMBEDDED_METADATA_BYTES = 4 * 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_MEMBERS = 10_000
SUPPORTED_ASTERIA_SCHEMAS = {"asteria.generation.v1"}


@dataclass(frozen=True, slots=True)
class _SpatialPromptPreviewState:
    prompt: str
    prompt_input: PromptInput | None
    prompt_layout: PromptLayout | None
    status: Literal["ready", "partial", "legacy", "unsupported"] = "ready"
    warning_code: str | None = None
    legacy_snapshot: dict[str, Any] | None = None


def build_image_metadata_preview(
    path: Path,
    *,
    preview_token: str,
    expires_at: datetime,
    history_payload: dict[str, object] | None = None,
    model_library: ModelLibrary | None = None,
    available_style_ids: set[str] | None = None,
    available_preset_ids: set[str] | None = None,
    available_workflow_ids: set[str] | None = None,
    sidecar_path: Path | None = None,
    data_dir: Path | None = None,
) -> ImageMetadataPreview:
    image_info, metadata_values, image_format, width, height = read_image_metadata_values(path)
    warnings: list[str] = []
    unsupported: list[str] = []
    sources: list[tuple[str, dict[str, Any]]] = []

    if isinstance(history_payload, dict):
        sources.append(("asteria", _canonical_payload(history_payload)))

    sidecar = _read_associated_sidecar(path, warnings, sidecar_path=sidecar_path)
    if sidecar is not None:
        sources.append(("asteria", _canonical_payload(sidecar)))

    embedded_asteria = _json_object(
        image_info.get("Asteria"),
        warnings,
        warning_prefix="embedded_asteria",
    )
    if embedded_asteria is not None:
        sources.append(("asteria", _canonical_payload(embedded_asteria)))

    parameters_text = first_generation_parameters_text(image_info, metadata_values)
    if parameters_text:
        parsed = parse_generation_parameters_text(parameters_text)
        if parsed is not None:
            parsed["_negative_present"] = "Negative prompt:" in parameters_text
            sources.append(("a1111_forge", _canonical_payload(parsed)))

    comfy_graph = _json_object(
        image_info.get("prompt"),
        warnings,
        warning_prefix="embedded_comfy",
    )
    if comfy_graph is None:
        comfy_graph = _json_object(
            image_info.get("workflow"),
            warnings,
            warning_prefix="embedded_comfy",
        )
    if comfy_graph is not None:
        mapped, comfy_warnings = _map_comfy_graph(comfy_graph)
        warnings.extend(comfy_warnings)
        if mapped:
            sources.append(("comfyui", mapped))
        else:
            unsupported.append("comfy_graph_unrecognized")

    merged = _merge_sources(sources)
    positive_selection = _select_positive_prompt_state(sources, data_dir=data_dir)
    if positive_selection is None:
        merged.pop("prompt", None)
        prompt_input = None
        prompt_layout = None
        spatial_prompt_status = "ready"
        spatial_prompt_warning = None
        spatial_prompt_legacy_snapshot = None
    else:
        merged["prompt"] = positive_selection.prompt
        prompt_input = positive_selection.prompt_input
        prompt_layout = positive_selection.prompt_layout
        spatial_prompt_status = positive_selection.status
        spatial_prompt_warning = positive_selection.warning_code
        spatial_prompt_legacy_snapshot = positive_selection.legacy_snapshot
        if spatial_prompt_warning:
            warnings.append(spatial_prompt_warning)
    settings, setting_warnings = _compatible_settings(merged)
    warnings.extend(setting_warnings)
    resources = resolve_resource_references(
        _resource_references(merged, warnings),
        library=model_library,
        available_style_ids=available_style_ids,
        available_preset_ids=available_preset_ids,
        available_workflow_ids=available_workflow_ids,
    )
    positive = _metadata_text(merged, "prompt")
    negative = _metadata_text(merged, "negative_prompt")
    seed = _integer_value(merged.get("seed"))
    recognized = _recognized_codes(
        positive,
        negative,
        settings,
        seed,
        resources,
        has_prompt_layout=prompt_layout is not None,
    )

    family = sources[0][0] if sources else "unknown"
    if not recognized:
        status = "unsupported"
    elif warnings or unsupported:
        status = "partial"
    else:
        status = "complete"

    return ImageMetadataPreview(
        preview_token=preview_token,
        expires_at=expires_at,
        source=ImageMetadataSource(
            filename=path.name,
            format=image_format,
            width=width,
            height=height,
        ),
        source_family=family,
        status=status,
        positive_prompt=positive,
        prompt_input=prompt_input,
        prompt_layout=prompt_layout,
        spatial_prompt_status=spatial_prompt_status,
        spatial_prompt_warning_code=spatial_prompt_warning,
        spatial_prompt_legacy_snapshot=spatial_prompt_legacy_snapshot,
        negative_prompt=negative,
        settings=settings,
        seed=seed,
        resources=resources,
        recognized_codes=recognized,
        warning_codes=list(dict.fromkeys(warnings)),
        unsupported_codes=list(dict.fromkeys(unsupported)),
    )


def _json_object(
    value: object,
    warnings: list[str],
    *,
    warning_prefix: str,
) -> dict[str, Any] | None:
    if isinstance(value, dict):
        parsed: object = {str(key): item for key, item in value.items()}
    else:
        text = metadata_value_to_text(value).strip()
        if not text or text[0] != "{":
            return None
        if len(text) > MAX_EMBEDDED_METADATA_BYTES or len(
            text.encode("utf-8", errors="replace")
        ) > MAX_EMBEDDED_METADATA_BYTES:
            warnings.append(f"{warning_prefix}_too_large")
            return None
        try:
            parsed = json.loads(text)
        except RecursionError:
            warnings.append(f"{warning_prefix}_too_deep")
            return None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    if not isinstance(parsed, dict):
        return None
    depth, members = _json_shape(parsed)
    if depth > MAX_JSON_DEPTH:
        warnings.append(f"{warning_prefix}_too_deep")
        return None
    if members > MAX_JSON_MEMBERS:
        warnings.append(f"{warning_prefix}_too_many_members")
        return None
    return parsed


def _read_associated_sidecar(
    path: Path,
    warnings: list[str],
    *,
    sidecar_path: Path | None = None,
) -> dict[str, Any] | None:
    sidecar = sidecar_path or path.with_suffix(".asteria.json")
    try:
        if not sidecar.is_file():
            return None
        if sidecar.stat().st_size > MAX_SIDECAR_BYTES:
            warnings.append("sidecar_too_large")
            return None
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except RecursionError:
        warnings.append("sidecar_too_deep")
        return None
    except (OSError, json.JSONDecodeError, UnicodeError):
        warnings.append("sidecar_unreadable")
        return None
    if not isinstance(payload, dict):
        warnings.append("sidecar_invalid_root")
        return None
    depth, members = _json_shape(payload)
    if depth > MAX_JSON_DEPTH:
        warnings.append("sidecar_too_deep")
        return None
    if members > MAX_JSON_MEMBERS:
        warnings.append("sidecar_too_many_members")
        return None
    if payload.get("schema") not in SUPPORTED_ASTERIA_SCHEMAS:
        warnings.append("sidecar_schema_unsupported")
        return None
    if not _sidecar_matches_image(payload, path):
        warnings.append("sidecar_unassociated")
        return None
    return {str(key): value for key, value in payload.items()}


def _json_shape(value: object, depth: int = 1) -> tuple[int, int]:
    maximum = depth
    members = 0
    pending: list[tuple[object, int]] = [(value, depth)]
    while pending:
        current, current_depth = pending.pop()
        maximum = max(maximum, current_depth)
        if maximum > MAX_JSON_DEPTH:
            return maximum, members
        if isinstance(current, dict):
            members += len(current)
            children = current.values()
        elif isinstance(current, list):
            members += len(current)
            children = current
        else:
            continue
        if members > MAX_JSON_MEMBERS:
            return maximum, members
        pending.extend((child, current_depth + 1) for child in children)
    return maximum, members


def _sidecar_matches_image(payload: dict[str, Any], image_path: Path) -> bool:
    candidates: list[object] = []
    for key in ("output_path", "filename"):
        if key in payload:
            candidates.append(payload[key])
    output_paths = payload.get("output_paths")
    if isinstance(output_paths, list):
        candidates.extend(output_paths)
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        candidate_path = Path(candidate)
        if candidate_path.name.casefold() == image_path.name.casefold():
            return True
        try:
            if candidate_path.resolve() == image_path.resolve():
                return True
        except OSError:
            continue
    return False


def _canonical_payload(payload: dict[str, object]) -> dict[str, Any]:
    canonical = {str(key): value for key, value in payload.items()}
    settings = canonical.get("settings")
    canonical["settings"] = dict(settings) if isinstance(settings, dict) else {}
    profile = canonical.get("profile")
    if isinstance(profile, str) and profile.strip().casefold() in {"", "unknown"}:
        canonical.pop("profile", None)
        canonical["settings"].pop("profile", None)
    elif isinstance(profile, str) and "profile" not in canonical["settings"]:
        canonical["settings"]["profile"] = profile
    if "seed" not in canonical and "seed" in canonical["settings"]:
        canonical["seed"] = canonical["settings"]["seed"]
    return canonical


def _merge_sources(sources: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    merged: dict[str, Any] = {"settings": {}}
    for _family, payload in sources:
        for key, value in payload.items():
            if key == "settings" and isinstance(value, dict):
                for setting_key, setting_value in value.items():
                    merged["settings"].setdefault(setting_key, setting_value)
                continue
            if key == "negative_prompt" and not payload.get("_negative_present", True):
                continue
            if key not in merged:
                merged[key] = value
    return merged


def _select_positive_prompt_state(
    sources: list[tuple[str, dict[str, Any]]],
    *,
    data_dir: Path | None,
) -> _SpatialPromptPreviewState | None:
    for _family, payload in sources:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            continue
        settings = payload.get("settings")
        source_settings = settings if isinstance(settings, dict) else {}
        raw_prompt_input = source_settings.get("prompt_input")
        raw_prompt_layout = source_settings.get("prompt_layout")
        mode = source_settings.get("mode")
        generation_mode = mode if mode in {"txt2img", "img2img", "inpaint"} else "txt2img"
        recovery = recover_saved_spatial_prompt_contract(
            prompt,
            raw_prompt_input,
            raw_prompt_layout,
            generation_mode,
        )
        if recovery.status == "fallback":
            return _SpatialPromptPreviewState(
                prompt=prompt,
                prompt_input=recovery.snapshot.prompt_input,
                prompt_layout=None,
                status="partial",
                warning_code=recovery.warning_code,
                legacy_snapshot=recovery.legacy_snapshot,
            )

        layout = recovery.snapshot.prompt_layout
        if layout is not None and data_dir is not None:
            try:
                for target in layout.targets:
                    if target.mask_source is not None:
                        resolve_spatial_mask_source(target.mask_source, data_dir)
            except (SpatialAssetResolutionError, OSError, ValueError):
                fallback = recover_saved_spatial_prompt_contract(
                    prompt,
                    raw_prompt_input,
                    {"schema_version": "asteria.prompt-layout.invalid"},
                    generation_mode,
                )
                return _SpatialPromptPreviewState(
                    prompt=prompt,
                    prompt_input=fallback.snapshot.prompt_input,
                    prompt_layout=None,
                    status="partial",
                    warning_code="spatial_prompt_invalid_fallback",
                    legacy_snapshot={
                        "prompt_input": deepcopy(raw_prompt_input),
                        "prompt_layout": deepcopy(raw_prompt_layout),
                    },
                )

        if raw_prompt_input is None and raw_prompt_layout is None:
            legacy_units = source_settings.get("anima_lllite_units")
            if isinstance(legacy_units, list):
                for unit_index, unit in enumerate(legacy_units):
                    if not isinstance(unit, dict) or "regional_prompts" not in unit:
                        continue
                    legacy = import_legacy_anima_layout(
                        unit,
                        source_identity=f"metadata:{_family}",
                        unit_index=unit_index,
                    )
                    return _SpatialPromptPreviewState(
                        prompt=prompt,
                        prompt_input=legacy.prompt_input,
                        prompt_layout=legacy.prompt_layout,
                        status="ready" if legacy.status == "ready" else (
                            "unsupported" if legacy.status == "unsupported" else "legacy"
                        ),
                        warning_code=legacy.reason,
                        legacy_snapshot={
                            "anima_lllite_units": deepcopy(legacy_units),
                        },
                    )

        return _SpatialPromptPreviewState(
            prompt=prompt,
            prompt_input=recovery.snapshot.prompt_input,
            prompt_layout=layout,
        )
    return None


def _compatible_settings(payload: dict[str, Any]) -> tuple[CompatibleTxt2ImgSettings, list[str]]:
    source = payload.get("settings")
    source_settings = source if isinstance(source, dict) else {}
    allowed = CompatibleTxt2ImgSettings.model_fields
    filtered = {key: value for key, value in source_settings.items() if key in allowed and value is not None}
    warnings: list[str] = []
    try:
        return CompatibleTxt2ImgSettings.model_validate(filtered), warnings
    except ValidationError:
        accepted: dict[str, Any] = {}
        for key, value in filtered.items():
            try:
                CompatibleTxt2ImgSettings.model_validate({key: value})
            except ValidationError:
                warnings.append(f"setting_invalid_{key}")
            else:
                accepted[key] = value
        return CompatibleTxt2ImgSettings.model_validate(accepted), warnings


def _metadata_text(payload: dict[str, Any], key: str) -> MetadataTextAbsent | MetadataTextPresent:
    if key not in payload or not isinstance(payload.get(key), str):
        return MetadataTextAbsent()
    return MetadataTextPresent(value=str(payload[key]))


def _integer_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None and str(value).strip() else None
    except (TypeError, ValueError):
        return None


def _resource_references(
    payload: dict[str, Any],
    warnings: list[str],
) -> list[ResourceReference]:
    resources: list[ResourceReference] = []
    model_slots = payload.get("model_slots")
    if isinstance(model_slots, dict):
        for slot_id, value in model_slots.items():
            if not isinstance(value, str) or not value.strip():
                continue
            resources.append(
                ProfileModelSlotReference(
                    kind="vae" if "vae" in str(slot_id).casefold() else "checkpoint",
                    slot_id=str(slot_id),
                    source_name=value,
                    status="missing",
                )
            )
    settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
    for key, model in (
        ("refiner_checkpoint", RefinerCheckpointReference),
        ("hires_checkpoint", HiresCheckpointReference),
    ):
        value = settings.get(key)
        if isinstance(value, str) and value.strip() and not value.casefold().startswith("use same"):
            resources.append(model(source_name=value, status="missing"))
    loras = payload.get("loras")
    if isinstance(loras, list):
        for index, item in enumerate(loras):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            weight = _metadata_weight(
                item["weight"] if "weight" in item else 1.0,
                minimum=-10.0,
                maximum=10.0,
            )
            if weight is None:
                warnings.append("resource_invalid_lora_weight")
                continue
            if "clip_weight" in item:
                clip_weight_value = item["clip_weight"]
            elif "clipWeight" in item:
                clip_weight_value = item["clipWeight"]
            else:
                clip_weight_value = weight
            clip_weight = _metadata_weight(
                clip_weight_value,
                minimum=-10.0,
                maximum=10.0,
            )
            if clip_weight is None:
                warnings.append("resource_invalid_lora_clip_weight")
                continue
            resources.append(
                LoRAReference(
                    source_name=name,
                    status="missing",
                    order=index,
                    enabled=bool(item.get("enabled", True)),
                    weight=weight,
                    clip_weight=clip_weight,
                    block_weight=str(item.get("block_weight", item.get("blockWeight", "")) or ""),
                    block_weight_preset=str(
                        item.get("block_weight_preset", item.get("blockWeightPreset", "")) or ""
                    ),
                    hierarchy_mode=str(item.get("hierarchy_mode", item.get("hierarchyMode", "chain")) or "chain"),
                )
            )
    embeddings = payload.get("embeddings")
    if isinstance(embeddings, list):
        for name in embeddings:
            if isinstance(name, str) and name.strip():
                resources.append(
                    InformationalReference(
                        kind="embedding",
                        application="prompt_embedded",
                        source_name=name,
                        status="missing",
                    )
                )
    settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
    advanced_styles = settings.get("advanced_styles")
    if isinstance(advanced_styles, list):
        for index, item in enumerate(advanced_styles):
            if not isinstance(item, dict):
                continue
            style_id = str(item.get("id") or item.get("name") or "").strip()
            if style_id:
                weight = _metadata_weight(item["weight"] if "weight" in item else 1.0)
                if weight is None:
                    warnings.append("resource_invalid_advanced_style_weight")
                    continue
                resources.append(
                    InformationalReference(
                        kind="preset",
                        application="advanced_style",
                        source_name=style_id,
                        status="missing",
                        order=index,
                        weight=weight,
                    )
                )
    for key, kind in (
        ("preset_id", "preset"),
        ("workflow_template_id", "workflow"),
        ("workflow_mapping_id", "workflow"),
    ):
        value = settings.get(key)
        if isinstance(value, str) and value.strip():
            resources.append(
                InformationalReference(
                    kind=kind,
                    application="preview_only",
                    source_name=value,
                    status="missing",
                )
            )
    return resources


def _metadata_weight(
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not isfinite(parsed):
        return None
    if minimum is not None and parsed < minimum:
        return None
    if maximum is not None and parsed > maximum:
        return None
    return parsed


def resolve_resource_references(
    resources: list[ResourceReference],
    *,
    library: ModelLibrary | None,
    available_style_ids: set[str] | None = None,
    available_preset_ids: set[str] | None = None,
    available_workflow_ids: set[str] | None = None,
) -> list[ResourceReference]:
    resolved: list[ResourceReference] = []
    for reference in resources:
        if reference.status == "unsupported":
            resolved.append(reference)
            continue
        if isinstance(reference, InformationalReference):
            identifiers: set[str] | None = None
            if reference.application == "advanced_style":
                identifiers = available_style_ids
            elif reference.kind == "preset":
                identifiers = available_preset_ids
            elif reference.kind == "workflow":
                identifiers = available_workflow_ids
            if identifiers is not None:
                match = next(
                    (item for item in identifiers if item.casefold() == reference.source_name.casefold()),
                    None,
                )
                resolved.append(
                    reference.model_copy(
                        update={
                            "status": "available" if match else "missing",
                            "resolved_id": match,
                        }
                    )
                )
                continue
            if reference.kind != "embedding":
                resolved.append(reference)
                continue
        if library is None:
            resolved.append(reference)
            continue
        model_type = _resource_model_type(reference)
        if model_type is None:
            resolved.append(reference)
            continue
        candidates = [item for item in library.items if _normalized_model_type(item.type) == model_type]
        exact = [item for item in candidates if _resource_exact_match(reference.source_name, item)]
        matches = exact if exact else [
            item for item in candidates if item.path.name.casefold() == Path(reference.source_name).name.casefold()
        ]
        if len(matches) == 1:
            item = matches[0]
            resolved.append(
                reference.model_copy(
                    update={
                        "status": "available",
                        "resolved_id": item.stable_model_id or item.relative_path.as_posix(),
                        "resolved_path": str(item.path),
                    }
                )
            )
        elif len(matches) > 1:
            resolved.append(reference.model_copy(update={"status": "ambiguous", "resolved_id": None, "resolved_path": None}))
        else:
            resolved.append(reference.model_copy(update={"status": "missing", "resolved_id": None, "resolved_path": None}))
    return resolved


def _resource_model_type(reference: ResourceReference) -> str | None:
    if isinstance(reference, LoRAReference):
        return "lora"
    if isinstance(reference, InformationalReference):
        return "embedding" if reference.kind == "embedding" else None
    if isinstance(reference, ProfileModelSlotReference):
        return _normalized_model_type(reference.kind)
    if isinstance(reference, (RefinerCheckpointReference, HiresCheckpointReference)):
        return "checkpoint"
    return None


def _normalized_model_type(value: str) -> str:
    return "checkpoint" if value == "diffusion_model" else value


def _resource_exact_match(source_name: str, item: ModelLibraryItem) -> bool:
    source = source_name.replace("\\", "/").lstrip("./").casefold()
    values = {
        str(item.path).replace("\\", "/").casefold(),
        item.relative_path.as_posix().lstrip("./").casefold(),
        item.name.casefold(),
    }
    external_relative = item.metadata.get("external_relative_path")
    if external_relative:
        values.add(external_relative.replace("\\", "/").lstrip("./").casefold())
    if item.stable_model_id:
        values.add(item.stable_model_id.casefold())
    return source in values


def _recognized_codes(
    positive: MetadataTextAbsent | MetadataTextPresent,
    negative: MetadataTextAbsent | MetadataTextPresent,
    settings: CompatibleTxt2ImgSettings,
    seed: int | None,
    resources: list[ResourceReference],
    *,
    has_prompt_layout: bool = False,
) -> list[str]:
    codes: list[str] = []
    if positive.presence == "present":
        codes.append("positive_prompt")
    if negative.presence == "present":
        codes.append("negative_prompt")
    codes.extend(f"setting_{key}" for key in settings.model_dump(exclude_none=True))
    if seed is not None:
        codes.append("seed")
    if resources:
        codes.append("resources")
    if has_prompt_layout:
        codes.append("prompt_layout")
    return codes


def _map_comfy_graph(graph: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    samplers = [node for node in graph.values() if _node_class(node) in {"KSampler", "KSamplerAdvanced"}]
    if len(samplers) != 1:
        return {}, ["comfy_sampler_ambiguous"] if samplers else ["comfy_sampler_missing"]
    sampler = samplers[0]
    inputs = sampler.get("inputs") if isinstance(sampler, dict) else None
    if not isinstance(inputs, dict):
        return {}, ["comfy_sampler_inputs_invalid"]
    settings: dict[str, Any] = {}
    for source_key, target_key in (
        ("steps", "steps"),
        ("cfg", "cfg_scale"),
        ("sampler_name", "sampler"),
        ("scheduler", "scheduler"),
    ):
        if source_key in inputs:
            settings[target_key] = inputs[source_key]
    latent = _linked_node(graph, inputs.get("latent_image"))
    latent_inputs = latent.get("inputs") if isinstance(latent, dict) else None
    if isinstance(latent_inputs, dict):
        for key in ("width", "height", "batch_size"):
            if key in latent_inputs:
                settings[key] = latent_inputs[key]
    positive = _linked_text(graph, inputs.get("positive"))
    negative = _linked_text(graph, inputs.get("negative"))
    model_slots: dict[str, str] = {}
    model_node = _linked_node(graph, inputs.get("model"))
    model_inputs = model_node.get("inputs") if isinstance(model_node, dict) else None
    if isinstance(model_inputs, dict):
        checkpoint = model_inputs.get("ckpt_name")
        if isinstance(checkpoint, str) and checkpoint.strip():
            model_slots["checkpoint"] = checkpoint
    payload: dict[str, Any] = {
        "settings": settings,
        "model_slots": model_slots,
        "seed": inputs.get("seed", inputs.get("noise_seed")),
    }
    if positive is not None:
        payload["prompt"] = positive
    if negative is not None:
        payload["negative_prompt"] = negative
    return payload, []


def _node_class(value: object) -> str:
    return str(value.get("class_type") or "") if isinstance(value, dict) else ""


def _linked_node(graph: dict[str, Any], link: object) -> dict[str, Any] | None:
    if not isinstance(link, list) or not link:
        return None
    node = graph.get(str(link[0]))
    return node if isinstance(node, dict) else None


def _linked_text(graph: dict[str, Any], link: object) -> str | None:
    node = _linked_node(graph, link)
    inputs = node.get("inputs") if isinstance(node, dict) else None
    text = inputs.get("text") if isinstance(inputs, dict) else None
    return text if isinstance(text, str) else None
