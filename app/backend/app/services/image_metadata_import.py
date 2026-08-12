from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

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
from app.services.image_metadata_text import (
    first_generation_parameters_text,
    metadata_value_to_text,
    read_image_metadata_values,
)
from app.services.metadata import parse_generation_parameters_text
from app.services.model_library import ModelLibrary, ModelLibraryItem


MAX_SIDECAR_BYTES = 4 * 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_MEMBERS = 10_000
SUPPORTED_ASTERIA_SCHEMAS = {"asteria.generation.v1"}


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

    embedded_asteria = _json_object(image_info.get("Asteria"))
    if embedded_asteria is not None:
        sources.append(("asteria", _canonical_payload(embedded_asteria)))

    parameters_text = first_generation_parameters_text(image_info, metadata_values)
    if parameters_text:
        parsed = parse_generation_parameters_text(parameters_text)
        if parsed is not None:
            parsed["_negative_present"] = "Negative prompt:" in parameters_text
            sources.append(("a1111_forge", _canonical_payload(parsed)))

    comfy_graph = _json_object(image_info.get("prompt"))
    if comfy_graph is None:
        comfy_graph = _json_object(image_info.get("workflow"))
    if comfy_graph is not None:
        mapped, comfy_warnings = _map_comfy_graph(comfy_graph)
        warnings.extend(comfy_warnings)
        if mapped:
            sources.append(("comfyui", mapped))
        else:
            unsupported.append("comfy_graph_unrecognized")

    merged = _merge_sources(sources)
    settings, setting_warnings = _compatible_settings(merged)
    warnings.extend(setting_warnings)
    resources = resolve_resource_references(
        _resource_references(merged),
        library=model_library,
        available_style_ids=available_style_ids,
        available_preset_ids=available_preset_ids,
        available_workflow_ids=available_workflow_ids,
    )
    positive = _metadata_text(merged, "prompt")
    negative = _metadata_text(merged, "negative_prompt")
    seed = _integer_value(merged.get("seed"))
    recognized = _recognized_codes(positive, negative, settings, seed, resources)

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
        negative_prompt=negative,
        settings=settings,
        seed=seed,
        resources=resources,
        recognized_codes=recognized,
        warning_codes=list(dict.fromkeys(warnings)),
        unsupported_codes=list(dict.fromkeys(unsupported)),
    )


def _json_object(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    text = metadata_value_to_text(value).strip()
    if not text or text[0] != "{":
        return None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


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
    if isinstance(value, dict):
        maximum = depth
        members = len(value)
        for item in value.values():
            child_depth, child_members = _json_shape(item, depth + 1)
            maximum = max(maximum, child_depth)
            members += child_members
        return maximum, members
    if isinstance(value, list):
        maximum = depth
        members = len(value)
        for item in value:
            child_depth, child_members = _json_shape(item, depth + 1)
            maximum = max(maximum, child_depth)
            members += child_members
        return maximum, members
    return depth, 0


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


def _resource_references(payload: dict[str, Any]) -> list[ResourceReference]:
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
            weight = float(item.get("weight", 1.0))
            resources.append(
                LoRAReference(
                    source_name=name,
                    status="missing",
                    order=index,
                    enabled=bool(item.get("enabled", True)),
                    weight=weight,
                    clip_weight=float(item.get("clip_weight", item.get("clipWeight", weight))),
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
                resources.append(
                    InformationalReference(
                        kind="preset",
                        application="advanced_style",
                        source_name=style_id,
                        status="missing",
                        order=index,
                        weight=float(item.get("weight", 1.0)),
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
