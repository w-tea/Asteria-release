from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote, unquote

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from app.services.image_metadata_text import first_generation_parameters_text, read_image_metadata_values


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
PARAMETER_LABELS = [
    "Raw prompt",
    "Raw negative prompt",
    "Prompt selections",
    "Template",
    "Negative Template",
    "Steps",
    "Sampler",
    "Schedule type",
    "CFG scale",
    "Distilled CFG scale",
    "Distilled CFG Scale",
    "Seed",
    "Variation seed",
    "Variation seed strength",
    "Seed resize from",
    "Size",
    "Batch size",
    "Batch count",
    "Eta",
    "ENSD",
    "Eta noise seed delta",
    "RNG",
    "Generated seeds",
    "Profile",
    "Workflow template",
    "Workflow mapping",
    "Mode",
    "Model",
    "VAE",
    "Model slots",
    "Model override options",
    "Clip skip",
    "Face restoration",
    "Tiling",
    "Asteria generation",
    "Runtime adapter",
    "Engine info",
    "Runtime details",
    "Runtime job",
    "Runtime jobs",
    "Asteria tags",
    "Hires upscale",
    "Hires resize",
    "Hires steps",
    "Hires upscaler",
    "Hires checkpoint",
    "Denoising strength",
    "Hires CFG scale",
    "Hires sampler",
    "Hires schedule type",
    "Hires prompt",
    "Hires negative prompt",
    "Hires distilled CFG scale",
    "Hires noise mode",
    "Hires keep first pass",
    "Hires metadata",
    "Mask blur",
    "Masked content",
    "Inpaint area",
    "Refiner",
    "Styles",
    "Style options",
    "ControlNet units",
    "ControlNet",
    "ControlNet artifacts",
    "Detailer units",
    "Detailer",
    "Detailer metadata",
    "LoRAs",
    "LoRA options",
    "Embeddings",
    "Topaz postprocess",
    "Topaz Photo AI",
    "Topaz upscale",
    "Topaz adjustments",
    "Topaz options",
    "Model hash",
    "VAE hash",
    "Lora hashes",
    "TI hashes",
    "Hashes",
    "freeu_enabled",
    "freeu_b1",
    "freeu_b2",
    "freeu_s1",
    "freeu_s2",
    "sag_enabled",
    "sag_scale",
    "sag_blur_sigma",
    "Script",
    "X Type",
    "X Values",
    "Y Type",
    "Y Values",
    "Fixed Y Values",
    "Version",
]


def parse_generation_parameters_text(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    lines = stripped.splitlines()
    params_text = lines[-1].strip()
    extension_params, params_text = _extract_extension_parameter_fields(params_text)
    params = _split_parameter_fields(params_text)
    if not params:
        return None

    prompt_lines = lines[:-1]
    negative_prompt = ""
    negative_index = next(
        (index for index, line in enumerate(prompt_lines) if line.startswith("Negative prompt:")),
        None,
    )
    if negative_index is None:
        rendered_prompt = "\n".join(prompt_lines).strip()
    else:
        rendered_prompt = "\n".join(prompt_lines[:negative_index]).strip()
        first_negative = prompt_lines[negative_index].removeprefix("Negative prompt:").strip()
        negative_tail = [line.strip() for line in prompt_lines[negative_index + 1 :] if line.strip()]
        negative_prompt = "\n".join([value for value in [first_negative, *negative_tail] if value]).strip()

    raw_prompt = params.get("raw prompt")
    settings = _settings_from_parameters(params)
    _apply_extension_parameter_fields(settings, extension_params)
    if raw_prompt and raw_prompt != rendered_prompt:
        settings["raw_prompt"] = raw_prompt
    prompt_selections = _prompt_selections_from_parameters(params.get("prompt selections"))
    if prompt_selections:
        settings["prompt_expansion_selections"] = prompt_selections
    model_slots = _model_slots_from_parameters(params)
    loras = _apply_lora_options(
        _merge_loras(
            _loras_from_parameters(params.get("loras")),
            _lora_tokens_from_prompt_texts(raw_prompt, rendered_prompt, negative_prompt),
            _lora_hash_names_from_parameters(params.get("lora hashes")),
            _lora_hash_names_from_hashes_field(params.get("hashes")),
        ),
        params.get("lora options"),
    )
    embeddings = _merge_strings(
        _comma_list(params.get("embeddings")),
        _hash_names_from_parameters(params.get("ti hashes")),
        _embedding_hash_names_from_hashes_field(params.get("hashes")),
    )
    metadata_tags = _tags_from_parameters(params.get("asteria tags"))
    expanded_prompt = rendered_prompt or raw_prompt or ""
    profile = params.get("profile") or "unknown"
    parsed: dict[str, Any] = {
        "status": "completed",
        "profile": profile,
        "prompt": raw_prompt or expanded_prompt,
        "expanded_prompt": expanded_prompt,
        "negative_prompt": negative_prompt,
        "settings": settings,
        "model_slots": model_slots,
        "loras": loras,
        "embeddings": embeddings,
    }
    if metadata_tags:
        parsed["metadata_tags"] = metadata_tags
    return parsed


def read_generation_parameters_from_image(path: Path) -> dict[str, Any] | None:
    try:
        image_info, metadata_values, _format, _width, _height = read_image_metadata_values(path)
    except (OSError, ValueError):
        return None
    parameters = first_generation_parameters_text(image_info, metadata_values)
    if not parameters:
        return None
    return parse_generation_parameters_text(parameters)


def _split_parameter_fields(text: str) -> dict[str, str]:
    labels = "|".join(re.escape(label) for label in sorted(PARAMETER_LABELS, key=len, reverse=True))
    pattern = re.compile(rf"(?:^|, )(?P<label>{labels}): ", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[value_start:value_end].strip().rstrip(",").strip()
        if value:
            fields[match.group("label").lower()] = value
    return fields


def _extract_extension_parameter_fields(text: str) -> tuple[dict[str, Any], str]:
    controlnet_sections, text = _extract_controlnet_sections(text)
    adetailer_fields, text = _extract_adetailer_fields(text)
    return {"controlnet": controlnet_sections, "adetailer": adetailer_fields}, _cleanup_parameter_text(text)


def _extract_controlnet_sections(text: str) -> tuple[list[str], str]:
    starts = list(re.finditer(r"(?:^|, )ControlNet\s+\d+:\s+", text, re.IGNORECASE))
    if not starts:
        return [], text
    sections: list[str] = []
    spans: list[tuple[int, int]] = []
    for index, match in enumerate(starts):
        value_start = match.end()
        next_controlnet = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        value_end = min(
            next_controlnet,
            _next_extension_terminator(text, value_start, ignored_labels={"Model"}),
        )
        value = text[value_start:value_end].strip().rstrip(",").strip()
        if value:
            sections.append(value)
        spans.append((match.start(), value_end))
    return sections, _remove_spans(text, spans)


def _extract_adetailer_fields(text: str) -> tuple[list[dict[str, str]], str]:
    starts = list(re.finditer(r"(?:^|, )(ADetailer\s+[^:]+):\s+", text, re.IGNORECASE))
    if not starts:
        return [], text
    grouped_fields: dict[str, dict[str, str]] = {}
    group_order: list[str] = []
    spans: list[tuple[int, int]] = []
    for index, match in enumerate(starts):
        value_start = match.end()
        next_field = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        value_end = min(next_field, _next_extension_terminator(text, value_start))
        label, group_key = _adetailer_label_and_group(match.group(1))
        value = text[value_start:value_end].strip().rstrip(",").strip()
        if label and value:
            if group_key not in grouped_fields:
                grouped_fields[group_key] = {}
                group_order.append(group_key)
            grouped_fields[group_key][label] = value
        spans.append((match.start(), value_end))
    return [grouped_fields[key] for key in group_order], _remove_spans(text, spans)


def _adetailer_label_and_group(label: str) -> tuple[str, str]:
    normalized = re.sub(r"\s+", " ", label).strip().lower().removeprefix("adetailer ").strip()
    match = re.search(r"\s+((?:\d+(?:st|nd|rd|th))|(?:second|third|fourth|fifth))$", normalized)
    if match is None:
        return normalized, "0"
    group_key = match.group(1)
    field_label = normalized[: match.start()].strip()
    return field_label, group_key


def _next_extension_terminator(
    text: str,
    start: int,
    *,
    ignored_labels: set[str] | None = None,
) -> int:
    ignored = {label.lower() for label in (ignored_labels or set())}
    labels = "|".join(
        re.escape(label)
        for label in sorted(PARAMETER_LABELS, key=len, reverse=True)
        if label.lower() not in ignored
    )
    terminators = [
        r",\s*ADetailer\s+[^:]+:\s+",
        r",\s*ControlNet\s+\d+:\s+",
        rf",\s*(?:{labels}):\s+",
    ]
    positions = [
        match.start()
        for pattern in terminators
        if (match := re.search(pattern, text[start:], re.IGNORECASE)) is not None
    ]
    if not positions:
        return len(text)
    return start + min(positions)


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    cleaned = text
    for start, end in sorted(spans, reverse=True):
        cleaned = cleaned[:start] + cleaned[end:]
    return _cleanup_parameter_text(cleaned)


def _cleanup_parameter_text(text: str) -> str:
    cleaned = re.sub(r"(?:,\s*){2,}", ", ", text)
    cleaned = re.sub(r"^\s*,\s*", "", cleaned)
    cleaned = re.sub(r"\s*,\s*$", "", cleaned)
    return cleaned.strip()


def _settings_from_parameters(params: dict[str, str]) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    for key, target in (
        ("steps", "steps"),
        ("seed", "seed"),
        ("variation seed", "subseed"),
        ("batch size", "batch_size"),
        ("batch count", "batch_count"),
        ("clip skip", "clip_skip"),
        ("asteria generation", "source_generation_id"),
        ("mask blur", "mask_blur"),
        ("hires steps", "hires_steps"),
        ("controlnet units", "controlnet_units_count"),
        ("detailer units", "detailer_units_count"),
        ("ensd", "eta_noise_seed_delta"),
        ("eta noise seed delta", "eta_noise_seed_delta"),
    ):
        value = _int_value(params.get(key))
        if value is not None:
            settings[target] = value
    for key, target in (
        ("cfg scale", "cfg_scale"),
        ("distilled cfg scale", "distilled_cfg_scale"),
        ("variation seed strength", "subseed_strength"),
        ("hires upscale", "hires_scale"),
        ("hires cfg scale", "hires_cfg_scale"),
        ("hires distilled cfg scale", "hires_distilled_cfg_scale"),
        ("eta", "eta"),
        ("freeu_b1", "freeu_b1"),
        ("freeu_b2", "freeu_b2"),
        ("freeu_s1", "freeu_s1"),
        ("freeu_s2", "freeu_s2"),
        ("sag_scale", "sag_scale"),
        ("sag_blur_sigma", "sag_blur_sigma"),
    ):
        value = _float_value(params.get(key))
        if value is not None:
            settings[target] = value
    for key, target in (
        ("sampler", "sampler"),
        ("schedule type", "scheduler"),
        ("raw negative prompt", "raw_negative_prompt"),
        ("template", "prompt_template"),
        ("negative template", "negative_prompt_template"),
        ("workflow template", "workflow_template_id"),
        ("workflow mapping", "workflow_mapping_id"),
        ("mode", "mode"),
        ("runtime adapter", "runtime_adapter"),
        ("engine info", "engine_info"),
        ("runtime job", "runtime_job_id"),
        ("hires upscaler", "hires_upscaler"),
        ("hires checkpoint", "hires_checkpoint"),
        ("hires sampler", "hires_sampler"),
        ("hires schedule type", "hires_scheduler"),
        ("hires prompt", "hires_prompt"),
        ("hires negative prompt", "hires_negative_prompt"),
        ("hires noise mode", "hires_noise_mode"),
        ("topaz postprocess", "topaz_postprocess"),
        ("masked content", "masked_content"),
        ("inpaint area", "inpaint_area"),
        ("model hash", "model_hash"),
        ("vae hash", "vae_hash"),
        ("rng", "rng_source"),
        ("version", "source_application_version"),
    ):
        value = params.get(key)
        if value:
            settings[target] = value
    _normalize_inpaint_settings(settings)
    _apply_hashes_field(settings, params.get("hashes"))
    runtime_job_ids = _comma_list(params.get("runtime jobs"))
    if runtime_job_ids:
        settings["runtime_job_ids"] = runtime_job_ids
    generated_seeds = _int_list(params.get("generated seeds"))
    if generated_seeds:
        settings["generated_seeds"] = generated_seeds
    _apply_runtime_details(settings, params.get("runtime details"))
    _apply_face_restoration(settings, params.get("face restoration"))
    for key, target in (
        ("tiling", "tiling"),
        ("hires keep first pass", "hires_keep_first_pass"),
        ("freeu_enabled", "freeu_enabled"),
        ("sag_enabled", "sag_enabled"),
    ):
        value = _bool_value(params.get(key))
        if value is not None:
            settings[target] = value
    hires_metadata = _hires_metadata_from_parameters(params.get("hires metadata"))
    if hires_metadata:
        settings["hires_metadata"] = hires_metadata
    _apply_size(settings, params.get("size"), "width", "height")
    _apply_size(settings, params.get("seed resize from"), "seed_resize_from_w", "seed_resize_from_h")
    _apply_size(settings, params.get("hires resize"), "hires_resize_width", "hires_resize_height")
    denoising_strength = _float_value(params.get("denoising strength"))
    if denoising_strength is not None:
        if _has_hires_parameters(params):
            settings["hires_denoising_strength"] = denoising_strength
        else:
            settings["denoising_strength"] = denoising_strength
    if any(key.startswith("hires_") for key in settings) or "hires_denoising_strength" in settings:
        settings["hires_enabled"] = True
    refiner = params.get("refiner")
    if refiner:
        checkpoint, _, switch_at = refiner.partition(" @ ")
        if checkpoint.strip():
            settings["refiner_enabled"] = True
            settings["refiner_checkpoint"] = checkpoint.strip()
        parsed_switch = _float_value(switch_at)
        if parsed_switch is not None:
            settings["refiner_switch_at"] = parsed_switch
    controlnet_units = _controlnet_units_from_parameters(params.get("controlnet"))
    if controlnet_units:
        settings["controlnet_units"] = controlnet_units
    controlnet_artifacts = _controlnet_artifacts_from_parameters(params.get("controlnet artifacts"))
    if controlnet_artifacts:
        settings["controlnet_artifacts"] = controlnet_artifacts
    detailer_units = _detailer_units_from_parameters(params.get("detailer"))
    if detailer_units:
        settings["detailer_units"] = detailer_units
    detailer_metadata = _detailer_metadata_from_parameters(params.get("detailer metadata"))
    if detailer_metadata:
        settings["detailer_metadata"] = detailer_metadata
    model_slot_overrides = _model_override_options_from_parameters(params.get("model override options"))
    if model_slot_overrides:
        settings["model_slot_overrides"] = model_slot_overrides
    advanced_styles = _apply_advanced_style_options(
        _advanced_styles_from_parameters(params.get("styles")),
        params.get("style options"),
    )
    if advanced_styles:
        settings["advanced_styles"] = advanced_styles
    _apply_topaz_settings(settings, params)
    for key, target in (
        ("script", "source_script"),
        ("x type", "source_xyz_x_type"),
        ("x values", "source_xyz_x_values"),
        ("y type", "source_xyz_y_type"),
        ("y values", "source_xyz_y_values"),
        ("fixed y values", "source_xyz_fixed_y_values"),
    ):
        value = _unquoted_text(params.get(key))
        if value:
            settings[target] = value
    return settings


def _normalize_inpaint_settings(settings: dict[str, Any]) -> None:
    masked_content = settings.get("masked_content")
    if isinstance(masked_content, str):
        normalized = _normalized_choice(masked_content)
        if normalized:
            settings["masked_content"] = normalized
    inpaint_area = settings.get("inpaint_area")
    if isinstance(inpaint_area, str):
        normalized = _normalized_choice(inpaint_area)
        if normalized in {"only_masked", "whole_picture"}:
            settings["inpaint_area"] = normalized


def _normalized_choice(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _apply_face_restoration(settings: dict[str, Any], value: str | None) -> None:
    parsed = _bool_value(value)
    if parsed is not None:
        settings["restore_faces"] = parsed
        return
    method = str(value or "").strip()
    if not method or method.lower() in {"none", "disabled", "off"}:
        return
    settings["restore_faces"] = True
    settings["face_restoration_model"] = method


def _apply_runtime_details(settings: dict[str, Any], value: str | None) -> None:
    if not value:
        return
    targets = {
        "controlnet": "controlnet_runtime",
        "detailer": "detailer_runtime",
        "lora_block_weight": "lora_block_weight_runtime",
    }
    tokens = value.split()
    index = 0
    while index + 1 < len(tokens):
        target = targets.get(tokens[index])
        if target:
            settings[target] = tokens[index + 1]
            index += 2
        else:
            index += 1


def _has_hires_parameters(params: dict[str, str]) -> bool:
    return any(
        params.get(key)
        for key in (
            "hires upscale",
            "hires resize",
            "hires steps",
            "hires upscaler",
            "hires checkpoint",
            "hires cfg scale",
            "hires sampler",
            "hires schedule type",
            "hires prompt",
            "hires negative prompt",
            "hires distilled cfg scale",
            "hires noise mode",
            "hires keep first pass",
        )
    )


def _apply_extension_parameter_fields(settings: dict[str, Any], extension_params: dict[str, Any]) -> None:
    controlnet_units = [
        unit
        for section in extension_params.get("controlnet", [])
        if (unit := _controlnet_unit_from_extension_section(section)) is not None
    ]
    if controlnet_units:
        settings["controlnet_units"] = [*settings.get("controlnet_units", []), *controlnet_units]
    detailer_units = [
        unit
        for fields in extension_params.get("adetailer", [])
        if (unit := _detailer_unit_from_adetailer_fields(fields)) is not None
    ]
    if detailer_units:
        settings["detailer_units"] = [*settings.get("detailer_units", []), *detailer_units]


def _model_slots_from_parameters(params: dict[str, str]) -> dict[str, str]:
    slots: dict[str, str] = {}
    if params.get("model"):
        slots["checkpoint"] = params["model"]
    if params.get("vae"):
        slots["vae"] = params["vae"]
    slots.update(_model_slot_entries_from_parameters(params.get("model slots")))
    return slots


def _model_slot_entries_from_parameters(value: str | None) -> dict[str, str]:
    slots: dict[str, str] = {}
    for item in _semicolon_list(value):
        key_text, separator, value_text = item.partition(" ")
        if not separator:
            continue
        key = _decoded_inline_text(key_text)
        slot_value = _decoded_inline_text(value_text)
        if key and slot_value:
            slots[key] = slot_value
    return slots


def _model_override_options_from_parameters(value: str | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in _semicolon_list(value):
        key_text, separator, value_text = item.partition(" ")
        if not separator:
            continue
        key = _decoded_inline_text(key_text)
        option_value = _decoded_inline_text(value_text)
        if key and option_value and key.startswith("merge."):
            overrides[key] = option_value
    return overrides


def _prompt_selections_from_parameters(value: str | None) -> list[str]:
    selections: list[str] = []
    for item in _semicolon_list(value):
        decoded = _decoded_inline_text(item)
        if decoded is not None:
            selections.append(decoded)
    return selections


def _tags_from_parameters(value: str | None) -> list[str]:
    tags: list[str] = []
    for item in _semicolon_list(value):
        decoded = _decoded_inline_text(item)
        if decoded is not None:
            tags.append(decoded)
    return tags


def _loras_from_parameters(value: str | None) -> list[dict[str, Any]]:
    loras: list[dict[str, Any]] = []
    for item in _comma_list(value):
        parts = [part.strip() for part in item.split(":")]
        if not parts or not parts[0]:
            continue
        lora: dict[str, Any] = {"name": parts[0]}
        if len(parts) >= 2 and (weight := _float_value(parts[1])) is not None:
            lora["weight"] = weight
        if len(parts) >= 3 and (clip_weight := _float_value(parts[2])) is not None:
            lora["clip_weight"] = clip_weight
        loras.append(lora)
    return loras


def _apply_lora_options(loras: list[dict[str, Any]], value: str | None) -> list[dict[str, Any]]:
    options = _lora_options_from_parameters(value)
    if not options:
        return loras
    seen_by_name = {str(item.get("name") or "").strip().lower(): item for item in loras}
    for name, option in options.items():
        key = name.lower()
        target = seen_by_name.get(key)
        if target is None:
            target = {"name": name}
            loras.append(target)
            seen_by_name[key] = target
        for field in ("block_weight", "block_weight_preset", "hierarchy_mode"):
            if option.get(field) not in {"", None}:
                target[field] = option[field]
    return loras


def _lora_options_from_parameters(value: str | None) -> dict[str, dict[str, str]]:
    options: dict[str, dict[str, str]] = {}
    labels = ["block_weight", "block_weight_preset", "hierarchy"]
    for item in _semicolon_list(value):
        name_text, fields = _split_inline_field_tail(item, labels)
        name = _decoded_inline_text(name_text)
        if not name:
            continue
        option: dict[str, str] = {}
        for source, target in (
            ("block_weight", "block_weight"),
            ("block_weight_preset", "block_weight_preset"),
            ("hierarchy", "hierarchy_mode"),
        ):
            decoded = _decoded_inline_text(fields.get(source))
            if decoded:
                option[target] = decoded
        if option:
            options[name] = option
    return options


def _lora_tokens_from_prompt_texts(*values: str | None) -> list[dict[str, Any]]:
    loras: list[dict[str, Any]] = []
    pattern = re.compile(r"<\s*lora\s*:\s*([^:<>]+?)\s*(?::\s*([-+]?\d+(?:\.\d+)?))?\s*>", re.IGNORECASE)
    for value in values:
        if not value:
            continue
        for match in pattern.finditer(value):
            name = match.group(1).strip()
            if not name:
                continue
            lora: dict[str, Any] = {"name": name}
            weight = _float_value(match.group(2))
            if weight is not None:
                lora["weight"] = weight
            loras.append(lora)
    return loras


def _merge_loras(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_by_name: dict[str, int] = {}
    for group in groups:
        for item in group:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            key = name.lower()
            if key in seen_by_name:
                existing = merged[seen_by_name[key]]
                if existing.get("weight") in {"", None} and item.get("weight") not in {"", None}:
                    existing["weight"] = item["weight"]
                if existing.get("clip_weight") in {"", None} and item.get("clip_weight") not in {"", None}:
                    existing["clip_weight"] = item["clip_weight"]
                continue
            seen_by_name[key] = len(merged)
            merged.append(item)
    return merged


def _lora_hash_names_from_parameters(value: str | None) -> list[dict[str, Any]]:
    return [{"name": name} for name in _hash_names_from_parameters(value)]


def _lora_hash_names_from_hashes_field(value: str | None) -> list[dict[str, Any]]:
    return [{"name": name} for name in _prefixed_hash_names_from_hashes_field(value, ("lora:", "lora/"))]


def _embedding_hash_names_from_hashes_field(value: str | None) -> list[str]:
    return _prefixed_hash_names_from_hashes_field(value, ("ti:", "embed:", "embedding:"))


def _prefixed_hash_names_from_hashes_field(value: str | None, prefixes: tuple[str, ...]) -> list[str]:
    hashes = _json_object(value)
    if not hashes:
        return []
    names: list[str] = []
    for key in hashes:
        if not isinstance(key, str):
            continue
        normalized = key.strip()
        normalized_lower = normalized.lower()
        for prefix in prefixes:
            if normalized_lower.startswith(prefix):
                name = normalized[len(prefix) :].strip()
                if name:
                    names.append(name)
                break
    return _merge_strings(names)


def _hash_names_from_parameters(value: str | None) -> list[str]:
    names: list[str] = []
    for item in _comma_list(value):
        name, separator, _hash_value = item.partition(":")
        if not separator:
            continue
        cleaned = name.strip().strip('"').strip("'").strip()
        if cleaned:
            names.append(cleaned)
    return _merge_strings(names)


def _merge_strings(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            value = str(item).strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(value)
    return merged


def _controlnet_unit_from_extension_section(section: str) -> dict[str, Any] | None:
    fields = _split_named_parameter_fields(
        section,
        [
            "Module",
            "Preprocessor",
            "Model",
            "Weight",
            "Resize Mode",
            "Guidance Start",
            "Guidance End",
            "Starting Control Step",
            "Ending Control Step",
            "Pixel Perfect",
            "Control Mode",
            "Processor Res",
            "Threshold A",
            "Threshold B",
            "Low Vram",
        ],
    )
    model = fields.get("model", "").strip()
    if not model:
        return None
    unit: dict[str, Any] = {"enabled": True, "model": model}
    preprocessor = fields.get("module") or fields.get("preprocessor")
    if preprocessor:
        unit["preprocessor"] = preprocessor
    weight = _float_value(fields.get("weight"))
    if weight is not None:
        unit["weight"] = weight
    guidance_start = _float_value(fields.get("guidance start") or fields.get("starting control step"))
    if guidance_start is not None:
        unit["guidance_start"] = guidance_start
    guidance_end = _float_value(fields.get("guidance end") or fields.get("ending control step"))
    if guidance_end is not None:
        unit["guidance_end"] = guidance_end
    resize_mode = fields.get("resize mode")
    if resize_mode:
        unit["resize_mode"] = resize_mode
    pixel_perfect = _bool_value(fields.get("pixel perfect"))
    if pixel_perfect is not None:
        unit["pixel_perfect"] = pixel_perfect
    control_mode = fields.get("control mode")
    if control_mode:
        unit["control_mode"] = control_mode
    for source, target in (
        ("processor res", "processor_res"),
        ("threshold a", "threshold_a"),
        ("threshold b", "threshold_b"),
    ):
        value = _int_value(fields.get(source))
        if value is not None:
            unit[target] = value
    low_vram = _bool_value(fields.get("low vram"))
    if low_vram is not None:
        unit["low_vram"] = low_vram
    return unit


def _controlnet_units_from_parameters(value: str | None) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    inline_labels = [
        "start",
        "end",
        "resize_mode",
        "pixel",
        "control",
        "processor",
        "threshold_a",
        "threshold_b",
        "low_vram",
    ]
    for item in _semicolon_list(value):
        text, fields = _split_inline_field_tail(item, inline_labels)
        guidance_start = _float_value(fields.get("start"))
        guidance_end = _float_value(fields.get("end"))
        model_part, _, preprocessor = text.partition("/")
        model, _, weight_text = model_part.partition("@")
        unit: dict[str, Any] = {"enabled": True, "model": model.strip()}
        weight = _float_value(weight_text)
        if weight is not None:
            unit["weight"] = weight
        if preprocessor.strip():
            unit["preprocessor"] = preprocessor.strip()
        if guidance_start is not None:
            unit["guidance_start"] = guidance_start
        if guidance_end is not None:
            unit["guidance_end"] = guidance_end
        if fields.get("resize_mode"):
            unit["resize_mode"] = fields["resize_mode"]
        pixel_perfect = _bool_value(fields.get("pixel"))
        if pixel_perfect is not None:
            unit["pixel_perfect"] = pixel_perfect
        if fields.get("control"):
            unit["control_mode"] = fields["control"]
        processor_res = _int_value(fields.get("processor"))
        if processor_res is not None:
            unit["processor_res"] = processor_res
        for source, target in (("threshold_a", "threshold_a"), ("threshold_b", "threshold_b")):
            threshold = _int_value(fields.get(source))
            if threshold is not None:
                unit[target] = threshold
        low_vram = _bool_value(fields.get("low_vram"))
        if low_vram is not None:
            unit["low_vram"] = low_vram
        if unit["model"]:
            units.append(unit)
    return units


def _controlnet_artifacts_from_parameters(value: str | None) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    labels = [
        "unit",
        "effective_strength",
        "control_mode_normalized",
        "control_mode_mapping",
        "control_mode_runtime",
        "control_mode_base_multiplier",
        "control_mode_uncond_multiplier",
        "preprocessor_runtime",
        "pixel_target",
    ]
    for item in _semicolon_list(value):
        fields = _split_inline_fields(item, labels)
        artifact: dict[str, Any] = {}
        unit_index = _int_value(fields.get("unit"))
        if unit_index is not None:
            artifact["unit_index"] = unit_index
        effective_strength = _float_value(fields.get("effective_strength"))
        if effective_strength is not None:
            artifact["effective_strength"] = effective_strength
        for key in (
            "control_mode_base_multiplier",
            "control_mode_uncond_multiplier",
        ):
            multiplier = _float_value(fields.get(key))
            if multiplier is not None:
                artifact[key] = multiplier
        for key in (
            "control_mode_normalized",
            "control_mode_mapping",
            "control_mode_runtime",
            "preprocessor_runtime",
        ):
            if fields.get(key):
                artifact[key] = fields[key]
        pixel_target = _size_dict(fields.get("pixel_target"))
        if pixel_target is not None:
            artifact["pixel_perfect_target_size"] = pixel_target
        if artifact:
            artifacts.append(artifact)
    return artifacts


def _detailer_unit_from_adetailer_fields(fields: dict[str, str]) -> dict[str, Any] | None:
    detector = _unquoted_text(fields.get("model")) or ""
    if not detector:
        return None
    unit: dict[str, Any] = {"enabled": True, "detector": detector}
    for source, target in (
        ("prompt", "prompt"),
        ("negative prompt", "negative_prompt"),
        ("sampler", "sampler"),
        ("scheduler", "scheduler"),
    ):
        value = _unquoted_text(fields.get(source))
        if value:
            unit[target] = value
    for source, target in (
        ("model classes", "model_classes"),
        ("mask merge mode", "mask_merge_mode"),
    ):
        if fields.get(source):
            unit[target] = fields[source]
    for source, target in (
        ("confidence", "confidence"),
        ("denoising strength", "denoising_strength"),
        ("cfg scale", "cfg_scale"),
        ("mask min ratio", "mask_min_ratio"),
        ("mask max ratio", "mask_max_ratio"),
    ):
        value = _float_value(fields.get(source))
        if value is not None:
            unit[target] = value
    for source, target in (
        ("mask blur", "mask_blur"),
        ("steps", "steps"),
        ("mask padding", "mask_padding"),
        ("inpaint padding", "mask_padding"),
        ("inpaint only masked padding", "mask_padding"),
        ("inpaint width", "inpaint_width"),
        ("inpaint height", "inpaint_height"),
        ("dilate/erode", "bbox_dilation"),
        ("bbox dilation", "bbox_dilation"),
        ("guide size", "guide_size"),
        ("max size", "max_size"),
        ("x offset", "x_offset"),
        ("y offset", "y_offset"),
    ):
        value = _int_value(fields.get(source))
        if value is not None:
            unit[target] = value
    bbox_crop_factor = _float_value(fields.get("bbox crop factor"))
    if bbox_crop_factor is not None:
        unit["bbox_crop_factor"] = bbox_crop_factor
    inpaint_only_masked = _bool_value(fields.get("inpaint only masked"))
    if inpaint_only_masked is not None:
        unit["inpaint_only_masked"] = inpaint_only_masked
    use_separate_dimensions = _bool_value(
        fields.get("use separate width/height") or fields.get("use separate width height")
    )
    if use_separate_dimensions is not None:
        unit["use_separate_dimensions"] = use_separate_dimensions
    mask_merge_invert = _bool_value(fields.get("mask merge invert"))
    if mask_merge_invert is not None:
        unit["mask_merge_invert"] = mask_merge_invert
    return unit


def _detailer_metadata_from_parameters(value: str | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    execution_order: list[dict[str, Any]] = []
    labels = [
        "output_policy",
        "unit",
        "detector",
        "prompt_override",
        "negative_prompt_override",
        "steps",
        "cfg",
        "sampler",
        "scheduler",
    ]
    for item in _semicolon_list(value):
        fields = _split_inline_fields(item, labels)
        if fields.get("output_policy") and not metadata.get("output_policy"):
            metadata["output_policy"] = fields["output_policy"]
        unit: dict[str, Any] = {}
        unit_index = _int_value(fields.get("unit"))
        if unit_index is not None:
            unit["unit_index"] = unit_index
        if fields.get("detector"):
            unit["detector"] = fields["detector"]
        prompt_override = _bool_value(fields.get("prompt_override"))
        if prompt_override is not None:
            unit["has_prompt_override"] = prompt_override
        negative_prompt_override = _bool_value(fields.get("negative_prompt_override"))
        if negative_prompt_override is not None:
            unit["has_negative_prompt_override"] = negative_prompt_override
        steps = _int_value(fields.get("steps"))
        if steps is not None:
            unit["steps"] = steps
        cfg_scale = _float_value(fields.get("cfg"))
        if cfg_scale is not None:
            unit["cfg_scale"] = cfg_scale
        if fields.get("sampler"):
            unit["sampler"] = fields["sampler"]
        if fields.get("scheduler"):
            unit["scheduler"] = fields["scheduler"]
        if unit:
            execution_order.append(unit)
    if execution_order:
        metadata["execution_order"] = execution_order
    return metadata


def _hires_metadata_from_parameters(value: str | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    labels = [
        "seed",
        "steps",
        "sampler",
        "scheduler",
        "cfg",
        "size",
        "denoise",
        "prompt",
        "negative",
        "resize",
        "scale",
        "upscaler",
        "noise",
    ]
    for item in _semicolon_list(value):
        pass_name, separator, fields_text = item.partition(" ")
        if not separator:
            continue
        normalized_pass = pass_name.strip().lower()
        if normalized_pass not in {"first", "second"}:
            continue
        parsed = _hires_metadata_pass_from_fields(normalized_pass, _split_inline_fields(fields_text, labels))
        if parsed:
            metadata[f"{normalized_pass}_pass"] = parsed
    return metadata


def _hires_metadata_pass_from_fields(pass_name: str, fields: dict[str, str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for source, target in (("seed", "seed"), ("steps", "steps")):
        value = _int_value(fields.get(source))
        if value is not None:
            parsed[target] = value
    for source, target in (
        ("sampler", "sampler"),
        ("scheduler", "scheduler"),
        ("prompt", "prompt"),
        ("negative", "negative_prompt"),
        ("upscaler", "upscaler"),
        ("noise", "noise_mode"),
    ):
        value = _decoded_inline_text(fields.get(source))
        if value:
            parsed[target] = value
    for source, target in (("cfg", "cfg_scale"), ("denoise", "denoising_strength"), ("scale", "scale")):
        value = _float_value(fields.get(source))
        if value is not None:
            parsed[target] = value
    if pass_name == "first":
        size = _size_dict(fields.get("size"))
        if size is not None:
            parsed["width"] = size["width"]
            parsed["height"] = size["height"]
    if pass_name == "second":
        resize = _size_dict(fields.get("resize"))
        if resize is not None:
            parsed["resize_width"] = resize["width"]
            parsed["resize_height"] = resize["height"]
    return parsed


def _detailer_units_from_parameters(value: str | None) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    inline_labels = [
        "prompt_text",
        "negative_prompt_text",
        "model_classes",
        "confidence",
        "mask_blur",
        "inpaint_only_masked",
        "use_separate_dimensions",
        "mask_padding",
        "inpaint_width",
        "inpaint_height",
        "bbox_dilation",
        "guide_size",
        "max_size",
        "bbox_crop_factor",
        "mask_min_ratio",
        "mask_max_ratio",
        "x_offset",
        "y_offset",
        "mask_merge_mode",
        "mask_merge_invert",
        "steps",
        "cfg",
        "sampler",
        "scheduler",
    ]
    for item in _semicolon_list(value):
        detector_part, _, tail = item.partition(" ")
        detector, _, denoise_text = detector_part.partition("@")
        if not detector.strip():
            continue
        unit: dict[str, Any] = {"enabled": True, "detector": detector.strip()}
        denoise = _float_value(denoise_text)
        if denoise is not None:
            unit["denoising_strength"] = denoise
        fields = _split_inline_fields(tail, inline_labels)
        for source, target in (("prompt_text", "prompt"), ("negative_prompt_text", "negative_prompt")):
            decoded = _decoded_inline_text(fields.get(source))
            if decoded:
                unit[target] = decoded
        for source, target in (("model_classes", "model_classes"), ("mask_merge_mode", "mask_merge_mode")):
            decoded = _decoded_inline_text(fields.get(source))
            if decoded:
                unit[target] = decoded
        confidence = _float_value(fields.get("confidence"))
        if confidence is not None:
            unit["confidence"] = confidence
        for source, target in (("mask_min_ratio", "mask_min_ratio"), ("mask_max_ratio", "mask_max_ratio")):
            parsed_float = _float_value(fields.get(source))
            if parsed_float is not None:
                unit[target] = parsed_float
        for source, target in (
            ("mask_blur", "mask_blur"),
            ("mask_padding", "mask_padding"),
            ("inpaint_width", "inpaint_width"),
            ("inpaint_height", "inpaint_height"),
            ("bbox_dilation", "bbox_dilation"),
            ("guide_size", "guide_size"),
            ("max_size", "max_size"),
            ("x_offset", "x_offset"),
            ("y_offset", "y_offset"),
        ):
            value_for_key = _int_value(fields.get(source))
            if value_for_key is not None:
                unit[target] = value_for_key
        bbox_crop_factor = _float_value(fields.get("bbox_crop_factor"))
        if bbox_crop_factor is not None:
            unit["bbox_crop_factor"] = bbox_crop_factor
        inpaint_only_masked = _bool_value(fields.get("inpaint_only_masked"))
        if inpaint_only_masked is not None:
            unit["inpaint_only_masked"] = inpaint_only_masked
        use_separate_dimensions = _bool_value(fields.get("use_separate_dimensions"))
        if use_separate_dimensions is not None:
            unit["use_separate_dimensions"] = use_separate_dimensions
        mask_merge_invert = _bool_value(fields.get("mask_merge_invert"))
        if mask_merge_invert is not None:
            unit["mask_merge_invert"] = mask_merge_invert
        steps = _int_value(fields.get("steps"))
        if steps is not None:
            unit["steps"] = steps
        cfg_scale = _float_value(fields.get("cfg"))
        if cfg_scale is not None:
            unit["cfg_scale"] = cfg_scale
        if fields.get("sampler"):
            unit["sampler"] = fields["sampler"]
        if fields.get("scheduler"):
            unit["scheduler"] = fields["scheduler"]
        units.append(unit)
    return units


def _apply_topaz_settings(settings: dict[str, Any], params: dict[str, str]) -> None:
    legacy = _unquoted_text(params.get("topaz photo ai"))
    if legacy:
        settings["topaz_postprocess"] = legacy
        normalized = legacy.lower()
        for label, key in (
            ("lighting", "topaz_lighting"),
            ("color", "topaz_color"),
            ("sharpen", "topaz_sharpen"),
            ("denoise", "topaz_denoise"),
        ):
            if label in normalized:
                settings[key] = True
    upscale = params.get("topaz upscale")
    if upscale:
        model, separator, scale_text = upscale.rpartition(" x")
        if separator:
            if model.strip():
                settings["topaz_upscale_model"] = model.strip()
            scale = _float_value(scale_text)
            if scale is not None:
                settings["topaz_upscale_scale"] = int(scale) if scale.is_integer() else scale
        else:
            settings["topaz_upscale_model"] = upscale
    adjustments = {item.lower() for item in _comma_list(params.get("topaz adjustments"))}
    for label, key in (
        ("lighting", "topaz_lighting"),
        ("color", "topaz_color"),
        ("sharpen", "topaz_sharpen"),
        ("denoise", "topaz_denoise"),
    ):
        if label in adjustments:
            settings[key] = True
    options = _split_inline_fields(params.get("topaz options") or "", ["override", "png_compression"])
    override = _bool_value(options.get("override"))
    if override is not None:
        settings["topaz_override"] = override
    png_compression = _int_value(options.get("png_compression"))
    if png_compression is not None:
        settings["topaz_png_compression"] = png_compression


def _advanced_styles_from_parameters(value: str | None) -> list[dict[str, Any]]:
    styles: list[dict[str, Any]] = []
    for item in _comma_aware_list(value):
        text = item.strip()
        if not text:
            continue
        style: dict[str, Any] = {}
        match = re.match(r"(?P<name>.*?)\((?P<details>.*)\)$", text)
        if match is None:
            style["name"] = text
        else:
            name = match.group("name").strip()
            details = [detail.strip() for detail in match.group("details").split(",") if detail.strip()]
            if name:
                style["name"] = name
            if details:
                style["id"] = details[0]
            if len(details) >= 2:
                style["source_type"] = details[1]
            for detail in details[2:]:
                if detail.lower().startswith("seed "):
                    seed = _int_value(detail.removeprefix("seed").strip())
                    if seed is not None:
                        style["selection_seed"] = seed
        if style.get("name") or style.get("id"):
            styles.append(style)
    return styles


def _apply_advanced_style_options(styles: list[dict[str, Any]], value: str | None) -> list[dict[str, Any]]:
    options = _advanced_style_options_from_parameters(value)
    if not options:
        return styles
    seen_by_id = {str(item.get("id") or "").strip().lower(): item for item in styles if item.get("id")}
    seen_by_name = {str(item.get("name") or "").strip().lower(): item for item in styles if item.get("name")}
    for key, option in options.items():
        normalized_key = key.lower()
        target = seen_by_id.get(normalized_key) or seen_by_name.get(normalized_key)
        if target is None:
            target = {"id": key}
            styles.append(target)
            seen_by_id[normalized_key] = target
        target.update(option)
    return styles


def _advanced_style_options_from_parameters(value: str | None) -> dict[str, dict[str, Any]]:
    labels = ["group", "prompt", "negative", "selected_from_group", "order"]
    options: dict[str, dict[str, Any]] = {}
    for item in _semicolon_list(value):
        key, fields = _split_inline_field_tail(item, labels)
        decoded_key = _decoded_inline_text(key)
        if not decoded_key:
            continue
        option: dict[str, Any] = {}
        for field_key in ("group", "prompt", "negative", "selected_from_group"):
            decoded = _decoded_inline_text(fields.get(field_key))
            if decoded is not None:
                option[field_key] = decoded
        order = _int_value(fields.get("order"))
        if order is not None:
            option["order"] = order
        if option:
            options[decoded_key] = option
    return options


def _comma_aware_list(value: str | None) -> list[str]:
    if not value:
        return []
    items: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            item = value[start:index].strip()
            if item:
                items.append(item)
            start = index + 1
    tail = value[start:].strip()
    if tail:
        items.append(tail)
    return items


def _apply_hashes_field(settings: dict[str, Any], value: str | None) -> None:
    hashes = _json_object(value)
    if hashes is None:
        return
    for source, target in (("model", "model_hash"), ("vae", "vae_hash")):
        hash_value = hashes.get(source)
        if isinstance(hash_value, str) and hash_value.strip() and not settings.get(target):
            settings[target] = hash_value.strip()


def _json_object(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _comma_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _int_list(value: str | None) -> list[int]:
    items: list[int] = []
    for item in _comma_list(value):
        parsed = _int_value(item)
        if parsed is not None:
            items.append(parsed)
    return items


def _semicolon_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def _inline_float(text: str, label: str) -> float | None:
    match = re.search(rf"(?:^|\s){re.escape(label)}\s+(-?\d+(?:\.\d+)?)", text)
    if match is None:
        return None
    return _float_value(match.group(1))


def _split_inline_fields(text: str, labels: list[str]) -> dict[str, str]:
    if not text.strip():
        return {}
    label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    pattern = re.compile(rf"(?:^|\s)(?P<label>{label_pattern})\s+", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[value_start:value_end].strip()
        if value:
            fields[match.group("label").lower()] = value
    return fields


def _split_inline_field_tail(text: str, labels: list[str]) -> tuple[str, dict[str, str]]:
    if not text.strip():
        return "", {}
    label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    pattern = re.compile(rf"(?:^|\s)(?P<label>{label_pattern})\s+", re.IGNORECASE)
    first_match = pattern.search(text)
    fields = _split_inline_fields(text, labels)
    if first_match is None:
        return text.strip(), fields
    return text[: first_match.start()].strip(), fields


def _split_named_parameter_fields(text: str, labels: list[str]) -> dict[str, str]:
    label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    pattern = re.compile(rf"(?:^|,\s*)(?P<label>{label_pattern}):\s*", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[value_start:value_end].strip().rstrip(",").strip()
        if value:
            fields[match.group("label").lower()] = value
    return fields


def _apply_size(settings: dict[str, Any], value: str | None, width_key: str, height_key: str) -> None:
    if not value:
        return
    width_text, separator, height_text = value.lower().partition("x")
    if not separator:
        return
    width = _int_value(width_text)
    height = _int_value(height_text)
    if width is not None:
        settings[width_key] = width
    if height is not None:
        settings[height_key] = height


def _size_dict(value: str | None) -> dict[str, int] | None:
    if not value:
        return None
    width_text, separator, height_text = value.lower().partition("x")
    if not separator:
        return None
    width = _int_value(width_text)
    height = _int_value(height_text)
    if width is None or height is None:
        return None
    return {"width": width, "height": height}


def _int_value(value: str | None) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _float_value(value: str | None) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _bool_value(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    return None


def _encoded_inline_text(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return quote(text, safe="")


def _decoded_inline_text(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    decoded = unquote(text).strip()
    return decoded or None


def _unquoted_text(value: str | None) -> str | None:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text or None


class MetadataWriter:
    def write_generation_sidecars(self, metadata: dict[str, Any], output_paths: list[str]) -> list[str]:
        sidecar_paths: list[str] = []
        for output_path in output_paths:
            path = self._safe_path(output_path)
            if path is None:
                continue
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            payload = {
                "schema": "asteria.generation.v1",
                "written_at": datetime.now(UTC).isoformat(),
                **metadata,
                "image_path": str(path),
            }
            self._write_png_metadata(path, payload)
            sidecar_path = path.with_suffix(".asteria.json")
            try:
                sidecar_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError:
                continue
            sidecar_paths.append(str(sidecar_path))
        return sidecar_paths

    def update_curation_sidecars(self, item: dict[str, Any]) -> list[str]:
        updated: list[str] = []
        for output_path in item.get("output_paths", []):
            path = self._safe_path(output_path)
            if path is None:
                continue
            for sidecar_path in self._curation_sidecar_paths(path):
                if not self._path_exists(sidecar_path):
                    continue
                try:
                    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                payload["rating"] = item.get("rating", 0)
                payload["favorite"] = item.get("favorite", False)
                payload["tags"] = item.get("tags", [])
                payload["notes"] = item.get("notes", "")
                payload["folder_id"] = item.get("folder_id")
                payload["folder_name"] = item.get("folder_name")
                payload["updated_at"] = datetime.now(UTC).isoformat()
                try:
                    sidecar_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                except OSError:
                    continue
                updated.append(str(sidecar_path))
        return updated

    def _curation_sidecar_paths(self, path: Path) -> list[Path]:
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return [path.with_suffix(".asteria.json"), path.with_suffix(".asteria.merge-candidate.json")]
        if path.name.endswith(".asteria.json") or path.name.endswith(".asteria.merge-candidate.json"):
            return [path]
        return []

    def _safe_path(self, value: Any) -> Path | None:
        try:
            raw = os.fspath(value)
        except TypeError:
            return None
        if not isinstance(raw, str) or not raw:
            return None
        return Path(raw)

    def _path_exists(self, path: Path) -> bool:
        try:
            return path.exists()
        except OSError:
            return False

    def _write_png_metadata(self, path: Path, payload: dict[str, Any]) -> None:
        if path.suffix.lower() != ".png":
            return
        try:
            with Image.open(path) as image:
                pnginfo = PngInfo()
                for key, value in image.info.items():
                    if isinstance(value, str):
                        pnginfo.add_text(key, value)
                pnginfo.add_text("Asteria", json.dumps(payload, ensure_ascii=False))
                pnginfo.add_text("parameters", self._parameters_text(payload))
                image.save(path, pnginfo=pnginfo)
        except OSError:
            return

    def _parameters_text(self, payload: dict[str, Any]) -> str:
        settings = payload.get("settings", {})
        model_slots = payload.get("model_slots", {})
        lines = [str(payload.get("expanded_prompt") or payload.get("prompt", ""))]
        negative_prompt = payload.get("negative_prompt")
        if negative_prompt:
            lines.append(f"Negative prompt: {negative_prompt}")
        params: list[tuple[str, Any]] = [
            ("Raw prompt", self._raw_prompt(settings, payload)),
            ("Raw negative prompt", self._raw_negative_prompt(settings, payload)),
            ("Prompt selections", self._prompt_selections_summary(payload, settings)),
            ("Template", settings.get("prompt_template")),
            ("Negative Template", settings.get("negative_prompt_template")),
            ("Steps", settings.get("steps")),
            ("Sampler", settings.get("sampler")),
            ("Schedule type", settings.get("scheduler")),
            ("CFG scale", settings.get("cfg_scale")),
            ("Distilled CFG scale", self._positive_value(settings.get("distilled_cfg_scale"))),
            ("Seed", settings.get("seed")),
            ("Variation seed", self._positive_value(settings.get("subseed"))),
            ("Variation seed strength", self._positive_value(settings.get("subseed_strength"))),
            ("Seed resize from", self._seed_resize_from(settings)),
            ("Size", f"{settings.get('width')}x{settings.get('height')}"),
            ("Batch size", self._non_default_value(settings.get("batch_size"), 1)),
            ("Batch count", self._non_default_value(settings.get("batch_count"), 1)),
            ("Eta", settings.get("eta")),
            ("ENSD", settings.get("eta_noise_seed_delta")),
            ("RNG", settings.get("rng_source")),
            ("Generated seeds", self._generated_seeds(settings)),
            ("Profile", payload.get("profile") or settings.get("profile")),
            ("Workflow template", self._workflow_id(payload, settings, "workflow_template_id", "workflow_template")),
            ("Workflow mapping", self._workflow_id(payload, settings, "workflow_mapping_id", "workflow_mapping")),
            ("Mode", self._mode_value(settings)),
            ("Model", self._display_model_value(model_slots.get("checkpoint") or model_slots.get("diffusion_model"))),
            ("Model hash", settings.get("model_hash")),
            ("VAE", self._display_model_value(model_slots.get("vae"))),
            ("VAE hash", settings.get("vae_hash")),
            ("Model slots", self._model_slots_summary(model_slots)),
            ("Model override options", self._model_override_options_summary(settings.get("model_slot_overrides"))),
            ("Clip skip", settings.get("clip_skip")),
            ("Face restoration", self._face_restoration_value(settings)),
            ("Tiling", settings.get("tiling") if settings.get("tiling") else None),
            ("Asteria generation", payload.get("generation_id")),
            ("Runtime adapter", settings.get("runtime_adapter") or payload.get("runtime_adapter_id")),
            ("Engine info", settings.get("engine_info")),
            ("Runtime details", self._runtime_details(settings)),
            ("Runtime job", settings.get("runtime_job_id")),
            ("Runtime jobs", self._runtime_jobs(settings)),
            ("Asteria tags", self._metadata_tags(payload)),
        ]
        if settings.get("mode") in {"img2img", "inpaint"}:
            params.append(("Denoising strength", settings.get("denoising_strength")))
        if settings.get("mode") == "inpaint":
            params.extend(
                [
                    ("Mask blur", settings.get("mask_blur")),
                    ("Masked content", settings.get("masked_content")),
                    ("Inpaint area", settings.get("inpaint_area")),
                ]
            )
        if settings.get("hires_enabled"):
            params.extend(
                [
                    ("Hires upscale", settings.get("hires_scale")),
                    ("Hires resize", self._hires_resize(settings)),
                    ("Hires steps", settings.get("hires_steps") or settings.get("steps")),
                    ("Hires upscaler", settings.get("hires_upscaler")),
                    ("Hires checkpoint", self._hires_checkpoint(settings, model_slots)),
                    ("Denoising strength", settings.get("hires_denoising_strength")),
                    ("Hires CFG scale", self._positive_value(settings.get("hires_cfg_scale"))),
                    ("Hires sampler", self._hires_setting(settings, "hires_sampler", "sampler", "Use same sampler")),
                    (
                        "Hires schedule type",
                        self._hires_setting(settings, "hires_scheduler", "scheduler", "Use same scheduler"),
                    ),
                    ("Hires prompt", settings.get("hires_prompt")),
                    ("Hires negative prompt", settings.get("hires_negative_prompt")),
                    ("Hires distilled CFG scale", self._positive_value(settings.get("hires_distilled_cfg_scale"))),
                    ("Hires noise mode", settings.get("hires_noise_mode")),
                    (
                        "Hires keep first pass",
                        settings.get("hires_keep_first_pass") if settings.get("hires_keep_first_pass") else None,
                    ),
                    ("Hires metadata", self._hires_metadata_summary(settings.get("hires_metadata"))),
                ]
            )
        if settings.get("refiner_enabled") and settings.get("refiner_checkpoint"):
            params.append(("Refiner", f"{settings.get('refiner_checkpoint')} @ {settings.get('refiner_switch_at')}"))
        styles_summary = self._advanced_styles_summary(settings.get("advanced_styles"))
        if styles_summary:
            params.append(("Styles", styles_summary))
        style_options = self._advanced_style_options_summary(settings.get("advanced_styles"))
        if style_options:
            params.append(("Style options", style_options))
        controlnet_units = self._enabled_count(settings.get("controlnet_units"))
        if controlnet_units:
            params.append(("ControlNet units", controlnet_units))
        controlnet_summary = self._controlnet_summary(settings.get("controlnet_units"))
        if controlnet_summary:
            params.append(("ControlNet", controlnet_summary))
        controlnet_artifacts = self._controlnet_artifact_summary(settings.get("controlnet_artifacts"))
        if controlnet_artifacts:
            params.append(("ControlNet artifacts", controlnet_artifacts))
        detailer_units = self._enabled_count(settings.get("detailer_units"))
        if detailer_units:
            params.append(("Detailer units", detailer_units))
        detailer_summary = self._detailer_summary(settings.get("detailer_units"))
        if detailer_summary:
            params.append(("Detailer", detailer_summary))
        detailer_metadata = self._detailer_metadata_summary(settings.get("detailer_metadata"))
        if detailer_metadata:
            params.append(("Detailer metadata", detailer_metadata))
        lora_summary = self._lora_summary(payload.get("loras"))
        if lora_summary:
            params.append(("LoRAs", lora_summary))
        lora_options = self._lora_options_summary(payload.get("loras"))
        if lora_options:
            params.append(("LoRA options", lora_options))
        embedding_summary = self._embedding_summary(payload.get("embeddings"))
        if embedding_summary:
            params.append(("Embeddings", embedding_summary))
        if settings.get("topaz_postprocess"):
            params.append(("Topaz postprocess", settings.get("topaz_postprocess")))
            topaz_upscale = self._topaz_upscale(settings)
            if topaz_upscale:
                params.append(("Topaz upscale", topaz_upscale))
            topaz_adjustments = self._topaz_adjustments(settings)
            if topaz_adjustments:
                params.append(("Topaz adjustments", topaz_adjustments))
            topaz_options = self._topaz_options(settings)
            if topaz_options:
                params.append(("Topaz options", topaz_options))
        if settings.get("source_application_version"):
            params.append(("Version", settings.get("source_application_version")))
        lines.append(", ".join(f"{key}: {value}" for key, value in params if value not in {"", None, "NonexNone"}))
        return "\n".join(lines)

    def _display_model_value(self, value: Any) -> str | None:
        if value in {"", None}:
            return None
        return Path(str(value)).name

    def _model_slots_summary(self, model_slots: Any) -> str:
        if not isinstance(model_slots, dict):
            return ""
        summaries: list[str] = []
        for key, value in model_slots.items():
            if key in {"checkpoint", "vae"}:
                continue
            slot = _encoded_inline_text(key)
            model = _encoded_inline_text(self._display_model_value(value))
            if slot and model:
                summaries.append(f"{slot} {model}")
        return "; ".join(summaries)

    def _model_override_options_summary(self, value: Any) -> str:
        if not isinstance(value, dict):
            return ""
        summaries: list[str] = []
        for key, option_value in value.items():
            key_text = str(key or "").strip()
            value_text = str(option_value or "").strip()
            if not key_text.startswith("merge.") or "metadata" in key_text.lower():
                continue
            if self._looks_like_local_path(value_text):
                continue
            encoded_key = _encoded_inline_text(key_text)
            encoded_value = _encoded_inline_text(value_text)
            if encoded_key and encoded_value:
                summaries.append(f"{encoded_key} {encoded_value}")
        return "; ".join(summaries)

    def _looks_like_local_path(self, value: str) -> bool:
        if "\\" in value or "/" in value:
            return True
        return re.match(r"^[A-Za-z]:", value) is not None

    def _raw_prompt(self, settings: dict[str, Any], payload: dict[str, Any]) -> str | None:
        raw_prompt = str(settings.get("raw_prompt") or "").strip()
        if not raw_prompt:
            return None
        rendered_prompt = str(payload.get("expanded_prompt") or payload.get("prompt") or "").strip()
        if raw_prompt == rendered_prompt:
            return None
        return raw_prompt

    def _raw_negative_prompt(self, settings: dict[str, Any], payload: dict[str, Any]) -> str | None:
        raw_negative_prompt = str(settings.get("raw_negative_prompt") or "").strip()
        if not raw_negative_prompt:
            return None
        rendered_negative_prompt = str(payload.get("negative_prompt") or "").strip()
        if raw_negative_prompt == rendered_negative_prompt:
            return None
        return raw_negative_prompt

    def _prompt_selections_summary(self, payload: dict[str, Any], settings: dict[str, Any]) -> str:
        value = payload.get("prompt_expansion_selections") or settings.get("prompt_expansion_selections")
        if not isinstance(value, list):
            return ""
        encoded = [_encoded_inline_text(item) for item in value]
        return "; ".join(item for item in encoded if item)

    def _workflow_id(self, payload: dict[str, Any], settings: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = payload.get(key) or settings.get(key)
            if value not in {"", None}:
                return str(value)
        return None

    def _runtime_jobs(self, settings: dict[str, Any]) -> str:
        value = settings.get("runtime_job_ids")
        if not isinstance(value, list):
            return ""
        return ", ".join(str(item).strip() for item in value if str(item).strip())

    def _generated_seeds(self, settings: dict[str, Any]) -> str:
        value = settings.get("generated_seeds")
        if not isinstance(value, list):
            return ""
        return ", ".join(str(item).strip() for item in value if str(item).strip())

    def _metadata_tags(self, payload: dict[str, Any]) -> str:
        value = payload.get("metadata_tags")
        if not isinstance(value, list):
            return ""
        encoded = [_encoded_inline_text(item) for item in value]
        return "; ".join(item for item in encoded if item)

    def _runtime_details(self, settings: dict[str, Any]) -> str:
        details: list[str] = []
        for key, label in (
            ("controlnet_runtime", "controlnet"),
            ("detailer_runtime", "detailer"),
            ("lora_block_weight_runtime", "lora_block_weight"),
        ):
            value = str(settings.get(key) or "").strip()
            if value:
                details.append(f"{label} {value}")
        return " ".join(details)

    def _mode_value(self, settings: dict[str, Any]) -> str | None:
        mode = str(settings.get("mode") or "").strip()
        if mode and mode != "txt2img":
            return mode
        return None

    def _face_restoration_value(self, settings: dict[str, Any]) -> Any:
        if not settings.get("restore_faces"):
            return None
        method = str(settings.get("face_restoration_model") or "").strip()
        return method or settings.get("restore_faces")

    def _hires_setting(self, settings: dict[str, Any], key: str, fallback_key: str, same_label: str) -> Any:
        value = settings.get(key)
        if not value or value == same_label:
            return settings.get(fallback_key)
        return value

    def _hires_checkpoint(self, settings: dict[str, Any], model_slots: dict[str, Any]) -> str | None:
        value = str(settings.get("hires_checkpoint") or "").strip()
        if not value or value == "Use same checkpoint":
            return None
        checkpoint = self._display_model_value(model_slots.get("checkpoint") or model_slots.get("diffusion_model"))
        if checkpoint and Path(value).name == checkpoint:
            return None
        return self._display_model_value(value)

    def _enabled_count(self, value: Any) -> int:
        if not isinstance(value, list):
            return 0
        return sum(1 for item in value if not isinstance(item, dict) or item.get("enabled", True))

    def _enabled_dicts(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict) and item.get("enabled", True)]

    def _controlnet_summary(self, value: Any) -> str:
        summaries: list[str] = []
        for item in self._enabled_dicts(value):
            model = self._display_model_value(item.get("model")) or "unknown"
            weight = item.get("weight")
            preprocessor = item.get("preprocessor") or "none"
            summary = model
            if weight not in {"", None}:
                summary = f"{summary}@{weight}"
            if preprocessor:
                summary = f"{summary}/{preprocessor}"
            for key, label in (("guidance_start", "start"), ("guidance_end", "end")):
                numeric = self._non_default_float(item.get(key), 0.0 if key == "guidance_start" else 1.0)
                if numeric is not None:
                    summary = f"{summary} {label} {numeric}"
            for key, label in (
                ("resize_mode", "resize_mode"),
                ("pixel_perfect", "pixel"),
                ("control_mode", "control"),
                ("processor_res", "processor"),
                ("threshold_a", "threshold_a"),
                ("threshold_b", "threshold_b"),
                ("low_vram", "low_vram"),
            ):
                value_for_key = item.get(key)
                if value_for_key not in {"", None}:
                    summary = f"{summary} {label} {value_for_key}"
            summaries.append(summary)
        return "; ".join(summaries)

    def _controlnet_artifact_summary(self, value: Any) -> str:
        if not isinstance(value, list):
            return ""
        summaries: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            fields: list[str] = []
            unit_index = item.get("unit_index")
            if unit_index not in {"", None}:
                fields.append(f"unit {unit_index}")
            for key, label in (
                ("effective_strength", "effective_strength"),
                ("control_mode_normalized", "control_mode_normalized"),
                ("control_mode_mapping", "control_mode_mapping"),
                ("control_mode_runtime", "control_mode_runtime"),
                ("control_mode_base_multiplier", "control_mode_base_multiplier"),
                ("control_mode_uncond_multiplier", "control_mode_uncond_multiplier"),
                ("preprocessor_runtime", "preprocessor_runtime"),
            ):
                value_for_key = item.get(key)
                if value_for_key not in {"", None}:
                    fields.append(f"{label} {value_for_key}")
            pixel_target = item.get("pixel_perfect_target_size")
            if isinstance(pixel_target, dict):
                width = pixel_target.get("width")
                height = pixel_target.get("height")
                if width not in {"", None} and height not in {"", None}:
                    fields.append(f"pixel_target {width}x{height}")
            if fields:
                summaries.append(" ".join(fields))
        return "; ".join(summaries)

    def _advanced_styles_summary(self, value: Any) -> str:
        if not isinstance(value, list):
            return ""
        summaries: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("id") or "").strip()
            if not name:
                continue
            details = [str(item.get("id") or "").strip(), str(item.get("source_type") or "").strip()]
            selection_seed = item.get("selection_seed")
            if selection_seed not in {"", None}:
                details.append(f"seed {selection_seed}")
            compact_details = ", ".join(detail for detail in details if detail)
            if compact_details:
                summaries.append(f"{name}({compact_details})")
            else:
                summaries.append(name)
        return ", ".join(summaries)

    def _advanced_style_options_summary(self, value: Any) -> str:
        if not isinstance(value, list):
            return ""
        summaries: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            key = _encoded_inline_text(item.get("id") or item.get("name"))
            if not key:
                continue
            fields: list[str] = []
            for source_key, label in (
                ("group", "group"),
                ("prompt", "prompt"),
                ("negative", "negative"),
                ("selected_from_group", "selected_from_group"),
            ):
                encoded = _encoded_inline_text(item.get(source_key))
                if encoded is not None:
                    fields.append(f"{label} {encoded}")
            order = item.get("order")
            if order not in {"", None, 0}:
                fields.append(f"order {order}")
            if fields:
                summaries.append(f"{key} {' '.join(fields)}")
        return "; ".join(summaries)

    def _topaz_upscale(self, settings: dict[str, Any]) -> str | None:
        model = str(settings.get("topaz_upscale_model") or "").strip()
        scale = settings.get("topaz_upscale_scale")
        if not model and scale in {"", None}:
            return None
        if model and scale not in {"", None}:
            return f"{model} x{scale}"
        return model or f"x{scale}"

    def _topaz_adjustments(self, settings: dict[str, Any]) -> str:
        enabled = [
            label
            for key, label in (
                ("topaz_lighting", "lighting"),
                ("topaz_color", "color"),
                ("topaz_sharpen", "sharpen"),
                ("topaz_denoise", "denoise"),
            )
            if settings.get(key)
        ]
        return ", ".join(enabled)

    def _topaz_options(self, settings: dict[str, Any]) -> str:
        fields: list[str] = []
        if settings.get("topaz_override"):
            fields.append(f"override {settings.get('topaz_override')}")
        png_compression = settings.get("topaz_png_compression")
        if png_compression not in {"", None}:
            fields.append(f"png_compression {png_compression}")
        return " ".join(fields)

    def _detailer_summary(self, value: Any) -> str:
        summaries: list[str] = []
        for item in self._enabled_dicts(value):
            detector = str(item.get("detector") or "face_yolo")
            summary = detector
            denoise = item.get("denoising_strength")
            if denoise not in {"", None}:
                summary = f"{summary}@{denoise}"
            prompt_text = _encoded_inline_text(item.get("prompt"))
            if prompt_text:
                summary = f"{summary} prompt_text {prompt_text}"
            negative_prompt_text = _encoded_inline_text(item.get("negative_prompt"))
            if negative_prompt_text:
                summary = f"{summary} negative_prompt_text {negative_prompt_text}"
            for key, label in (
                ("model_classes", "model_classes"),
                ("mask_merge_mode", "mask_merge_mode"),
            ):
                encoded_value = _encoded_inline_text(item.get(key))
                if encoded_value:
                    summary = f"{summary} {label} {encoded_value}"
            for key, label in (
                ("confidence", "confidence"),
                ("mask_blur", "mask_blur"),
                ("inpaint_only_masked", "inpaint_only_masked"),
                ("use_separate_dimensions", "use_separate_dimensions"),
                ("mask_padding", "mask_padding"),
                ("inpaint_width", "inpaint_width"),
                ("inpaint_height", "inpaint_height"),
                ("bbox_dilation", "bbox_dilation"),
                ("guide_size", "guide_size"),
                ("max_size", "max_size"),
                ("bbox_crop_factor", "bbox_crop_factor"),
                ("mask_min_ratio", "mask_min_ratio"),
                ("mask_max_ratio", "mask_max_ratio"),
                ("x_offset", "x_offset"),
                ("y_offset", "y_offset"),
                ("mask_merge_invert", "mask_merge_invert"),
                ("steps", "steps"),
                ("cfg_scale", "cfg"),
                ("sampler", "sampler"),
                ("scheduler", "scheduler"),
            ):
                value_for_key = item.get(key)
                if value_for_key not in {"", None, 0, 0.0}:
                    summary = f"{summary} {label} {value_for_key}"
            summaries.append(summary)
        return "; ".join(summaries)

    def _detailer_metadata_summary(self, value: Any) -> str:
        if not isinstance(value, dict):
            return ""
        output_policy = str(value.get("output_policy") or "").strip()
        execution_order = value.get("execution_order")
        if not isinstance(execution_order, list) or not execution_order:
            return f"output_policy {output_policy}" if output_policy else ""
        summaries: list[str] = []
        for index, item in enumerate(execution_order):
            if not isinstance(item, dict):
                continue
            fields: list[str] = []
            if index == 0 and output_policy:
                fields.append(f"output_policy {output_policy}")
            unit_index = item.get("unit_index")
            if unit_index not in {"", None}:
                fields.append(f"unit {unit_index}")
            detector = str(item.get("detector") or "").strip()
            if detector:
                fields.append(f"detector {detector}")
            for key, label in (
                ("has_prompt_override", "prompt_override"),
                ("has_negative_prompt_override", "negative_prompt_override"),
            ):
                value_for_key = item.get(key)
                if value_for_key not in {"", None}:
                    fields.append(f"{label} {value_for_key}")
            for key, label in (
                ("steps", "steps"),
                ("cfg_scale", "cfg"),
                ("sampler", "sampler"),
                ("scheduler", "scheduler"),
            ):
                value_for_key = item.get(key)
                if value_for_key not in {"", None}:
                    fields.append(f"{label} {value_for_key}")
            if fields:
                summaries.append(" ".join(fields))
        return "; ".join(summaries)

    def _hires_metadata_summary(self, value: Any) -> str:
        if not isinstance(value, dict):
            return ""
        summaries: list[str] = []
        for pass_name, key in (("first", "first_pass"), ("second", "second_pass")):
            summary = self._hires_metadata_pass_summary(pass_name, value.get(key))
            if summary:
                summaries.append(summary)
        return "; ".join(summaries)

    def _hires_metadata_pass_summary(self, pass_name: str, value: Any) -> str:
        if not isinstance(value, dict):
            return ""
        fields: list[str] = [pass_name]
        for key, label in (("seed", "seed"), ("steps", "steps")):
            value_for_key = value.get(key)
            if value_for_key not in {"", None}:
                fields.append(f"{label} {value_for_key}")
        for key, label in (("sampler", "sampler"), ("scheduler", "scheduler")):
            encoded = _encoded_inline_text(value.get(key))
            if encoded:
                fields.append(f"{label} {encoded}")
        cfg_scale = value.get("cfg_scale")
        if cfg_scale not in {"", None}:
            fields.append(f"cfg {cfg_scale}")
        if pass_name == "first":
            width = int(value.get("width") or 0)
            height = int(value.get("height") or 0)
            if width or height:
                fields.append(f"size {width}x{height}")
        if pass_name == "second":
            denoise = value.get("denoising_strength")
            if denoise not in {"", None}:
                fields.append(f"denoise {denoise}")
            for key, label in (("prompt", "prompt"), ("negative_prompt", "negative")):
                encoded = _encoded_inline_text(value.get(key))
                if encoded:
                    fields.append(f"{label} {encoded}")
            resize_width = int(value.get("resize_width") or 0)
            resize_height = int(value.get("resize_height") or 0)
            if resize_width or resize_height:
                fields.append(f"resize {resize_width}x{resize_height}")
            scale = value.get("scale")
            if scale not in {"", None}:
                fields.append(f"scale {scale}")
            for key, label in (("upscaler", "upscaler"), ("noise_mode", "noise")):
                encoded = _encoded_inline_text(value.get(key))
                if encoded:
                    fields.append(f"{label} {encoded}")
        return " ".join(fields) if len(fields) > 1 else ""

    def _lora_summary(self, value: Any) -> str:
        if not isinstance(value, list):
            return ""
        summaries: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            name = self._display_model_value(item.get("name"))
            if not name:
                continue
            weight = item.get("weight")
            clip_weight = item.get("clip_weight")
            if clip_weight not in {"", None}:
                summaries.append(f"{name}:{weight}:{clip_weight}")
            elif weight not in {"", None}:
                summaries.append(f"{name}:{weight}")
            else:
                summaries.append(name)
        return ", ".join(summaries)

    def _lora_options_summary(self, value: Any) -> str:
        if not isinstance(value, list):
            return ""
        summaries: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            name = self._display_model_value(item.get("name"))
            if not name:
                continue
            fields: list[str] = []
            block_weight = _encoded_inline_text(item.get("block_weight"))
            if block_weight:
                fields.append(f"block_weight {block_weight}")
            block_weight_preset = _encoded_inline_text(item.get("block_weight_preset"))
            if block_weight_preset:
                fields.append(f"block_weight_preset {block_weight_preset}")
            hierarchy = str(item.get("hierarchy_mode") or "").strip()
            if hierarchy and hierarchy != "chain":
                encoded_hierarchy = _encoded_inline_text(hierarchy)
                if encoded_hierarchy:
                    fields.append(f"hierarchy {encoded_hierarchy}")
            if fields:
                summaries.append(f"{_encoded_inline_text(name)} {' '.join(fields)}")
        return "; ".join(summaries)

    def _embedding_summary(self, value: Any) -> str:
        if not isinstance(value, list):
            return ""
        embeddings = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(embeddings)

    def _positive_value(self, value: Any) -> Any:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric <= 0:
            return None
        return value

    def _non_default_value(self, value: Any, default: int) -> Any:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return None
        if numeric == default:
            return None
        return value

    def _non_default_float(self, value: Any, default: float) -> Any:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric == default:
            return None
        return value

    def _seed_resize_from(self, settings: dict[str, Any]) -> str | None:
        width = int(settings.get("seed_resize_from_w") or 0)
        height = int(settings.get("seed_resize_from_h") or 0)
        if not width and not height:
            return None
        return f"{width}x{height}"

    def _hires_resize(self, settings: dict[str, Any]) -> str | None:
        width = int(settings.get("hires_resize_width") or 0)
        height = int(settings.get("hires_resize_height") or 0)
        if not width and not height:
            return None
        return f"{width}x{height}"
