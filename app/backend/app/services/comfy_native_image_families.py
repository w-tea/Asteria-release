"""Native Krea 2 / MOSS-Image inputs for the existing Comfy sampler tail."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.services.workflow_mapping import WorkflowMapping


NATIVE_IMAGE_DEFAULTS = {
    "krea2": {"steps": 8, "cfg_scale": 1.0, "sampler": "Euler"},
    "moss_image": {"steps": 25, "cfg_scale": 2.5, "sampler": "ER SDE"},
}
NATIVE_IMAGE_COMMON_DEFAULTS = {
    "width": 1024, "height": 1024, "scheduler": "Simple",
    "hires_enabled": False, "refiner_enabled": False, "clip_skip": 1,
}


def unsupported_native_image_settings(profile: str, settings: dict, *, loras=()) -> list[str]:
    if profile not in NATIVE_IMAGE_DEFAULTS:
        return []
    unsupported = []
    if settings.get("mode", "txt2img") != "txt2img":
        unsupported.append("mode")
    for field in ("hires_enabled", "refiner_enabled", "topaz_enabled", "restore_faces", "tiling", "embeddings",
                  "record_word_masks", "record_branch_step", "branch_source_id"):
        if settings.get(field):
            unsupported.append(field)
    for field in ("anima_lllite_units", "detailer_units"):
        if any(isinstance(unit, dict) and unit.get("enabled", True) for unit in settings.get(field, ())):
            unsupported.append(field)
    if settings.get("remaster", {}).get("enabled"):
        unsupported.append("remaster")
    if int(settings.get("clip_skip", 1)) != 1:
        unsupported.append("clip_skip")
    if float(settings.get("denoise", settings.get("denoising_strength", 1))) != 1:
        unsupported.append("denoise")
    if settings.get("lora_block_weight_runtime") == "requested" or any(
        (lora.get("block_weight") or lora.get("block_weight_preset"))
        and lora.get("hierarchy_mode") != "disabled" for lora in loras
    ):
        unsupported.append("lora_block_weight")
    return unsupported


def native_image_model_slot_checks(mapping: WorkflowMapping) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (slot, directory, mapping.class_type(node, ""), input_name)
        for slot, directory, node, input_name in (
            ("diffusion_model", "diffusion_models", "unet_loader", "unet_name"),
            ("text_encoder", "text_encoders", "clip_loader", "clip_name"),
            ("vae", "vae", "vae_loader", "vae_name"),
        )
    )


def native_image_required_nodes(mapping: WorkflowMapping, *, requires_lora: bool = False,
                                requires_control: bool = False) -> set[str]:
    return {value for key, value in mapping.class_types.items()
            if not key.startswith("generation_stage_") and (key != "lora_loader" or requires_lora)
            and (not key.startswith('control_') or requires_control)}


def native_image_runtime_contract_issues(mapping: WorkflowMapping, object_info: dict) -> list[dict[str, str]]:
    issues = []
    for key, contract in mapping.node_contracts.items():
        class_type = mapping.class_type(key, "")
        info = object_info.get(class_type, {})
        inputs = info.get("input", {}) if isinstance(info, dict) else {}
        declared = {**inputs.get("required", {}), **inputs.get("optional", {})}
        missing = [name for name in contract.required_inputs if name not in declared]
        for name, required_values in contract.required_input_values.items():
            descriptor = declared.get(name, ())
            values = descriptor[0] if isinstance(descriptor, (list, tuple)) and descriptor else ()
            if not isinstance(values, (list, tuple)) or not set(required_values).issubset(values):
                missing.append(name)
        outputs = info.get("output", ()) if isinstance(info, dict) else ()
        if any(index >= len(outputs) or outputs[index] != value for index, value in contract.outputs_by_index.items()):
            missing.append("output")
        if missing:
            issues.append({"field": "profile", "reason": "native_image_runtime_contract_unavailable",
                           "node": class_type, "inputs": ",".join(sorted(set(missing)))})
    return issues


@dataclass(frozen=True)
class NativeImageInputs:
    workflow: dict
    model: list
    positive: list
    negative: list
    latent: list
    vae: list


def build_native_image_inputs(
    request: dict, mapping: WorkflowMapping, model_name: Callable[[str, str], str],
    lora_loader: Callable[[dict, list, list, WorkflowMapping, dict], dict],
    control_preprocessor: Callable = lambda *args: None,
) -> NativeImageInputs:
    profile = request["profile"]
    settings = request["settings"]
    unsupported = unsupported_native_image_settings(profile, settings, loras=request.get("loras", ()))
    if request.get("embeddings"):
        unsupported.append("embeddings")
    if unsupported:
        raise ValueError("unsupported_native_image_settings:" + ",".join(unsupported))
    slots = request.get("model_slots", {})
    missing = [key for key in ("diffusion_model", "text_encoder", "vae") if not slots.get(key)]
    if missing:
        raise ValueError("native_image_model_slots_required:" + ",".join(missing))
    ids = {key: mapping.node(key, "") for key in
           ("unet_loader", "clip_loader", "vae_loader", "positive", "negative", "latent")}
    if not all(ids.values()) or len(set(ids.values())) != len(ids):
        raise ValueError("native_image_mapping_node_ids_invalid")
    model = [ids["unet_loader"], 0]
    clip = [ids["clip_loader"], 0]
    vae = [ids["vae_loader"], 0]
    positive = [ids["positive"], 0]
    negative = [ids["negative"], 0]
    workflow = {
        ids["unet_loader"]: {
            "class_type": mapping.class_type("unet_loader", "UNETLoader"),
            "inputs": {"unet_name": model_name(slots["diffusion_model"], "diffusion_models"),
                       "weight_dtype": "default"},
        },
        ids["clip_loader"]: {
            "class_type": mapping.class_type("clip_loader", "CLIPLoader"),
            "inputs": {"clip_name": model_name(slots["text_encoder"], "text_encoders"),
                       "type": profile},
        },
        ids["vae_loader"]: {
            "class_type": mapping.class_type("vae_loader", "VAELoader"),
            "inputs": {"vae_name": model_name(slots["vae"], "vae")},
        },
        ids["positive"]: {
            "class_type": mapping.class_type("positive", "CLIPTextEncode"),
            "inputs": {"clip": clip, "text": request["expanded_prompt"]},
        },
        ids["latent"]: {
            "class_type": mapping.class_type("latent", ""),
            "inputs": {"width": settings["width"], "height": settings["height"],
                       "batch_size": settings.get("batch_size", 1)},
        },
    }
    from app.services.native_image_control import append_native_controls, active_controls
    model, positive, negative = append_native_controls(
        workflow, request, mapping, model_name, model, vae, positive, negative, control_preprocessor)
    # Control adapters replace the base input projection before ordinary LoRAs.
    # Reserve every mapped node, including MOSS guidance, before adding the chain.
    next_id = max([int(value) for value in mapping.nodes.values()] + [int(value) for value in workflow]) + 1
    for lora in request.get("loras", ()):
        node_id = str(next_id)
        workflow[node_id] = lora_loader(lora, model, clip, mapping, settings)
        if profile == 'moss_image' and active_controls(settings):
            workflow[node_id]['class_type'] = mapping.class_type('control_lora_loader', '')
        model, clip = [node_id, 0], [node_id, 1]
        next_id += 1
    workflow[ids["positive"]]["inputs"]["clip"] = clip
    if profile == "krea2":
        workflow[ids["negative"]] = {
            "class_type": mapping.class_type("negative", "ConditioningZeroOut"),
            "inputs": {"conditioning": [ids['positive'], 0]},
        }
        settings["negative_prompt_runtime"] = "conditioning_zero_out"
    else:
        workflow[ids["negative"]] = {
            "class_type": mapping.class_type("negative", "CLIPTextEncode"),
            "inputs": {"clip": clip, "text": request.get("negative_prompt", "")},
        }
        slg_id = mapping.node("skip_layer_guidance", "")
        if not slg_id or slg_id in workflow:
            raise ValueError("native_image_mapping_node_ids_invalid")
        workflow[slg_id] = {
            "class_type": mapping.class_type("skip_layer_guidance", "SkipLayerGuidanceDiTSimple"),
            "inputs": {"model": model, "double_layers": "", "single_layers": "22",
                       "start_percent": 0.0, "end_percent": 0.5},
        }
        model = [slg_id, 0]
        settings["negative_prompt_runtime"] = "text_conditioning"
    return NativeImageInputs(workflow, model, positive, negative, [ids["latent"], 0], vae)
