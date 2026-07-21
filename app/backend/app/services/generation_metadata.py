from hashlib import sha256
from pathlib import Path
import re


def record_hires_metadata(settings: dict) -> None:
    if not settings.get("hires_enabled"):
        return
    second_pass_steps = settings.get("hires_steps") or settings.get("steps")
    second_pass_sampler = settings.get("hires_sampler")
    if not second_pass_sampler or second_pass_sampler == "Use same sampler":
        second_pass_sampler = settings.get("sampler")
    second_pass_scheduler = settings.get("hires_scheduler")
    if not second_pass_scheduler or second_pass_scheduler == "Use same scheduler":
        second_pass_scheduler = settings.get("scheduler")
    second_pass = {
        "seed": settings.get("seed"),
        "steps": second_pass_steps,
        "sampler": second_pass_sampler,
        "scheduler": second_pass_scheduler,
        "cfg_scale": settings.get("hires_cfg_scale") or settings.get("cfg_scale"),
        "denoising_strength": settings.get("hires_denoising_strength"),
        "prompt": settings.get("hires_prompt") or None,
        "negative_prompt": settings.get("hires_negative_prompt") or None,
        "resize_width": settings.get("hires_resize_width") or 0,
        "resize_height": settings.get("hires_resize_height") or 0,
        "scale": settings.get("hires_scale"),
        "upscaler": settings.get("hires_upscaler"),
    }
    hires_checkpoint = str(settings.get("hires_checkpoint") or "").strip()
    if hires_checkpoint and hires_checkpoint != "Use same checkpoint":
        second_pass["checkpoint"] = hires_checkpoint
    if settings.get("hires_checkpoint_runtime"):
        second_pass["checkpoint_runtime"] = settings.get("hires_checkpoint_runtime")
    if settings.get("hires_checkpoint_kind"):
        second_pass["model_kind"] = settings.get("hires_checkpoint_kind")
    if settings.get("hires_noise_mode"):
        second_pass["noise_mode"] = settings.get("hires_noise_mode")
    settings["hires_metadata"] = {
        "first_pass": {
            "seed": settings.get("seed"),
            "steps": settings.get("steps"),
            "sampler": settings.get("sampler"),
            "scheduler": settings.get("scheduler"),
            "cfg_scale": settings.get("cfg_scale"),
            "width": settings.get("width"),
            "height": settings.get("height"),
        },
        "second_pass": second_pass,
    }


def record_controlnet_artifacts(settings: dict) -> None:
    artifacts: list[dict] = []
    for index, unit in enumerate(settings.get("controlnet_units", [])):
        artifact = {
            "unit_index": index,
            "preprocessor": unit.get("preprocessor", "none"),
            "source_image_path": unit.get("source_image_path") or unit.get("image_path"),
            "processed_image_path": unit.get("processed_image_path"),
            "uploaded_image_name": unit.get("uploaded_image_name"),
        }
        for key in (
            "model",
            "weight",
            "effective_strength",
            "control_mode",
            "control_mode_normalized",
            "control_mode_mapping",
            "control_mode_runtime",
            "control_mode_base_multiplier",
            "control_mode_uncond_multiplier",
            "preprocessor_runtime",
            "controlnet_union_type",
            "controlnet_union_type_node_id",
            "resize_mode",
            "pixel_perfect",
            "pixel_perfect_target_size",
        ):
            if unit.get(key) is not None:
                artifact[key] = unit.get(key)
        artifacts.append(artifact)
    if artifacts:
        settings["controlnet_artifacts"] = artifacts


def record_anima_lllite_artifacts(settings: dict) -> None:
    artifacts: list[dict] = []
    for index, unit in enumerate(settings.get("anima_lllite_units", [])):
        artifact = {
            "unit_index": index,
            "model": unit.get("model"),
            "source_image_path": unit.get("source_image_path") or unit.get("image_path"),
            "uploaded_image_name": unit.get("uploaded_image_name"),
            "strength": unit.get("strength"),
            "guidance_start": unit.get("guidance_start"),
            "guidance_end": unit.get("guidance_end"),
            "preserve_wrapper": unit.get("preserve_wrapper", True),
            "runtime": unit.get("runtime"),
            "regional_prompt_runtime": unit.get("regional_prompt_runtime"),
            "regional_prompts": unit.get("regional_prompts"),
        }
        for field in ("source_image_path", "mask_source_image_path"):
            path = unit.get(field)
            if path:
                digest = _file_sha256(path)
                if digest:
                    artifact[f"{field}_sha256"] = digest
        if unit.get("mask_source_image_path"):
            artifact["mask_source_image_path"] = unit.get("mask_source_image_path")
            artifact["uploaded_mask_name"] = unit.get("uploaded_mask_name")
        artifacts.append({key: value for key, value in artifact.items() if value not in (None, "")})
    if artifacts:
        settings["anima_lllite_artifacts"] = artifacts


def record_detailer_metadata(settings: dict) -> None:
    units = settings.get("detailer_units", [])
    if not units:
        return
    settings["detailer_metadata"] = {
        "execution_order": [
            {
                "unit_index": index,
                "detector": unit.get("detector", "face_yolo"),
                "has_prompt_override": bool(unit.get("prompt")),
                "has_negative_prompt_override": bool(unit.get("negative_prompt")),
                "steps": unit.get("steps") or settings.get("steps"),
                "cfg_scale": unit.get("cfg_scale") or settings.get("cfg_scale"),
                "sampler": unit.get("sampler") or settings.get("sampler"),
                "scheduler": unit.get("scheduler") or settings.get("scheduler"),
            }
            for index, unit in enumerate(units)
        ],
        "output_policy": "sequential_final_image",
    }


def metadata_tags(*, profile: str, prompt: str, model_slots: dict[str, str], embeddings: list[str], loras: list[dict] | None = None) -> list[str]:
    tags: list[str] = ["asteria", profile]
    for value in model_slots.values():
        if value:
            tags.append(Path(value).stem)
    tags.extend(Path(embedding).stem for embedding in embeddings if embedding)
    for lora in loras or []:
        name = lora.get("name") if isinstance(lora, dict) else None
        if name:
            tags.append(Path(str(name)).stem)
    for token in prompt.split(","):
        cleaned = token.strip().strip("()[]{}<>:\"'`")
        cleaned = re.sub(r":\s*-?\d+(?:\.\d+)?$", "", cleaned).strip()
        if 2 <= len(cleaned) <= 64:
            tags.append(cleaned)
    seen: set[str] = set()
    unique: list[str] = []
    for tag in tags:
        normalized = tag.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


def _file_sha256(path_value: object) -> str:
    try:
        path = Path(str(path_value))
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""
