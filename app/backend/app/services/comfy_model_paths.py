from __future__ import annotations


def comfy_model_name(value: str, model_dir_name: str) -> str:
    normalized = value.replace("\\", "/")
    if model_dir_name == "checkpoints" and "/merged/checkpoints/" in normalized:
        return normalized.split("/merged/checkpoints/", 1)[1].replace("/", "\\")
    if model_dir_name == "loras" and "/merged/lora/" in normalized:
        return normalized.split("/merged/lora/", 1)[1].replace("/", "\\")
    directory_aliases = [model_dir_name]
    if model_dir_name == "diffusion_models":
        directory_aliases.extend(["checkpoints", "unet"])
    for directory in directory_aliases:
        marker = f"/{directory}/"
        if marker in normalized:
            return normalized.split(marker, 1)[1].replace("/", "\\")
        if normalized.startswith(f"{directory}/"):
            return normalized[len(directory) + 1 :].replace("/", "\\")
    return normalized.replace("/", "\\")
