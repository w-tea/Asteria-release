from __future__ import annotations


def comfy_model_name(value: str, model_dir_name: str) -> str:
    normalized = value.replace("\\", "/")
    folded = normalized.casefold()
    if model_dir_name == "checkpoints" and "/merged/checkpoints/" in folded:
        marker = "/merged/checkpoints/"
        return normalized[folded.index(marker) + len(marker) :].replace("/", "\\")
    if model_dir_name == "loras" and "/merged/lora/" in folded:
        marker = "/merged/lora/"
        return normalized[folded.index(marker) + len(marker) :].replace("/", "\\")
    directory_aliases = [model_dir_name]
    if model_dir_name == "diffusion_models":
        directory_aliases.extend(["checkpoints", "unet"])
    for directory in directory_aliases:
        marker = f"/{directory}/"
        folded_marker = marker.casefold()
        if folded_marker in folded:
            return normalized[folded.index(folded_marker) + len(marker) :].replace("/", "\\")
        prefix = f"{directory}/"
        if folded.startswith(prefix.casefold()):
            return normalized[len(prefix) :].replace("/", "\\")
    return normalized.replace("/", "\\")
